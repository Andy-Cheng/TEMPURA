"""Video highlight detection (VHD) on QVHighlights: per-clip saliency scores for a query.

    python -m src.inference.inference_vhd --config configs/eval/qvhighlights_qwen.json --model_path <ckpt>

Writes ``<output_dir>/vhd/result_<qid>.json`` =
``{"qid", "vid", "query", "response", "timestamps", "scores", "pred_saliency_scores"}`` where
``pred_saliency_scores`` has one entry per 2-second clip of the video (0 = not a highlight).

``--pipeline dvc_refine`` additionally densely captions the video (cached under ``<output_dir>/dvc/``)
and refines the answer with a text-only prompt that sees the caption, mirroring the Charades pipeline.
"""
import os

from tqdm import tqdm

from src.inference import prompts as P
from src.inference.inference_base import InferBase, common_parser, parse_args, resolve_video_path
from src.inference.inference_dvc import DVCInference
from src.inference.parsers import highlights_to_clip_scores, parse_highlights


class VHDInference(InferBase):
    task = "vhd"

    def _task_init(self):
        self.dvc = None
        if self.args.pipeline == "dvc_refine":
            self.dvc = DVCInference.__new__(DVCInference)
            self.dvc.__dict__.update(self.__dict__)
            self.dvc.task = DVCInference.task
            self.dvc.output_dir = os.path.join(self.args.output_dir, DVCInference.task)
            os.makedirs(self.dvc.output_dir, exist_ok=True)

    def detect(self, item: dict) -> dict:
        qid = item["qid"]
        cached = self.load_result(f"result_{qid}")
        if cached is not None:
            return cached
        duration = float(item.get("duration", 150.0))
        video_path = resolve_video_path(self.args.videos_dir, item["vid"])
        result = {"qid": qid, "vid": item["vid"], "query": item["query"], "duration": duration,
                  "response": "", "timestamps": [], "scores": [], "pred_saliency_scores": []}
        if not os.path.exists(video_path):
            print(f"[vhd] missing video {video_path}")
            return result
        response, _ = self.generate_from_video(video_path, P.PROMPTS[self.args.prompt_type], item["query"], self.args.max_new_tokens)
        result["direct_response"] = response
        if self.dvc is not None:
            dense_caption = self.dvc.caption_video(item["vid"])
            if dense_caption:
                prompt = P.VHD_REFINE.format(duration=duration, dense_caption=dense_caption, query=item["query"], initial_answer=response)
                refined = self.generate_from_text(prompt, self.args.max_new_tokens)
                result["refine_response"] = refined
                if parse_highlights(refined)[0]:  # keep the direct answer if the refinement is unparsable
                    response = refined
        timestamps, scores = parse_highlights(response)
        result.update(response=response, timestamps=timestamps, scores=scores,
                      pred_saliency_scores=highlights_to_clip_scores(timestamps, scores, duration, self.args.clip_length).tolist())
        self.save_result(f"result_{qid}", result)
        return result

    def run(self):
        for item in tqdm(sorted(self.test_data, key=lambda x: int(x["qid"])), desc="VHD"):
            self.detect(item)
        print(f"[vhd] done: {len(self.test_data)} queries -> {self.output_dir}")


def vhd_parser():
    parser = common_parser("TEMPURA video highlight detection", default_max_new_tokens=256)
    parser.set_defaults(videos_dir="data/eval/qvhighlights/videos", gt_json_file="data/eval/highlight_val_release.jsonl",
                        output_dir="results/qvhighlights")
    parser.add_argument("--prompt_type", type=str, default="vhd_ft", choices=["vhd_ft"])
    parser.add_argument("--pipeline", type=str, default="direct", choices=["direct", "dvc_refine"],
                        help="direct = one highlight-detection pass; dvc_refine = also caption the video and refine with the caption")
    parser.add_argument("--clip_length", type=float, default=2.0, help="QVHighlights clips are 2 seconds")
    return parser


if __name__ == "__main__":
    VHDInference(parse_args(vhd_parser())).run()
