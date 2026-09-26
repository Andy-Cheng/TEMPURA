"""Video temporal grounding (VTG) on Charades-STA style data: one time window per query.

    python -m src.inference.inference_vtg --config configs/eval/charades_qwen.json --model_path <ckpt>

Writes ``<output_dir>/vtg/result_<qid>.json`` = ``{"qid", "vid", "query", "response", "relevant_windows"}``.
"""
import os

from tqdm import tqdm

from src.inference import prompts as P
from src.inference.inference_base import InferBase, common_parser, parse_args, resolve_video_path
from src.inference.parsers import parse_windows


class VTGInference(InferBase):
    task = "vtg"

    def ground(self, item: dict) -> dict:
        qid = item["qid"]
        cached = self.load_result(f"result_{qid}")
        if cached is not None:
            return cached
        video_path = resolve_video_path(self.args.videos_dir, item["vid"])
        result = {"qid": qid, "vid": item["vid"], "query": item["query"], "response": "", "relevant_windows": []}
        if not os.path.exists(video_path):
            print(f"[vtg] missing video {video_path}")
            return result
        template = P.PROMPTS[self.args.prompt_type]
        response, _ = self.generate_from_video(video_path, template, item["query"], self.args.max_new_tokens)
        result.update(response=response, relevant_windows=parse_windows(response))
        self.save_result(f"result_{qid}", result)
        return result

    def run(self):
        for item in tqdm(self.test_data, desc="VTG"):
            self.ground(item)
        print(f"[vtg] done: {len(self.test_data)} queries -> {self.output_dir}")


def vtg_parser():
    parser = common_parser("TEMPURA video temporal grounding", default_max_new_tokens=512)
    parser.add_argument("--prompt_type", type=str, default="vtg", choices=["vtg", "vtg_ft"],
                        help="vtg = zero-shot JSON prompt; vtg_ft = short prompt for Charades-finetuned checkpoints")
    return parser


if __name__ == "__main__":
    VTGInference(parse_args(vtg_parser())).run()
