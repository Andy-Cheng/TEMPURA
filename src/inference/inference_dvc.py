"""Dense video captioning (DVC): one timestamped event list per video.

    python -m src.inference.inference_dvc --config configs/eval/charades_qwen.json --model_path <ckpt>

Writes ``<output_dir>/dvc/video_<vid>.json`` = ``{"vid", "response", "events": [[start, end, text], ...]}``.
"""
import os

from tqdm import tqdm

from src.inference import prompts as P
from src.inference.inference_base import InferBase, common_parser, parse_args, resolve_video_path
from src.inference.parsers import parse_dvc_events


class DVCInference(InferBase):
    task = "dvc"

    def caption_video(self, vid: str) -> str:
        """Return the dense caption of ``vid`` (cached on disk); ``""`` if the video is missing."""
        cached = self.load_result(f"video_{vid}")
        if cached is not None:
            return cached.get("response", "")
        video_path = resolve_video_path(self.args.videos_dir, vid)
        if not os.path.exists(video_path):
            print(f"[dvc] missing video {video_path}")
            return ""
        response, _ = self.generate_from_video(video_path, P.DVC, None, self.args.max_new_tokens)
        self.save_result(f"video_{vid}", {"vid": vid, "response": response, "events": parse_dvc_events(response)})
        return response

    def run(self):
        vids = list(dict.fromkeys(item["vid"] for item in self.test_data))  # unique, keep order
        for vid in tqdm(vids, desc="DVC"):
            self.caption_video(vid)
        print(f"[dvc] done: {len(vids)} videos -> {self.output_dir}")


if __name__ == "__main__":
    args = parse_args(common_parser("TEMPURA dense video captioning", default_max_new_tokens=2048))
    DVCInference(args).run()
