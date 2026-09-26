"""Minimal example: dense video captioning (and masked event prediction) with a TEMPURA checkpoint.

    python -m src.inference.dense_video_captioning_demo --video test_videos_demo/hotdog.mp4
    python -m src.inference.dense_video_captioning_demo --video test_videos_demo/hotdog.mp4 --task mep --mask 5 10

Works with any released checkpoint (Qwen2.5-VL or InternVL3 based); the family is auto-detected.
"""
import argparse

from src.inference import prompts as P
from src.inference.model_utils import build_messages, generate, load_model
from src.inference.parsers import parse_dvc_events
from src.inference.video_utils import load_video_for_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="andaba/TEMPURA-Qwen2.5-VL-3B")
    ap.add_argument("--video", default="test_videos_demo/hotdog.mp4", help="video file or a folder of 1 fps frames")
    ap.add_argument("--task", default="dvc", choices=["dvc", "mep"])
    ap.add_argument("--mask", type=float, nargs=2, metavar=("START", "END"), default=(5.0, 10.0), help="MEP: masked segment in seconds")
    ap.add_argument("--fps", type=float, default=1.0, help="1 for Qwen checkpoints, 0.5 for InternVL checkpoints")
    ap.add_argument("--min_pixels", type=int, default=336 * 336)
    ap.add_argument("--max_pixels", type=int, default=336 * 336)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--no_timestamp", action="store_true", help="do not draw timestamps on the frames")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    processor, model, model_type = load_model(args.model_path, device=args.device)
    frames, timestamps = load_video_for_model(args.video, args.fps, add_timestamp=not args.no_timestamp)
    if args.task == "mep":
        s, e = args.mask
        # Masked event prediction: grey out the frames inside the masked span and ask the model to infer the event.
        for f, t in zip(frames, timestamps):
            if s <= t <= e:
                f.paste((128, 128, 128), (0, 0, *f.size))
        prompt = P.MEP.replace("<start>", f"{s:.1f}").replace("<end>", f"{e:.1f}")
    else:
        prompt = P.DVC
    messages = build_messages(frames, prompt, model_type, args.min_pixels, args.max_pixels)
    response = generate(processor, model, model_type, messages, args.max_new_tokens)
    print(f"\n=== {args.task.upper()} on {args.video} ({len(frames)} frames @ {args.fps} fps) ===\n{response}\n")
    if args.task == "dvc":
        for start, end, desc in parse_dvc_events(response):
            print(f"[{start:6.1f}s - {end:6.1f}s] {desc}")


if __name__ == "__main__":
    main()
