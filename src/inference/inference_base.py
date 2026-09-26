"""Common driver for the benchmark inference tasks (DVC, VTG, refined VTG, VHD).

Every task reads a ground-truth file (``.json`` list or ``.jsonl``) whose items carry at least
``vid`` (video id = ``<videos_dir>/<vid>.mp4``) and, for query tasks, ``qid`` and ``query``.
Results are written one file per item under ``<output_dir>/<task>/`` so runs can resume.
"""
import argparse
import datetime
import json
import os
from typing import List, Optional

import torch

from src.inference import prompts as P
from src.inference.config_utils import parse_args_with_config, save_config, setup_parser_with_config_support
from src.inference.model_utils import build_messages, generate, load_model
from src.inference.video_utils import load_video_for_model


def load_gt(path: str) -> List[dict]:
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def resolve_video_path(videos_dir: str, vid: str) -> str:
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov", ""):
        p = os.path.join(videos_dir, f"{vid}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(videos_dir, f"{vid}.mp4")


class InferBase:
    task: str = "base"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.verbose = args.verbose
        self.output_dir = os.path.join(args.output_dir, self.task)
        os.makedirs(self.output_dir, exist_ok=True)
        self.processor, self.model, self.model_type = load_model(
            args.model_path, model_type=(args.model_type or None), device=args.device, attn_implementation=args.attn_implementation)
        self.test_data = load_gt(args.gt_json_file)
        if args.max_items > 0:
            self.test_data = self.test_data[: args.max_items]
        print(f"[tempura] {self.task}: {len(self.test_data)} items from {args.gt_json_file}; results -> {self.output_dir}")
        self._save_run_config()
        self._task_init()

    def _task_init(self):
        pass

    def _save_run_config(self):
        cfg = {k: v for k, v in vars(self.args).items() if not callable(v)}
        cfg["model_type"] = self.model_type
        cfg["timestamp"] = str(datetime.datetime.now())
        with open(os.path.join(self.output_dir, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    # ---------------------------------------------------------------- generation
    def load_frames(self, video_path: str):
        return load_video_for_model(video_path, sample_fps=self.args.fps, add_timestamp=self.args.add_timestamp,
                                    max_num_frames=self.args.max_num_frames)

    def format_prompt(self, template: str, query: Optional[str] = None, timestamps=None) -> str:
        prompt = template.replace("<your_query>", query) if query is not None else template
        if self.args.add_time_instruction and timestamps:
            prompt = P.time_instruction(self.args.fps, timestamps) + prompt
        return prompt

    @torch.inference_mode()
    def generate_from_video(self, video_path: str, template: str, query: Optional[str], max_new_tokens: int):
        """Sample frames, build the prompt and decode. Returns ``(response, timestamps)``."""
        frames, timestamps = self.load_frames(video_path)
        prompt = self.format_prompt(template, query, timestamps)
        messages = build_messages(frames, prompt, self.model_type, self.args.min_pixels, self.args.max_pixels)
        response = generate(self.processor, self.model, self.model_type, messages, max_new_tokens)
        if self.verbose:
            print(f"\n[prompt] {prompt}\n[response] {response}")
        return response, timestamps

    @torch.inference_mode()
    def generate_from_text(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        response = generate(self.processor, self.model, self.model_type, messages, max_new_tokens)
        if self.verbose:
            print(f"\n[prompt] {prompt}\n[response] {response}")
        return response

    # ---------------------------------------------------------------- io
    def result_path(self, name: str) -> str:
        return os.path.join(self.output_dir, f"{name}.json")

    def load_result(self, name: str) -> Optional[dict]:
        p = self.result_path(name)
        if os.path.exists(p) and not self.args.overwrite:
            try:
                with open(p) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None

    def save_result(self, name: str, result: dict):
        with open(self.result_path(name), "w") as f:
            json.dump(result, f, indent=2)

    def run(self):
        raise NotImplementedError


def common_parser(description: str, default_max_new_tokens: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_path", type=str, default="andaba/TEMPURA-Qwen2.5-VL-3B", help="HF id or local checkpoint dir")
    parser.add_argument("--model_type", type=str, default="", choices=["", "qwenvl", "internvl"], help="auto-detected from config.json when empty")
    parser.add_argument("--videos_dir", type=str, default="data/eval/Charades_v1_480")
    parser.add_argument("--gt_json_file", type=str, default="data/eval/charades_sta_test_tvr_format.json")
    parser.add_argument("--output_dir", type=str, default="results/charades", help="results are written to <output_dir>/<task>/")
    parser.add_argument("--fps", type=float, default=1.0, help="frames sampled per second (Qwen: 1, InternVL: 0.5)")
    parser.add_argument("--max_num_frames", type=int, default=-1, help="cap on sampled frames (-1 = no cap)")
    parser.add_argument("--min_pixels", type=int, default=336 * 336, help="Qwen only: min pixels per frame")
    parser.add_argument("--max_pixels", type=int, default=336 * 336, help="Qwen only: max pixels per frame")
    parser.add_argument("--add_timestamp", action=argparse.BooleanOptionalAction, default=True, help="draw the timestamp on each frame")
    parser.add_argument("--add_time_instruction", action=argparse.BooleanOptionalAction, default=False, help="prepend the frame timestamps to the prompt")
    parser.add_argument("--max_new_tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("--max_items", type=int, default=-1, help="process only the first N items (-1 = all)")
    parser.add_argument("--overwrite", action="store_true", help="recompute items that already have a result file")
    parser.add_argument("--attn_implementation", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_config", type=str, default="", help="write the resolved arguments to this JSON file")
    return setup_parser_with_config_support(parser)


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parse_args_with_config(parser)
    if args.save_config:
        save_config(args, args.save_config)
    return args
