"""TEMPURA temporal grounding pipeline: VTG -> dense video captioning -> caption-guided refinement.

For each query the model (1) grounds the query directly (VTG), (2) densely captions the video once
(DVC, cached per video) and (3) refines the VTG window with a text-only prompt that sees the dense
caption. All three stages use the same checkpoint.

    python -m src.inference.inference_vtg_refined --config configs/eval/charades_qwen.json --model_path <ckpt>

Writes ``<output_dir>/vtg_refined/result_<qid>.json`` with ``vtg_windows`` (stage 1),
``refined_windows`` (stage 3) and ``relevant_windows`` = the windows selected by ``--final_windows``:

* ``vtg_first`` (default, paper protocol): the VTG windows; the refined window is used only when VTG returned nothing.
* ``refined``: the refined window (falls back to the VTG window when refinement returned nothing).
* ``vtg``: the plain VTG window (the DVC/refinement stages are still run and stored).

Stage-1 and DVC results are cached in the sibling folders ``<output_dir>/vtg/`` and ``<output_dir>/dvc/``.
"""
import argparse
import copy
import os

from tqdm import tqdm

from src.inference import prompts as P
from src.inference.inference_base import parse_args
from src.inference.inference_dvc import DVCInference
from src.inference.inference_vtg import VTGInference, vtg_parser
from src.inference.parsers import parse_windows


def ensure_min_duration(windows, min_duration: float):
    out = []
    for s, e in windows:
        if e - s < min_duration:
            ext = min_duration - (e - s)
            s_ext = min(ext / 2, s)
            s, e = s - s_ext, e + (ext - s_ext)
        out.append([round(s, 2), round(e, 2)])
    return out


def _share_model(parent, cls, args):
    """Build a task helper that reuses the parent's loaded model but writes to its own task folder."""
    helper = cls.__new__(cls)
    helper.__dict__.update(parent.__dict__)
    helper.args = args
    helper.task = cls.task
    helper.output_dir = os.path.join(args.output_dir, cls.task)
    os.makedirs(helper.output_dir, exist_ok=True)
    return helper


class RefinedVTGInference(VTGInference):
    task = "vtg_refined"

    def _task_init(self):
        dvc_args = copy.copy(self.args)
        dvc_args.max_new_tokens = self.args.max_new_tokens_dvc
        self.vtg = _share_model(self, VTGInference, self.args)
        self.dvc = _share_model(self, DVCInference, dvc_args)

    def refine(self, item: dict, vtg_windows, dense_caption: str):
        windows_str = "[" + ", ".join(f"[{s:.1f}, {e:.1f}]" for s, e in vtg_windows) + "]"
        prompt = P.VTG_REFINE.format(dense_caption=dense_caption, query=item["query"], windows=windows_str,
                                     min_window_duration=self.args.min_window_duration)
        response = self.generate_from_text(prompt, self.args.max_new_tokens_refine)
        return response, ensure_min_duration(parse_windows(response), self.args.min_window_duration)

    def select_windows(self, vtg_windows, refined_windows):
        mode = self.args.final_windows
        if mode == "vtg":
            return vtg_windows
        if mode == "refined":
            return refined_windows or vtg_windows
        return vtg_windows or refined_windows  # vtg_first

    def run(self):
        stats = {"vtg_empty": 0, "refined_empty": 0}
        for item in tqdm(self.test_data, desc="VTG+DVC+refine"):
            qid = item["qid"]
            cached = self.load_result(f"result_{qid}")
            if cached is not None:
                if cached.get("final_windows_mode") != self.args.final_windows:  # re-select without re-running the model
                    cached["final_windows_mode"] = self.args.final_windows
                    cached["relevant_windows"] = self.select_windows(cached.get("vtg_windows", []), cached.get("refined_windows", []))
                    self.save_result(f"result_{qid}", cached)
                continue
            vtg_result = self.vtg.ground(item)
            vtg_windows = ensure_min_duration(vtg_result.get("relevant_windows", []), self.args.min_window_duration)
            dense_caption = self.dvc.caption_video(item["vid"])
            if dense_caption:
                refine_response, refined_windows = self.refine(item, vtg_windows, dense_caption)
            else:
                refine_response, refined_windows = "", []
            stats["vtg_empty"] += not vtg_windows
            stats["refined_empty"] += not refined_windows
            self.save_result(f"result_{qid}", {
                "qid": qid, "vid": item["vid"], "query": item["query"],
                "vtg_response": vtg_result.get("response", ""), "vtg_windows": vtg_windows,
                "refine_response": refine_response, "refined_windows": refined_windows,
                "final_windows_mode": self.args.final_windows,
                "relevant_windows": self.select_windows(vtg_windows, refined_windows),
            })
        print(f"[vtg_refined] done: {len(self.test_data)} queries -> {self.output_dir}; {stats}")


def refined_parser() -> argparse.ArgumentParser:
    parser = vtg_parser()
    parser.description = "TEMPURA temporal grounding: VTG + dense captioning + caption-guided refinement"
    parser.add_argument("--max_new_tokens_dvc", type=int, default=2048, help="generation budget for the dense caption")
    parser.add_argument("--max_new_tokens_refine", type=int, default=1024, help="generation budget for the refinement step")
    parser.add_argument("--min_window_duration", type=float, default=4.0, help="windows shorter than this are expanded symmetrically")
    parser.add_argument("--final_windows", type=str, default="vtg_first", choices=["vtg_first", "refined", "vtg"])
    return parser


if __name__ == "__main__":
    RefinedVTGInference(parse_args(refined_parser())).run()
