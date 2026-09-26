"""Re-derive the window fields of Charades result files from the stored model responses.

    python scripts/eval/reparse_charades.py results/charades/TEMPURA-Qwen2.5-VL-3B/vtg_refined [--min_window_duration 4]

Useful after a parser change: no model call is needed because every result file keeps the raw
``vtg_response`` and ``refine_response`` texts.
"""
import argparse
import glob
import json
import os

from src.inference.inference_vtg_refined import ensure_min_duration
from src.inference.parsers import parse_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inference_dir")
    ap.add_argument("--min_window_duration", type=float, default=4.0)
    ap.add_argument("--final_windows", default="vtg_first", choices=["vtg_first", "refined", "vtg"])
    args = ap.parse_args()
    n = 0
    for f in glob.glob(os.path.join(args.inference_dir, "result_*.json")):
        d = json.load(open(f))
        vtg = ensure_min_duration(parse_windows(d.get("vtg_response", "")), args.min_window_duration)
        ref = ensure_min_duration(parse_windows(d.get("refine_response", "")), args.min_window_duration)
        d["vtg_windows"], d["refined_windows"], d["final_windows_mode"] = vtg, ref, args.final_windows
        d["relevant_windows"] = vtg if args.final_windows == "vtg" else (ref or vtg) if args.final_windows == "refined" else (vtg or ref)
        json.dump(d, open(f, "w"), indent=2)
        n += 1
    print(f"re-parsed {n} result files in {args.inference_dir}")


if __name__ == "__main__":
    main()
