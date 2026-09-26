"""Print the README result tables from results/ (one sub-folder per model under charades/ and qvhighlights/).

    python scripts/eval/collect_results.py --results_dir results

Runs the Charades scorer for the two reported protocols and reads the QVHighlights evaluation.json.
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

ORDER = ["TEMPURA-Qwen2.5-VL-3B", "TEMPURA-Qwen2.5-VL-7B", "TEMPURA-InternVL3-2B", "TEMPURA-InternVL3-8B"]


def charades(inference_dir, windows, reduce):
    out = os.path.join(inference_dir, f"evaluation_{windows}_{reduce}.json")
    subprocess.run([sys.executable, "-m", "src.evaluate.evaluate_charades", "--inference_dir", inference_dir,
                    "--windows", windows, "--reduce", reduce, "--output", out], check=True, capture_output=True)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()
    ch, qv = os.path.join(args.results_dir, "charades"), os.path.join(args.results_dir, "qvhighlights")
    models = [m for m in ORDER if os.path.isdir(os.path.join(ch, m)) or os.path.isdir(os.path.join(qv, m))]
    models += sorted(m for m in set(os.listdir(ch) if os.path.isdir(ch) else []) | set(os.listdir(qv) if os.path.isdir(qv) else []) if m not in models)

    def f(x):
        return f"{x:.1f}"

    lines = ["**Charades-STA, any-window protocol** (VTG stage; a query counts when any returned window reaches the IoU threshold)", "",
             "| Model | mIoU | R@0.3 | R@0.5 | R@0.7 | windows / query |", "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        d = os.path.join(ch, m, "vtg_refined")
        if not os.path.isdir(d):
            continue
        r = charades(d, "vtg_first", "max")
        nw = statistics.mean(len(json.load(open(p))["vtg_windows"]) for p in glob.glob(os.path.join(d, "result_*.json")))
        lines.append(f"| {m} | {f(100*r['mIoU'])} | {f(100*r['R@1,IoU=0.3'])} | {f(100*r['R@1,IoU=0.5'])} | {f(100*r['R@1,IoU=0.7'])} | {nw:.1f} |")
    lines += ["", "**Charades-STA, single-window protocol** (final caption-refined answer; only its first window is scored)", "",
              "| Model | mIoU | R@0.3 | R@0.5 | R@0.7 |", "|---|---:|---:|---:|---:|"]
    for m in models:
        d = os.path.join(ch, m, "vtg_refined")
        if not os.path.isdir(d):
            continue
        r = charades(d, "refined_first", "first")
        lines.append(f"| {m} | {f(100*r['mIoU'])} | {f(100*r['R@1,IoU=0.3'])} | {f(100*r['R@1,IoU=0.5'])} | {f(100*r['R@1,IoU=0.7'])} |")
    lines += ["", "**QVHighlights highlight detection**", "",
              "| Model | mAP | HIT@1 | mAP (Moment-DETR, Very Good) | HIT@1 (Moment-DETR, Very Good) |", "|---|---:|---:|---:|---:|"]
    for m in models:
        p = os.path.join(qv, m, "vhd", "evaluation.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        a, b = r["paper_protocol"], r["HL-min-VeryGood"]
        lines.append(f"| {m} | {f(a['mAP'])} | {f(a['HIT@1'])} | {f(b['mAP'])} | {f(b['HIT@1'])} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
