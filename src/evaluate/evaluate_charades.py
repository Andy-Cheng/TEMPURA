"""Charades-STA temporal grounding metrics: R@1 at IoU {0.3, 0.5, 0.7} and mIoU.

    python -m src.evaluate.evaluate_charades --inference_dir results/charades/vtg_refined \
        --gt_json_path data/eval/charades_sta_test_tvr_format.json

Predictions and ground truth are joined by ``qid``. Paper protocol (default): the windows of the direct
VTG stage are used, falling back to the caption-refined windows when VTG returned nothing
(``--windows vtg_first``), and a query counts as correct when any of its predicted windows reaches the
IoU threshold (``--reduce max``; the prompt asks for every occurrence of the query). ``--windows`` can also
score ``vtg_windows``, ``refined_windows`` or the stored ``relevant_windows``, and ``--reduce first`` scores
only the first predicted window (strict R@1). ``--protocol timelens`` applies the TimeChat / TimeLens rule
instead: the first timestamp pair of the raw answer, no minimum-window expansion, misses for inverted pairs.
"""
import argparse
import json
import os
from typing import Dict, List

from src.inference.parsers import parse_windows


def load_gt(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()] if path.endswith(".jsonl") else json.load(f)


def load_predictions(inference_dir: str) -> Dict[int, dict]:
    preds = {}
    for fn in os.listdir(inference_dir):
        if fn.startswith("result_") and fn.endswith(".json"):
            with open(os.path.join(inference_dir, fn)) as f:
                d = json.load(f)
            preds[int(d["qid"])] = d
    return preds


def iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def clean_windows(windows) -> List[List[float]]:
    out = []
    for w in windows or []:
        if isinstance(w, (list, tuple)) and len(w) == 2:
            try:
                s, e = float(w[0]), float(w[1])
            except (TypeError, ValueError):
                continue
            if e > s:
                out.append([s, e])
    return out


def timelens_first_pair(p: dict, windows_key: str) -> List[List[float]]:
    """TimeChat / TimeLens convention: the first timestamp pair of the raw answer, no expansion; an inverted
    or zero-length pair counts as a miss."""
    field = "refine_response" if windows_key.startswith("refined") else "vtg_response"
    w = parse_windows(p.get(field, ""))
    if not w or w[0][1] <= w[0][0]:
        return []
    return w[:1]


def evaluate(gt: List[dict], preds: Dict[int, dict], windows_key: str = "vtg_first", reduce: str = "max",
             thresholds=(0.3, 0.5, 0.7), protocol: str = "tempura") -> dict:
    ious, empty, missing = [], 0, 0
    for item in gt:
        p = preds.get(int(item["qid"]))
        if p is None:
            missing += 1
            ious.append(0.0)
            continue
        if protocol == "timelens":
            pw = timelens_first_pair(p, windows_key)
        elif windows_key == "vtg_first":
            pw = clean_windows(p.get("vtg_windows")) or clean_windows(p.get("refined_windows"))
        elif windows_key == "refined_first":
            pw = clean_windows(p.get("refined_windows")) or clean_windows(p.get("vtg_windows"))
        else:
            pw = clean_windows(p.get(windows_key))
        gw = clean_windows(item["relevant_windows"])
        if not pw:
            empty += 1
            ious.append(0.0)
            continue
        cand = pw if reduce == "max" else pw[:1]
        ious.append(max(iou(c, g) for c in cand for g in gw) if gw else 0.0)
    n = len(gt)
    res = {"num_samples": n, "num_missing_predictions": missing, "num_empty_predictions": empty,
           "windows": windows_key, "reduce": reduce, "protocol": protocol, "mIoU": sum(ious) / n if n else 0.0}
    for t in thresholds:
        res[f"R@1,IoU={t}"] = sum(i >= t for i in ious) / n if n else 0.0
    return res


def main():
    ap = argparse.ArgumentParser(description="Evaluate Charades-STA temporal grounding results")
    ap.add_argument("--gt_json_path", default="data/eval/charades_sta_test_tvr_format.json")
    ap.add_argument("--inference_dir", required=True, help="folder with result_<qid>.json files")
    ap.add_argument("--windows", default="vtg_first", choices=["vtg_first", "refined_first", "vtg_windows", "refined_windows", "relevant_windows"],
                    help="which predicted windows to score (vtg_first = paper protocol)")
    ap.add_argument("--reduce", default="max", choices=["max", "first"], help="max = any predicted window (paper protocol); first = strict top-1")
    ap.add_argument("--protocol", default="tempura", choices=["tempura", "timelens"],
                    help="timelens = TimeChat/TimeLens rule: first pair of the raw answer, no minimum-window expansion")
    ap.add_argument("--output", default="", help="metrics JSON path (default: <inference_dir>/evaluation_<windows>_<reduce>.json)")
    args = ap.parse_args()

    gt, preds = load_gt(args.gt_json_path), load_predictions(args.inference_dir)
    res = evaluate(gt, preds, args.windows, args.reduce, protocol=args.protocol)
    print(f"Charades-STA ({args.windows}, reduce={args.reduce}, protocol={args.protocol}): {res['num_samples']} queries, "
          f"{res['num_missing_predictions']} missing, {res['num_empty_predictions']} empty predictions")
    print(f"  mIoU        : {100 * res['mIoU']:.2f}")
    for k, v in res.items():
        if k.startswith("R@1"):
            print(f"  {k:<12}: {100 * v:.2f}")
    if res["num_missing_predictions"]:
        # Partial run (e.g. --max_items): also score only the queries that have a prediction.
        sub = evaluate([g for g in gt if int(g["qid"]) in preds], preds, args.windows, args.reduce, protocol=args.protocol)
        res["subset_with_predictions"] = sub
        print(f"  [subset of {sub['num_samples']} predicted queries] mIoU {100 * sub['mIoU']:.2f}  "
              + "  ".join(f"{k} {100 * v:.2f}" for k, v in sub.items() if k.startswith("R@1")))
    out = args.output or os.path.join(args.inference_dir, f"evaluation_{args.windows}_{args.reduce}" + ("_timelens" if args.protocol == "timelens" else "") + ".json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
