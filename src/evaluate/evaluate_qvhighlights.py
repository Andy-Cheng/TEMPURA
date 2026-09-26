"""QVHighlights highlight-detection metrics (mAP and HIT@1).

    python -m src.evaluate.evaluate_qvhighlights --inference_dir results/qvhighlights/vhd \
        --gt_json_path data/eval/highlight_val_release.jsonl

Paper protocol (reported first): every clip in ``relevant_clip_ids`` is a positive; mAP is the average
precision of the predicted per-clip saliency vector against these positives, HIT@1 checks whether the
top-scored clip is one of them. For reference the Moment-DETR ``eval_highlight`` variants are also printed:
positives are clips whose annotator saliency (0-4, three annotators) reaches a threshold ("Fair" >= 2,
"Good" >= 3, "VeryGood" >= 4), with per-annotator AP averaged.

Predictions are ``result_<qid>.json`` files written by ``src.inference.inference_vhd``; the ``response`` text is
re-parsed here (``--parse strict`` = paper protocol, ``--parse tolerant`` also keeps truncated answers).
"""
import argparse
import json
import os
import re
import warnings
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score

from src.inference.parsers import highlights_to_clip_scores, parse_highlights

LEVELS = {"Fair": 2, "Good": 3, "VeryGood": 4}


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


def gt_scores_full(item: dict, clip_length: float = 2.0) -> np.ndarray:
    """(#clips, 3) annotator saliency scores over the whole video (0 outside the relevant clips)."""
    num_clips = int(item["duration"] / clip_length)
    full = np.zeros((num_clips, 3))
    ids = np.array(item["relevant_clip_ids"], dtype=int)
    scores = np.array(item["saliency_scores"], dtype=float)
    keep = ids < num_clips
    full[ids[keep]] = scores[keep]
    return full


def pred_scores_for(item: dict, pred: dict, num_clips: int, clip_length: float, parse: str = "strict") -> np.ndarray:
    """Per-clip saliency from the response text. ``strict`` (paper protocol) ignores responses whose score
    list is missing; ``tolerant`` keeps their timestamps with a default score of 1."""
    ts, sc = parse_highlights(pred.get("response", ""), require_scores=(parse == "strict"))
    s = highlights_to_clip_scores(ts, sc, float(item["duration"]), clip_length)
    s = np.asarray(s, dtype=float)
    if len(s) < num_clips:
        s = np.pad(s, (0, num_clips - len(s)))
    return s[:num_clips]


def ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(average_precision_score(y_true, y_score))


def paper_saliency_vector(response: str, duration: float, clip_length: float, parse: str) -> np.ndarray:
    """Per-clip saliency exactly as in the original evaluation script.

    The response must contain ``"... in the <timestamps> seconds. Their saliency scores are <scores>"``;
    ``<timestamps>`` is a comma list or a range ``"A to B"`` (expanded to len(scores) points). Later mentions of
    a clip overwrite earlier ones and a missing score reuses the last one. ``parse="tolerant"`` additionally
    accepts answers whose score list was cut off (scores default to 1).
    """
    num_clips = int(np.ceil(duration / clip_length))
    out = np.zeros(num_clips)
    m = re.search(r"in the (.*?) seconds?\.?\s*Their saliency scores are ([\d\.,\s]+)", response or "", re.IGNORECASE | re.DOTALL)
    if m:
        ts_part, sc_part = m.group(1), m.group(2)
        scores = [float(x) for x in re.findall(r"\d+\.?\d*", sc_part)]
    elif parse == "tolerant":
        ts, scores = parse_highlights(response)
        ts_part = ", ".join(str(t) for t in ts)
    else:
        return out
    if " to " in ts_part:
        nums = [float(x) for x in re.findall(r"\d+\.?\d*", ts_part)]
        a, b = (nums[0], nums[-1]) if nums else (0.0, 0.0)
        n = max(len(scores), 1)
        timestamps = [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 else [a]
    else:
        timestamps = [float(t) for t in re.findall(r"\d+\.?\d*", ts_part)]
    for i, t in enumerate(timestamps):
        score = scores[i] if i < len(scores) else (scores[-1] if scores else 0.0)
        out[min(int(t / clip_length), num_clips - 1)] = score
    return out


def paper_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AP as in the original script: undefined (0) when there are no positives or only positives."""
    if np.all(y_true == 0) or np.all(y_true == 1):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(average_precision_score(y_true, y_score))


def evaluate(gt: List[dict], preds: Dict[int, dict], clip_length: float = 2.0, parse: str = "strict") -> dict:
    per_level = {name: {"ap": [], "hit": []} for name in LEVELS}
    legacy_ap, legacy_hit, missing = [], [], 0
    for item in gt:
        p = preds.get(int(item["qid"]))
        full = gt_scores_full(item, clip_length)  # (#clips, 3)
        num_clips = full.shape[0]
        if p is None:
            missing += 1
            scores = np.zeros(num_clips)
        else:
            scores = pred_scores_for(item, p, num_clips, clip_length, parse)
        top = int(np.argmax(scores)) if len(scores) else 0
        for name, thr in LEVELS.items():
            binary = (full >= thr).astype(float)  # (#clips, 3)
            per_level[name]["ap"].append(np.mean([ap(binary[:, w], scores) for w in range(3)]))
            per_level[name]["hit"].append(float(binary[top].max()) if num_clips else 0.0)
        # Paper protocol (faithful port of the original evaluate_vhd.py): every relevant clip is a positive,
        # the saliency vector has ceil(duration / clip) entries padded to int(duration / clip), AP is 0 when a
        # query has no or only positives, and HIT@1 uses the highest-index clip among tied top scores.
        expected = int(item["duration"] / clip_length)
        ps = paper_saliency_vector(p.get("response", "") if p else "", float(item["duration"]), clip_length, parse)
        if len(ps) < expected:
            ps = np.pad(ps, (0, expected - len(ps)))
        y = np.zeros(len(ps))
        for idx in item["relevant_clip_ids"]:
            if idx < len(y):
                y[idx] = 1
        legacy_ap.append(paper_ap(y, ps))
        top_paper = int(np.argsort(ps)[-1]) if len(ps) else 0
        legacy_hit.append(float(y[top_paper]) if len(ps) else 0.0)
    res = {"num_samples": len(gt), "num_missing_predictions": missing,
           "paper_protocol": {"mAP": 100 * float(np.mean(legacy_ap)), "HIT@1": 100 * float(np.mean(legacy_hit))}}
    for name in LEVELS:
        res[f"HL-min-{name}"] = {"mAP": 100 * float(np.mean(per_level[name]["ap"])),
                                 "HIT@1": 100 * float(np.mean(per_level[name]["hit"]))}
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate QVHighlights highlight detection results")
    parser.add_argument("--gt_json_path", default="data/eval/highlight_val_release.jsonl")
    parser.add_argument("--inference_dir", required=True, help="folder with result_<qid>.json files")
    parser.add_argument("--clip_length", type=float, default=2.0)
    parser.add_argument("--parse", default="strict", choices=["strict", "tolerant"],
                        help="strict = paper protocol (responses without a score list count as no prediction); tolerant = keep their timestamps with score 1")
    parser.add_argument("--output", default="", help="metrics JSON path (default: <inference_dir>/evaluation.json)")
    args = parser.parse_args()

    gt, preds = load_gt(args.gt_json_path), load_predictions(args.inference_dir)
    res = evaluate(gt, preds, args.clip_length, args.parse)

    def show(r, label):
        print(label)
        x = r["paper_protocol"]
        print(f"  paper protocol (relevant clips as positives): mAP {x['mAP']:.2f}  HIT@1 {x['HIT@1']:.2f}")
        for name in LEVELS:
            x = r[f"HL-min-{name}"]
            print(f"  Moment-DETR HL-min-{name:<9}: mAP {x['mAP']:.2f}  HIT@1 {x['HIT@1']:.2f}")

    show(res, f"QVHighlights highlight detection: {res['num_samples']} queries, {res['num_missing_predictions']} missing predictions")
    if res["num_missing_predictions"]:
        # Partial run (e.g. --max_items): also score only the queries that have a prediction.
        sub = evaluate([g for g in gt if int(g["qid"]) in preds], preds, args.clip_length, args.parse)
        res["subset_with_predictions"] = sub
        show(sub, f"  [subset of {sub['num_samples']} predicted queries]")
    out = args.output or os.path.join(args.inference_dir, "evaluation.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
