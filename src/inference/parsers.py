"""Parsers that turn free-form model responses into evaluable predictions."""
import json
import re
from typing import List, Optional, Tuple

import numpy as np

_NUM = r"\d+(?:\.\d+)?"


def _strip_code_fence(text: str) -> str:
    return re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text.strip(), flags=re.DOTALL)


def _to_seconds(v) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(_NUM, v)
        return float(m.group(0)) if m else None
    return None


def parse_windows(response: str) -> List[List[float]]:
    """Extract ``[[start, end], ...]`` (seconds) from a VTG / refinement response.

    Accepts the JSON format requested by the prompt, dict-style windows, bare number pairs
    and phrasings such as ``"from 3.2 to 8.5 seconds"``. Zero-length windows (``end == start``, which the
    models emit for single-second moments) are kept, as in the paper's evaluation, and later widened to the
    minimum window duration; windows with ``end < start`` are dropped.
    """
    text = _strip_code_fence(response or "")
    windows: List[List[float]] = []
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        raw = obj.get("relevant_windows", []) if isinstance(obj, dict) else obj
        if not isinstance(raw, list):
            raw = [raw]
        flat: List[float] = []
        for w in raw:
            if isinstance(w, dict):
                s, e = _to_seconds(w.get("start_time", w.get("start"))), _to_seconds(w.get("end_time", w.get("end")))
                if s is not None and e is not None:
                    windows.append([s, e])
            elif isinstance(w, (list, tuple)) and len(w) == 2:
                s, e = _to_seconds(w[0]), _to_seconds(w[1])
                if s is not None and e is not None:
                    windows.append([s, e])
            else:
                v = _to_seconds(w)
                if v is not None:
                    flat.append(v)
        windows += [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]
    except (json.JSONDecodeError, AttributeError, TypeError):
        seg = text
        m = re.search(r"relevant_windows\"?\s*[:=]?\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if m:
            seg = m.group(1)
        pairs = re.findall(rf"\[\s*({_NUM})\s*(?:seconds?)?\s*,\s*({_NUM})\s*(?:seconds?)?\s*\]", seg)
        if not pairs:
            pairs = re.findall(rf"(?:from\s+)?({_NUM})\s*(?:s|sec|seconds)?\s*(?:to|-|–)\s*({_NUM})", seg, re.IGNORECASE)
        windows = [[float(a), float(b)] for a, b in pairs]
    return [[s, e] for s, e in windows if e >= s]


def parse_dvc_events(response: str) -> List[Tuple[float, float, str]]:
    """``"From A to B seconds, desc."`` paragraphs -> ``[(A, B, desc), ...]``."""
    return [(float(a), float(b), d.strip()) for a, b, d in
            re.findall(rf"From\s+({_NUM})\s+to\s+({_NUM})\s+seconds?,\s*(.*?)(?=\n\s*\n\s*From\s+\d|\Z)", response or "", re.S)]


def parse_highlights(response: str, require_scores: bool = False) -> Tuple[List[float], List[float]]:
    """Return ``(timestamps, scores)`` from a highlight-detection response.

    Handles the fine-tuned format ``"The highlight timestamps are in the 82, 84, ... seconds. Their saliency
    scores are 1.3, 1.7, ..."`` (also ``"There are N highlight moments in the ... second."`` and ranges
    ``"78.0 to 96.0"``), and the zero-shot ``"Highlight k: A - B seconds, saliency score: s"`` lines.
    ``require_scores=True`` discards responses whose score list is missing (truncated generations).
    """
    text = response or ""
    m = re.search(r"in the (.*?) seconds?\.?(?:\s*Their saliency scores are ([\d\.,\s]*))?", text, re.IGNORECASE | re.DOTALL)
    if m:
        ts_part, sc_part = m.group(1), m.group(2) or ""
        # A response cut off by max_new_tokens may list timestamps without (all of) the scores. With
        # ``require_scores`` (the paper protocol) such a response yields no prediction; otherwise missing
        # scores default to 1.0 so the clips still count as predicted highlights.
        if require_scores and not sc_part.strip():
            return [], []
        scores = [float(s) for s in re.findall(_NUM, sc_part)]
        if " to " in ts_part:
            nums = [float(x) for x in re.findall(_NUM, ts_part)]
            a, b = (nums[0], nums[-1]) if nums else (0.0, 0.0)
            n = max(len(scores), 1)
            timestamps = [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 else [a]
        else:
            timestamps = [float(t) for t in re.findall(_NUM, ts_part)]
        if len(scores) < len(timestamps):
            scores = scores + [1.0] * (len(timestamps) - len(scores))
        return timestamps, scores[: len(timestamps)]
    timestamps, scores = [], []
    for a, b, s in re.findall(rf"Highlight\s*\d+:\s*\[?\s*({_NUM})\s*-\s*({_NUM})\s*(?:seconds?)?\s*\]?"
                              rf"(?:\s*seconds?)?(?:,\s*saliency score:\s*({_NUM}))?", text):
        a, b = float(a), float(b)
        score = float(s) if s else 1.0
        t = a
        while t < b:
            timestamps.append(t)
            scores.append(score)
            t += 2.0
    return timestamps, scores


def highlights_to_clip_scores(timestamps: List[float], scores: List[float], duration: float, clip_len: float = 2.0) -> np.ndarray:
    """Saliency score per ``clip_len``-second clip (0 where the model said nothing)."""
    num_clips = max(1, int(np.ceil(duration / clip_len)))
    out = np.zeros(num_clips, dtype=float)
    for i, t in enumerate(timestamps):
        s = scores[i] if i < len(scores) else (scores[-1] if scores else 0.0)
        idx = min(max(int(t / clip_len), 0), num_clips - 1)
        out[idx] = max(out[idx], s)
    return out
