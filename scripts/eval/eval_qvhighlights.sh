#!/usr/bin/env bash
# QVHighlights highlight detection, then scoring.
# Usage: bash scripts/eval/eval_qvhighlights.sh <model_path_or_hf_id> <output_dir> [qwen|internvl] [gpu_id] [extra args...]
set -euo pipefail
MODEL=${1:?model path or HF id}; OUT=${2:?output dir}; FAMILY=${3:-qwen}; GPU=${4:-0}; shift $(( $# < 4 ? $# : 4 ))
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=.
python -m src.inference.inference_vhd --config configs/eval/qvhighlights_${FAMILY}.json --model_path "$MODEL" --output_dir "$OUT" "$@"
python -m src.evaluate.evaluate_qvhighlights --inference_dir "$OUT/vhd" --gt_json_path data/eval/highlight_val_release.jsonl
