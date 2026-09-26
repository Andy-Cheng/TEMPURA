#!/usr/bin/env bash
# Charades-STA: VTG -> DVC -> caption-guided refinement, then scoring.
# Usage: bash scripts/eval/eval_charades.sh <model_path_or_hf_id> <output_dir> [qwen|internvl] [gpu_id] [extra args...]
set -euo pipefail
MODEL=${1:?model path or HF id}; OUT=${2:?output dir}; FAMILY=${3:-qwen}; GPU=${4:-0}; shift $(( $# < 4 ? $# : 4 ))
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=.
python -m src.inference.inference_vtg_refined --config configs/eval/charades_${FAMILY}.json --model_path "$MODEL" --output_dir "$OUT" "$@"
python -m src.evaluate.evaluate_charades --inference_dir "$OUT/vtg_refined" --gt_json_path data/eval/charades_sta_test_tvr_format.json
