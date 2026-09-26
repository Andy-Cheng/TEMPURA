<div align="center">

# TEMPURA: Temporal Event Masked Prediction and Understanding for Reasoning in Action

[Jen-Hao Cheng](https://jen-haocheng.com/)<sup>1</sup> &nbsp;·&nbsp;
[Yi-Hao Peng](https://www.yihaopeng.tw/)<sup>2</sup> &nbsp;·&nbsp;
[Huapeng Zhou](https://huapengzhou.com/)<sup>1</sup> &nbsp;·&nbsp;
[Vivian Wang](https://www.linkedin.com/in/vivian-wang-bb14a4225/)<sup>1</sup> &nbsp;·&nbsp;
[Huayu Wang](https://huayuww.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Hsiang-Wei Huang](https://hsiangwei0903.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Wenhao Chai](https://wenhaochai.com/)<sup>3</sup> &nbsp;·&nbsp;
[Hou-I Liu](https://www.linkedin.com/in/hoiliu0801/)<sup>4</sup> &nbsp;·&nbsp;
[Kuang-Ming Chen](https://gorden0413.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Cheng-Yen Yang](https://yangchris11.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Yi-Ling Chen](https://www.linkedin.com/in/yiling-chen-tw)<sup>5</sup> &nbsp;·&nbsp;
[Vibhav Vineet](https://vibhav-vineet.github.io/)<sup>5</sup> &nbsp;·&nbsp;
[Qin Cai](https://www.linkedin.com/in/qin-cai-4329a195)<sup>6</sup> &nbsp;·&nbsp;
[Jenq-Neng Hwang](https://people.ece.uw.edu/hwang/)<sup>1</sup>

<sup>1</sup> University of Washington &nbsp;&nbsp; <sup>2</sup> Carnegie Mellon University &nbsp;&nbsp; <sup>3</sup> Princeton University &nbsp;&nbsp;
<sup>4</sup> National Yang Ming Chiao Tung University &nbsp;&nbsp; <sup>5</sup> Microsoft &nbsp;&nbsp; <sup>6</sup> Independent Researcher

**Conference on Language Modeling (COLM) 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-1f2d3d)](https://andy-cheng.github.io/TEMPURA/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.01583-b31b1b)](https://arxiv.org/abs/2505.01583)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20TEMPURA-Models-yellow)](https://huggingface.co/collections/andaba/tempura-681c325777c23f72666a0995)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20VER-Dataset-yellow)](https://huggingface.co/datasets/andaba/TEMPURA-VER)

</div>

---

## Overview

**TEMPURA** teaches video-language models to reason about causal event structure and to produce
fine-grained, timestamp-aligned descriptions of untrimmed videos. Training follows a two-stage curriculum:

1. **Masked Event Prediction (MEP)** – a segment of the video is hidden and the model reasons step by step
   about what must have happened there from the surrounding events.
2. **Dense Video Captioning (DVC)** – the model partitions the whole video into consecutive, non-overlapping
   events and writes a timestamped description for each.

Both stages are trained on **VER** (Video Event Reasoning), our dataset of 500K YouTube videos with temporally
aligned event descriptions and structured reasoning traces. The resulting models improve strong base VLMs on
video temporal grounding and highlight detection across model families and scales.

![TEMPURA Teaser](assets/teaser.png)

## Highlights

- **Event-level reasoning + fine-grained segmentation** – one recipe that transfers across Qwen2.5-VL and InternVL3 backbones.
- **Released checkpoints** – four models (2B to 8B) behind a unified inference API; the model family is auto-detected from the checkpoint.
- **Reproducible benchmark pipeline** – Charades-STA temporal grounding and QVHighlights highlight detection, from raw videos to metrics, in one command each.
- **VER dataset** – 500K dense-captioning and 100K masked-event-reasoning samples on Hugging Face.

## News

- **2026-09** – Inference and benchmark-evaluation code released, together with TEMPURA checkpoints for Qwen2.5-VL-3B/7B and InternVL3-2B/8B.
- **2026-09** – The VER dataset is available at [andaba/TEMPURA-VER](https://huggingface.co/datasets/andaba/TEMPURA-VER).
- **2026** – TEMPURA is accepted to COLM 2026.

## Installation

Tested with Python 3.12, PyTorch 2.10 (CUDA 12.8) and transformers 5.3 on H100 GPUs.

```bash
git clone https://github.com/Andy-Cheng/TEMPURA.git && cd TEMPURA
bash scripts/install/install.sh          # creates .venv, installs torch + requirements (+ flash-attn when it builds)
source .venv/bin/activate
```

Or by hand:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional; SDPA attention is used when it is missing
```

All commands below are run from the repository root with `PYTHONPATH=.` (the shell scripts set it for you).

## Model Zoo

| Model | Base model | Frame input | Hugging Face |
|---|---|---|---|
| TEMPURA-Qwen2.5-VL-3B | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 1 fps, 336×336 px budget per frame | [andaba/TEMPURA-Qwen2.5-VL-3B](https://huggingface.co/andaba/TEMPURA-Qwen2.5-VL-3B) |
| TEMPURA-Qwen2.5-VL-7B | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 1 fps, 336×336 px budget per frame | [andaba/TEMPURA-Qwen2.5-VL-7B](https://huggingface.co/andaba/TEMPURA-Qwen2.5-VL-7B) |
| TEMPURA-InternVL3-2B | [OpenGVLab/InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | 0.5 fps, one 448×448 tile per frame | [andaba/TEMPURA-InternVL3-2B](https://huggingface.co/andaba/TEMPURA-InternVL3-2B) |
| TEMPURA-InternVL3-8B | [OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) | 0.5 fps, one 448×448 tile per frame | [andaba/TEMPURA-InternVL3-8B](https://huggingface.co/andaba/TEMPURA-InternVL3-8B) |

All checkpoints expect the video as a sequence of frames with the timestamp (seconds) drawn on the top-left
corner of every frame; `src/inference/video_utils.py` produces exactly this input. The 3B checkpoints of the
preprint ([-s1](https://huggingface.co/andaba/TEMPURA-Qwen2.5-VL-3B-s1), [-s2](https://huggingface.co/andaba/TEMPURA-Qwen2.5-VL-3B-s2)) remain available.

### Results

All numbers below were produced with the released checkpoints, the scripts in `scripts/eval/` and the default
configs, and are reproduced by `python scripts/eval/collect_results.py` from your own `results/` folder.

<!-- RESULTS_TABLE_START -->
**Charades-STA, any-window protocol** (Video Temporal Grounding stage)

| Model | mIoU | R@0.3 | R@0.5 | R@0.7 | windows / query |
|---|---:|---:|---:|---:|---:|
| TEMPURA-Qwen2.5-VL-3B | 49.5 | 82.3 | 52.6 | 20.4 | 5.9 |
| TEMPURA-Qwen2.5-VL-7B | 46.8 | 76.7 | 49.4 | 19.5 | 3.6 |
| TEMPURA-InternVL3-2B | 56.6 | 95.9 | 60.3 | 22.9 | 6.5 |
| TEMPURA-InternVL3-8B | 54.9 | 91.3 | 59.3 | 23.7 | 11.8 |

**Charades-STA, single-window protocol** (final caption-refined answer; only its first window is scored)

| Model | mIoU | R@0.3 | R@0.5 | R@0.7 |
|---|---:|---:|---:|---:|
| TEMPURA-Qwen2.5-VL-3B | 31.4 | 50.8 | 28.7 | 10.3 |
| TEMPURA-Qwen2.5-VL-7B | 34.4 | 55.5 | 34.4 | 13.5 |
| TEMPURA-InternVL3-2B | 25.0 | 38.6 | 23.1 | 10.5 |
| TEMPURA-InternVL3-8B | 31.4 | 50.7 | 29.2 | 11.4 |

**QVHighlights highlight detection**

| Model | mAP | HIT@1 | mAP (Moment-DETR, Very Good) | HIT@1 (Moment-DETR, Very Good) |
|---|---:|---:|---:|---:|
| TEMPURA-Qwen2.5-VL-3B | 47.4 | 43.7 | 21.6 | 45.7 |
| TEMPURA-Qwen2.5-VL-7B | 49.5 | 58.3 | 23.9 | 51.4 |
| TEMPURA-InternVL3-2B | 34.2 | 31.0 | 16.0 | 31.2 |
| TEMPURA-InternVL3-8B | 53.9 | 68.0 | 25.5 | 54.3 |
<!-- RESULTS_TABLE_END -->

How to read the Charades-STA tables. The grounding prompt asks the model for every occurrence of the query, so the
first table credits a query when any returned window reaches the IoU threshold and therefore measures recall over
the returned candidates. The
second table scores only the model's final single-window answer, the convention of single-prediction grounding
evaluations such as [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat) and
[TimeLens](https://github.com/TencentARC/TimeLens), and reflects localization precision. Both are computed from the same
result files. QVHighlights reports mAP and HIT@1 with every relevant clip as a positive, and, for reference, the
Moment-DETR highlight-detection metric at the "Very Good" saliency threshold.

## Inference

Dense video captioning on any video file (or a folder of pre-extracted frames named `0.jpg, 1.jpg, ...`):

```bash
python -m src.inference.dense_video_captioning_demo \
    --model_path andaba/TEMPURA-Qwen2.5-VL-3B --video test_videos_demo/hotdog.mp4
```

Masked event prediction: hide a segment and let the model reason about what happened there.

```bash
python -m src.inference.dense_video_captioning_demo \
    --model_path andaba/TEMPURA-Qwen2.5-VL-3B --video test_videos_demo/hotdog.mp4 --task mep --mask 5 10
```

For InternVL3 checkpoints add `--fps 0.5`. The building blocks are small and reusable:

```python
from src.inference.model_utils import load_model, build_messages, generate
from src.inference.video_utils import load_video_for_model
from src.inference import prompts

processor, model, family = load_model("andaba/TEMPURA-Qwen2.5-VL-3B")            # family: "qwenvl" | "internvl"
frames, timestamps = load_video_for_model("video.mp4", sample_fps=1.0, add_timestamp=True)
messages = build_messages(frames, prompts.DVC, family, min_pixels=336 * 336, max_pixels=336 * 336)
print(generate(processor, model, family, messages, max_new_tokens=2048))
```

A Gradio demo is available with `python src/serve/app.py --model-path andaba/TEMPURA-Qwen2.5-VL-3B` (`pip install gradio`; the app uses the legacy loader in `src/utils.py`).

## Benchmark evaluation

### Data

The ground-truth files are included under `data/eval/`:

- `charades_sta_test_tvr_format.json` – Charades-STA test split (3,720 queries), in the format released with [Moment-DETR](https://github.com/jayleicn/moment_detr).
- `highlight_val_release.jsonl` – QVHighlights validation split (1,550 queries), from the [Moment-DETR](https://github.com/jayleicn/moment_detr) release.

Videos must be obtained from the original sources and placed (or symlinked) as:

```
data/eval/Charades_v1_480/<vid>.mp4         # Charades v1 480p videos: https://prior.allenai.org/projects/charades
data/eval/qvhighlights/videos/<vid>.mp4     # QVHighlights clips (vid = <youtube_id>_<start>_<end>): https://github.com/jayleicn/moment_detr
```

### Charades-STA temporal grounding

Three steps, all with the same checkpoint: (1) **VTG** – answer the query with the time window(s) of every
occurrence; (2) **DVC** – densely caption the video once (cached per video); (3) **refine** – re-read the dense
caption and the VTG windows in a text-only prompt and output one final window. Every result file keeps the VTG
windows, the refined windows and both raw answers.

```bash
# <model> <output_dir> <qwen|internvl> <gpu>
bash scripts/eval/eval_charades.sh andaba/TEMPURA-Qwen2.5-VL-3B results/charades/TEMPURA-Qwen2.5-VL-3B qwen 0
bash scripts/eval/eval_charades.sh andaba/TEMPURA-InternVL3-8B results/charades/TEMPURA-InternVL3-8B internvl 0
```

The script runs `src.inference.inference_vtg_refined` and then `src.evaluate.evaluate_charades`, whose flags select
the protocol without re-running the model:

| Table | Flags |
|---|---|
| any-window (default) | `--windows vtg_first --reduce max` |
| single-window | `--windows refined_first --reduce first` |
| [TimeLens](https://github.com/TencentARC/TimeLens) / TimeChat script rule (first pair of the raw answer, no 4 s widening) | `--windows refined_first --protocol timelens` |

Runs resume automatically; add `--max_items N` for a quick check (the scorer then also reports the predicted subset).

### QVHighlights highlight detection

The model is asked for the highlight timestamps (2-second clips) and a 1-5 saliency score per clip with a
256-token budget; the answer becomes a per-clip saliency vector.

```bash
bash scripts/eval/eval_qvhighlights.sh andaba/TEMPURA-Qwen2.5-VL-3B results/qvhighlights/TEMPURA-Qwen2.5-VL-3B qwen 0
```

`src.evaluate.evaluate_qvhighlights` prints mAP and HIT@1 with every clip in `relevant_clip_ids` as a positive
(the table above) and the Moment-DETR `eval_highlight` variants at the Fair / Good / Very Good thresholds. Answers
cut off before their score list count as no prediction (`--parse strict`); `--parse tolerant` keeps their
timestamps. `--pipeline dvc_refine` on the inference script runs the DVC-then-refine variant used for Charades.

### Configuration

`configs/eval/*.json` hold the per-family defaults (frame rate, pixel budget, prompts, generation budgets);
every key can be overridden on the command line. Results are written as one JSON per item under
`<output_dir>/<task>/` together with a `run_config.json`.

## VER dataset

The Video Event Reasoning dataset is released on Hugging Face: [andaba/TEMPURA-VER](https://huggingface.co/datasets/andaba/TEMPURA-VER)
(`DVC500k_gpt4o`: 500K dense-captioning samples, `MEP100k_gpt4o`: 100K masked-event-reasoning samples; videos are
identified by their YouTube ids from YT-Temporal-1B). Frames were sampled at 1 fps with the timestamp overlaid,
exactly as at inference time.

## Training (reference only)

The released checkpoints were trained with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) using
full-parameter SFT on VER (stage 1: MEP, stage 2: DVC) with the frame/timestamp formatting implemented in this
repository. The original in-house trainer under `src/training/` and `scripts/train/` (a DeepSpeed recipe for
Qwen2-VL / Qwen2.5-VL adapted from [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)) is kept for
reference only and is **obsolete**: it targets older `transformers` releases and is not maintained. To train your
own models, convert VER to LLaMA-Factory ShareGPT format (`conversations` plus one image per frame) and follow the
LLaMA-Factory multi-image SFT recipe.

## Citation

If you find TEMPURA useful in your research, please cite our paper:

```bibtex
@inproceedings{
cheng2026tempura,
title={{TEMPURA}: Temporal Event Masked Prediction and Understanding for Reasoning in Action},
author={Cheng, Jen-Hao and Peng, Yi-Hao and Zhou, Huapeng and Wang, Vivian and Wang, Huayu and Huang, Hsiang-Wei and Chai, Wenhao and Liu, Hou-I and Chen, Kuang-Ming and Yang, Cheng-Yen and Chen, Yi-Ling and Vineet, Vibhav and Cai, Qin and Hwang, Jenq-Neng},
booktitle={Third Conference on Language Modeling},
year={2026}
}
```

## Acknowledgements

We build on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [InternVL3](https://github.com/OpenGVLab/InternVL),
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)
[Moment-DETR](https://github.com/jayleicn/moment_detr) (benchmark annotations and highlight-detection metrics) and
[TimeLens](https://github.com/TencentARC/TimeLens) / [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat) (the
single-window temporal-grounding scoring rule). Video links in VER come from
[YT-Temporal-1B](https://rowanzellers.com/merlotreserve/).
