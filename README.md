# TEMPURA: Temporal Event Masked Prediction and Understanding for Reasoning in Action



[Model Weights](https://huggingface.co/andaba/TEMPURA) | [VER Dataset](https://huggingface.co/datasets/andaba/TEMPURA-VER) | [Project Page](https://andy-cheng.github.io/TEMPURA/)

TEMPURA enables video-language models to reason about causal event relationships and generate fine-grained, timestamped descriptions of untrimmed videos.

![TEMPURA Teaser](assets/teaser.png)



## Installation
```bash
bash scripts/install/install.sh
```


## VER Dataset Preparation
Please refer to [VER Dataset on Hugging Face](https://huggingface.co/datasets/andaba/TEMPURA-VER) for dataset downloading.


Place the downloaded JSON files in data/VER/jsons/, and store the processed video frames (from [YT-Temporal-1B](https://rowanzellers.com/merlotreserve/)) in the appropriate directory (data/yt1b/processed_frames). Our experiments used 1 FPS sampling, but you may adjust the frame rate based on your specific needs.

## Training
By default, we use DeepSpeed ZeRO-2. You may switch to ZeRO-3 to further reduce GPU memory usage.

### Masked Event Prediction

```bash
bash scripts/train/masked_event_prediction_3B_8H100.sh
```

### Video Event Segmentation and Temporal Dense Captioning
You can swap the MODEL_PATH for the model weight from the masked event prediction training stage.

```bash
bash scripts/train/dense_event_caption_3B_8H100.sh
```


<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Qwen2-VL model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--tune_merger` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--merger_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--image_min_pixels` (int): Option for minimum input pixels for image.
- `--image_max_pixles` (int): Option for maximum maxmimum pixels for image.
- `--video_min_pixels` (int): Option for minimum input pixels for video.
- `--video_max_pixles` (int): Option for maximum maxmimum pixels for video.
- `--lora_enable` (bool): Option for using LoRA.
- `--vision_lora` (bool): Option for including `vision_tower` in LoRA module. `lora_enable` should be `True` to use this option.
- `--use_dora` (bool): Option for using DoRA instead of LoRA. `lora_enable` should be `True` to use this option.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>



## Inference & Evaluation




## Codebase Supported Features

- Deepspeed
- LoRA/QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image and video training
- Training optimized with liger kernel



## Citing TEMPURA
If you find our paper or dataset useful, please consider citing our work!


```tex
@article{tempura,
       title={TEMPURA: Temporal Event Masked Prediction
and Understanding for Reasoning in Action}, 
       author={Jen-Hao Cheng and Vivian Wang and Huayu Wang and Huapeng Zhou and Yi-Hao Peng and Hou-I Liu and
       Hsiang-Wei Huang and Kuang-Ming Chen and Cheng-Yen Yang and Wenhao Chai and Yi-Ling Chen and
       Vibhav Vineet and Qin Cai and Jenq-Neng Hwang},
       journal={},
       year={2025}
}


```


## Acknowledgement

We build upon the following repositories:

- [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune): An amazing open-source project of Qwen2-VL and Qwen2.5-VL finetuning.

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5): Awesome pretrained MLLM based on Qwen2.5-VL.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): Collection of Tirton kernels designed specifically for LLM training.
