#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME=$(basename "$MODEL_PATH")
GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
DATA_NAME="masked_event_prediction_split_0"
SYSTEM_PROMPT_VERSION="default"
PROJECT_NAME="event_vlm_stage2"
FREEZE_VISION=False
FREEZE_LLM=False
ADD_TIMESTAMP=True
VIDEO_INPUT_TYPE="image"


# If your dataset is mixed with images and videos, you need to use zero2.
deepspeed src/training/train.py \
    --deepspeed scripts/zero2.json \
    --model_path $MODEL_PATH \
    --data_path "data/VER/jsons/$DATA_NAME.json" \
    --image_folder data \
    --remove_unused_columns False \
    --freeze_vision_tower $FREEZE_VISION \
    --freeze_llm $FREEZE_LLM \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/${DATA_NAME}_${SYSTEM_PROMPT_VERSION}_${FREEZE_VISION}_${FREEZE_LLM}_${GLOBAL_BATCH_SIZE}_${VIDEO_INPUT_TYPE}_${ADD_TIMESTAMP}_${MODEL_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((180 * 320)) \
    --video_min_pixels $((180 * 320)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 4 \
    --system_message_version $SYSTEM_PROMPT_VERSION \
    --project_name $PROJECT_NAME \
    --video_input_type $VIDEO_INPUT_TYPE \
    --add_timestamp $ADD_TIMESTAMP
