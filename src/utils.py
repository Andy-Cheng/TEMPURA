from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
import warnings
import os
from decord import VideoReader
from PIL import Image
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_name="", model_base =None, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=True, **kwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        # Don't use float16, it will cause NaN!!
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'

    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(model_base)
        print('Loading Qwen2-VL from base model...')
        if "Qwen2.5" in model_base:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Qwen2-VL weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()

        print('Model Loaded!!!')

    else:
        processor = AutoProcessor.from_pretrained(model_path, padding='left', use_fast=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ['rand', 'middle']: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif 'fps' in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_decord(
        video_path, num_frames=10, sample='rand', fix_start=None,
         clip=None, min_num_frames=4, frame_indices=None
):
    video_reader = VideoReader(video_path)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if frame_indices is  None:
        if clip:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps)
            start_index = int(start * fps)
        # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
        frame_indices = get_frame_indices(
            t_num_frames, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps
        )
        if clip:
            frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames, [f / fps for f in frame_indices]

def video_frame_decoration(video_frame_list, time_list, mode):
    if mode == "visual_timestamp":
        video_frame_list = [Image.open(f) for f in video_frame_list] if isinstance(video_frame_list[0], str) else video_frame_list
        for i, video_frame in enumerate(video_frame_list):
            timestamp = time_list[i]
            text = f"{timestamp:.2f}"
            img_width = video_frame.size[1]
            draw = ImageDraw.Draw(video_frame)
            font_size = img_width // 10
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            # Get final text dimensions
            text_height = text_bbox[3] - text_bbox[1]
            # Position text in bottom-right corner with padding
            text_x = 0
            text_y = 0
            # Add semi-transparent background
            draw.rectangle([text_x-5, text_y-5, text_x+text_width+5, text_y+text_height+5], 
                        fill=(0, 0, 0, 128))
            draw.text((text_x, text_y), text, fill='white', font=font)
    elif mode is None:
        pass
    else: 
        raise NotImplementedError
    return video_frame_list

def prompt_decoration(prompt, fps=None, time_list=None, mode=None):
    if mode == "time_instruction":
        prompt = f"You are given a video sampled at {fps} frame per second, so input video frames are sampled at {', '.join([f'{t:.2f}' for t in time_list])} second sequentially, so in total the video is {time_list[-1]+1:.2f} seconds long. {prompt}"
    elif mode is None:
        prompt = prompt
    else: 
        raise NotImplementedError
    return prompt

def prepare_message(prompt, param, video_frame_list):
    video_encoding_type = param["video_encoding_type"]
    fps = param["video_fps"]
    if video_encoding_type == "video":
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frame_list,
                        "fps": fps,
                        "max_pixels": param["max_pixels"],
                        "min_pixels": param["min_pixels"],
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    elif video_encoding_type == "image":
        message = [
            {
                "role": "user",
            "content": [
                *[
                {
                    "type": "image",
                    "image": video_frame,
                    "max_pixels": param["max_pixels"],
                    "min_pixels": param["min_pixels"],
                }
                for video_frame in video_frame_list
                ],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return message