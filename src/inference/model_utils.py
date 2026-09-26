"""Model loading and generation helpers shared by the TEMPURA inference tasks.

Two model families are supported and auto-detected from ``config.json``:

* ``qwenvl``   – Qwen2-VL / Qwen2.5-VL (frames are passed as a list of images with a pixel budget)
* ``internvl`` – InternVL3 (HF ``internvl`` architecture; frames are passed as images, no tiling)
"""
import importlib.util
from typing import List, Optional

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def detect_model_type(model_path: str) -> str:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    mt = (getattr(cfg, "model_type", "") or "").lower()
    if "internvl" in mt:
        return "internvl"
    if "qwen" in mt:
        return "qwenvl"
    raise ValueError(f"Unsupported model_type '{mt}' at {model_path}; expected a Qwen2(.5)-VL or InternVL3 checkpoint")


def resolve_attn_implementation(preference: str = "auto") -> str:
    """``auto`` -> flash_attention_2 when installed, else sdpa."""
    if preference == "auto":
        return "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"
    return preference


def load_model(model_path: str, model_type: Optional[str] = None, device: str = "cuda:0",
               attn_implementation: str = "auto", dtype: torch.dtype = torch.bfloat16):
    """Return ``(processor, model, model_type)`` in eval mode on ``device``."""
    model_type = model_type or detect_model_type(model_path)
    attn = resolve_attn_implementation(attn_implementation)
    if model_type == "internvl":
        processor = AutoProcessor.from_pretrained(model_path, crop_to_patches=False)
    else:
        processor = AutoProcessor.from_pretrained(model_path)
    kwargs = dict(device_map=device, attn_implementation=attn)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    model.eval()
    print(f"[tempura] loaded {model_type} model from {model_path} (attn={attn}, dtype={dtype})")
    return processor, model, model_type


def build_messages(frames: List, prompt: str, model_type: str, min_pixels: int, max_pixels: int) -> List[dict]:
    """One user turn: all frames as images followed by the text prompt."""
    if model_type == "internvl":
        content = [{"type": "image", "image": f} for f in frames]
    else:
        content = [{"type": "image", "image": f, "min_pixels": min_pixels, "max_pixels": max_pixels} for f in frames]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def generate(processor, model, model_type: str, messages: List[dict], max_new_tokens: int) -> str:
    """Greedy-decode a response for ``messages`` (frames already embedded as PIL images)."""
    if model_type == "internvl":
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    else:
        from src.qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs or None, videos=video_inputs or None,
                           padding=True, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
