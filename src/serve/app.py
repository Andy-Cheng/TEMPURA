import os
import torch
import warnings
import argparse
from threading import Thread
import gradio as gr
from transformers import TextIteratorStreamer
from functools import partial
from src.qwen_vl_utils import process_vision_info
from src.utils import (
    load_pretrained_model, 
    get_model_name_from_path, 
    disable_torch_init,
    read_frames_decord,
    video_frame_decoration,
)

warnings.filterwarnings("ignore")
selected_video_directory = None
def set_video_directory(dir_path):
    global selected_video_directory
    selected_video_directory = dir_path
    return f"Video directory set to {dir_path}"

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def process_video_content(file_path, tempura_confg):
    if tempura_confg["video_encoding_type"] == "image":
        video_frame_list, time_list = read_frames_decord(file_path, sample=f"fps{tempura_confg['video_fps']}")
        video_frame_list = video_frame_decoration(
            video_frame_list, 
            time_list, 
            mode="visual_timestamp" if tempura_confg["add_visual_timestamp"] else None
        )
        return [
            {
                "type": "image",
                "image": video_frame,
                "max_pixels": tempura_confg["max_pixels"] * tempura_confg["max_pixels"],
                "min_pixels": tempura_confg["min_pixels"] * tempura_confg["min_pixels"],
            }
            for video_frame in video_frame_list
        ]
    else:
        return [{"type": "video", "video": file_path, "fps": tempura_confg["video_fps"]}]

def bot_streaming(message, history, generation_args, tempura_confg):
    # Initialize variables
    images = []
    videos = []
    # If no files uploaded but a directory was set via the UI, use it
    if not message.get("files") and selected_video_directory:
        message["files"] = [selected_video_directory]
    if message["files"]:
        for file_item in message["files"]:
            if isinstance(file_item, dict):
                file_path = file_item["path"]
            else:
                file_path = file_item
            if is_video_file(file_path):
                videos.append(file_path)
            elif os.path.isdir(file_path):
                video_frame_list = [os.path.join(file_path, f) for f in sorted(os.listdir(file_path), key=lambda x: int(x.split(".")[0]))]
                videos.append(video_frame_list)
            else:
                images.append(file_path)
    conversation = []
    for user_turn, assistant_turn in history:
        user_content = []
        if isinstance(user_turn, tuple):
            file_paths = user_turn[0]
            user_text = user_turn[1] if len(user_turn) > 1 else None
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            for file_path in file_paths:
                if is_video_file(file_path):
                    user_content.extend(process_video_content(file_path, tempura_confg))
                else:
                    user_content.append({"type": "image", "image": file_path})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            user_content.append({"type": "text", "text": user_turn})
        conversation.append({"role": "user", "content": user_content})

        if assistant_turn is not None:
            assistant_content = [{"type": "text", "text": assistant_turn}]
            conversation.append({"role": "assistant", "content": assistant_content})
    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
    for video in videos:
        user_content.extend(process_video_content(video, tempura_confg))
    user_text = message['text']
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    conversation.append({"role": "user", "content": user_content})
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device) 
    streamer = TextIteratorStreamer(processor.tokenizer, **{"skip_special_tokens": True, "skip_prompt": True, 'clean_up_tokenization_spaces':False,}) 
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

def clear_cuda_cache():
    """Clear PyTorch CUDA cache."""
    torch.cuda.empty_cache()
    return "CUDA cache cleared."

def main(args):
    global processor, model, device
    device = args.device
    disable_torch_init()
    use_flash_attn = True
    model_name = get_model_name_from_path(args.model_path)
    if args.disable_flash_attention:
        use_flash_attn = False
    processor, model = load_pretrained_model(model_path = args.model_path, 
                                                device_map=args.device, model_name=model_name, 
                                                load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                                device=args.device, use_flash_attn=use_flash_attn
    )
    chatbot = gr.Chatbot(scale=2)
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image", "video", "directory"],
        placeholder="Enter message or upload file…",
        show_label=False,
        lines=6,            # allow six lines
        max_lines=10        # optional upper limit
    )
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }

    tempura_confg = {
        "video_encoding_type": args.video_encoding_type,
        "video_fps": args.video_fps,
        "add_visual_timestamp": args.add_visual_timestamp,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
    }
    bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args, tempura_confg=tempura_confg)
    with gr.Blocks(fill_height=True) as demo:
        # UI controls for user-configurable video settings
        with gr.Row():
            fps_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=args.video_fps, label="Video FPS")
            video_encoding_type = gr.Radio(choices=["image", "video"], value=args.video_encoding_type, label="Video Encoding Type")
        # Bind controls to update tempura_confg
        fps_slider.change(lambda fps: tempura_confg.update({"video_fps": fps}), inputs=fps_slider)
        video_encoding_type.change(lambda itype: tempura_confg.update({"video_encoding_type": itype}), inputs=video_encoding_type)
        add_visual_timestamp = gr.Checkbox(value=args.add_visual_timestamp, label="Add Visual Timestamp")
        add_visual_timestamp.change(lambda add_visual_timestamp: tempura_confg.update({"add_visual_timestamp": add_visual_timestamp}), inputs=add_visual_timestamp)
        # Default prompt dropdown and insert button
        with gr.Row():
            default_prompts = [
                "Partition and identify events by dividing the video into a series of non-overlapping segments, determining the start and end time for each event, and arranging them in chronological order to ensure complete coverage of all video frames. Each event should be accompanied by a detailed description. Please follow the format: 'From <start time1> to <end time1>, <detailed description1>.\n\nFrom <start time2> to <end time2>, <detailed description2>.\n\n...'"
            ]
            prompt_dropdown = gr.Dropdown(choices=default_prompts, value=default_prompts[0], label="Default Prompt", interactive=True)
            insert_prompt_btn = gr.Button("Insert Prompt")
        insert_prompt_btn.click(lambda p: p, inputs=prompt_dropdown, outputs=chat_input)
        # Button to clear CUDA cache
        with gr.Row():
            clear_cache_btn = gr.Button("Clear CUDA Cache")
            cache_status = gr.Textbox(label="Cache Status", interactive=False)
        clear_cache_btn.click(fn=clear_cuda_cache, inputs=None, outputs=cache_status)
        # UI control to adjust max tokens dynamically
        max_tokens_slider = gr.Slider(minimum=128, maximum=4096, step=128, value=args.max_new_tokens, label="Max New Tokens")
        max_tokens_slider.change(lambda x: generation_args.update({"max_new_tokens": int(x)}), inputs=max_tokens_slider)
        # UI control to adjust temperature dynamically
        temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=args.temperature, label="Temperature")
        temperature_slider.change(lambda x: generation_args.update({"temperature": float(x), "do_sample": True if x > 0 else False}), inputs=temperature_slider)
        # UI control to adjust repetition penalty dynamically
        repetition_penalty_slider = gr.Slider(minimum=1.0, maximum=2.0, step=0.1, value=args.repetition_penalty, label="Repetition Penalty")
        repetition_penalty_slider.change(lambda x: generation_args.update({"repetition_penalty": float(x)}), inputs=repetition_penalty_slider)
        # UI field to select a default video directory
        with gr.Row():
            video_dir_input = gr.Textbox(label="Video Directory", placeholder="Enter or browse to a directory", interactive=True)
            set_dir_btn = gr.Button("Set Video Directory")
            dir_status = gr.Textbox(label="Directory Status", interactive=False)
        set_dir_btn.click(fn=set_video_directory, inputs=video_dir_input, outputs=dir_status)
        # UI fields to adjust min and max pixels
        with gr.Row():
            min_pixels_input = gr.Number(value=args.min_pixels, label="Min Pixel width", precision=0)
            max_pixels_input = gr.Number(value=args.max_pixels, label="Max Pixel width", precision=0)
        min_pixels_input.change(lambda v: tempura_confg.update({"min_pixels": int(v)}), inputs=min_pixels_input)
        max_pixels_input.change(lambda v: tempura_confg.update({"max_pixels": int(v)}), inputs=max_pixels_input)
        gr.ChatInterface(
            fn=bot_streaming_with_args,
            title=f"{args.model_path}",
            stop_btn="Stop Generation",
            multimodal=True,
            textbox=chat_input,
            chatbot=chatbot,
        )
    demo.queue(api_open=False)
    demo.launch(show_api=False, share=False, server_name='0.0.0.0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="andaba/TEMPURA-Qwen2.5-VL-3B-s2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--video-encoding-type", type=str, default="image", choices=["image", "video"])
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--add-visual-timestamp", type=bool, default=True)
    parser.add_argument("--min-pixels", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=540)
    args = parser.parse_args()
    main(args)