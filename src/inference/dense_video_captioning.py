import os
from src.qwen_vl_utils import process_vision_info
from src.utils import load_pretrained_model, read_frames_decord, video_frame_decoration, prompt_decoration, prepare_message
from pprint import pprint

def inference(prompt, model, video_path, param):
    add_time_instruction = param["add_time_instruction"]
    add_visual_timestamp = param["add_visual_timestamp"]
    fps = param["video_fps"]
    # Video Loading
    # Can be a directory containing video frames, or a video file
    if os.path.isdir(video_path):
        video_frame_list = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path), key=lambda x: int(x.split(".")[0]))]
        time_list = [i * (1 / fps) for i in range(len(video_frame_list))]
    else:
        video_frame_list, time_list = read_frames_decord(video_path, sample=f"fps{fps}")
    video_frame_list = video_frame_decoration(video_frame_list, time_list, mode="visual_timestamp" if add_visual_timestamp else None)
    prompt = prompt_decoration(prompt, fps=fps, time_list=time_list, mode="time_instruction" if add_time_instruction else None)
  
    # Preparation for inference
    messages = [prepare_message(prompt, param, video_frame_list)]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # Generation of the text output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

if __name__ == "__main__":
    # Video Related Parameters
    # You can change video_fps, min_pixels, and max_pixels according to your GPU memory
    param = {
        "video_encoding_type": "image",
        "add_time_instruction": False,
        "add_visual_timestamp": True,
        "video_fps": 1,
        "min_pixels": 336*336,
        "max_pixels": 336*336,
    }
    model_id = "andaba/TEMPURA-Qwen2.5-VL-3B-s2"
    processor, model = load_pretrained_model(model_path = model_id, 
                                                device_map="cuda:0",  
                                                device="cuda", use_flash_attn=True
    )
    video_path = "test_videos_demo/hotdog.mp4"
    prompt = "Partition and identify events by dividing the video into a series of non-overlapping segments, determining the start and end time for each event, and arranging them in chronological order to ensure complete coverage of all video frames. Each event should be accompanied by a detailed description. Please follow the format: 'From <start time1> to <end time1>, <detailed description1>.\n\nFrom <start time2> to <end time2>, <detailed description2>.\n\n...'"
    output_text = inference(prompt, model, video_path, param)
    pprint(f"User Prompt: {prompt}")
    pprint(f"Model Response: {output_text}")