import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageFont
from params import DataArguments
from constants import *
import glob
import time
from math import ceil

def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixel": min_pixel,
                "max_pixel": max_pixel

            }
            ]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, fps, masked_start_end_frame_idxs, masked_strategy_version, masked_image, add_timestamp, font, input_type = "image"):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    # If the video_path is a folder, we need to sort the images by the frame number
    if not "." in video_path:
        video_path = sorted(glob.glob(os.path.join(video_path, "*.jpg")), key=lambda x: int(os.path.basename(x).split(".")[0]))
        time_stamps = [f"{i * (1 / fps):.1f}" for i in range(len(video_path))] # hardcoded for now
        if len(video_path) == 0:
            return None, None

    if input_type == "video":
        messages = [
            {"role": "user", 
            "content": [
                {
                    "type": "video", 
                    "video": video_path,
                    "timestamps": time_stamps,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "fps": fps,
                    "masked_start_end_frame_idxs": masked_start_end_frame_idxs
                }
                ]
            }
        ]

    elif input_type == "image":
        messages = [
            {
                "role": "user",
                "content": [
                *[
                {
                    "type": "image",
                    "image": frame_name,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "timestamp": time_stamp,
                }
                for frame_name, time_stamp in zip(video_path, time_stamps)
                ],
            ],
        }
    ]



    image_input, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True, masked_strategy_version=masked_strategy_version, masked_image=masked_image, add_timestamp=add_timestamp, font=font)
    # No difference between video_input[0] and image_input, video_kwargs: {'fps': [1.0]}
    if input_type == "video":
        return video_input[0], video_kwargs
    else:
        return image_input, None


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        # TODO: remove after debug
        # list_data_dict = list_data_dict[:6]

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.fps = data_args.video_fps
        self.masked_image = None
        self.system_message = SYSTEM_MESSAGE[data_args.system_message_version]
        if data_args.masked_strategy_version == "ver0":
            # load in a masked image
            # hardcoded now
            img_width, img_height = 640, 360
            image = Image.new('RGB', (img_width, img_height), (0, 0, 0))
            self.masked_image = image
        self.timestamp_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", img_width//10)


    def __len__(self):
        return len(self.list_data_dict)
    

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep .5s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(.5)
        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass
        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
        


    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            is_dummy = False
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel))

        elif "video" in sources:
            is_dummy = False
            is_video = True 
            is_video_input_type_video = self.data_args.video_input_type == "video"
            images=None if is_video_input_type_video else []
            grid_key = "video_grid_thw" if is_video_input_type_video else "image_grid_thw"
            pixel_key = "pixel_values_videos" if is_video_input_type_video else "pixel_values"

            video_files = sources["video"]
            masked_start_end_frame_idxs =  sources.get("masked_start_end_frame_idxs", [])
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = [] if is_video_input_type_video else None
            for video_file in video_files:
                video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.data_args.video_fps, masked_start_end_frame_idxs, self.data_args.masked_strategy_version, self.masked_image, self.data_args.add_timestamp, self.timestamp_font, self.data_args.video_input_type)
                if video_input is None:
                    print(f"Video file {video_file} not found")
                    raise Exception(f"Video file {video_file} not found")
                if is_video_input_type_video:
                    videos.append(video_input)
                else:
                    images.append(video_input)
            if not is_video_input_type_video:
                images = [image for image_list in images for image in image_list]
        else:
            is_dummy=True
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video, video_input_type=self.data_args.video_input_type, frame_count=len(images) if self.data_args.video_input_type == "image" else 0)) # we only need to replace with the image tokens for image type of video input

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(self.system_message) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{self.system_message}\n{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            is_dummy=is_dummy,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_dummy_flags = []
        batch_second_per_grid_ts = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_dummy_flags.append(example["is_dummy"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'is_dummy': torch.tensor(batch_dummy_flags, dtype=torch.bool)
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    

def replace_image_tokens(input_string, is_video=False, video_input_type="image", frame_count=0):

    if is_video:
        if video_input_type == "image":
            input_string = input_string.replace(LLAVA_VIDEO_TOKEN+'\n', ''.join([VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN] * frame_count))
        else:
            input_string = input_string.replace(LLAVA_VIDEO_TOKEN+'\n', VISION_START_TOKEN+DEFAULT_VIDEO_TOKEN+VISION_END_TOKEN)

    else:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN+'\n', VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN)

    return input_string

def llava_to_openai(conversations, is_video=False, video_input_type="image", frame_count=0):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video, video_input_type=video_input_type, frame_count=frame_count)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data




role_mapping = {
    "human": "user",
    "gpt": "assistant"
}



class EventVLMDataSet(Dataset): # for toy demo, for train_data/data.json
    def __init__(self, data_args, data_path, media_dir, masking_strategy_version, video_min_pixels, video_max_pixels, video_fps=1, default_fps=1, fixed_frames=-1):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)
        # TODO: remove after debug
        # self.data = self.data[:6]
        self.media_dir = media_dir
        self.masking_strategy_version = masking_strategy_version
        self.video_fps = video_fps
        self.default_fps = default_fps
        self.fixed_frames = fixed_frames
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.system_message = SYSTEM_MESSAGE[data_args.system_message_version]

    def __len__(self):
        return len(self.data)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep .5s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(.5)
        num_base_retries = 10
        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.data) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass
        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
        
    def build_video_info(self, video_path, masked_start_end_frame_idxs, min_pixels, max_pixels, fps=1, default_fps=1):
        if not "." in video_path:
            video_path = sorted(glob.glob(os.path.join(video_path, "*.jpg")), key=lambda x: int(os.path.basename(x).split(".")[0]))
            if len(video_path) == 0:
                return None
        if self.fixed_frames > -1:
            video_path = video_path[::ceil(len(video_path) / self.fixed_frames)]
        else:
            sample_ratio = ceil(default_fps / fps)
            video_path = video_path[::sample_ratio]

        return{
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps,
                "masked_start_end_frame_idxs": masked_start_end_frame_idxs
            }
            


    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        messages = []
        messages.append({"role": "system", "content": self.system_message})
        image_idx = 0
        video_idx = 0
        for conversation in self.data[i]["conversations"]:
            role = role_mapping[conversation["from"]]
            content_list = []
            # only user can send images and videos
            content = conversation["value"]
            if role == "user":
                while "<video>" in content:
                    # Find the index of the current <image> tag
                    img_start = content.find("<video>")
                    content = content[:img_start] + content[img_start + len("<video>"):]
                    video_path = self.data[i]["video"][video_idx] if isinstance(self.data[i]["video"], list) else self.data[i]["video"]
                    video_info = self.build_video_info(os.path.join(self.media_dir, video_path), masked_start_end_frame_idxs=self.data[i].get('masked_start_end_frame_idxs', []), min_pixels=self.video_min_pixels, max_pixels=self.video_max_pixels, fps=self.video_fps, default_fps=self.default_fps)
                    if video_info is None:
                        raise ValueError(f"Failed to build video info for video {video_path}")
                    content_list.append(video_info)
                    video_idx += 1
                # Handle multiple images
                while "<image>" in content:
                    # Find the index of the current <image> tag
                    img_start = content.find("<image>")
                    content = content[:img_start] + content[img_start + len("<image>"):]
                    image_path = self.data[i]["image"][image_idx] if isinstance(self.data[i]["image"], list) else self.data[i]["image"]
                    content_list.append({"type": "image", "image": os.path.join(self.media_dir, image_path)})
                    image_idx += 1
            content = content.lstrip('\n')
            content_list.append({"type": "text", "text": content})
            messages.append({"role": role, "content": content_list})

        return {"messages": messages}


def find_assistant_content_sublist_indexes(l):
    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def collate_fn(batch, processor, pad_token_id):
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    # Qwen2_5_VLProcessor
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    attention_mask = inputs['input_ids'] != pad_token_id

    data_dict = {
        **inputs,
        'labels': labels_ids,
        'attention_mask': attention_mask,
    }
    return data_dict




class Collator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, processor, pad_token_id):
        self.processor = processor
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        return collate_fn(examples, self.processor, self.pad_token_id)



def make_supervised_data_module2(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = EventVLMDataSet(data_args=data_args, data_path=data_args.data_path, media_dir=data_args.image_folder, masking_strategy_version=data_args.masked_strategy_version, video_min_pixels=data_args.video_min_pixels, video_max_pixels=data_args.video_max_pixels, video_fps=data_args.video_fps, default_fps=data_args.default_fps, fixed_frames=data_args.fixed_frames)

    data_collator = Collator(processor, processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)