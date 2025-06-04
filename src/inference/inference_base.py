import sys
import torch
import os
import json
from tqdm import tqdm
import re
from PIL import Image, ImageDraw, ImageFont
from misc.pre_process_video import read_frames_decord
from src.utils import load_pretrained_model
from src.qwen_vl_utils import process_vision_info
import datetime

class InferBase:
    def __init__(self, args):
        self.args = args
        self.verbose = args.verbose
        self.videos_dir = args.videos_dir
        self.task = self._get_task_name()
        dir_list = args.model_path.split('/')
        self.model_name = dir_list[-1] if not dir_list[-1] == "" else dir_list[-2]
        self.output_dir = f"{args.output_dir}/{self.task}/{self.model_name}"
        if args.add_timestamp:
            self.output_dir += "_timestamp"
        if args.add_time_instruction:
            self.output_dir += "_time_instruction"
        self.output_dir += f"_{args.input_type}"
        print(f"output_dir: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.processor, self.model = load_pretrained_model(model_path=args.model_path, 
                                            device_map="cuda:0",  
                                            device="cuda", 
                                            use_flash_attn=not args.disable_flash_attn2)
        print(f"disable_flash_attn2: {args.disable_flash_attn2}")
        # Load data based on file extension
        if args.gt_json_file.endswith('.jsonl'):
            # Load JSONL file (each line is a JSON object)
            with open(args.gt_json_file, 'r') as f:
                self.test_data = [json.loads(line) for line in f]
            print(f"Loaded {len(self.test_data)} items from JSONL file: {args.gt_json_file}")
        else:
            # Load regular JSON file
            with open(args.gt_json_file, 'r') as f:
                self.test_data = json.load(f)
            print(f"Loaded data from JSON file: {args.gt_json_file}")

        if self.args.overwrite:
            print(f'Your results will be overwritten')
        else:
            print(f'Your results will not be overwritten, if you want to overwrite, please use the --overwrite flag')
        
        # Define base prompts dictionary
        self.base_prompts = {
            "vtg": '''You are a highly capable AI assistant trained to analyze video content and accurately identify the relevant time intervals in response to natural language queries.

            ### **Task Description**
            Your goal is to perform **video temporal grounding** by identifying the start and end timestamps in a video that match a given user query.

            ### **Guidelines**
            1. **Comprehend Video Content**: Analyze the sequence of frames to understand the event described in the query.
            2. **Understand Temporal Dynamics**: Consider motion, object interactions, and scene transitions to determine when the event occurs.
            3. **Handle Multiple Occurrences**: The same query may appear **multiple times** throughout the video. Identify and return all relevant time intervals.
            4. **Output Format**: Provide the results in **JSON format** as specified below.

            ### **Expected Output JSON Format**
            { "query": "<user query>", "relevant_windows": [ [<start time>, <end time>], [<start time>, <end time>], ... ] }
            User Query: <your_query>
            ''',

            "dvc": 
            '''
            Partition and identify events by dividing the video into a series of non-overlapping segments,
            determining the start and end time for each event, and arranging them in chronological order to ensure complete coverage of all video frames.
            Each event should be accompanied by a detailed description. Please follow the format:
            'From <start time1> to <end time1>, <detailed description1>.
            \n\nFrom <start time2> to <end time2>, <detailed description2>.\n\n...'
            ''',
            
            "dvc_yc":
            '''
            Analyze the cooking video by:
            1. Dividing into logical steps matching recipe steps
            2. Identifying <ingredients>, <tools>, and <actions> per step
            3. Using cooking-specific verbs: chop, simmer, knead, etc.

            Generate timestamps with:
            From [start] to [end]s: 
            - Primary action (boil water) 
            - Tool usage (using spatula) 
            - Ingredient state (chopped onions)
            - Temperature/time (medium heat for 3 mins)

            Format requirements:
            • English lowercase except proper nouns
            • Max 12 steps per video
            • Avoid decorative adjectives
            Please follow the format:
            'From <start time1> to <end time1>, <cooking step1>.
            \n\nFrom <start time2> to <end time2>, <cooking step2>.\n\n...'
            ''',

            "vhd_no_saliency": 
            '''
            You are a highly capable AI assistant trained to analyze video content and accurately identify the relevant time intervals in response to natural language queries. 
            Please watch the video carefully and identify relevant time intervals in response to the user's query. 
            For each identified time interval, provide the start and end timestamps in seconds.
            Format your answer as follows:
            Highlight 1: [start_time] - [end_time] seconds\n
            Highlight 2: [start_time] - [end_time] seconds\n
            and so on. The start and end timestamps should be within the video duration. 
            For example, if the video is 150 seconds long, and the user query is 'A person is eating.', the output is:\n
            Highlight 1: 5.0 - 10.0 seconds\n
            Highlight 2: 90.0 - 95.0 seconds\n
            In the example above, 'A person is eating.' happens at 5.0 - 10.0 seconds and 90.0 - 95.0 seconds, so we should return both of them.
            
            User Query: <your_query>
            ''',

            "vhd": 
            '''
            1. Please watch the video carefully and identify the highlight moments based on the user's query. 
            2. For each identified highlight moment, provide the timestamp in seconds and a saliency score from 0.0 to 1.0 
            indicating how relevant the moment is to the query (higher score means more relevant). 
            3. Format your answer as follows:
            Highlight 1: [start_time] - [end_time] seconds, saliency score: [score]
            Highlight 2: [start_time] - [end_time] seconds, saliency score: [score]
            The start and end time should be within the video duration. 
            4. Output less than 5 Highlights.
            5. The different Highlight windows may have different saliency scores.
            For example, if the video is 150 seconds long, and the user query is 'A person is eating.', the output is:\n
            Highlight 1: 5.0 - 10.0 seconds, saliency score: 0.5\n
            Highlight 2: 90.0 - 95.0 seconds, saliency score: 0.8\n
            In the example above, 'A person is eating.' happens at 5.0 - 10.0 seconds and 90.0 - 95.0 seconds, so we should return both of them.
            User Query: <your_query>
            ''',

            "dvc_vhd":
            '''
            You are a highly capable AI assistant trained to analyze video content and accurately identify the relevant time intervals in response to natural language queries.

            ### **Task Description**
            Your goal is to perform **video highlight detection** by identifying the start and end timestamps in a video that match a given user query.

            ### **Guidelines**
            Please watch the video carefully and identify the highlight moments based on the user's query. 

            1. Partition and identify events by dividing the video into a series of non-overlapping segments,
            determining the start and end time for each event, and arranging them in chronological order to ensure complete coverage of all video frames.
            Each event should be accompanied by a detailed description. Please follow the format:
            'From <start time1> to <end time1>, <detailed description1>.
            \n\nFrom <start time2> to <end time2>, <detailed description2>.\n\n...'
            2. Compare the user query with the detailed description of each event, and determine if the event is a highlight moment.
            3. For each identified highlight moment, provide the timestamp in seconds and a saliency score from 0.0 to 1.0 "
                "indicating how relevant the moment is to the query (higher score means more relevant). "
                "Format your answer as follows:\n"
                "Highlight 1: [start_time] - [end_time] seconds, saliency score: [score]\n"
                "Highlight 2: [start_time] - [end_time] seconds, saliency score: [score]\n"
                "And so on for each highlight moment.
            User Query: <your_query>
            '''
                ,
        }
        
        # Time instruction template
        self.time_instruction_template = "You are given a video sampled at {fps} frame per second, so input video frames are sampled at {time_points} second sequentially, so in total the video is {duration:.2f} seconds long.\n "
        
        # Initialize task-specific settings
        self._task_init()
    
    def _get_task_name(self):
        """Get the task name for the current class."""
        raise NotImplementedError("_get_task_name() must be implemented by subclasses")
    
    def _task_init(self):
        """Initialize task-specific variables and settings."""
        raise NotImplementedError("_task_init() must be implemented by subclasses")
    
    def process_video_with_timestamps(self, video_path, fps=1, add_timestamp=True):
        """Process video and optionally add timestamps to frames."""
        if os.path.isdir(video_path):
            video_frame_list = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path), key=lambda x: int(x.split(".")[0]))]
            time_list = [i * (1 / fps) for i in range(len(video_frame_list))]
        else:
            video_frame_list, time_list = read_frames_decord(video_path, sample=f"fps{fps}")
        
        if add_timestamp:
            video_frame_list = [Image.open(f) if isinstance(f, str) else f for f in video_frame_list]
            for i, video_frame in enumerate(video_frame_list):
                timestamp = time_list[i]
                text = f"{timestamp:.2f}"
                draw = ImageDraw.Draw(video_frame)
                
                img_width = video_frame.size[0]
                font_size = img_width // 10
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = 0
                text_y = 0
                
                draw.rectangle([text_x-5, text_y-5, text_x+text_width+5, text_y+text_height+5], 
                            fill=(0, 0, 0, 128))
                draw.text((text_x, text_y), text, fill='white', font=font)
        
        return video_frame_list, time_list
    
    def get_task_prompt(self, query=None, time_list=None, fps=1, add_time_instruction=False, add_dense_video_cap=False, dense_video_cap_dir=None, vid=None):
        """
        Generate a task-specific prompt, optionally with time information and dense video captions.
        
        This base implementation should be overridden by subclasses for task-specific customization.
        """
        raise NotImplementedError("get_task_prompt() must be implemented by subclasses")
    
    def generate_messages(self, video_path, prompt, fps=1, input_type="video", add_timestamp=True, max_pixels=336*336, min_pixels=336*336):
        """Generate message structure for the model based on input type."""
        # Get video frames and timestamps
        video_frame_list, time_list = self.process_video_with_timestamps(
            video_path, 
            fps=fps, 
            add_timestamp=add_timestamp,
        )
        
        # For images input type, create a message with multiple images
        if input_type == "images":
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[
                        {
                            "type": "image",
                            "image": video_frame,
                            "max_pixels": max_pixels,
                            "min_pixels": min_pixels,
                        }
                        for video_frame in video_frame_list
                        ],
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        # For video input type, create a message with a video
        elif input_type == "video":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": fps,
                            "max_pixels": max_pixels,
                            "min_pixels": min_pixels,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            raise ValueError(f"Invalid input_type: {input_type}. Must be 'video' or 'images'")
        
        return messages, time_list
    
    def inference(self, video_path, prompt, max_new_tokens=2048, max_pixels=336*336, min_pixels=336*336, fps=1):
        """Run inference on a video with a specific prompt."""
        messages, time_list = self.generate_messages(
            video_path, 
            prompt, 
            fps=fps, 
            input_type=self.args.input_type,
            add_timestamp=self.args.add_timestamp,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
        fps_inputs = video_kwargs.get('fps', fps)
        
        if video_inputs and len(video_inputs) > 0:
            num_frames, _, resized_height, resized_width = video_inputs[0].shape
            if self.verbose:
                print("User Query: ", prompt)
                print('video fps: ', fps_inputs)
                print("video input:", video_inputs[0].shape)
                print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
        
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Get video basename for parameter saving
        # video_basename = os.path.basename(video_path)
        # video_name = os.path.splitext(video_basename)[0]
        
        # # Save inference parameters
        # self.save_inference_params(video_name, prompt, max_new_tokens, max_pixels, min_pixels, fps, fps_inputs)
        
        return output_text[0], time_list
    
    def save_inference_params(self, prompt, max_new_tokens, max_pixels, min_pixels, fps):
        """
        Only save at the start of the inference
        Save inference parameters, model information, prompts, and other settings to a JSON file.
        
        Args:
            prompt (str): The prompt used for inference.
            max_new_tokens (int): Maximum number of new tokens to generate.
            max_pixels (int): Maximum number of pixels in the processed video.
            min_pixels (int): Minimum number of pixels in the processed video.
            fps (float): Frames per second used for processing.
        """
        # Create params directory if it doesn't exist
        params_dir = os.path.join(self.output_dir, "params")
        os.makedirs(params_dir, exist_ok=True)
        
        # Prepare parameter dictionary
        params = {
            "timestamp": str(datetime.datetime.now()),
            "model": {
                "model_path": self.args.model_path,
                "model_name": self.model_name
            },
            "task": self.task,
            "input_type": self.args.input_type,
            "base_prompt": prompt,
            "inference_params": {
                "max_new_tokens": max_new_tokens,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "fps": fps,
            },
            "args": {
                k: v for k, v in vars(self.args).items() 
                if not callable(v) and not k.startswith('__')
            }
        }
        
        # Save to JSON file
        params_file = os.path.join(params_dir, f"params_{str(datetime.datetime.now())}.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        if self.verbose:
            print(f"Inference parameters saved to {params_file}")
    
    def parse_response(self, response_text):
        """Parse the model's response. To be implemented by subclasses."""
        raise NotImplementedError("parse_response() must be implemented by subclasses")
    
    def run(self):
        """
        Main method to run the inference pipeline.
        This should be implemented by the subclasses.
        """
        raise NotImplementedError("run() must be implemented by subclasses")