import os
import json
from tqdm import tqdm
from inference.inference_base import InferBase
from inference.config_utils import setup_parser_with_config_support, parse_args_with_config, save_config

class DVCInference(InferBase):
    def _get_task_name(self):
        return "dvc"
    
    def _task_init(self):
        # Initialize any task-specific variables
        self.videos_set = set()
    
    def get_task_prompt(self, query=None, time_list=None, fps=1, add_time_instruction=False, add_dense_video_cap=False, dense_video_cap_dir=None, vid=None):
        """Generate a task-specific prompt for Dense Video Captioning task."""
        base_prompt = self.base_prompts["dvc"]
        
        # For dense_video_cap, prepend the time instruction if requested
        if add_time_instruction and time_list is not None:
            time_points = ", ".join([f"{t:.2f}" for t in time_list])
            duration = time_list[-1] + 1
            time_instruction = self.time_instruction_template.format(
                fps=fps,
                time_points=time_points,
                duration=duration
            )
            
            formatted_prompt = time_instruction + base_prompt
        else:
            formatted_prompt = base_prompt
            
        return formatted_prompt
    
    def parse_response(self, response_text):
        """
        For Dense Video Captioning, we don't need to parse the response.
        We just return the raw text as it's already in the desired format.
        """
        return response_text
    
    def run(self):
        """Run the Dense Video Captioning inference pipeline on the test dataset."""
        # Get a list of unique videos
        total_videos = list(set([gt_data["vid"] for gt_data in self.test_data]))
        
        # Apply max_items limit if specified
        if hasattr(self.args, 'max_items') and self.args.max_items > 0:
            total_videos = total_videos[:self.args.max_items]
            if self.verbose:
                print(f"Limiting processing to {len(total_videos)} videos as specified by max_items")
        
        # Count variables for progress tracking
        total_items = len(total_videos)
        processed_items = 0
        skipped_items = 0
        successful_items = 0
        failed_items = 0
        
        # Create a progress bar with information
        progress_bar = tqdm(total=total_items, desc="Processing Dense Video Captioning", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for vid in total_videos:
            # Skip if already processed
            if vid in self.videos_set:
                continue
            
            # Generate output file path
            output_file = os.path.join(self.output_dir, f"video_{vid}.json")
            
            # Skip if output file already exists and overwrite is not enabled
            if os.path.exists(output_file) and not self.args.overwrite:
                if self.verbose:
                    print(f"Output file for vid {vid} already exists. Skipping...")
                skipped_items += 1
                processed_items += 1
                progress_bar.set_postfix(successful=successful_items, failed=failed_items, skipped=skipped_items)
                progress_bar.update(1)
                continue
            
            # Get video path
            video_path = os.path.join(self.videos_dir, f"{vid}.mp4")
            
            if not os.path.exists(video_path):
                if self.verbose:
                    print(f"Video file {video_path} not found. Skipping vid {vid}.")
                skipped_items += 1
                processed_items += 1
                progress_bar.set_postfix(successful=successful_items, failed=failed_items, skipped=skipped_items)
                progress_bar.update(1)
                continue
            
            # Process video to get time_list for prompt generation
            _, time_list = self.process_video_with_timestamps(
                video_path, 
                fps=self.args.fps, 
                add_timestamp=self.args.add_timestamp,
            )
            
            # Generate task-specific prompt with time information
            formatted_prompt = self.get_task_prompt(
                time_list=time_list,
                fps=self.args.fps,
                add_time_instruction=self.args.add_time_instruction
            )
            
            try:
                # Run inference with a larger max_new_tokens for dense captioning
                response, _ = self.inference(
                    video_path, 
                    formatted_prompt, 
                    max_new_tokens=2048, 
                    max_pixels=self.args.max_pixels, 
                    min_pixels=self.args.min_pixels, 
                    fps=self.args.fps
                )
                
                # Mark as processed
                self.videos_set.add(vid)
                
                # Create result structure
                result = {
                    "vid": vid,
                    "response": response
                }
                
                # Save results
                with open(output_file, 'w') as out_f:
                    json.dump(result, out_f, indent=2)
                
                successful_items += 1
                
            except Exception as e:
                print(f"Failed to process vid {vid}: {e}")
                failed_items += 1
            
            processed_items += 1
            progress_bar.set_postfix(successful=successful_items, failed=failed_items, skipped=skipped_items)
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Save inference parameters
        self.save_inference_params(self.base_prompts["dvc"], self.args.max_new_tokens, self.args.max_pixels, self.args.min_pixels, self.args.fps)
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total videos: {total_items}")
        print(f"Successfully processed: {successful_items}")
        print(f"Failed: {failed_items}")
        print(f"Skipped: {skipped_items}")
        print(f"Results saved to: {self.output_dir}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    def arg_parser():
        parser = ArgumentParser()
        parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
        parser.add_argument("--verbose", action="store_true", help="verbose")
        parser.add_argument("--max_pixels", type=int, default=336*336, help="max pixels of each video frame")
        parser.add_argument("--min_pixels", type=int, default=336*336, help="min pixels of each video frame")
        parser.add_argument("--fps", type=int, default=1, help="fps of the video")
        parser.add_argument("--videos_dir", type=str, default="../event_vlm/eval_data/Charades_v1_480/")
        parser.add_argument("--gt_json_file", type=str, default="data/charades/charades_gt.json")
        parser.add_argument("--output_dir", type=str, default="results/charades", help="output root directory")
        parser.add_argument("--add_timestamp", action="store_true", help="Add timestamps to frames")
        parser.add_argument("--add_time_instruction", action="store_true", help="Add time information to prompt")
        parser.add_argument("--input_type", type=str, default="images", choices=["video", "images"])
        parser.add_argument("--hardware", type=str, default="h100", choices=["h100", "3090", "l40s"])
        parser.add_argument("--overwrite", action="store_true", help="overwrite existing results")
        parser.add_argument("--disable_flash_attn2", action="store_true", help="disable flash attention 2")
        parser.add_argument("--max_items", type=int, default=-1, help="Maximum number of videos to process, -1 for all videos")
        
        # Add support for configuration files
        parser = setup_parser_with_config_support(parser)
        
        # Add option to save current configuration
        parser.add_argument("--save_config", type=str, help="Save current configuration to specified file")
        
        return parse_args_with_config(parser)
    
    args = arg_parser()
    
    # Save configuration if requested
    if hasattr(args, 'save_config') and args.save_config:
        save_config(args, args.save_config)
    
    dvc_inference = DVCInference(args)
    dvc_inference.run()
