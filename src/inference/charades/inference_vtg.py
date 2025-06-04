import os
import json
import re
from tqdm import tqdm
from inference.inference_base import InferBase
from inference.config_utils import setup_parser_with_config_support, parse_args_with_config, save_config

class VTGInference(InferBase):
    def _get_task_name(self):
        return "vtg"
    
    def _task_init(self):
        # List of query IDs that cause CUDA errors (known problematic cases)
        self.qid_cause_cuda_error = set({13404, 14515, 14757, 14758, 14901, 14902, 14903, 15154, 15155, 15156, 15157, 15158, 16003, 16004, 16005, 16006, 16007, 16008, 16013, 16014, 16020})
    
    def get_task_prompt(self, query=None, time_list=None, fps=1, add_time_instruction=False):
        """Generate a task-specific prompt for VTG task."""
        base_prompt = self.base_prompts["vtg"]
        # Replace query placeholder
        formatted_prompt = base_prompt.replace("<your_query>", query)
        # Add time instruction if requested and time_list is available
        if add_time_instruction and time_list is not None:
            time_points = ", ".join([f"{t:.2f}" for t in time_list])
            duration = time_list[-1] + 1
            time_instruction = self.time_instruction_template.format(
                fps=fps,
                time_points=time_points,
                duration=duration
            )
            # Insert the time instruction at an appropriate point
            formatted_prompt = formatted_prompt.replace("### **Task Description**", 
                                                      time_instruction + "### **Task Description**")
        
        return formatted_prompt
    
    def _clean_response_text(self, response_text):
        """Clean and prepare the response text for JSON parsing."""
        # Remove whitespace
        cleaned_text = response_text.strip()
        
        # Remove markdown code blocks if present
        cleaned_text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', cleaned_text, flags=re.DOTALL)
        
        return cleaned_text
    
    def _normalize_window_format(self, window, i, windows):
        """
        Normalize a single window format to [start_time, end_time] format.
        
        Args:
            window: The window to normalize
            i: Current index in the windows list
            windows: Full list of windows
            
        Returns:
            tuple: (normalized_window, next_index)
                normalized_window: [start_time, end_time] or None if cannot normalize
                next_index: The next index to process after this window
        """
        next_idx = i + 1
        
        # Standard format [start, end]
        if isinstance(window, list) and len(window) == 2:
            try:
                # Handle format like [0.0, 1.0 seconds] by extracting just the numbers
                start = window[0]
                end = window[1]
                
                # Convert string with "seconds" to float
                if isinstance(start, str) and "seconds" in start:
                    start = float(start.replace("seconds", "").strip())
                elif isinstance(start, str):
                    start = float(start)
                    
                if isinstance(end, str) and "seconds" in end:
                    end = float(end.replace("seconds", "").strip())
                elif isinstance(end, str):
                    end = float(end)
                    
                return [float(start), float(end)], next_idx
            except (ValueError, TypeError):
                print(f"Warning: Cannot convert window {window} to float list")
                return None, next_idx
        
        # Dictionary format {'start_time': '0.0', 'end_time': '6.5'}
        elif isinstance(window, dict) and 'start_time' in window and 'end_time' in window:
            try:
                start = window['start_time']
                end = window['end_time']
                
                # Handle "seconds" in dictionary values
                if isinstance(start, str) and "seconds" in start:
                    start = float(start.replace("seconds", "").strip())
                else:
                    start = float(start)
                    
                if isinstance(end, str) and "seconds" in end:
                    end = float(end.replace("seconds", "").strip())
                else:
                    end = float(end)
                    
                return [start, end], next_idx
            except (ValueError, TypeError):
                print(f"Warning: Cannot convert dictionary window {window} to float list")
                return None, next_idx
        
        # Single number, try to pair with next number
        elif isinstance(window, (int, float)) or (isinstance(window, str) and window.replace('.', '', 1).isdigit()):
            if i < len(windows) - 1:
                next_window = windows[i + 1]
                if isinstance(next_window, (int, float)) or (isinstance(next_window, str) and next_window.replace('.', '', 1).isdigit()):
                    try:
                        start = float(window)
                        end = float(next_window)
                        if start < end:
                            return [start, end], next_idx + 1  # Skip next number
                    except (ValueError, TypeError):
                        print(f"Warning: Cannot convert numeric window pair {window},{next_window} to float list")
        
        return None, next_idx
    
    def _normalize_windows(self, windows):
        """Normalize all windows to a consistent format."""
        if not isinstance(windows, list):
            windows = [windows]
        
        normalized_windows = []
        i = 0
        while i < len(windows):
            normalized_window, next_idx = self._normalize_window_format(windows[i], i, windows)
            if normalized_window is not None:
                normalized_windows.append(normalized_window)
            i = next_idx
        
        return normalized_windows
    
    def _extract_from_text(self, text):
        """
        Extract query and windows from text when JSON parsing fails.
        
        Args:
            text: The text to extract information from
            
        Returns:
            dict: A dictionary with query and relevant_windows
        """
        # Try to extract query
        match = re.search(r'"query"\s*:\s*"([^"]+)"', text)
        query = match.group(1) if match else "unknown query"
        
        # Find time windows, including processing formats with "seconds"
        windows_match = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*(?:seconds)?\s*,\s*(\d+(?:\.\d+)?)\s*(?:seconds)?\s*\]', text)
        windows = [[float(start), float(end)] for start, end in windows_match]
        
        if self.verbose:
            print(f"Extracted query: {query}")
            print(f"Extracted windows: {windows}")
        
        return {
            "query": query,
            "relevant_windows": windows
        }
    
    def parse_response(self, response_text):
        """Parse JSON response for VTG task, handling various formats."""
        # Clean the response text
        cleaned_text = self._clean_response_text(response_text)
        
        # Try to parse the JSON
        try:
            result = json.loads(cleaned_text)
            
            # If result is a list, take the first element if it exists
            if isinstance(result, list) and len(result) > 0:
                if self.verbose:
                    print("Response was a list, taking first element")
                result = result[0]
            
            # Normalize relevant_windows format
            if 'relevant_windows' in result:
                result['relevant_windows'] = self._normalize_windows(result['relevant_windows'])
                
            return result
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Failed to parse JSON: {e}")
                print(f"Cleaned text: {cleaned_text}")
            
            # If parsing fails, try to extract information from text
            return self._extract_from_text(cleaned_text)
    
    def run(self):
        """Run the VTG inference pipeline on the test dataset."""
        # Count total items to process for better progress tracking
        total_items = len(self.test_data) if self.args.max_items == -1 else self.args.max_items
        if hasattr(self.args, 'max_items') and self.args.max_items > 0:
            total_items = min(total_items, self.args.max_items)
            if self.verbose:
                print(f"Limiting processing to {total_items} items as specified by max_items")
        
        processed_items = 0
        skipped_items = 0
        successful_items = 0
        failed_items = 0
        progress_bar = tqdm(total=total_items, desc="Processing VTG", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for gt_data in self.test_data[:total_items]:
            vid = gt_data["vid"]
            qid = gt_data["qid"]
            
            # Skip known problematic cases
            if qid in self.qid_cause_cuda_error:
                skipped_items += 1
                processed_items += 1
                progress_bar.set_postfix(successful=successful_items, failed=failed_items, skipped=skipped_items)
                progress_bar.update(1)
                continue
            
            # Check if output file already exists
            output_file = os.path.join(self.output_dir, f"result_{qid}.json")
            if os.path.exists(output_file) and not self.args.overwrite:
                if self.verbose:
                    print(f"Output file for qid {qid} already exists. Skipping...")
                skipped_items += 1
                processed_items += 1
                progress_bar.set_postfix(successful=successful_items, failed=failed_items, skipped=skipped_items)
                progress_bar.update(1)
                continue
            
            # Get video path and query
            query = gt_data["query"]
            video_path = os.path.join(self.videos_dir, f"{vid}.mp4")
            
            if not os.path.exists(video_path):
                if self.verbose:
                    print(f"Video file {video_path} not found. Skipping qid {qid}.")
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
            
            # Generate task-specific prompt with query and optional enhancements
            formatted_prompt = self.get_task_prompt(
                query=query,
                time_list=time_list,
                fps=self.args.fps,
                add_time_instruction=self.args.add_time_instruction
            )
            
            # Run inference
            try:
                response, time_list = self.inference(
                    video_path, 
                    formatted_prompt, 
                    max_new_tokens=512, 
                    max_pixels=self.args.max_pixels, 
                    min_pixels=self.args.min_pixels, 
                    fps=self.args.fps
                )
                
                if self.verbose:
                    print(f"qid: {qid}, vid: {vid}")
                    print(f"Response: {response}")
                
                # Parse the response
                result = self.parse_response(response)
                
                # Add metadata
                result["qid"] = qid
                result["vid"] = vid

                # Save results
                with open(output_file, 'w') as out_f:
                    json.dump(result, out_f, indent=2)
                
                successful_items += 1
                
            except Exception as e:
                print(f"Failed to process qid {qid}: {e}")
                failed_items += 1
            
            processed_items += 1
            progress_bar.set_postfix(successful=successful_items, failed=failed_items, skipped=skipped_items)
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total items: {total_items}")
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
        parser.add_argument("--max_items", type=int, default=-1, help="Maximum number of items to process, -1 for all items")
        
        # Add support for configuration files
        parser = setup_parser_with_config_support(parser)
        
        # Add option to save current configuration
        parser.add_argument("--save_config", type=str, help="Save current configuration to specified file")
        
        return parse_args_with_config(parser)
    
    args = arg_parser()
    
    # Save configuration if requested
    if hasattr(args, 'save_config') and args.save_config:
        save_config(args, args.save_config)
    
    vtg_inference = VTGInference(args)
    vtg_inference.run()
