import os
import json
import re
import torch
from tqdm import tqdm
from inference.inference_vtg import VTGInference
from inference.inference_dvc import DVCInference
from inference.config_utils import setup_parser_with_config_support, parse_args_with_config, save_config

class FinalVTG(VTGInference):
    """
    A class that combines VTG inference with DVC results to refine temporal grounding predictions.
    This is an implementation that integrates the functionality of the postprocess.py script.
    """
    
    def _task_init(self):
        """Initialize task-specific variables and settings."""
        super()._task_init()  # Initialize VTG-specific variables
        
        # Additional attributes for refinement
        self.dvc_output_dir = None
        self.min_window_duration = 4.0  # Default value, can be set via args
        self.refined_results = {}
        self.dvc_inference = None  # Will be initialized when needed
        
    def _get_task_name(self):
        """Get the task name for the current class."""
        return "refined_vtg"
    
    def set_dvc_dir(self, dvc_output_dir):
        """Set the directory containing DVC inference results."""
        self.dvc_output_dir = dvc_output_dir
        if not os.path.exists(dvc_output_dir):
            print(f"Warning: DVC output directory '{dvc_output_dir}' does not exist")
            # Create the directory since we might generate captions
            os.makedirs(dvc_output_dir, exist_ok=True)
    
    def _init_dvc_inference(self):
        """Initialize DVCInference if not already initialized."""
        if self.dvc_inference is None:
            # Create a new args object for DVC that copies relevant settings from this instance
            from types import SimpleNamespace
            dvc_args = SimpleNamespace()
            
            # Copy relevant attributes from self.args
            for attr in dir(self.args):
                if not attr.startswith('_'):
                    setattr(dvc_args, attr, getattr(self.args, attr))
            
            # Set DVC-specific output directory
            if self.dvc_output_dir:
                dvc_args.output_dir = self.dvc_output_dir
            
            # Initialize DVCInference with the same model
            self.dvc_inference = DVCInference(dvc_args)
            self.dvc_inference.processor = self.processor
            self.dvc_inference.model = self.model
            
            if self.verbose:
                print("Initialized DVCInference for generating dense captions")
    
    @torch.inference_mode()
    def generate_dense_cap(self, vid):
        """Generate dense video caption for a specific video using DVCInference."""
        if self.dvc_inference is None:
            self._init_dvc_inference()
            
        video_path = os.path.join(self.videos_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            if self.verbose:
                print(f"Video file {video_path} not found. Cannot generate dense caption.")
            return ""
            
        # Generate output file path 
        output_file = os.path.join(self.dvc_output_dir, f"video_{vid}.json")
            
        try:
            if self.verbose:
                print(f"Generating dense caption for video {vid}...")
                
            # Process video to get time_list for prompt generation
            _, time_list = self.dvc_inference.process_video_with_timestamps(
                video_path, 
                fps=self.args.fps, 
                add_timestamp=self.args.add_timestamp,
            )
            
            # Generate task-specific prompt
            formatted_prompt = self.dvc_inference.get_task_prompt(
                time_list=time_list,
                fps=self.args.fps,
                add_time_instruction=self.args.add_time_instruction
            )
            
            # Run inference with a larger max_new_tokens for dense captioning
            response, _ = self.dvc_inference.inference(
                video_path, 
                formatted_prompt, 
                max_new_tokens=2048, 
                max_pixels=self.args.max_pixels, 
                min_pixels=self.args.min_pixels, 
                fps=self.args.fps
            )
            
            # Create result structure
            result = {
                "vid": vid,
                "response": response
            }
            
            # Save results
            with open(output_file, 'w') as out_f:
                json.dump(result, out_f, indent=2)
                
            if self.verbose:
                print(f"Dense caption generated and saved to {output_file}")
                
            return response
                
        except Exception as e:
            print(f"Failed to generate dense caption for vid {vid}: {e}")
            return ""
    
    def load_dense_cap(self, vid):
        """
        Load dense video caption for a specific video.
        If caption file doesn't exist, generate it on the fly.
        """
        if not self.dvc_output_dir:
            if self.verbose:
                print("DVC output directory not set. Setting to default location.")
            # Create a default DVC output directory
            base_output_dir = os.path.dirname(os.path.dirname(self.output_dir))
            self.dvc_output_dir = os.path.join(base_output_dir, "dvc", f"{os.path.basename(self.args.model_path)}_dvc_captions")
            os.makedirs(self.dvc_output_dir, exist_ok=True)
            if self.verbose:
                print(f"Using DVC output directory: {self.dvc_output_dir}")
            
        caption_file = os.path.join(self.dvc_output_dir, f"video_{vid}.json")
        if os.path.exists(caption_file):
            try:
                with open(caption_file, 'r') as f:
                    caption_data = json.load(f)
                    if self.verbose:
                        print(f"Loaded existing dense caption for video {vid}")
                    return caption_data.get("response", "")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading dense caption for {vid}: {e}")
                    print(f"Will attempt to generate a new caption.")
        else:
            if self.verbose:
                print(f"Dense caption file not found: {caption_file}")
                print(f"Generating new dense caption for video {vid}...")
                
        # If we reach here, we need to generate the caption
        return self.generate_dense_cap(vid)
    
    def create_refinement_prompt(self, vtg_result, dense_caption):
        """Create prompt for refining VTG result using dense caption."""
        query = vtg_result.get("query", "")
        windows = vtg_result.get("relevant_windows", [])
        
        # Format original time windows for better readability
        windows_str = "["
        for i, window in enumerate(windows):
            # Handle windows that might be strings or floats
            try:
                start = float(window[0])
                end = float(window[1])
                windows_str += f"[{start:.1f}, {end:.1f}]"
            except (ValueError, TypeError, IndexError):
                # If conversion fails, use the raw values
                windows_str += f"[{window[0]}, {window[1]}]"
                
            if i < len(windows) - 1:
                windows_str += ", "
        windows_str += "]"
        
        prompt = f"""You are an intelligent video understanding assistant. I need your help to refine video temporal grounding results.

I have a video with the following dense caption description of its content:
{dense_caption}

For the query: "{query}"

Instructions:
1. First analyze the dense caption and the initial time windows ({windows_str}) to find the most relevant time segments for the query.
2. Merge time windows if they describe the same or closely related events.
3. Each time window MUST be at least {self.min_window_duration} seconds long. Extend shorter windows to include context.
4. Structure your response as follows:
   - Put your reasoning, analysis, and explanations inside `<think>` tags.
   - Put the final refined time window(s) inside `<answer>` tags in the format [[start_time, end_time]]. Ensure there is only one time window list inside the answer tags.

Example:
<think>
Based on the dense caption, the person performs the action X between time A and B. This matches the query. The duration is sufficient.
</think>
<answer>[[A, B]]</answer>

Another Example (window extension needed):
<think>
The query describes action Y, which happens between time C and D. The duration D-C is less than {self.min_window_duration}s. I will extend it to [{self.min_window_duration}s]. The extended window is [C_new, D_new].
</think>
<answer>[[C_new, D_new]]</answer>

IMPORTANT: Always provide the final answer within the <answer> tags using the specified format [[start_time, end_time]].
"""
        return prompt
    
    def ensure_min_window_duration(self, windows):
        """Ensure all time windows have at least the minimum duration."""
        adjusted_windows = []
        
        for window in windows:
            if len(window) != 2:
                # Skip invalid windows
                continue
                
            start, end = float(window[0]), float(window[1])
            duration = end - start
            
            if duration < self.min_window_duration:
                if self.verbose:
                    print(f"Window [{start:.2f}, {end:.2f}] duration ({duration:.2f}s) is less than minimum ({self.min_window_duration}s)")
                
                # Expand the window to meet minimum duration
                # Try to expand equally on both sides when possible
                extension = self.min_window_duration - duration
                start_extension = extension / 2
                end_extension = extension / 2
                
                # Ensure start time doesn't go below 0
                if start - start_extension < 0:
                    start_extension = start
                    end_extension = extension - start_extension
                
                new_start = max(0, start - start_extension)
                new_end = end + end_extension
                
                if self.verbose:
                    print(f"Adjusted to [{new_start:.2f}, {new_end:.2f}] (duration: {new_end - new_start:.2f}s)")
                
                adjusted_windows.append([new_start, new_end])
            else:
                adjusted_windows.append([start, end])
        
        return adjusted_windows
    
    def parse_refinement_response(self, response_text, original_windows):
        """Parse time windows from the <answer> tag in the model response."""
        # Clean up response
        cleaned_text = response_text.strip()

        # Find content within <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned_text, re.DOTALL | re.IGNORECASE)

        if answer_match:
            answer_content = answer_match.group(1).strip()
            # Update regex to match [start, end seconds] or [start, end]
            windows_match = re.search(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)[^\]]*\]', answer_content)

            if windows_match:
                try:
                    # Extract numbers and build time window list
                    start_time = float(windows_match.group(1))
                    end_time = float(windows_match.group(2))
                    windows = [[start_time, end_time]] # Build list directly

                    # Ensure time windows meet minimum duration requirements
                    windows = self.ensure_min_window_duration(windows)
                    if self.verbose:
                        print(f"Successfully parsed windows from <answer>: {windows}")
                    return {"relevant_windows": windows}
                except (ValueError, IndexError) as e:
                    if self.verbose:
                        print(f"Error extracting/converting numbers inside <answer>: {e}")
                        print(f"Content inside <answer>: {answer_content}")
            else:
                 if self.verbose:
                    print(f"Could not find valid window format [start, end...] inside <answer> tag.")
                    print(f"Content inside <answer>: {answer_content}")

        # If <answer> tag not found or parsing failed, try previous JSON or text extraction logic as fallback
        if self.verbose:
            print("Could not parse from <answer> tag, falling back to previous parsing methods...")

        # --- Fallback Logic (maintain previous parsing logic) ---
        try:
            # Remove markdown code blocks if present
            json_text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', cleaned_text, flags=re.DOTALL)
            result = json.loads(json_text)
            if result.get("relevant_windows"):
                windows = self.ensure_min_window_duration(result["relevant_windows"])
                if self.verbose:
                     print(f"Fallback: Parsed JSON format: {windows}")
                return {"relevant_windows": windows}
        except json.JSONDecodeError:
            if self.verbose:
                print("Fallback: Failed to parse JSON format, trying to extract time windows from text...")

        windows = []
        time_patterns = [
            r"from (\d+\.?\d*) to (\d+\.?\d*) seconds",
            r"(\d+\.?\d*) to (\d+\.?\d*) seconds",
            r"(\d+\.?\d*)-(\d+\.?\d*) seconds",
            r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]" # Also try to find bracket format
        ]
        extracted_windows = []
        for pattern in time_patterns:
            matches = re.findall(pattern, cleaned_text.lower())
            if matches:
                for match in matches:
                    try:
                        # Handle different group structures from regex
                        start_time = float(match[0])
                        end_time = float(match[1])
                        extracted_windows.append([start_time, end_time])
                    except (ValueError, IndexError) as e:
                        if self.verbose:
                            print(f"Fallback: Error parsing time values: {match}, Error: {e}")

        if extracted_windows:
            relevant_window = None
            relevance_patterns = [
                r"most relevant.*?(\d+\.?\d*) to (\d+\.?\d*) seconds",
                r"relevant window.*?(\d+\.?\d*) to (\d+\.?\d*) seconds",
                r"best match.*?(\d+\.?\d*) to (\d+\.?\d*) seconds"
            ]
            for pattern in relevance_patterns:
                match = re.search(pattern, cleaned_text.lower())
                if match:
                    try:
                        start_time = float(match.group(1))
                        end_time = float(match.group(2))
                        # Find the closest extracted window to the identified relevant one
                        min_diff = float('inf')
                        for w in extracted_windows:
                            diff = abs(w[0] - start_time) + abs(w[1] - end_time)
                            if diff < min_diff:
                                min_diff = diff
                                relevant_window = w
                        break
                    except (ValueError, IndexError):
                        continue

            if relevant_window:
                windows = [relevant_window]
            elif extracted_windows: # If relevance not found, take the last found window? Or first? Let's take the last one for now.
                 windows = [extracted_windows[-1]]

            windows = self.ensure_min_window_duration(windows)
            if self.verbose:
                print(f"Fallback: Extracted windows from text: {windows}")

        # --- End of Fallback Logic ---

        # If all methods fail, use original windows
        if not windows and original_windows:
            if self.verbose:
                print("All parsing methods failed, using original windows")
            windows = self.ensure_min_window_duration(original_windows)
        elif not windows and not original_windows:
             if self.verbose:
                 print("All parsing methods failed and no original windows available.")
             windows = [] # Return empty if nothing found and no original

        return {
            "relevant_windows": windows
        }
    
    @torch.inference_mode()
    def refine_vtg_result(self, qid, vtg_result):
        """Refine a VTG result using dense video caption."""
        vid = vtg_result.get("vid", "")
        if not vid:
            if self.verbose:
                print(f"Missing video ID in VTG result for qid {qid}. Cannot refine.")
            return vtg_result
        
        # Load dense caption or generate it if not found
        dense_caption = self.load_dense_cap(vid)
        if not dense_caption:
            if self.verbose:
                print(f"Failed to generate dense caption for video {vid}. Using original VTG result.")
            return vtg_result
        
        # Create refinement prompt
        prompt = self.create_refinement_prompt(vtg_result, dense_caption)
        
        if self.verbose:
            print(f"Refining query: {vtg_result.get('query', '')}")
            print(f"Refinement prompt: {prompt}")
        
        # Prepare messages for model - text only
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        # Process with model
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # No image or video inputs for refinement
        inputs = self.processor(
            text=[text], 
            padding=True, 
            return_tensors="pt"
        )
        inputs = inputs.to('cuda')
        
        output_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        if self.verbose:
            print(f"Refinement output: {output_text}")
            
        # Parse response with original windows as fallback
        original_windows = vtg_result.get("relevant_windows", [])
        
        # Ensure original windows also meet minimum duration requirements
        if original_windows:
            original_windows = self.ensure_min_window_duration(original_windows)
        
        refinement_result = self.parse_refinement_response(output_text, original_windows)
        
        # Create final refined result
        final_result = {
            "qid": qid,
            "vid": vid,
            "query": vtg_result.get("query", ""),
            "original_windows": original_windows,
            "relevant_windows": refinement_result.get("relevant_windows", original_windows),
            "model_response": output_text
        }
        
        # Store refined result
        self.refined_results[qid] = final_result
        
        return final_result
    
    @torch.inference_mode()
    def run_refinement(self):
        """Run only the refinement part without VTG inference."""
        # Initialize DVC inference if needed
        if self.dvc_output_dir and not os.path.exists(self.dvc_output_dir):
            if self.verbose:
                print(f"Creating DVC output directory: {self.dvc_output_dir}")
            os.makedirs(self.dvc_output_dir, exist_ok=True)
            
        # Refine the results using DVC captions
        print("Refining VTG results with dense video captions...")
        
        # Get list of all VTG result files
        vtg_files = [f for f in os.listdir(self.output_dir) if f.startswith("result_") and f.endswith(".json")]
        
        # Apply max_items limit if specified
        if hasattr(self.args, 'max_items') and self.args.max_items > 0:
            vtg_files = vtg_files[:self.args.max_items]
            print(f"Limiting refinement to {len(vtg_files)} items as specified by max_items={self.args.max_items}")
        
        # Create refined output directory
        refined_output_dir = os.path.join(self.output_dir, "refined")
        os.makedirs(refined_output_dir, exist_ok=True)
        
        # Process all files for refinement
        progress_bar = tqdm(total=len(vtg_files), desc="Refining VTG results", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        refined_count = 0
        skipped_count = 0
        
        for vtg_file in vtg_files:
            # Extract qid from filename
            qid = vtg_file.replace("result_", "").replace(".json", "")
            
            # Define output file path for refined result
            output_file = os.path.join(refined_output_dir, f"result_{qid}.json")
            
            # Skip if output file already exists and overwrite is not enabled
            if os.path.exists(output_file) and not self.args.overwrite:
                if self.verbose:
                    print(f"Refined output file for qid {qid} already exists. Skipping...")
                skipped_count += 1
                progress_bar.update(1)
                continue
                
            # Load VTG result
            vtg_result_path = os.path.join(self.output_dir, vtg_file)
            try:
                with open(vtg_result_path, 'r') as f:
                    vtg_result = json.load(f)
            except Exception as e:
                print(f"Error loading VTG result for qid {qid}: {e}")
                skipped_count += 1
                progress_bar.update(1)
                continue
                
            # Refine the result
            try:
                refined_result = self.refine_vtg_result(qid, vtg_result)
                
                # Save refined result
                with open(output_file, 'w') as f:
                    json.dump(refined_result, f, indent=2)
                    
                refined_count += 1
                
            except Exception as e:
                print(f"Error refining result for qid {qid}: {e}")
                skipped_count += 1
                
            progress_bar.update(1)
            
        progress_bar.close()
        
        # Print summary
        print(f"\nRefinement complete!")
        print(f"Total items: {len(vtg_files)}")
        print(f"Successfully refined: {refined_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Refined results saved to: {refined_output_dir}")
    
    @torch.inference_mode()
    def run(self):
        """
        Run a streamlined pipeline that processes each item one by one:
        1. Run VTG inference for a single query
        2. Immediately refine that result with DVC
        3. Move to the next query
        This is more efficient than processing all VTG first, then all refinements.
        """        
        # Apply max_items limit if specified
        if hasattr(self.args, 'max_items') and self.args.max_items > 0:
            processed_data = self.test_data[:self.args.max_items]
            if self.verbose:
                print(f"Limiting processing to {len(processed_data)} items as specified by max_items")
        else:
            processed_data = self.test_data
        




        # Create output directories
        vtg_output_dir = self.output_dir
        refined_output_dir = os.path.join(self.output_dir, "refined")
        os.makedirs(vtg_output_dir, exist_ok=True)
        os.makedirs(refined_output_dir, exist_ok=True)
        

        # Save inference parameters
        base_prompts = f"vtg:\n{self.base_prompts["vtg"]}\n\ndvc:\n{self.base_prompts["dvc"]}"
        self.save_inference_params(base_prompts, self.args.max_new_tokens, self.args.max_pixels, self.args.min_pixels, self.args.fps)
        
        # Track progress
        processed_items = 0
        skipped_items = 0
        successful_vtg = 0
        successful_refine = 0
        failed_items = 0
        
        # Create a progress bar
        progress_bar = tqdm(total=len(processed_data), desc="Processing VTG+Refinement", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for gt_data in processed_data:
            qid = gt_data["qid"]
            vid = gt_data["vid"]
            query = gt_data["query"]
            
            # Define output files
            vtg_output_file = os.path.join(vtg_output_dir, f"result_{qid}.json")
            refined_output_file = os.path.join(refined_output_dir, f"result_{qid}.json")
            
            # Skip if refined output file already exists and overwrite is not enabled
            if os.path.exists(refined_output_file) and not self.args.overwrite:
                if self.verbose:
                    print(f"Refined output file for qid {qid} already exists. Skipping...")
                skipped_items += 1
                processed_items += 1
                progress_bar.set_postfix(vtg=successful_vtg, refined=successful_refine, failed=failed_items, skipped=skipped_items)
                progress_bar.update(1)
                continue
            
            # Get video path
            video_path = os.path.join(self.videos_dir, f"{vid}.mp4")
            
            if not os.path.exists(video_path):
                if self.verbose:
                    print(f"Video file {video_path} not found. Skipping qid {qid}.")
                failed_items += 1
                processed_items += 1
                progress_bar.set_postfix(vtg=successful_vtg, refined=successful_refine, failed=failed_items, skipped=skipped_items)
                progress_bar.update(1)
                continue
            
            vtg_result = None
            
            # STEP 1: VTG Inference (or load existing result)
            if os.path.exists(vtg_output_file) and not self.args.overwrite:
                # Load existing VTG result
                try:
                    with open(vtg_output_file, 'r') as f:
                        vtg_result = json.load(f)
                    if self.verbose:
                        print(f"Loaded existing VTG result for qid {qid}")
                    successful_vtg += 1
                except Exception as e:
                    print(f"Error loading VTG result for qid {qid}: {e}")
                    failed_items += 1
                    processed_items += 1
                    progress_bar.set_postfix(vtg=successful_vtg, refined=successful_refine, failed=failed_items, skipped=skipped_items)
                    progress_bar.update(1)
                    continue
            else:
                # Run VTG inference for this query
                try:
                    # Process video to get time_list for prompt generation
                    _, time_list = self.process_video_with_timestamps(
                        video_path, 
                        fps=self.args.fps, 
                        add_timestamp=self.args.add_timestamp,
                    )
                    
                    # Generate task-specific prompt
                    formatted_prompt = self.get_task_prompt(
                        query=query,
                        time_list=time_list,
                        fps=self.args.fps,
                        add_time_instruction=self.args.add_time_instruction
                    )
                    
                    # Run inference
                    response, _ = self.inference(
                        video_path, 
                        formatted_prompt, 
                        max_new_tokens=512, 
                        max_pixels=self.args.max_pixels, 
                        min_pixels=self.args.min_pixels, 
                        fps=self.args.fps
                    )
                    
                    # Parse response
                    result = self.parse_response(response)
                    
                    # Add metadata
                    vtg_result = {
                        "qid": qid,
                        "vid": vid,
                        "query": query,
                        "relevant_windows": result.get("relevant_windows", [])
                    }
                    
                    # Save VTG result
                    with open(vtg_output_file, 'w') as out_f:
                        json.dump(vtg_result, out_f, indent=2)
                    
                    successful_vtg += 1
                    
                except Exception as e:
                    print(f"Failed to process VTG for qid {qid}: {e}")
                    failed_items += 1
                    processed_items += 1
                    progress_bar.set_postfix(vtg=successful_vtg, refined=successful_refine, failed=failed_items, skipped=skipped_items)
                    progress_bar.update(1)
                    continue
            
            # STEP 2: Refinement with DVC
            try:
                # Refine the result
                refined_result = self.refine_vtg_result(qid, vtg_result)
                
                # Save refined result
                with open(refined_output_file, 'w') as f:
                    json.dump(refined_result, f, indent=2)
                
                successful_refine += 1
                
            except Exception as e:
                print(f"Error refining result for qid {qid}: {e}")
                # Still count as processed even if refinement fails
            
            processed_items += 1
            progress_bar.set_postfix(vtg=successful_vtg, refined=successful_refine, failed=failed_items, skipped=skipped_items)
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total items: {len(processed_data)}")
        print(f"Successfully processed VTG: {successful_vtg}")
        print(f"Successfully refined: {successful_refine}")
        print(f"Failed: {failed_items}")
        print(f"Skipped: {skipped_items}")
        print(f"Results saved to: {self.output_dir} and {refined_output_dir}")

    @torch.inference_mode()
    def run_only_refinement(self):
        """
        Run only the refinement part on existing VTG results,
        processing each result one by one from the specified VTG results directory.
        """
        if not os.path.exists(self.output_dir):
            print(f"VTG results directory not found: {self.output_dir}")
            return
            
        # Get list of all VTG result files
        vtg_files = [f for f in os.listdir(self.output_dir) if f.startswith("result_") and f.endswith(".json")]
        
        # Apply max_items limit if specified
        if hasattr(self.args, 'max_items') and self.args.max_items > 0:
            vtg_files = vtg_files[:self.args.max_items]
            print(f"Limiting refinement to {len(vtg_files)} items as specified by max_items={self.args.max_items}")
        
        # Create refined output directory
        refined_output_dir = os.path.join(self.output_dir, "refined")
        os.makedirs(refined_output_dir, exist_ok=True)
        
        # Process all files for refinement
        progress_bar = tqdm(total=len(vtg_files), desc="Refining VTG results", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        refined_count = 0
        skipped_count = 0
        failed_count = 0
        
        for vtg_file in vtg_files:
            # Extract qid from filename
            qid = vtg_file.replace("result_", "").replace(".json", "")
            
            # Define output file path for refined result
            output_file = os.path.join(refined_output_dir, f"result_{qid}.json")
            
            # Skip if output file already exists and overwrite is not enabled
            if os.path.exists(output_file) and not self.args.overwrite:
                if self.verbose:
                    print(f"Refined output file for qid {qid} already exists. Skipping...")
                skipped_count += 1
                progress_bar.update(1)
                continue
                
            # Load VTG result
            vtg_result_path = os.path.join(self.output_dir, vtg_file)
            try:
                with open(vtg_result_path, 'r') as f:
                    vtg_result = json.load(f)
            except Exception as e:
                print(f"Error loading VTG result for qid {qid}: {e}")
                failed_count += 1
                progress_bar.update(1)
                continue
                
            # Refine the result
            try:
                refined_result = self.refine_vtg_result(qid, vtg_result)
                
                # Save refined result
                with open(output_file, 'w') as f:
                    json.dump(refined_result, f, indent=2)
                    
                refined_count += 1
                
            except Exception as e:
                print(f"Error refining result for qid {qid}: {e}")
                failed_count += 1
                
            progress_bar.update(1)
            progress_bar.set_postfix(refined=refined_count, failed=failed_count, skipped=skipped_count)
            
        progress_bar.close()
        
        # Print summary
        print(f"\nRefinement complete!")
        print(f"Total items: {len(vtg_files)}")
        print(f"Successfully refined: {refined_count}")
        print(f"Failed: {failed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Refined results saved to: {refined_output_dir}")


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
        parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
        
        # Add refinement-specific arguments
        parser.add_argument("--dvc_output_dir", type=str, help="Directory containing DVC inference results")
        parser.add_argument("--min_window_duration", type=float, default=4.0, help="Minimum duration of time windows in seconds")
        parser.add_argument("--skip_vtg", action="store_true", help="Skip VTG inference and only do refinement")
        parser.add_argument("--vtg_results_dir", type=str, help="Directory containing existing VTG results (used with --skip_vtg)")
        
        # Add support for configuration files
        parser = setup_parser_with_config_support(parser)
        
        # Add option to save current configuration
        parser.add_argument("--save_config", type=str, help="Save current configuration to specified file")
        
        return parse_args_with_config(parser)
    
    args = arg_parser()
    
    # Ensure command line arguments take precedence over config file
    if args.verbose:
        print(f"Using max_items={args.max_items}")
    
    # Save configuration if requested
    if hasattr(args, 'save_config') and args.save_config:
        save_config(args, args.save_config)
    
    final_vtg = FinalVTG(args)
    
    if hasattr(args, 'min_window_duration'):
        final_vtg.min_window_duration = args.min_window_duration
    if hasattr(args, 'dvc_output_dir') and args.dvc_output_dir:
        final_vtg.set_dvc_dir(args.dvc_output_dir)
    if hasattr(args, 'vtg_results_dir') and args.vtg_results_dir:
        final_vtg.output_dir = args.vtg_results_dir
        print(f"Using {args.vtg_results_dir} as VTG output directory")
    
    if hasattr(args, 'skip_vtg') and args.skip_vtg:
        print("Skipping VTG inference and only running refinement")
        final_vtg.run_only_refinement()
    else:
        print("Using streamlined pipeline processing (VTG+refinement for each item)")
        final_vtg.run() 