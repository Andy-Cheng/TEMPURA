import os
import json
from typing import List, Dict, Tuple
import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json_path", type=str, default="data/charades/charades_gt.json")
    parser.add_argument("--inference_dir", type=str, default="results/charades/vtg/Qwen2.5-VL-3B-Instruct_images")
    return parser.parse_args()

def calculate_iou(interval_a: Tuple[float, float], 
                 interval_b: Tuple[float, float]) -> float:
    a_start, a_end = interval_a
    b_start, b_end = interval_b
    intersection_start = max(a_start, b_start)
    intersection_end = min(a_end, b_end)
    intersection = max(0.0, intersection_end - intersection_start)
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0

def batch_iou_pairs(inference_intervals: List[List[Tuple[float, float]]],
                   gt_intervals: List[List[Tuple[float, float]]]) -> Tuple[List[float], float, int]:
    total = []
    empty_count = 0
    
    for inf_windows, gt_windows in zip(inference_intervals, gt_intervals):
        if not inf_windows: 
            iou = 0.0
            empty_count += 1
        elif not gt_windows:  
            iou = 0.0
        else:
            max_iou = max(calculate_iou(inf, gt) for inf in inf_windows for gt in gt_windows)
            iou = max_iou
        total.append(iou)
    
    average = sum(total) / len(total) if total else 0.0
    return total, average, empty_count

def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_inference_results(inference_dir: str) -> List[Dict]:
    inference_data = []

    # for debug
    # ref_dir = 'eval_data/charades/inference_results/vtg/postprocessed_qwen_vl_7b_instruct'
    # for filename in os.listdir(inference_dir):
    for filename in os.listdir(inference_dir):
        if filename.startswith("result_") and filename.endswith(".json"):
            with open(os.path.join(inference_dir, filename), 'r') as f:
                inference_data.append(json.loads(f.read()))
    return inference_data

def parse_time(input_json: List[Dict], label_name: str) -> List[List[Tuple[float, float]]]:
    result = []
    for item in input_json:
        windows = []
        if label_name in item:
            # Check the type and content of relevant_windows
            relevant_windows = item[label_name]
            
            # Ensure relevant_windows is a list
            if isinstance(relevant_windows, list):
                for window in relevant_windows:
                    # Ensure window is a list or tuple
                    if isinstance(window, (list, tuple)) and len(window) == 2:
                        try:
                            windows.append((float(window[0]), float(window[1])))
                        except (ValueError, TypeError):
                            print(f"Warning: Cannot convert window {window} to float tuple")
                    else:
                        print(f"Warning: Invalid window format: {window}")
            else:
                print(f"Warning: relevant_windows is not a list: {relevant_windows}")
        else:
            print(f"Warning: {label_name} not found in item {item}")
        result.append(windows)
    return result

def eval_all(gt_json_path: str, inference_dir: str):
    label_data = load_json(gt_json_path)
    inference_data = load_inference_results(inference_dir)
    label_data.sort(key=lambda x: x['qid'])
    inference_data.sort(key=lambda x: x['qid'])

    if len(label_data) != len(inference_data):
        print(f"Warning: Number of ground truth ({len(label_data)}) and inference results ({len(inference_data)}) do not match!")
        min_len = min(len(label_data), len(inference_data))
        label_data = label_data[:min_len]
        inference_data = inference_data[:min_len]

    label_name = 'relevant_windows'
    test_time = parse_time(inference_data, label_name) 
    label_time = parse_time(label_data, label_name)    

    total, average, empty_count = batch_iou_pairs(test_time, label_time)

    print(f'Total samples: {len(label_data)}')
    print(f'Empty predictions: {empty_count} ({empty_count/len(label_data)*100:.2f}%)')
    print(f'Average IoU (including empty predictions): {average:.4f}')

    for threshold in [0.3, 0.5, 0.7]:
        success_rate = sum(1 for iou in total if iou >= threshold) / len(total) if total else 0.0
        print(f'Success Rate @ IoU>{threshold}: {success_rate:.4f}')

if __name__ == "__main__":
    # python text_parser_charade.py --gt_json_path eval/data/charades_sta_test_tvr_format.json --inference_dir eval/charades/inference_results/vtg/Qwen2.5-VL-3B-Instruct
    args = arg_parser()
    eval_all(args.gt_json_path, args.inference_dir)