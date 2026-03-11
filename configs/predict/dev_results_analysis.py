import json
import re
import os
import csv
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Dev Set Error Analysis")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--predict_file", type=str, required=True, help="Path to model predictions JSONL")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the error CSV")
    return parser.parse_args()

def extract_label(text):
    """
    Extract label from text or boolean: returns 'valid' or 'invalid'
    """
    # 1. If already a boolean
    if isinstance(text, bool):
        return "valid" if text else "invalid"

    if not text: return "none"

    # 2. Clean string
    clean_text = str(text).replace("", "").strip().lower()

    # 3. Handle Chain-of-Thought (CoT): focus on part after 
    if "" in clean_text:
        final_part = clean_text.split("")[-1].strip()
    else:
        final_part = clean_text

    # 4. Keyword matching
    if final_part.startswith("false") or final_part == "0" or re.search(r'\binvalid\b', final_part):
        return "invalid"
    if final_part.startswith("true") or final_part == "1" or re.search(r'\bvalid\b', final_part):
        return "valid"

    return "none"

def get_ground_truth(item):
    """
    Automatically detect ground truth format:
    1. Original format (with 'validity' key)
    2. ShareGPT format (inside 'conversations')
    """
    # Case A: Original JSON format
    if 'validity' in item:
        return extract_label(item['validity'])

    # Case B: ShareGPT format (last assistant message)
    if 'conversations' in item:
        last_msg = item['conversations'][-1]['value']
        return extract_label(last_msg)

    return "none"

def main():
    args = parse_args()
    csv_output = os.path.join(args.output_dir, "error_cases.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Analyzing...\n GT: {args.dev_file}\n Pred: {args.predict_file}")

    # --- 1. Load data ---
    with open(args.dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    predictions = []
    with open(args.predict_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    # --- 2. Comparison logic ---
    error_list = []
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    limit = min(len(dev_data), len(predictions))

    print(f"Processing {limit} items...")

    for i in range(limit):
        gt_item = dev_data[i]
        pred_item = predictions[i]

        # Get ground truth label (handles multiple formats)
        gt_label = get_ground_truth(gt_item)

        # Get model prediction
        model_output = pred_item.get('predict', '')
        model_label = extract_label(model_output)

        # Get quadrant label for statistics (default to 'Overall' if missing)
        q_label = gt_item.get('quadrant', 'Overall')

        is_correct = (model_label == gt_label)

        stats[q_label]["total"] += 1
        if is_correct:
            stats[q_label]["correct"] += 1
        else:
            # Extract original text for error case
            if 'syllogism' in gt_item:
                syllogism_text = gt_item['syllogism']
            elif 'conversations' in gt_item:
                syllogism_text = gt_item['conversations'][0]['value']  # Human question
            else:
                syllogism_text = "N/A"

            error_list.append({
                "Index": i,
                "Quadrant": q_label,
                "Ground_Truth": gt_label,
                "Model_Prediction": model_label,
                "Syllogism": syllogism_text[:100] + "...",
                "Full_Output": model_output
            })

    # --- 3. Export and print ---
    if error_list:
        with open(csv_output, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=error_list[0].keys())
            writer.writeheader()
            writer.writerows(error_list)

    print("\n" + "="*45)
    print(f"{'Quadrant':<15} | {'Acc (%)':<12} | {'Count':<8}")
    print("-" * 45)
    t_correct, t_total = 0, 0
    for q in sorted(stats.keys()):
        val = stats[q]
        acc = (val["correct"] / val["total"] * 100) if val["total"] > 0 else 0
        print(f"{q:<15} | {acc:>10.2f}% | {val['total']:<8}")
        t_correct += val["correct"]
        t_total += val["total"]
    total_acc = (t_correct / t_total * 100) if t_total > 0 else 0
    print("-" * 45)
    print(f"{'TOTAL':<15} | {total_acc:>10.2f}% | {t_total:<8}")
    print("="*45)
    print(f"Error cases saved to: {csv_output}\n")

if __name__ == "__main__":
    main()