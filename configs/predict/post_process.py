import json
import re
import os
import sys

def extract_validity(text):
    """
    Extracts the logical conclusion from the model's prediction string.
    Supports both Boolean labels (true/false) and verbal labels (valid/invalid).
    """
    # 1. Pre-processing: Remove ChatML tags and strip whitespace
    clean_text = text.replace("<|im_end|>", "").strip().lower()

    # 2. Focus on post-thought content for CoT (Chain-of-Thought) models
    if "</think>" in clean_text:
        final_answer = clean_text.split("</think>")[-1].strip()
    else:
        final_answer = clean_text

    # 3. Priority 1: Direct boolean string matching (e.g., "false", "true")
    # Using exact match or start-of-string match for robustness
    if final_answer.startswith("false") or final_answer == "0":
        return False
    if final_answer.startswith("true") or final_answer == "1":
        return True

    # 4. Priority 2: Keyword search using Regular Expressions
    # \b ensures we match the whole word (e.g., won't match "validation" as "valid")
    if re.search(r'\binvalid\b', final_answer):
        return False
    if re.search(r'\bvalid\b', final_answer):
        return True

    # 5. Default Fallback
    # If no clear conclusion is found, default to False
    return False

def main():
    # Ensure a model directory is provided via command line argument
    if len(sys.argv) < 2:
        print("Usage: python post_process.py <model_dir_name>")
        return
    
    model_dir = sys.argv[1]

    # --- Path Configuration ---
    # The original test set containing the 'id' field
    original_test_path = "data/test_data_subtask_1.json" 
    # The raw output from LLaMA-Factory / inference engine
    predict_file = f"outputs/{model_dir}/predict_test/generated_predictions.jsonl"
    # The final JSON file formatted for submission
    output_path = f"outputs/{model_dir}/predict_test/predictions.json"

    # Check for file existence (supporting both .jsonl and .json)
    if not os.path.exists(predict_file):
        alternative = predict_file.replace(".jsonl", ".json")
        if os.path.exists(alternative):
            predict_file = alternative
        else:
            print(f"❌ Error: Prediction file not found at {predict_file}")
            return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load original data to retrieve 'id'
    with open(original_test_path, "r", encoding="utf-8") as f:
        original_test = json.load(f)
    
    # Load model predictions
    predictions = []
    with open(predict_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
            
    final_results = []
    num_predictions = len(predictions)
    
    # Iterate through original test items and map model outputs to them
    for i in range(len(original_test)):
        item_id = original_test[i]['id']
        
        if i >= num_predictions:
            print(f"⚠️ Warning: Prediction missing for item index {i}.")
            break
            
        # Extract the 'predict' field from the jsonl entry
        predict_raw = predictions[i].get('predict', "")
        is_valid = extract_validity(predict_raw)
        
        # Format required for SemEval Task 11 Subtask 1
        final_results.append({
            "id": item_id,
            "validity": is_valid
        })

    # Save to the final submission format
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    # --- Statistics Output ---
    valid_count = sum(1 for x in final_results if x['validity'])
    print(f"\n--- Experiment: {model_dir} ---")
    print(f"✅ Processing complete! Output: {output_path}")
    print(f"📊 Total processed: {len(final_results)}")
    print(f"💡 Result: Valid ({valid_count}) | Invalid ({len(final_results) - valid_count})")
    
    # Conclusion pattern matching check
    unmatched = 0
    for p in predictions:
        text = p['predict'].split("</think>")[-1].lower() if "</think>" in p['predict'] else p['predict'].lower()
        if not re.search(r'\b(valid|invalid|true|false)\b', text):
            unmatched += 1
    if unmatched > 0:
        print(f"⚠️ Unrecognized outputs (defaulted to False): {unmatched}\n")

if __name__ == "__main__":
    main()