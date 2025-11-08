import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os
import argparse

def load_model(base_model_path, adapter_path=None):
    """Carga modelo con o sin adaptadores LoRA"""
    
    print(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("No adapter path provided, loading base model only.")
        
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_prediction(model, tokenizer, syllogism):
    """Genera predicción con few-shot prompting"""
    prompt = f"""Task: Determine if logical arguments are valid or invalid.

Example 1:
Syllogism: All dogs are animals. All animals are living things. Therefore, all dogs are living things.
Answer: valid

Example 2:
Syllogism: No cats are dogs. Some animals are dogs. Therefore, some animals are cats.
Answer: invalid

Example 3:
Syllogism: {syllogism}
Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response_clean = response.strip().lower()
    
    if response_clean.startswith("valid") and "invalid" not in response_clean[:10]:
        return True
    elif "invalid" in response_clean[:15]:
        return False
    else:
        return False

def generate_predictions_file(model_path, adapter_path, valid_file, output_file):
    """
    Evalúa el modelo y guarda SÓLO las predicciones en un JSON.
    """
    model, tokenizer = load_model(model_path, adapter_path)
    
    with open(valid_file, 'r') as f:
        valid_data = json.load(f)
    
    print(f"\nGenerating predictions for {len(valid_data)} samples...")
    
    predictions_list = []
    for item in tqdm(valid_data, desc="Evaluating"):
        pred = generate_prediction(model, tokenizer, item['syllogism'])
        predictions_list.append({
            'id': item['id'],
            'validity': pred
        })
    
    with open(output_file, 'w') as f:
        json.dump(predictions_list, f, indent=2)
    
    print(f"\nPredictions saved to: {output_file}")


if __name__ == "__main__":
    # Setup the argument parser
    parser = argparse.ArgumentParser(description="Run LLM evaluation and save predictions.")
    
    # Define the arguments
    parser.add_argument(
        '--base_model', 
        type=str, 
        default="LLaMAX/LLaMAX3-8B", 
        help="Base model ID from Hugging Face."
    )
    
    parser.add_argument(
        '--adapter', 
        type=str, 
        default=None,
        help="Optional adapter ID from Hugging Face (maytemuma/llamax3-8b-lora-zeroshot / maytemuma/llamax3-8b-lora-fewshot'). If not provided, runs base model only."
    )
    
    parser.add_argument(
        '--valid_file', 
        type=str, 
        default="/home1/s4779428/shared-task/valid_data.json",
        help="Path to the validation (ground truth) JSON file."
    )
    
    parser.add_argument(
        '--output_file', 
        type=str,  
        default="/home1/s4779428/shared-task/predictions.json",
        help="Path to save the output predictions JSON."
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Run the main function with the parsed arguments
    generate_predictions_file(
        args.base_model, 
        args.adapter, 
        args.valid_file, 
        args.output_file
    )