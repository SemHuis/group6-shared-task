import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def load_base_model(model_path):
    """Carga modelo base SIN adaptadores LoRA"""
    print(f"Loading base model (NO LoRA): {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_prediction_fewshot(model, tokenizer, syllogism):
    """Genera predicción con few-shot examples (mismo formato que con LoRA)"""
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
    
    # DEBUG primeras 5 respuestas
    import sys
    if not hasattr(sys, '_debug_count'):
        sys._debug_count = 0
    if sys._debug_count < 5:
        print(f"\n[BASELINE DEBUG {sys._debug_count}]")
        print(f"Syllogism: {syllogism[:80]}...")
        print(f"Raw response: '{response}'")
        sys._debug_count += 1
    
    response_clean = response.strip().lower()
    
    if response_clean.startswith("valid") and "invalid" not in response_clean[:10]:
        return True
    elif "invalid" in response_clean[:15]:
        return False
    else:
        return False


def calculate_content_effect_metrics(df):
    """Calcula métricas de Content Effect"""
    metrics = {}
    
    metrics['accuracy'] = df['correct'].mean()
    
    # Intra-Plausibility Content Effect
    intra_diffs = []
    for plaus in [True, False]:
        subset = df[df['plausibility'] == plaus]
        if len(subset) > 0:
            valid_acc = subset[subset['validity'] == True]['correct'].mean()
            invalid_acc = subset[subset['validity'] == False]['correct'].mean()
            diff = abs(valid_acc - invalid_acc)
            intra_diffs.append(diff)
            metrics[f'acc_plaus_{plaus}_valid'] = valid_acc
            metrics[f'acc_plaus_{plaus}_invalid'] = invalid_acc
    
    metrics['intra_plausibility_ce'] = np.mean(intra_diffs) if intra_diffs else 0.0
    
    # Cross-Plausibility Content Effect
    cross_diffs = []
    for val in [True, False]:
        subset = df[df['validity'] == val]
        if len(subset) > 0:
            plaus_acc = subset[subset['plausibility'] == True]['correct'].mean()
            implaus_acc = subset[subset['plausibility'] == False]['correct'].mean()
            diff = abs(plaus_acc - implaus_acc)
            cross_diffs.append(diff)
            metrics[f'acc_valid_{val}_plaus'] = plaus_acc
            metrics[f'acc_valid_{val}_implaus'] = implaus_acc
    
    metrics['cross_plausibility_ce'] = np.mean(cross_diffs) if cross_diffs else 0.0
    
    # Total Content Effect
    metrics['total_content_effect'] = (metrics['intra_plausibility_ce'] + metrics['cross_plausibility_ce']) / 2
    
    # Ranking score
    if metrics['total_content_effect'] > 0:
        metrics['ranking_score'] = metrics['accuracy'] / metrics['total_content_effect']
    else:
        metrics['ranking_score'] = metrics['accuracy']
    
    return metrics


def evaluate_baseline(model_path, valid_file, output_file):
    """Evalúa modelo base sin LoRA"""
    # Cargar modelo base (SIN LoRA)
    model, tokenizer = load_base_model(model_path)
    
    # Cargar valid_data.json
    with open(valid_file, 'r') as f:
        valid_data = json.load(f)
    
    print(f"\nEvaluating {len(valid_data)} samples (BASELINE - NO LoRA)")
    
    # Hacer predicciones
    results = []
    for item in tqdm(valid_data, desc="Evaluating Baseline"):
        pred = generate_prediction_fewshot(model, tokenizer, item['syllogism'])
        
        results.append({
            'id': item['id'],
            'syllogism': item['syllogism'],
            'validity': item['validity'],
            'plausibility': item['plausibility'],
            'prediction': pred,
            'correct': pred == item['validity']
        })
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    # Calcular métricas
    metrics = calculate_content_effect_metrics(df)
    
    # Imprimir resultados
    print("\n" + "=" * 60)
    print("BASELINE RESULTS (LLaMAX3-8B WITHOUT LoRA)")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Análisis de errores
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    errors = df[~df['correct']]
    
    fp_plausible = len(errors[(errors['validity'] == False) & (errors['plausibility'] == True)])
    fp_implausible = len(errors[(errors['validity'] == False) & (errors['plausibility'] == False)])
    fn_plausible = len(errors[(errors['validity'] == True) & (errors['plausibility'] == True)])
    fn_implausible = len(errors[(errors['validity'] == True) & (errors['plausibility'] == False)])
    
    print("\nFalse Positives (Invalid predicted as Valid):")
    print(f"  - Plausible: {fp_plausible}")
    print(f"  - Implausible: {fp_implausible}")
    
    print("\nFalse Negatives (Valid predicted as Invalid):")
    print(f"  - Plausible: {fn_plausible}")
    print(f"  - Implausible: {fn_implausible}")
    
    # Distribución de predicciones
    total_valid_pred = df['prediction'].sum()
    total_invalid_pred = len(df) - total_valid_pred
    print(f"\nPrediction distribution:")
    print(f"  - Predicted Valid: {total_valid_pred}/{len(df)} ({100*total_valid_pred/len(df):.1f}%)")
    print(f"  - Predicted Invalid: {total_invalid_pred}/{len(df)} ({100*total_invalid_pred/len(df):.1f}%)")
    
    # Guardar resultados
    results_dict = {
        'model': 'LLaMAX3-8B-baseline',
        'method': 'few-shot (no LoRA)',
        'metrics': metrics,
        'predictions': results,
        'error_analysis': {
            'fp_plausible': int(fp_plausible),
            'fp_implausible': int(fp_implausible),
            'fn_plausible': int(fn_plausible),
            'fn_implausible': int(fn_implausible)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved: {output_file}")
    
    return metrics


if __name__ == "__main__":
    BASE_MODEL = "LLaMAX/LLaMAX3-8B"
    VALID_FILE = "/gaueko1/users/mmartin/shared_task/data/valid_data.json"
    OUTPUT_FILE = "/gaueko1/users/mmartin/shared_task/results/evaluation_baseline.json"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    evaluate_baseline(BASE_MODEL, VALID_FILE, OUTPUT_FILE)
