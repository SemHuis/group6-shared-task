# SemEval 2026 Task 11 — LLaMAX3-8B + LoRA for Syllogistic Reasoning

LoRA adapters on top of LLaMAX3-8B for syllogistic validity classification and content-effect evaluation. Includes zero-shot training (best accuracy with few-shot inference) and few-shot training (best ranking score with low content effect).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Pretrained Adapters (Hugging Face)](#pretrained-adapters-hugging-face)
- [Usage Examples](#usage-examples)
- [SLURM Example (tcsh)](#slurm-example-tcsh)
- [Good Practices](#good-practices)
- [License](#license)
- [Citation](#citation)

## Overview
This project fine-tunes LoRA adapters for LLaMAX3-8B to perform binary syllogism validity classification while measuring content effect (intra/cross) and the ranking score used in the shared task. Two variants are provided:
- Zero-shot trained adapter, used with few-shot prompting at inference for best accuracy.
- Few-shot trained adapter, used with the same few-shot template at inference for best ranking score and slightly lower content effect.

## Requirements
- Python 3.10+
- CUDA-enabled GPU (bfloat16 recommended)
- Install dependencies:
pip install -r requirements.txt


## Setup
If you track experiments with Weights & Biases, configure the API key as an environment variable. 

For tcsh/csh:
---
In ~/.tcshrc (or ~/.cshrc)

setenv WANDB_API_KEY "YOUR_WANDB_KEY"

source ~/.tcshrc


For bash/zsh:
---
echo 'export WANDB_API_KEY="YOUR_WANDB_KEY"' >> ~/.bashrc

source ~/.bashrc


## Data
- data/train_data.json: English syllogisms with fields:
  - id, syllogism, validity (bool), plausibility (bool)
- data/valid_data.json: held-out validation set used for reporting

Also dutch and spanish valid data (.json) for multilingual experiments.

## Training
Zero-shot training (concise prompt in the label):
---
python scripts/train_lora.py


Few-shot training (2 in-context examples in the prompt):
---
python scripts/train_lora_fewshot.py


Notes:
- Few-shot prompts are longer; set MAX_SEQ_LENGTH to 768 (or 1024 if needed) to avoid truncation.
- Final adapters are saved into models/llamax3-8b-final and models/llamax3-8b-fewshot-final respectively.

## Evaluation
Evaluate on the held-out validation set (240 items):
---
python scripts/evaluate.py


The report includes:
- accuracy
- intra_plausibility_ce
- cross_plausibility_ce
- total_content_effect
- ranking_score
- predictions with per-item correctness
- basic error analysis (FP/FN broken down by plausibility/validity)

## Results
Validation results (few-shot inference for both):

| Model                                 | Accuracy | Content Effect | Ranking Score |
|---------------------------------------|----------|----------------|---------------|
| LLaMAX3-8B (few-shot only, no LoRA)   | 60.8%    | 0.214          | 2.84          |
| Zero-shot trained + few-shot infer.   | 87.5%    | 0.058          | 15.03         |
| Few-shot trained + few-shot infer.    | 86.3%    | 0.056          | 15.45         |

Interpretation:
- Zero-shot trained + few-shot inference yields the highest accuracy.
- Few-shot trained + few-shot inference yields the best ranking score and slightly lower content effect.

## Pretrained Adapters (Hugging Face)
- Zero-shot trained: https://huggingface.co/maytemuma/llamax3-8b-lora-zeroshot
- Few-shot trained: https://huggingface.co/maytemuma/llamax3-8b-lora-fewshot

## Usage Examples

Load base model and adapter:
---
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

base_id = "LLaMAX/LLaMAX3-8B"

base_model = AutoModelForCausalLM.from_pretrained(
base_id, torch_dtype=torch.bfloat16, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_id)

if tokenizer.pad_token is None:

tokenizer.pad_token = tokenizer.eos_token

---
Choose one adapter:
---
adapter_id = "maytemuma/llamax3-8b-lora-zeroshot"

adapter_id = "maytemuma/llamax3-8b-lora-fewshot"

model = PeftModel.from_pretrained(base_model, adapter_id)

model.eval()


---
Few-shot inference helper:
---

def classify_syllogism(syllogism: str, model, tokenizer) -> bool:

"""
Returns True for 'valid', False for 'invalid'.
"""

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
response = tokenizer.decode(

    outputs[inputs["input_ids"].shape:],[11]
    skip_special_tokens=True
).strip().lower()

if response.startswith("valid") and "invalid" not in response[:10]:

    return True

if "invalid" in response[:15]:

    return False
    
return False

## SLURM Example (tcsh)
#!/bin/tcsh
#SBATCH --job-name=lora_fewshot
#SBATCH --partition=DGXA100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out


setenv WANDB_API_KEY "YOUR_WANDB_KEY"

source lora_env/bin/activate.csh

python scripts/train_lora_fewshot.py


## License
Apache-2.0. Adapters inherit compatibility with the base model license.




