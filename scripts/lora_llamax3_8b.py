import os
import json
import torch
import wandb
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ============================================
# CONFIGURACIÓN
# ============================================
class Config:
    # Paths
    BASE_DIR = "/gaueko1/users/mmartin/shared_task"
    DATA_DIR = f"{BASE_DIR}/data"
    OUTPUT_DIR = f"{BASE_DIR}/checkpoints/llamax3-8b-lora-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    FINAL_MODEL_DIR = f"{BASE_DIR}/models/llamax3-8b-final"
    
    # Model
    MODEL_NAME = "LLaMAX/LLaMAX3-8B"
    
    # LoRA Hyperparameters
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
    
    # Training Hyperparameters
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_SEQ_LENGTH = 512
    
    # Training settings
    LOGGING_STEPS = 10
    SAVE_STEPS = 50
    EVAL_STEPS = 50
    SAVE_TOTAL_LIMIT = 3
    
    # Split settings (como en el notebook BERT)
    DEV_SIZE = 0.1
    SEED = 42
    
    # WandB
    WANDB_PROJECT = "semeval2026-task11"
    WANDB_RUN_NAME = f"llamax3-8b-lora-r{LORA_R}-lr{LEARNING_RATE}"

# ============================================
# FUNCIONES AUXILIARES
# ============================================
def format_prompt_for_training(syllogism, validity):
    """Formato de prompt para training"""
    label = "valid" if validity else "invalid"
    return f"""Analyze the following syllogism and determine if it is logically valid or invalid. Consider only the formal logical structure, not the plausibility of the content.

Syllogism: {syllogism}

Is this syllogism logically valid? Answer with 'valid' or 'invalid'.
Answer: {label}"""

def transform_labels(example):
    """Transforma los datos"""
    example['label'] = int(example['validity'])
    example['text'] = format_prompt_for_training(example['syllogism'], example['validity'])
    return example


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokeniza los ejemplos"""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

def compute_metrics(eval_pred):
    """
    Calcula métricas de evaluación
    """
    predictions, labels = eval_pred
   
    return {
        "eval_metric": 0.0,  
    }

# ============================================
# MAIN TRAINING FUNCTION
# ============================================
def main():
    # Inicializar WandB
    wandb.init(
        project=Config.WANDB_PROJECT,
        name=Config.WANDB_RUN_NAME,
        config={
            "model": Config.MODEL_NAME,
            "lora_r": Config.LORA_R,
            "lora_alpha": Config.LORA_ALPHA,
            "lora_dropout": Config.LORA_DROPOUT,
            "learning_rate": Config.LEARNING_RATE,
            "num_epochs": Config.NUM_EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "gradient_accumulation_steps": Config.GRADIENT_ACCUMULATION_STEPS,
            "dev_size": Config.DEV_SIZE,
        }
    )
    
    print("=" * 50)
    print("SEMEVAL 2026 TASK 11 - LORA FINE-TUNING")
    print("=" * 50)
    
    # [1] CARGAR Y PREPARAR DATOS
    print(f"\ntraining dataset")
    
    # Cargar train_data.json
    train_file = f"{Config.DATA_DIR}/train_data.json"
    raw_dataset = load_dataset('json', data_files={'train': train_file})
    dataset = raw_dataset['train']
    
    print(f"  - Total samples: {len(dataset)}")
    
    # Split 90% train / 10% dev (como en el notebook BERT)
    print(f"\nsplit train/dev({100-Config.DEV_SIZE*100:.0f}%/{Config.DEV_SIZE*100:.0f}%)")

    indices = list(range(len(dataset)))
    labels = [example['validity'] for example in dataset]

    train_indices, dev_indices = train_test_split(
        indices,
        test_size=Config.DEV_SIZE,
        random_state=Config.SEED,
        stratify=labels
    )

    # Crear splits
    dataset_dict = DatasetDict({
        'train': dataset.select(train_indices),
        'dev': dataset.select(dev_indices)
    })
        
    # Aplicar transformación
    dataset_dict = dataset_dict.map(transform_labels)
    
    print(f"  - Train: {len(dataset_dict['train'])} sample")
    print(f"  - Dev: {len(dataset_dict['dev'])} samples")
    
    # Estadísticas
    train_valid = sum(1 for x in dataset_dict['train'] if x['validity'])
    train_invalid = len(dataset_dict['train']) - train_valid
    dev_valid = sum(1 for x in dataset_dict['dev'] if x['validity'])
    dev_invalid = len(dataset_dict['dev']) - dev_valid
    
    print(f"\n  Train - Valid: {train_valid}, Invalid: {train_invalid}")
    print(f"  Dev   - Valid: {dev_valid}, Invalid: {dev_invalid}")
    
    # [2] CARGAR TOKENIZER
    print(f"\ntokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # [3] CARGAR MODELO
    print(f"\nmodel: {Config.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.gradient_checkpointing_enable()

    # [4] CONFIGURAR LORA
    print(f"\nlora config (r={Config.LORA_R}, alpha={Config.LORA_ALPHA})")
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    
    print("\ntrainable parameters:")
    model.print_trainable_parameters()

    model = model.to(torch.bfloat16)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
    
    # [5] TOKENIZAR DATASETS
    print(f"\ntokenize dataset")
    
    train_dataset = dataset_dict['train'].map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=dataset_dict['train'].column_names
    )
    
    dev_dataset = dataset_dict['dev'].map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=dataset_dict['dev'].column_names
    )
    
    print(f"  - Train tokenized: {len(train_dataset)} samples")
    print(f"  - Dev tokenized: {len(dev_dataset)} samples")
    
    # [6] CONFIGURAR TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        logging_steps=Config.LOGGING_STEPS,
        save_steps=Config.SAVE_STEPS,
        eval_steps=Config.EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        bf16=True,
        report_to="wandb",
        logging_dir=f"{Config.BASE_DIR}/logs",
        #gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # [7] INICIALIZAR TRAINER
    print("\n" + "=" * 50)
    print("trainer")
    print("=" * 50)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # [8] ENTRENAR
    print("\n" + "=" * 50)
    print("training")
    print("=" * 50 + "\n")
    
    trainer.train()
    
    # [9] GUARDAR MODELO FINAL
    print(f"\nsaving final model in: {Config.FINAL_MODEL_DIR}")
    os.makedirs(Config.FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(Config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)
    
    # [10] EVALUACIÓN FINAL
    print("\n" + "=" * 50)
    print("final eval in dev set")
    print("=" * 50)
    eval_results = trainer.evaluate()
    
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    wandb.log({"final_eval": eval_results})
    wandb.finish()
    
    print("\n" + "=" * 50)
    print("training completed")
    print("=" * 50)
    
    # Guardar info del split para referencia
    split_info = {
        "train_size": len(dataset_dict['train']),
        "dev_size": len(dataset_dict['dev']),
        "train_valid": int(train_valid),
        "train_invalid": int(train_invalid),
        "dev_valid": int(dev_valid),
        "dev_invalid": int(dev_invalid),
        "seed": Config.SEED,
        "dev_ratio": Config.DEV_SIZE
    }
    
    with open(f"{Config.FINAL_MODEL_DIR}/split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSplit info saved in: {Config.FINAL_MODEL_DIR}/split_info.json")

if __name__ == "__main__":
    main()
