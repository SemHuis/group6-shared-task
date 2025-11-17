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
    EarlyStoppingCallback,
    default_data_collator
)
from transformers import set_seed
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
# CONFIGURATION
# ============================================
class Config:
    # Paths
    BASE_DIR = ".."
    DATA_DIR = f"{BASE_DIR}/data"
    OUTPUT_DIR = f"{BASE_DIR}/checkpoints/qwen3-8b-lora-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    FINAL_MODEL_DIR = f"{BASE_DIR}/models/qwen3-8b-final"
    
    # Model
    MODEL_NAME = "Qwen/Qwen3-8B"
    
    # LoRA Hyperparameters
    LORA_R = 32
    LORA_ALPHA = 64
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
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 8
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_SEQ_LENGTH = 512
    
    # Training settings
    LOGGING_STEPS = 10
    SAVE_STEPS = 50
    EVAL_STEPS = 50
    SAVE_TOTAL_LIMIT = 3
    
    # Data split settings
    DEV_SIZE = 0.1
    SEED = 42
    
    # Weights & Biases
    WANDB_PROJECT = "semeval2026-task11-zeroshot"
    WANDB_RUN_NAME = f"Qwen3-8b-lora-r{LORA_R}-lr{LEARNING_RATE}"


# ============================================
# HELPER FUNCTIONS
# ============================================
def format_prompt_for_training(syllogism, validity):
    """Format a training prompt for the syllogism validity classification task."""
    label = "valid" if validity else "invalid"
    return f"""Analyze the following syllogism and determine if it is logically valid or invalid. Consider only the formal logical structure, and do not be affected by the plausibility of content.

Syllogism: {syllogism}

Is this syllogism logically valid? Answer with 'valid' or 'invalid'.
Answer: {label}"""


def transform_labels(example):
    """Add numeric label and construct the full text prompt for each example."""
    example["label"] = int(example["validity"])
    example["text"] = format_prompt_for_training(example["syllogism"], example["validity"])
    return example


def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize a batch of examples and create labels that only supervise the answer part.

    - Input:  examples["text"] is a list of prompts.
    - We search for the token sequence corresponding to "Answer:".
    - All tokens before the answer are masked with -100 in the labels so they do not contribute to the loss.
    - The answer tokens themselves are kept as training targets.
    """
    # Batch input: examples is a dict, values are lists
    texts = examples["text"]

    # Token IDs for the "Answer:" prefix (no extra space, more robust)
    answer_tokens = tokenizer.encode("Answer:", add_special_tokens=False)
    answer_len = len(answer_tokens)

    # Batch tokenization (no padding yet)
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    # Start from a copy of input_ids and then mask out prompt tokens
    labels = [ids.copy() for ids in input_ids]

    # Masking: keep only the answer region as labels, mask everything before it
    for i, ids in enumerate(input_ids):
        start_pos = None

        # Find the position of the "Answer:" token sequence
        for j in range(len(ids) - answer_len + 1):
            if ids[j : j + answer_len] == answer_tokens:
                # Found "Answer:". Move to the token right after the prefix.
                start_pos = j + answer_len
                # Optionally skip a single space token right after "Answer:"
                if start_pos < len(ids) and tokenizer.decode([ids[start_pos]]) == " ":
                    start_pos += 1
                break

        if start_pos is not None:
            # Mask all tokens before the start of the answer
            for k in range(start_pos):
                labels[i][k] = -100
        else:
            # If "Answer:" is not found, mask the entire sequence (should not happen in clean data)
            labels[i] = [-100] * len(ids)

    # Pad inputs to max_length
    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    # Pad labels to max_length with -100
    padded_labels = []
    for label_seq in labels:
        pad_len = max_length - len(label_seq)
        padded_labels.append(label_seq + [-100] * pad_len)

    return {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "labels": padded_labels,
    }


# ============================================
# MAIN TRAINING FUNCTION
# ============================================
def main():
    """Main entry point: load data, prepare LoRA model, train, evaluate, and save artifacts."""
    # Set global random seeds for reproducibility
    set_seed(Config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Initialize Weights & Biases run
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
        },
    )

    print("=" * 50)
    print("SEMEVAL 2026 TASK 11 - LORA FINE-TUNING")
    print("=" * 50)

    # [1] Load and prepare data
    print("\nLoading training dataset")

    # Load train_data.json
    train_file = f"{Config.DATA_DIR}/train_data.json"
    raw_dataset = load_dataset("json", data_files={"train": train_file})
    dataset = raw_dataset["train"]

    print(f"  - Total samples: {len(dataset)}")  # type: ignore

    # Split into train / dev
    print(f"\nSplitting train/dev ({100 - Config.DEV_SIZE * 100:.0f}%/{Config.DEV_SIZE * 100:.0f}%)")

    indices = list(range(len(dataset)))  # type: ignore
    labels = [example["validity"] for example in dataset]  # type: ignore

    train_indices, dev_indices = train_test_split(
        indices,
        test_size=Config.DEV_SIZE,
        random_state=Config.SEED,
        stratify=labels,
    )

    # Create DatasetDict with train/dev splits
    dataset_dict = DatasetDict(
        {
            "train": dataset.select(train_indices),  # type: ignore
            "dev": dataset.select(dev_indices),  # type: ignore
        }
    )

    # Apply label and prompt transformations
    dataset_dict = dataset_dict.map(transform_labels)

    print(f"  - Train: {len(dataset_dict['train'])} samples")
    print(f"  - Dev: {len(dataset_dict['dev'])} samples")

    # Quick label distribution check
    train_valid = sum(1 for x in dataset_dict["train"] if x["validity"])  # type: ignore
    train_invalid = len(dataset_dict["train"]) - train_valid
    dev_valid = sum(1 for x in dataset_dict["dev"] if x["validity"])  # type: ignore
    dev_invalid = len(dataset_dict["dev"]) - dev_valid

    print(f"\n  Train - Valid: {train_valid}, Invalid: {train_invalid}")
    print(f"  Dev   - Valid: {dev_valid}, Invalid: {dev_invalid}")

    # Inspect a few raw transformed examples
    print("\n" + "=" * 50)
    print("Label and text sanity check (first 3 training samples)")
    print("=" * 50)
    for i in range(min(3, len(dataset_dict["train"]))):
        example = dataset_dict["train"][i]
        print(f"\nSample {i}:")
        print(f"  - text: {example['text'][:100]}...")  # Print first 100 characters
        print(f"  - label: {example['label']}")  # Should be 0 or 1
        print(f"  - validity: {example['validity']}")  # Original boolean label
    print("=" * 50 + "\n")

    # [2] Load tokenizer
    print(f"\nLoading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )

    # Ensure pad token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.model_max_length = Config.MAX_SEQ_LENGTH

    # [3] Load base model
    print(f"\nLoading model: {Config.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Disable cache during training to avoid warnings and reduce memory usage
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # [4] Configure LoRA
    print(f"\nConfiguring LoRA (r={Config.LORA_R}, alpha={Config.LORA_ALPHA})")
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    # Wrap the base model with LoRA adapters
    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters summary:")
    model.print_trainable_parameters()

    # [5] Tokenize datasets
    print(f"\nTokenizing datasets")

    train_dataset = dataset_dict["train"].map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )

    # Inspect one tokenized sample to ensure labels are correct
    print("\n" + "=" * 50)
    print("Tokenized label sanity check (first training sample)")
    print("=" * 50)
    sample = train_dataset[0]
    print(f"Length of input_ids: {len(sample['input_ids'])}")
    print(f"Length of labels: {len(sample['labels'])}")
    print(f"First 20 label values: {sample['labels'][:20]}")  # Should not all be -100
    print(f"Number of -100 in labels: {sample['labels'].count(-100)} / {len(sample['labels'])}")
    print(
        f"Non -100 label values (first 10): "
        f"{[v for v in sample['labels'] if v != -100][:10]}"
    )  # Should be meaningful token IDs
    print("=" * 50 + "\n")

    dev_dataset = dataset_dict["dev"].map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=dataset_dict["dev"].column_names,
    )

    print(f"  - Train tokenized: {len(train_dataset)} samples")
    print(f"  - Dev tokenized: {len(dev_dataset)} samples")

    # [6] Configure TrainingArguments
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
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        bf16=True,
        report_to="wandb",
        logging_dir=f"{Config.BASE_DIR}/logs",
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
    )

    # Data collator (simple default collator for already-tokenized inputs)
    data_collator = default_data_collator

    # [7] Initialize Trainer
    print("\n" + "=" * 50)
    print("Initializing Trainer")
    print("=" * 50)

    # Build a classification metric based on the first answer token (valid/invalid)
    def make_compute_metrics(tok):
        """
        Create a compute_metrics function that:
        - Extracts the first non-masked answer token from labels.
        - Compares logits at that position for 'valid' vs 'invalid'.
        - Computes accuracy and F1 over the predicted class.
        """
        valid_ids = tok(" valid", add_special_tokens=False).input_ids
        invalid_ids = tok(" invalid", add_special_tokens=False).input_ids
        if not valid_ids:
            valid_ids = tok("valid", add_special_tokens=False).input_ids
        if not invalid_ids:
            invalid_ids = tok("invalid", add_special_tokens=False).input_ids
        valid_first = valid_ids[0]
        invalid_first = invalid_ids[0]

        def _compute(eval_pred):
            preds = eval_pred.predictions
            if isinstance(preds, tuple):
                preds = preds[0]
            labels = eval_pred.label_ids

            # Move predictions and labels to CPU to avoid GPU memory build-up
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu()

            y_true, y_pred = [], []
            for i in range(labels.shape[0]):
                label_row = labels[i]
                idxs = np.where(label_row != -100)[0]
                if idxs.size == 0:
                    # No supervised position in this sample
                    continue
                start_idx = int(idxs[0])

                # Determine true class from the first answer token id
                true_first_id = int(label_row[start_idx])
                true_cls = 1 if true_first_id == valid_first else 0

                # Use logits at the same position to decide between valid/invalid
                logit_row = preds[i, start_idx]
                if isinstance(logit_row, torch.Tensor):
                    logit_row = logit_row.numpy()
                pred_cls = 1 if float(logit_row[valid_first]) > float(logit_row[invalid_first]) else 0

                y_true.append(true_cls)
                y_pred.append(pred_cls)

            if len(y_true) == 0:
                return {"accuracy": 0.0, "f1": 0.0, "eval_metric": 0.0}

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="binary")
            # Use accuracy as eval_metric (can be changed if needed)
            return {"accuracy": acc, "f1": f1, "eval_metric": acc}

        return _compute

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print("\n" + "=" * 50)
    print("Starting training")
    print("=" * 50 + "\n")

    trainer.train()

    # [8] Save final model and tokenizer
    print(f"\nSaving final model to: {Config.FINAL_MODEL_DIR}")
    os.makedirs(Config.FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(Config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)

    # [9] Final evaluation on dev set
    print("\n" + "=" * 50)
    print("Final evaluation on dev set")
    print("=" * 50)
    eval_results = trainer.evaluate()

    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

    # Log final evaluation to Weights & Biases and finish the run
    wandb.log({"final_eval": eval_results})
    wandb.finish()

    print("\n" + "=" * 50)
    print("Training completed")
    print("=" * 50)

    # Save split and label statistics for reproducibility
    split_info = {
        "train_size": len(dataset_dict["train"]),
        "dev_size": len(dataset_dict["dev"]),
        "train_valid": int(train_valid),
        "train_invalid": int(train_invalid),
        "dev_valid": int(dev_valid),
        "dev_invalid": int(dev_invalid),
        "seed": Config.SEED,
        "dev_ratio": Config.DEV_SIZE,
    }

    with open(f"{Config.FINAL_MODEL_DIR}/split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit info saved to: {Config.FINAL_MODEL_DIR}/split_info.json")


if __name__ == "__main__":
    main()
