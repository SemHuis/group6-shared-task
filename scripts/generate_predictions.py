import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


# ====== 1. 路径配置 ======
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "Qwen/Qwen3-8B"                       # 基座模型
#LORA_DIR = PROJECT_ROOT / "models" / "qwen3-8b-zeroshot"
LORA_DIR = "meichifan/Qwen3-8B_lora_syllogism_fewshot_prompt"
GT_FILE = PROJECT_ROOT / "data" / "spanish_valid_data.json"   # 你要评估的文件（也可以换成 test.json）
PRED_FILE = PROJECT_ROOT / "results" / "qwen_spanish" / "predictions_fewshot.json"  # 输出的预测文件


# ====== 2. 加载 tokenizer + 模型 + LoRA ======
def load_model_and_tokenizer():
    print(f"Base model   : {BASE_MODEL}")
    print(f"LoRA adapter : {LORA_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, str(LORA_DIR))
    model.eval()

    return model, tokenizer


# ====== 3. 构造和训练阶段一致的 prompt（去掉答案） ======
def build_prompt_from_syllogism(syllogism: str) -> str:
    return f"""Analyze the following syllogism and determine if it is logically valid or invalid. Consider only the formal logical structure, and do not be affected by the plausibility of content.
Example 1:
Syllogism: All dogs are animals. All animals are living things. Therefore, all dogs are living things.
Answer: valid

Example 2:
Syllogism: No cats are dogs. Some animals are dogs. Therefore, some animals are cats.
Answer: invalid

Syllogism: {syllogism}

Is this syllogism logically valid? Answer with 'valid' or 'invalid'.
Answer:"""


# ====== 4. 准备 ' valid' 和 ' invalid' 的 token id，并用 logits 做二分类 ======
def prepare_label_token_ids(tokenizer):
    valid_ids = tokenizer(" valid", add_special_tokens=False).input_ids
    invalid_ids = tokenizer(" invalid", add_special_tokens=False).input_ids
    valid_id = valid_ids[-1]
    invalid_id = invalid_ids[-1]

    print(f"' valid'   ids = {valid_ids}, last = {valid_id}, tokens = {tokenizer.convert_ids_to_tokens(valid_ids)}")
    print(f"' invalid' ids = {invalid_ids}, last = {invalid_id}, tokens = {tokenizer.convert_ids_to_tokens(invalid_ids)}")
    return valid_id, invalid_id


@torch.no_grad()
def predict_validity_label(model, tokenizer, text: str, valid_id: int, invalid_id: int, max_len: int = 512) -> bool:
    """
    返回布尔值：
      True  -> 预测 valid
      False -> 预测 invalid
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [1, seq_len, vocab]

    last_logits = logits[0, -1]
    logit_valid = last_logits[valid_id].item()
    logit_invalid = last_logits[invalid_id].item()

    # logit 大的那个作为预测
    if logit_valid > logit_invalid:
        return True   # valid
    else:
        return False  # invalid


def main():
    # 1) 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer()
    valid_id, invalid_id = prepare_label_token_ids(tokenizer)

    # 2) 读取 ground truth 文件（JSON 数组）
    gt_path = GT_FILE
    assert gt_path.exists(), f"Ground truth file not found: {gt_path}"
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nLoaded {len(data)} samples from {gt_path}")

    # 3) 逐条预测并写入列表
    predictions = []
    correct_count = 0

    for item in tqdm(data, desc="Predicting"):
        sid = item["id"]
        #syllogism = item["translated_syllogism"]
        syllogism = item["spanish"]

        prompt = build_prompt_from_syllogism(syllogism)
        pred_valid_bool = predict_validity_label(model, tokenizer, prompt, valid_id, invalid_id)

        # 如果 GT 文件里也有 validity，可以顺便算个 accuracy（方便 sanity check）
        if "validity" in item:
            gold_valid = bool(item["validity"])
            if pred_valid_bool == gold_valid:
                correct_count += 1

        predictions.append({
            "id": sid,
            "validity": pred_valid_bool,   # 注意：布尔值 True/False，符合官方格式
        })

    if "validity" in data[0]:
        acc = correct_count / len(data)
        print(f"\nSanity check accuracy on {len(data)} samples: {acc:.4f}")

    # 4) 保存为 predictions.json
    pred_path = PRED_FILE
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\nPredictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
