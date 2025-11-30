import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import logging
from typing import Dict, List
import sys

# === 1. 配置参数 ===
class Config:
    MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
    VALID_FILE = "data/chinese_valid_data.json"
    OUTPUT_DIR = "results"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 2. 设置日志 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generate_predictions_chinese.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def build_prompt_from_syllogism(syllogism: str) -> str:
    """构建中文三段论提示模板（专为中文优化）"""
    return f"""分析以下三段论，判断其逻辑上是否有效。请仅考虑形式逻辑结构，不要受内容真实性的影响。
示例1:
三段论: 所有狗都是动物。所有动物都是生物。因此，所有狗都是生物。
答案: 有效
示例2:
三段论: 没有猫是狗。有些动物是狗。因此，有些动物是猫。
答案: 无效
三段论: {syllogism}
这个三段论在逻辑上是否有效？请回答'有效'或'无效'。
答案:"""

def load_model_and_tokenizer():
    """加载量化模型和分词器"""
    logger.info(f"Loading model: {Config.MODEL_NAME}")
    
    # 量化配置
    quantization_config = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    logger.info("Model loaded successfully with 4-bit quantization")
    return model, tokenizer

def predict_validity_label(model, tokenizer, text: str, valid_id: int, invalid_id: int) -> bool:
    """预测逻辑有效性（返回布尔值）"""
    enc = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=Config.MAX_LENGTH,
        padding=True
    ).to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[0, -1]
        logit_valid = logits[valid_id].item()
        logit_invalid = logits[invalid_id].item()
        
        return logit_valid > logit_invalid

def main():
    # === 1. 创建输出目录 ===
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # === 2. 加载模型和分词器 ===
    model, tokenizer = load_model_and_tokenizer()
    
    # 获取有效/无效的token ID
    valid_id = tokenizer.encode("有效", add_special_tokens=False)[0]
    invalid_id = tokenizer.encode("无效", add_special_tokens=False)[0]
    
    logger.info(f"Valid token ID: {valid_id}, Invalid token ID: {invalid_id}")
    
    # === 3. 加载验证数据 ===
    logger.info(f"Loading validation data from {Config.VALID_FILE}")
    with open(Config.VALID_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} validation samples")
    
    # === 4. 逐批预测 ===
    predictions = []
    correct_count = 0
    
    # 分批处理数据
    for i in tqdm(range(0, len(data), Config.BATCH_SIZE), desc="Processing batches"):
        batch = data[i:i+Config.BATCH_SIZE]
        batch_prompts = [
            build_prompt_from_syllogism(item["syllogism"]) 
            for item in batch
        ]
        
        # 批量编码
        batch_enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding=True
        ).to(Config.DEVICE)
        
        # 批量预测
        with torch.no_grad():
            outputs = model(**batch_enc)
            logits = outputs.logits[:, -1, :]  # 取最后一个token的logits
            
            # 获取预测结果
            pred_valid = logits[:, valid_id] > logits[:, invalid_id]
            pred_valid = pred_valid.cpu().numpy()
        
        # 保存结果
        for j, item in enumerate(batch):
            is_valid = bool(pred_valid[j])
            predictions.append({
                "id": item["id"],
                "syllogism": item["syllogism"],
                "predicted_validity": is_valid
            })
            
            # 计算准确率（如果数据中有真实标签）
            if "validity" in item:
                if is_valid == bool(item["validity"]):
                    correct_count += 1
    
    # === 5. 保存结果 ===
    output_file = os.path.join(Config.OUTPUT_DIR, "chinese_predictions.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    # === 6. 打印结果 ===
    accuracy = correct_count / len(data) if len(data) > 0 else 0
    logger.info(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(data)})")
    logger.info(f"Results saved to: {output_file}")
    
    # === 7. 清理显存 ===
    torch.cuda.empty_cache()
    logger.info("CUDA cache cleared")

if __name__ == "__main__":
    # === 关键修复：使用 parse_known_args 忽略所有未知参数 ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='zh', help='Language for prediction (zh for Chinese)')
    
    # 重要：使用 parse_known_args 而不是 parse_args
    args, unknown = parser.parse_known_args()
    
    # 忽略所有未知参数（包括Jupyter添加的 -f 参数）
    if unknown:
        logger.warning(f"Ignoring unknown arguments: {unknown}")
    
    # 验证语言参数
    if args.language != 'zh':
        logger.error(f"Invalid language '{args.language}' for Chinese script. Only 'zh' is supported.")
        sys.exit(1)
    
    logger.info(f"Starting Chinese prediction (language: {args.language})")
    main()