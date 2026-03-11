#!/bin/bash
#SBATCH --job-name=14b_aug
#SBATCH --output=logs/14b_aug_%j.log
#SBATCH --error=logs/14b_aug_%j.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mem=80G

# ==========================================
# 1. Environment and Path Setup
# ==========================================
mkdir -p logs
module load CUDA/12.1.1
export CUDA_HOME=$CUDA_PATH

PROJECT_ROOT=$(pwd)

# Activate virtual environment
if [ -d "envs" ]; then
    VENV_PATH=$(find envs -name "bin/activate" -type f | head -n1)
    if [ -n "$VENV_PATH" ]; then
        source "$VENV_PATH"
    else
        echo "Error: No virtual environment found in envs/"
        exit 1
    fi
else
    echo "Error: envs/ directory not found."
    exit 1
fi

# Storage Redirection (Using $USER)
export SCRATCH_DIR="/scratch/$USER"
export TMPDIR="$SCRATCH_DIR/tmp"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/hf_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/hf_cache/transformers"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Disable Xet and HF Transfer
export HF_HUB_ENABLE_HF_TRANSFER=0
export HUGGINGFACE_HUB_DISABLE_XET=1
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_OPS=0

# ==========================================
# 2. Experiment Variables
# ==========================================
MODEL_NAME="14b_aug"
DATASET_TRAIN="logic_train_augmented"
DATASET_DEV="logic_dev_augmented"
DATASET_TEST="logic_s1_test"
GT_DEV_FILE="${PROJECT_ROOT}/data/train_dev/final_dev.json"

TRAIN_OUTPUT="${SCRATCH_DIR}/semeval2026_task11/${MODEL_NAME}"
PREDICT_DEV_DIR="${PROJECT_ROOT}/outputs/${MODEL_NAME}/predict_dev"
PREDICT_TEST_DIR="${PROJECT_ROOT}/outputs/${MODEL_NAME}/predict_test"

export USE_LOGIC_LOSS=false

cd src/LLaMA-Factory || exit
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo ">>> [Status] Mode: DEFAULT Loss | Dataset: AUGMENTED | Model: Qwen-14B"

# ==========================================
# 3. Training Phase
# ==========================================
echo ">>> [Step 1/4] Starting Training..."
python3 src/llamafactory/launcher.py \
    --stage sft \
    --do_train True \
    --do_eval True \
    --model_name_or_path Qwen/Qwen3-14B \
    --dataset ${DATASET_TRAIN} \
    --eval_dataset ${DATASET_DEV} \
    --dataset_dir ${PROJECT_ROOT}/data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 16 \
    --lora_alpha 32 \
    --output_dir ${TRAIN_OUTPUT} \
    --overwrite_output_dir True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 5.0 \
    --cutoff_len 2048 \
    --flash_attn fa2 \
    --eval_strategy steps \
    --eval_steps 25 \
    --save_strategy steps \
    --save_steps 100 \
    --save_safetensors True \
    --load_best_model_at_end True \
    --metric_for_best_model loss \
    --early_stopping_steps 4 \
    --logging_steps 5 \
    --save_total_limit 2 \
    --bf16 True \
    --plot_loss True \
    --seed 42 \
    --report_to wandb

# ==========================================
# 4. Dev Set Inference
# ==========================================
echo ">>> [Step 2/4] Running Inference on DEV set..."
python3 src/llamafactory/launcher.py \
    --stage sft \
    --do_predict True \
    --model_name_or_path Qwen/Qwen3-14B \
    --adapter_name_or_path ${TRAIN_OUTPUT} \
    --eval_dataset ${DATASET_DEV} \
    --dataset_dir ${PROJECT_ROOT}/data \
    --template qwen \
    --finetuning_type lora \
    --output_dir ${PREDICT_DEV_DIR} \
    --overwrite_output True \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --bf16 True \
    --max_new_tokens 2048 \
    --temperature 0.1

# ==========================================
# 5. Test Set Inference
# ==========================================
echo ">>> [Step 3/4] Running Inference on TEST set..."
python3 src/llamafactory/launcher.py \
    --stage sft \
    --do_predict True \
    --model_name_or_path Qwen/Qwen3-14B \
    --adapter_name_or_path ${TRAIN_OUTPUT} \
    --eval_dataset ${DATASET_TEST} \
    --dataset_dir ${PROJECT_ROOT}/data \
    --template qwen \
    --finetuning_type lora \
    --output_dir ${PREDICT_TEST_DIR} \
    --overwrite_output True \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --bf16 True \
    --max_new_tokens 2048 \
    --temperature 0.1

# ==========================================
# 6. Post-processing & Analysis
# ==========================================
echo ">>> [Step 4/4] Finalizing Results..."
cd "${PROJECT_ROOT}"

if [ -f "configs/predict/post_process.py" ]; then
    echo "Running Post-Process for Test Set..."
    python3 configs/predict/post_process.py ${MODEL_NAME}
fi

ANALYSIS_SCRIPT="configs/predict/dev_results_analysis.py"
if [ -f "$ANALYSIS_SCRIPT" ]; then
    echo "📊 Running Dev Set Error Analysis..."
    python3 "$ANALYSIS_SCRIPT" \
        --dev_file "$GT_DEV_FILE" \
        --predict_file "${PREDICT_DEV_DIR}/generated_predictions.jsonl" \
        --output_dir "${PREDICT_DEV_DIR}"
else
    echo "⚠️ Warning: Analysis script not found at $ANALYSIS_SCRIPT"
fi

echo "✅ All tasks for ${MODEL_NAME} completed successfully!"