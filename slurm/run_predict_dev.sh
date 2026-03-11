#!/bin/bash
#SBATCH --job-name=predict_dev_14b_aug
#SBATCH --output=logs/predict_dev_%j.log
#SBATCH --error=logs/predict_dev_%j.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G

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
export GIT_CONFIG_PARAMETERS="'filter.xet.clean=' 'filter.xet.smudge=' 'filter.lfs.clean=git-lfs clean -- %f' 'filter.lfs.smudge=git-lfs smudge -- %f'"

# ==========================================
# 2. Experiment Variables
# ==========================================
MODEL_NAME="14b_aug"  # Must match the training MODEL_NAME
DATASET_DEV="logic_dev_augmented"

# Paths
TRAIN_OUTPUT="${SCRATCH_DIR}/semeval2026_task11/${MODEL_NAME}"
PREDICT_DEV_DIR="${PROJECT_ROOT}/outputs/${MODEL_NAME}/predict_dev"

# Disable Logic Loss (must match training setting)
export USE_LOGIC_LOSS=false

# Navigate to LLaMA-Factory
cd src/LLaMA-Factory || exit
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

echo ">>> [Status] Running DEV inference for model: ${MODEL_NAME}"

# ==========================================
# 3. Dev Set Inference
# ==========================================
echo ">>> Starting Inference on DEV set..."
llamafactory-cli train \
    --stage sft \
    --do_predict True \
    --model_name_or_path Qwen/Qwen3-14B \
    --adapter_name_or_path "${TRAIN_OUTPUT}" \
    --eval_dataset ${DATASET_DEV} \
    --dataset_dir "${PROJECT_ROOT}/data" \
    --template qwen \
    --finetuning_type lora \
    --output_dir "${PREDICT_DEV_DIR}" \
    --per_device_eval_batch_size 8 \
    --cutoff_len 2048 \
    --bf16 True \
    --predict_with_generate True \
    --temperature 0.1 \
    --top_p 0.9

# ==========================================
# 4. Post-analysis
# ==========================================
cd "${PROJECT_ROOT}"

GT_DEV_FILE="data/train_dev/final_dev.json"
PRED_FILE="${PREDICT_DEV_DIR}/generated_predictions.jsonl"
OUT_DIR="${PREDICT_DEV_DIR}"

if [ -f "configs/predict/dev_results_analysis.py" ]; then
    echo "Running Dev Set Error Analysis..."
    python3 configs/predict/dev_results_analysis.py \
        --dev_file "$GT_DEV_FILE" \
        --predict_file "$PRED_FILE" \
        --output_dir "$OUT_DIR"
    echo "✅ Analysis complete! Results saved to $OUT_DIR"
else
    echo "⚠️ Warning: Analysis script not found at configs/predict/dev_results_analysis.py"
fi

echo "✅ All DEV prediction tasks completed successfully!"