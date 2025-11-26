#!/bin/bash
#SBATCH --job-name=evaluation_qwen_predictions
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/shared_task/logs/evaluation_qwen_spanish_spanish_prompt%j.out
#SBATCH --error=/gaueko1/users/mmartin/shared_task/logs/evaluation_qwen_spanish_spanish_prompt%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

echo "========================================"
echo "Evaluate predictions from Qwen3-8B LoRA models"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

source /gaueko1/users/mmartin/shared_task/lora_env/bin/activate

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Evaluate zero-shot predictions
python /gaueko1/users/mmartin/shared_task/scripts/evaluation_script.py \
  --gt_file /gaueko1/users/mmartin/shared_task/data/spanish_valid_data.json \
  --pred_file /gaueko1/users/mmartin/shared_task/results/qwen_spanish/predictions_zeroshot_spanish_prompt.json \
  --model_name "Qwen3-8B LoRA Zero-shot Spanish Valid Data With Spanish Prompt"

# Evaluate few-shot predictions (comment out if running separately)
python /gaueko1/users/mmartin/shared_task/scripts/evaluation_script.py \
  --gt_file /gaueko1/users/mmartin/shared_task/data/spanish_valid_data.json \
  --pred_file /gaueko1/users/mmartin/shared_task/results/qwen_spanish/predictions_fewshot_spanish_prompt.json \
  --model_name "Qwen3-8B LoRA Few-shot Spanish Valid Data With Spanish Prompt"

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
