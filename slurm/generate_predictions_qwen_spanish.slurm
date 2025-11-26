#!/bin/bash
#SBATCH --job-name=generate_predictions_qwen
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/shared_task/logs/generate_predictions_qwen_spanish_fewshot_spanish_prompt_%j.out
#SBATCH --error=/gaueko1/users/mmartin/shared_task/logs/generate_predictions_qwen_spanish__fewshot_spanish_prompt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

echo "========================================"
echo "Generate Predictions using Qwen3-8B LoRA models Fewshot"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

source /gaueko1/users/mmartin/shared_task/lora_env/bin/activate

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

export CUDA_VISIBLE_DEVICES=0

# Run zero-shot predictions
python /gaueko1/users/mmartin/shared_task/scripts/generate_predictions_spanish_prompt.py 

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
