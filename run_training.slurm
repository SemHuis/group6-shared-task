#!/bin/bash
#SBATCH --job-name=lora_semeval2026
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=/gaueko1/users/mmartin/shared_task/logs/lora_llamax3-8b_%j.out
#SBATCH --error=/gaueko1/users/mmartin/shared_task/logs/lora_llamax3-8b_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mariateresa.munoz@ehu.eus

echo "========================================"
echo "SemEval 2026 Task 11 - LoRA Fine-tuning fewshot"
echo "Model: LLaMAX3-8B"
echo "========================================"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# Activar entorno
source /gaueko1/users/mmartin/shared_task/lora_env/bin/activate

# Mostrar info de GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Variables de entorno para WandB
export WANDB_API_KEY="API-KEY"
export CUDA_VISIBLE_DEVICES=0

# Ejecutar script de entrenamiento
python /gaueko1/users/mmartin/shared_task/scripts/lora_llamax3_8b_fewshot.py

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "========================================"
