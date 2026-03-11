# Sylloscope: DeepSeek-Enhanced Distillation for Logical Reasoning

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache-2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of our SemEval-2026 Task 11 system: *Sylloscope - Decoupling Logic from Belief via DeepSeek-Enhanced Distillation in Qwen Models*.

---

## 🧠 Overview

This project implements a neuro-symbolic teacher-student framework for syllogistic reasoning. We use **DeepSeek-R1** as a Logical Auditor to generate high-fidelity Chain-of-Thought (CoT) traces, then distill these into **Qwen-3** models using **LoRA**. Our approach mitigates belief bias by teaching formal logic mechanics rather than simple label matching.

**Team**: Zhanyu Chen, María Teresa Muñoz Martín, Sem Huisman, Jingjing Lan  
**Institution**: University of Groningen, Netherlands  
**Task**: [SemEval-2026 Task 11](https://semeval.github.io/SemEval2026/tasks) - Disentangling Content and Formal Reasoning in LLMs  

**Repository**: https://github.com/SemHuis/group6-shared-task (branch: `deepseek-distillation`)

---

## 📊 Results

| Model | Subtask | Score | Accuracy | TCE |
|-------|---------|-------|----------|-----|
| Qwen-3 14B (Dataset 2) | 1 | 39.81 | 96.86% | — |
| Qwen-3 14B (Dataset 2) | 3 | 26.02 | 91.67% | 11.46 |

*See [Paper](paper.pdf) for full results and error analysis.*

---

## 🗂️ Repository Structure

```
group6-shared-task/
├── data/
│   └── train_dev/
│       ├── dev.json                 # Dataset 1 dev (CoT only)
│       ├── dev_base.json            # Unregistered raw base dev
│       ├── dev_base_lf.json         # Base dev (no CoT, ShareGPT) → dev_base
│       ├── final_dev.json           # Dataset 2 dev (augmented) → logic_dev_augmented
│       ├── final_train.json         # Dataset 2 train (augmented) → logic_train_augmented
│       ├── train.json               # Dataset 1 train (CoT only) → logic_train
│       ├── train_base.json          # Unregistered raw base train
│       ├── train_base_lf.json       # Base train (no CoT, ShareGPT) → train_base
│       └── dataset_info.json        # LLaMA-Factory dataset registration
├── slurm/
│   ├── run_14b_aug_default.sh      # Train Qwen-3 14B on Dataset 2
│   ├── run_predict_dev.sh          # Dev inference + error analysis
│   └── run_predict_test.sh         # Test inference + post-processing
├── configs/
│   └── predict/
│       ├── post_process.py         # Convert predictions to submission format
│       └── dev_results_analysis.py # Error analysis by quadrant
├── src/
│   └── LLaMA-Factory/              # LLaMA-Factory library (v1)
├── requirements.txt                # Python dependencies (from pip freeze)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository (default branch: deepseek-distillation)
git clone https://github.com/SemHuis/group6-shared-task.git
cd group6-shared-task

# Create and activate virtual environment
python -m venv envs/llama_factory_v1
source envs/llama_factory_v1/bin/activate  # Linux/Mac
# or: envs\llama_factory_v1\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data

All required data files are located in `data/train_dev/`.  
The `data/dataset_info.json` file registers dataset IDs for LLaMA-Factory.

**Main training data** (Dataset 2):
- `final_train.json` → dataset ID `logic_train_augmented`
- `final_dev.json` → dataset ID `logic_dev_augmented`

No additional download needed.

### 3. Training

Submit the main training job via Slurm:

```bash
sbatch slurm/run_14b_aug_default.sh
```

This will:
- Fine-tune Qwen-3 14B with LoRA (r=16, α=32) on `logic_train_augmented`
- Perform periodic evaluation on `logic_dev_augmented`
- Save best model (based on validation loss) to scratch
- Log to WandB (if configured)

**Expected duration**: ~1 hour on L40S (adjust `--time` in script as needed)

### 4. Inference

#### Dev set (with error analysis)
```bash
sbatch slurm/run_predict_dev.sh
```
Generates predictions and runs `configs/predict/dev_results_analysis.py` to produce `error_cases.csv` by quadrant.

#### Test set (submission format)
```bash
sbatch slurm/run_predict_test.sh
```
Generates `predictions.json` in `outputs/14b_aug/predict_test/` ready for submission.

---

## 📦 Dataset Details

### Registered Datasets (via `dataset_info.json`)

| Dataset ID | File | Format | Samples | Description |
|------------|------|--------|---------|-------------|
| `logic_train_augmented` | `final_train.json` | ShareGPT | 1,146 | **Main training set** (CoT + adversarial augmentation) |
| `logic_dev_augmented` | `final_dev.json` | ShareGPT | 114 | **Validation set** (ground truth) |
| `logic_train` | `train.json` | ShareGPT | ~940 | CoT-distilled only (baseline) |
| `logic_dev` | `dev.json` | ShareGPT | ~100 | Baseline validation |
| `train_base` | `train_base_lf.json` | ShareGPT | ~940 | Base data without CoT |
| `dev_base` | `dev_base_lf.json` | ShareGPT | ~100 | Base validation |

**Note**: Only files listed in `dataset_info.json` can be loaded by LLaMA-Factory using their dataset IDs. The `*_base.json` files (without `_lf`) are kept for reference but are **not** registered.

### Data Pipeline

1. **Raw data** (960 samples, labels only) → DeepSeek-R1 CoT distillation → `train.json`/`dev.json` (Dataset 1, ~940 high-quality samples after filtering)
2. **Dataset 1** + targeted adversarial augmentation → `final_train.json`/`final_dev.json` (Dataset 2, 1,260 samples total: 1,146 train + 114 dev)
3. **Base data** (no CoT) → `*_base_lf.json` (for ablation studies)

---

## 🔧 Training Configuration

### Model & Method
- **Base model**: Qwen-3 14B (`Qwen/Qwen3-14B`)
- **Fine-tuning method**: LoRA (Low-Rank Adaptation)
- **Target modules**: `q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj`
- **LoRA hyperparameters**: rank=16, alpha=32

### Optimizer & Scheduler
- **Optimizer**: AdamW (betas: 0.9, 0.95; weight_decay: 0.01)
- **Learning rate**: 5e-5 (cosine scheduler, warmup: 500 iters)
- **Training epochs**: 5.0
- **Batch size**: 4 per device × 4 gradient accumulation = 16 effective
- **Sequence length**: 2048
- **BF16**: enabled
- **Gradient checkpointing**: enabled
- **Early stopping**: patience 4 steps on validation loss

### Evaluation
- **Evaluation strategy**: steps
- **Eval steps**: 25
- **Save strategy**: steps
- **Save steps**: 100
- **Keep top**: 2 checkpoints
- **Load best at end**: True (based on validation loss)
- **WandB logging**: enabled

---

## 🛠️ Scripts

### Training Script (`slurm/run_14b_aug_default.sh`)

Key environment variables:
- `MODEL_NAME="14b_aug"` – output directory name under `outputs/` and scratch
- `USE_LOGIC_LOSS=false` – standard cross-entropy (not logic-specific loss)
- `PROJECT_ROOT` – automatically set to current directory
- `TRAIN_OUTPUT` – where LoRA adapters are saved (`/scratch/$USER/semeval2026_task11/${MODEL_NAME}`)
- `PREDICT_DEV_DIR`, `PREDICT_TEST_DIR` – prediction output paths

**Note**: The script expects to be run from the repository root directory.

### Prediction Scripts

- `slurm/run_predict_dev.sh`: loads adapter from `TRAIN_OUTPUT`, runs dev inference, then executes `configs/predict/dev_results_analysis.py` to produce quadrant-wise error analysis.
- `slurm/run_predict_test.sh`: runs test inference and calls `configs/predict/post_process.py` to produce submission JSON.

**Important**: Both scripts require `MODEL_NAME` to match the training run name.

---

## 📝 Citation

If you use this code or data, please cite our paper:

```bibtex
@inproceedings{chen2026sylloscope,
  title={Sylloscope at SemEval-2026 Task 11: Decoupling Logic from Belief via DeepSeek-Enhanced Distillation in Qwen Models},
  author={Chen, Zhanyu and Muñoz Martín, María Teresa and Huisman, Sem and Lan, Jingjing},
  booktitle={Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year={2026},
  organization={N/A}
}
```

---

## 📄 License

This project is licensed under the Apache-2.0 License - see [LICENSE](LICENSE) for details.

**Third-party code**: `src/LLaMA-Factory/` is a separate library under its own license (Apache-2.0). All credit to the original authors.

---

## 🙏 Acknowledgements

- **LLaMA-Factory** team for the efficient fine-tuning framework
- **DeepSeek-AI** for the DeepSeek-R1 reasoning model
- **Qwen** team for the multilingual Qwen-3 model
- **SemEval 2026 Task 11** organizers for the challenging benchmark

---

## 🔗 Related Links

- **Task homepage**: https://semeval.github.io/SemEval2026/tasks
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **DeepSeek**: https://github.com/deepseek-ai/DeepSeek
- **Qwen**: https://github.com/QwenLM

---

## 🛠️ Troubleshooting

### Dataset not found?

Ensure `data/dataset_info.json` exists and contains the dataset ID you are using (e.g., `logic_train_augmented`). The file must be present in the `data/` directory before launching training.

### Path errors?

All scripts assume they are executed from the repository root directory. Do not `cd` into subdirectories before running `sbatch`.

### Virtual environment missing?

The scripts automatically activate `envs/llama_factory_v1/bin/activate`. If you use a different environment name, update the `source` line in the slurm scripts.

---

**Last updated**: 2026-03-11