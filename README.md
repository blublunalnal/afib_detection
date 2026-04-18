# AFib Detection from PPG

This repository contains two independent projects for detecting atrial fibrillation (AFib) from photoplethysmography (PPG) signals:

1. **Revised DeepBeat** — a structurally improved retrain of the DeepBeat model using cleaner, updated data
2. **PPG Foundation Model Fine-tuning** — fine-tuning of two leading PPG foundation models (AnyPPG and PulsePPG) for AFib detection

---

## Repository Structure

```
afib_detection/
├── data_inspection.ipynb           # Data cleaning pipeline (deduplication, relabeling)
├── requirements.txt
├── pytorch/                        # DeepBeat (original + revised) model code
│   ├── deepbeat_model.py           # PyTorch port of the original DeepBeat architecture
│   ├── revised_deepbeat_model.py   # Structurally improved DeepBeat variant
│   ├── train_pytorch_model.py      # Training script for the original DeepBeat model
│   ├── train_revised_deepbeat.py   # Training script for the revised model (W&B integrated)
│   ├── benchmark_deepbeat.py       # Stratified evaluation metrics (by QA level)
│   ├── evaluate_per_qa_level.py    # Per-QA-level performance breakdown
│   ├── utils.py                    # Dataset, training loop, checkpointing utilities
│   ├── inspect_model.ipynb         # Layer-by-layer inspection and TF/PyTorch comparison
├── retrained_deepbeat_model/       # Trained artifacts for the revised DeepBeat
│   ├── infer.py                    # Inference script
│   ├── revised_deepbeat_diff_branch_2_best.pth   # Best checkpoint
│   └── model_thresholds.json       # Thresholds calibrated on validation data
├── fine_tuning/                    # PPG foundation model fine-tuning project
│   ├── fine_tuning_models.py       # FineTuning_rhythm and FineTuning_multitask model classes
│   ├── train_finetune_wandb.py     # Fine-tuning with W&B experiment tracking
│   ├── evaluate_fine_tune.py       # Evaluation for fine-tuned models
│   ├── params_tuning_finetune.py   # Hyperparameter search for fine-tuning
│   ├── preprocess_anyPPG.py        # Resampling + normalization for AnyPPG (32 Hz → 125 Hz)
│   ├── preprocess_pulsePPG.py      # Resampling + normalization for PulsePPG (32 Hz → 50 Hz)
│   ├── resnet1d.py                 # ResNet1D backbone (AnyPPG architecture)
│   ├── ResNet1D_Net.py             # ResNet1D backbone (PulsePPG architecture)
│   ├── anyppg_ckpt.pth             # Pre-trained AnyPPG weights
│   ├── plot_auroc.py               # AUROC curve plotting
```

---

## Project 1: Revised DeepBeat

### Background

[DeepBeat](https://www.nature.com/articles/s41746-020-00320-4) is a multi-task CNN that jointly classifies PPG signal quality (QA: poor / acceptable / good) and cardiac rhythm (Normal vs. AFib) from 25-second PPG windows sampled at 32 Hz.

### What changed

**Data:** The original DeepBeat training data contained duplicate segments with conflicting labels and outdated annotations. `data_inspection.ipynb` documents the full cleaning pipeline: duplicate removal, label conflict resolution, and augmentation with re-labeled samples.

**Architecture (`revised_deepbeat_model.py`):** The revised model follows the standard Conv → BN → Activation → Pooling → Dropout block ordering throughout the backbone, whereas the original (`deepbeat_model.py`) applies batch norm *after* pooling. The encoder part of the original model also omits batch norm entirely. Both models share the same dual-branch output structure (QA head + rhythm head), backbone channel sizes, and number of layers.

The QA branch's conv block intentionally retains the original ordering (Conv → ReLU → Pool → BN), as it empirically yields better QA prediction.

**Output & threshold (`retrained_deepbeat_model/model_thresholds.json`):** The original model produces two sigmoid outputs for AFib detection and uses `np.max(predictions)` to obtain a binary decision. The revised model uses softmax + cross-entropy, yielding calibrated class probabilities suitable for threshold tuning. The decision threshold is optimized per QA level by maximizing F1 on the validation set.

| Component | Original DeepBeat | Revised DeepBeat |
|---|---|---|
| Rhythm Conv Block order | Conv → ReLU → Pool → BN | Conv → BN → ReLU → Pool |
| Encoder batch norm | Missing | Present |
| AFib output | 2-class sigmoid + `np.max` | Softmax + calibrated threshold |

### Performance

> **Key result:** The revised model matches or exceeds the original DeepBeat at medium and high signal quality (QA 1–2) **without any autoencoder pretraining** — the original DeepBeat encoder was initialized from a self-supervised autoencoder, while the revised model is trained from scratch on cleaned data alone.

Metrics are stratified by QA level (0 = low, 1 = medium, 2 = high quality).

| QA Level | Model | TPR | TNR | PPV | F1 |
|---|---|---|---|---|---|
| 0 | Original DeepBeat | 0.640 | 0.780 | 0.530 | 0.580 |
| 0 | **Revised DeepBeat** | 0.472 | 0.841 | 0.540 | 0.503 |
| 1 | Original DeepBeat | 0.930 | 0.980 | 0.870 | 0.900 |
| 1 | **Revised DeepBeat** | 0.944 | 0.985 | 0.924 | **0.934** |
| 2 | Original DeepBeat | 0.980 | 0.990 | 0.940 | 0.960 |
| 2 | **Revised DeepBeat** | 0.992 | 0.997 | 0.980 | **0.986** |

At QA level 0 (low-quality signals), the revised model trades sensitivity for higher specificity; both models perform poorly in this regime and low-quality segments are typically excluded in practice. At QA 1 and QA 2, the revised model outperforms the original despite having no pretrained encoder.


### Training

```bash
cd pytorch
python train_revised_deepbeat.py \
    --file_name my_run \
    --train_data_path /path/to/train.pkl \
    --val_data_path   /path/to/val.pkl \
    --test_data_path  /path/to/test.pkl \
    --model revised_deepbeat \
    --epochs 100 \
    --scheduler plateau \
    --early_stopping \
    --auto_class_weights \
    --use_focal_rhythm
```

Key training features:
- **W&B integration** — metrics, gradients, artifacts, and model checkpoints are logged automatically
- **Focal loss** — optional focal loss for both rhythm and QA branches (`--use_focal_rhythm`, `--use_focal_qa`)
- **Class weighting** — manual or automatic inverse-frequency weights for the imbalanced AFib class
- **LR scheduling** — ReduceLROnPlateau or CosineAnnealingLR
- **Early stopping** — configurable patience and delta on any tracked metric
- **Optimal threshold** — validation-set F1-maximizing threshold found and applied at test time


### Pretrained checkpoint

The trained model is published on Hugging Face: **[llan00/revised_deepbeat](https://huggingface.co/llan00/revised_deepbeat)**

The `retrained_deepbeat_model/` folder contains everything needed for local use:

| File | Description |
|---|---|
| `revised_deepbeat_diff_branch_2_best.pth` | Best model checkpoint |
| `model_thresholds.json` | Per-QA-level decision thresholds calibrated on the validation set |
| `infer.py` | Inference script |

To load the checkpoint directly:

```python
import torch
from pytorch.revised_deepbeat_model import revised_DeepBeatModel

model = revised_DeepBeatModel()
ckpt  = torch.load('retrained_deepbeat_model/revised_deepbeat_diff_branch_2_best.pth',
                   map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

---

## Project 2: PPG Foundation Model Fine-tuning

### Background

Two self-supervised PPG foundation models are fine-tuned for AFib detection on the same DeepBeat dataset:

- **AnyPPG** — ResNet1D pre-trained on large-scale PPG data; expects input at **125 Hz**
- **PulsePPG** — ResNet1D pre-trained on PPG data; expects input at **50 Hz**

Both encoders output a 512-dimensional globally-pooled embedding, on top of which task-specific classification heads are attached.

### Model variants (`fine_tuning/fine_tuning_models.py`)

| Class | Output | Use case |
|---|---|---|
| `FineTuning_rhythm` | rhythm logits (2) | Rhythm-only classification |
| `FineTuning_multitask` | rhythm logits (2) + QA logits (3) | Joint rhythm + signal quality |

Both support `backbone='anyppg'` or `backbone='pulseppg'`, and optional encoder freezing (`freeze=True`).

### Preprocessing

Raw DeepBeat data is at 32 Hz. Before passing to the foundation models, signals must be resampled:

```bash
# For AnyPPG (32 Hz → 125 Hz)
python fine_tuning/preprocess_anyPPG.py

# For PulsePPG (32 Hz → 50 Hz)
python fine_tuning/preprocess_pulsePPG.py
```

### Training

```bash
# Standard training
python fine_tuning/train_finetune.py

# With W&B experiment tracking
python fine_tuning/train_finetune_wandb.py
```

### Hyperparameter tuning

```bash
python fine_tuning/params_tuning_finetune.py
```

### Pre-trained backbone weights

| File | Model |
|---|---|
| `fine_tuning/anyppg_ckpt.pth` | AnyPPG encoder weights |
| `fine_tuning/pulseppg_ckpt.pkl` | PulsePPG encoder weights (key: `net`) |

---

## Data
 The models are trained on the DeepBeat dataset. `data_inspection.ipynb` walks through the data cleaning steps applied before training. 
 
---

## Dependencies

```bash
pip install -r requirements.txt    
```
Core dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `optuna`, `wandb`, `scipy`, `tqdm`
