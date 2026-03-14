import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import json
import pickle
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

# -- local imports --
sys.path.insert(0, str(Path(__file__).parent))
from fine_tuning_models import DeepBeatDataset, FineTuning_rhythm, FineTuning_multitask

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent / 'pytorch'))
from utils import (
    EarlyStopping, restore_early_stopping_state,
    save_checkpoint, load_checkpoint, load_pickle_file,
    get_optimal_workers,
)
from benchmark_deepbeat import deepbeat_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AnyPPG backbone on DeepBeat dataset")

    # Data
    parser.add_argument("--train_data_path", required=True, help="Path to training pickle file")
    parser.add_argument("--val_data_path",   required=True, help="Path to validation pickle file")
    parser.add_argument("--output_path",     required=True, help="Root directory for outputs")

    # Experiment
    parser.add_argument("--file_name",   required=True, help="Run name (used in file/folder names)")
    parser.add_argument("--model_type",  required=True, choices=['rhythm', 'multitask'],
                        help="'rhythm': single-task rhythm classifier | 'multitask': rhythm + QA")

    # Backbone
    parser.add_argument("--backbone", type=str, default='anyppg', help='anyppg / pulseppg')
    parser.add_argument("--freeze_backbone", action='store_true',
                        help="Freeze AnyPPG encoder weights (train head only)")

    # Hyperparameters
    parser.add_argument("--batch_size",         type=int,   default=64)
    parser.add_argument("--epochs",             type=int,   default=50)
    parser.add_argument("--learning_rate",      type=float, default=1e-4)
    parser.add_argument("--weight_decay",       type=float, default=0.01)
    parser.add_argument("--dropout",            type=float, default=0.3)
    parser.add_argument("--backbone_lr_scale",  type=float, default=0.1,
                        help="Backbone LR = learning_rate * backbone_lr_scale (only when backbone is unfrozen). "
                             "Use a small value (e.g. 0.01-0.1) to prevent catastrophic forgetting.")
    parser.add_argument("--grad_clip",          type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled)")

    # multitask-only loss weights (ignored by rhythm model)
    parser.add_argument("--qa_loss_weight",     type=float, default=0.2)
    parser.add_argument("--rhythm_loss_weight", type=float, default=5.0)

    # Tuned params (JSON)
    parser.add_argument("--tuned_params_path", type=str, default=None,
                        help="Path to JSON file with tuned hyperparameters")

    # Best-model checkpointing
    parser.add_argument("--monitor_metric", type=str, default=None,
                        help="Metric to save best model. "
                             "rhythm: [f1, acc, loss] | multitask: [rhythm_f1, rhythm_acc, qa_acc, loss]")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")

    # Early stopping
    parser.add_argument("--early_stopping",           action='store_true')
    parser.add_argument("--early_stopping_patience",  type=int,   default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001)
    parser.add_argument("--early_stopping_metric",    type=str,   default=None,
                        help="Metric for early stopping (defaults to --monitor_metric)")

    # Evaluation
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Path to test pickle file. If provided, evaluation runs after training.")

    # Device
    parser.add_argument("--device",      type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers", type=int, default=None)

    # W&B
    parser.add_argument("--wandb_project", type=str, default="afib-detection",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity",  type=str, default=None,
                        help="W&B entity (team or username). Defaults to your default entity.")
    parser.add_argument("--wandb_tags",    type=str, nargs='*', default=None,
                        help="Optional tags for the W&B run (space-separated)")
    parser.add_argument("--notes",         type=str, default=None,
                        help="Free-text notes about the run, saved to W&B and the history file")
    parser.add_argument("--wandb_run_id",  type=str, default=None,
                        help="W&B run ID to resume a previous run (for --resume_from)")
    parser.add_argument("--wandb_offline", action='store_true',
                        help="Run W&B in offline mode (sync later with `wandb sync`)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_tuned_params(args):
    """Load hyperparameters from a JSON file and override args in-place."""
    if args.tuned_params_path is None:
        print("No tuned params file provided. Using command-line arguments.")
        return

    path = Path(args.tuned_params_path)
    if not path.exists():
        print(f"WARNING: Tuned params file not found: {path}. Using defaults.")
        return

    with open(path, 'r') as f:
        params = json.load(f)

    # Support both top-level and nested 'ready_to_use' format
    rtu = params.get('ready_to_use', params)
    args.batch_size    = int(rtu.get('batch_size', args.batch_size))
    args.learning_rate = rtu.get('lr', args.learning_rate)
    args.weight_decay  = rtu.get('weight_decay', args.weight_decay)
    args.dropout       = rtu.get('dropout', args.dropout)
    if args.model_type == 'multitask':
        args.qa_loss_weight     = rtu.get('qa_weight', args.qa_loss_weight)
        args.rhythm_loss_weight = rtu.get('rhythm_weight', args.rhythm_loss_weight)

    print("\n" + "=" * 60)
    print("LOADED TUNED HYPERPARAMETERS")
    print("=" * 60)
    print(f"  batch_size:    {args.batch_size}")
    print(f"  learning_rate: {args.learning_rate:.6f}")
    print(f"  weight_decay:  {args.weight_decay:.6f}")
    print(f"  dropout:       {args.dropout:.4f}")
    if args.model_type == 'multitask':
        print(f"  qa_weight:     {args.qa_loss_weight:.4f}")
        print(f"  rhythm_weight: {args.rhythm_loss_weight:.4f}")
    print("=" * 60 + "\n")


def _file_md5(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute MD5 hash of a file for data versioning."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def log_data_artifacts(args):
    """
    Version training and validation datasets as W&B Artifacts.
    Records file path, size, and MD5 so runs are reproducible.
    Returns the artifact objects (already logged).
    """
    def _make_artifact(label, data_path):
        path = Path(data_path)
        artifact = wandb.Artifact(
            name=f"{label}-dataset",
            type="dataset",
            description=f"{label} split for fine-tuning run '{args.file_name}'",
            metadata={
                "path":     str(path.resolve()),
                "filename": path.name,
                "size_mb":  round(path.stat().st_size / 1e6, 3),
                "md5":      _file_md5(data_path),
            },
        )
        artifact.add_reference(f"file://{path.resolve()}", name=path.name)
        wandb.log_artifact(artifact)
        return artifact

    train_artifact = _make_artifact("train", args.train_data_path)
    val_artifact   = _make_artifact("val",   args.val_data_path)
    print(f"  W&B data artifact logged: train={Path(args.train_data_path).name}, "
          f"val={Path(args.val_data_path).name}")
    return train_artifact, val_artifact


def setup_wandb(args, hyperparams: dict):
    """
    Initialise a W&B run.
    - Config stores all hyperparameters.
    - Tags and notes are set from CLI args.
    Returns the run object.
    """
    mode = "offline" if args.wandb_offline else "online"

    resume_mode = None
    if args.wandb_run_id is not None:
        resume_mode = "must"   # force-resume an existing run

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.file_name,
        id=args.wandb_run_id,
        resume=resume_mode,
        tags=args.wandb_tags,
        notes=args.notes,
        config=hyperparams,
        mode=mode,
    )
    # Expose W&B config so downstream code can call wandb.config.update()
    return run


def build_model(args, device):
    if args.model_type == 'rhythm':
        return FineTuning_rhythm(dropout=args.dropout, backbone=args.backbone, freeze=args.freeze_backbone).to(device)
    return FineTuning_multitask(dropout=args.dropout, backbone=args.backbone, freeze=args.freeze_backbone).to(device)


def default_monitor_metric(model_type):
    return 'f1' if model_type == 'rhythm' else 'rhythm_f1'


def init_history(model_type):
    base = {'loss': [], 'val_loss': []}
    if model_type == 'rhythm':
        base.update({'val_f1': [], 'val_acc': []})
    else:
        base.update({'val_rhythm_f1': [], 'val_rhythm_acc': [], 'val_qa_acc': []})
    return base


def update_history(history, train_m, val_m, model_type):
    history['loss'].append(train_m['loss'])
    history['val_loss'].append(val_m['loss'])
    if model_type == 'rhythm':
        history['val_f1'].append(val_m['f1'])
        history['val_acc'].append(val_m['acc'])
    else:
        history['val_rhythm_f1'].append(val_m['rhythm_f1'])
        history['val_rhythm_acc'].append(val_m['rhythm_acc'])
        history['val_qa_acc'].append(val_m['qa_acc'])


def log_wandb(train_m, val_m, epoch, model_type):
    """
    Log per-epoch metrics to W&B — mirrors what was logged to TensorBoard.

    Metric naming mirrors TensorBoard group/tag convention:
      Loss/train, Loss/val
      Gradients/train_grad_norm
      F1/train, F1/val  (rhythm)
      Accuracy/train, Accuracy/val  (rhythm)
      F1_Rhythm/train, F1_Rhythm/val  (multitask)
      Accuracy_Rhythm/train, Accuracy_Rhythm/val  (multitask)
      Accuracy_QA/train, Accuracy_QA/val  (multitask)
      Val/AUROC, Val/AUPRC
    """
    log_dict = {
        "epoch":              epoch,
        "Loss/train":         train_m['loss'],
        "Loss/val":           val_m['loss'],
        "Gradients/train_grad_norm": train_m['grad_norm'],
        "Val/AUROC":          val_m['auroc'],
        "Val/AUPRC":          val_m['auprc'],
    }

    if model_type == 'rhythm':
        log_dict.update({
            "F1/train":       train_m['f1'],
            "F1/val":         val_m['f1'],
            "Accuracy/train": train_m['acc'],
            "Accuracy/val":   val_m['acc'],
        })
    else:
        log_dict.update({
            "F1_Rhythm/train":       train_m['rhythm_f1'],
            "F1_Rhythm/val":         val_m['rhythm_f1'],
            "Accuracy_Rhythm/train": train_m['rhythm_acc'],
            "Accuracy_Rhythm/val":   val_m['rhythm_acc'],
            "Accuracy_QA/train":     train_m['qa_acc'],
            "Accuracy_QA/val":       val_m['qa_acc'],
            "Loss_Rhythm/train":     train_m['rhythm_loss'],
            "Loss_QA/train":         train_m['qa_loss'],
        })

    wandb.log(log_dict, step=epoch)


def print_epoch_summary(epoch, train_m, val_m, model_type):
    if model_type == 'rhythm':
        print(f"   -> Train Loss: {train_m['loss']:.4f} | F1: {train_m['f1']:.4f} | Acc: {train_m['acc']:.4f}")
        print(f"   -> Val   Loss: {val_m['loss']:.4f} | F1: {val_m['f1']:.4f} | Acc: {val_m['acc']:.4f} | AUROC: {val_m['auroc']:.4f}")
    else:
        print(f"   -> Train Loss: {train_m['loss']:.4f} | Rh F1: {train_m['rhythm_f1']:.4f} | Rh Acc: {train_m['rhythm_acc']:.4f} | QA Acc: {train_m['qa_acc']:.4f}")
        print(f"   -> Val   Loss: {val_m['loss']:.4f} | Rh F1: {val_m['rhythm_f1']:.4f} | Rh Acc: {val_m['rhythm_acc']:.4f} | QA Acc: {val_m['qa_acc']:.4f} | AUROC: {val_m['auroc']:.4f}")


def save_model_artifact(output_path, file_name, epoch, monitor_metric, best_metric_val):
    """Upload the best model checkpoint as a W&B Artifact."""
    ckpt_path = output_path / f"{file_name}_best.pth"
    if not ckpt_path.exists():
        return
    artifact = wandb.Artifact(
        name=f"{file_name}-best-model",
        type="model",
        metadata={
            "epoch":          epoch,
            "monitor_metric": monitor_metric,
            "metric_value":   best_metric_val,
        },
    )
    artifact.add_file(str(ckpt_path))
    wandb.log_artifact(artifact)
    print(f"   W&B model artifact uploaded: {ckpt_path.name}")


# ---------------------------------------------------------------------------
# Epoch runners  (unchanged from original)
# ---------------------------------------------------------------------------

def run_epoch_rhythm(model, dataloader, optimizer, device, epoch, is_training=True,
                     f1_average='macro', max_batches=None, verbose=True, grad_clip=1.0):
    model.train() if is_training else model.eval()
    desc = f"Epoch {epoch} [{'TRAIN' if is_training else 'VAL'}]"

    running_loss = 0.0
    total_grad_norm = 0.0
    all_preds, all_targets, all_logits = [], [], []

    criterion = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(is_training):
        for i, batch in enumerate(tqdm(dataloader, desc=desc, leave=True, ncols=120, disable=not verbose)):
            if max_batches is not None and i >= max_batches:
                break
            data   = batch['data'].to(device)
            target = batch['rhythm_label'].to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(data)
            loss   = criterion(logits, target)

            if is_training:
                loss.backward()
                clip = grad_clip if grad_clip > 0 else float('inf')
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                total_grad_norm += grad_norm.item()
                optimizer.step()

            running_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu())

    all_logits_cat = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits_cat, dim=1)[:, 1].numpy()

    metrics = {
        'loss':      running_loss / len(dataloader),
        'f1':        f1_score(all_targets, all_preds, average=f1_average, zero_division=0),
        'acc':       accuracy_score(all_targets, all_preds),
        'grad_norm': total_grad_norm / len(dataloader) if is_training else 0.0,
        'auroc':     roc_auc_score(all_targets, probs)           if not is_training else 0.0,
        'auprc':     average_precision_score(all_targets, probs) if not is_training else 0.0,
    }
    return metrics


def run_epoch_multitask(model, dataloader, optimizer, device, epoch,
                        qa_weight, rhythm_weight, is_training=True, f1_average='macro',
                        max_batches=None, verbose=True, grad_clip=1.0):
    model.train() if is_training else model.eval()
    desc = f"Epoch {epoch} [{'TRAIN' if is_training else 'VAL'}]"

    running_loss = 0.0
    running_qa_loss = 0.0
    running_rhythm_loss = 0.0
    total_grad_norm = 0.0
    all_rhythm_preds, all_rhythm_targets, all_rhythm_logits = [], [], []
    all_qa_preds, all_qa_targets = [], []

    criterion = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(is_training):
        for i, batch in enumerate(tqdm(dataloader, desc=desc, leave=True, ncols=120, disable=not verbose)):
            if max_batches is not None and i >= max_batches:
                break
            data          = batch['data'].to(device)
            rhythm_target = batch['rhythm_label'].to(device)
            qa_target     = batch['qa_label'].to(device)

            if is_training:
                optimizer.zero_grad()

            rhythm_logits, qa_logits = model(data)

            rhythm_loss = criterion(rhythm_logits, rhythm_target)
            qa_loss     = criterion(qa_logits, qa_target)
            loss        = rhythm_weight * rhythm_loss + qa_weight * qa_loss

            if is_training:
                loss.backward()
                clip = grad_clip if grad_clip > 0 else float('inf')
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                total_grad_norm += grad_norm.item()
                optimizer.step()

            running_loss        += loss.item()
            running_rhythm_loss += rhythm_loss.item()
            running_qa_loss     += qa_loss.item()

            all_rhythm_preds.extend(torch.argmax(rhythm_logits, dim=1).detach().cpu().numpy())
            all_rhythm_targets.extend(rhythm_target.detach().cpu().numpy())
            all_rhythm_logits.append(rhythm_logits.detach().cpu())
            all_qa_preds.extend(torch.argmax(qa_logits, dim=1).detach().cpu().numpy())
            all_qa_targets.extend(qa_target.detach().cpu().numpy())

    all_rhythm_logits_cat = torch.cat(all_rhythm_logits, dim=0)
    probs = F.softmax(all_rhythm_logits_cat, dim=1)[:, 1].numpy()

    metrics = {
        'loss':         running_loss        / len(dataloader),
        'rhythm_loss':  running_rhythm_loss / len(dataloader),
        'qa_loss':      running_qa_loss     / len(dataloader),
        'rhythm_f1':    f1_score(all_rhythm_targets, all_rhythm_preds, average=f1_average, zero_division=0),
        'rhythm_acc':   accuracy_score(all_rhythm_targets, all_rhythm_preds),
        'qa_acc':       accuracy_score(all_qa_targets, all_qa_preds),
        'grad_norm':    total_grad_norm / len(dataloader) if is_training else 0.0,
        'auroc':        roc_auc_score(all_rhythm_targets, probs)           if not is_training else 0.0,
        'auprc':        average_precision_score(all_rhythm_targets, probs) if not is_training else 0.0,
    }
    return metrics


def run_one_epoch(model, loader, optimizer, device, epoch, args, is_training):
    if args.model_type == 'rhythm':
        return run_epoch_rhythm(model, loader, optimizer, device, epoch,
                                is_training=is_training, grad_clip=args.grad_clip)
    return run_epoch_multitask(
        model, loader, optimizer, device, epoch,
        args.qa_loss_weight, args.rhythm_loss_weight,
        is_training=is_training, grad_clip=args.grad_clip,
    )


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

def run_evaluation(model, args, device, output_path):
    """Run inference on the test set and save predictions CSV + W&B metrics."""
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    # Load best checkpoint if available
    best_ckpt_path = output_path / f"{args.file_name}_best.pth"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print("Loaded best checkpoint for evaluation.")

    model.eval()

    print(f"Loading test data from: {args.test_data_path}")
    test_dict = load_pickle_file(args.test_data_path)
    is_preprocessed = test_dict.get('preprocessed', False)
    if is_preprocessed:
        print("Preprocessed data detected — skipping resampling and normalization.")

    test_dataset = DeepBeatDataset(
        test_dict['data'],
        test_dict['qa_label'],
        test_dict['rhythm_label'],
        preprocessed=is_preprocessed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    all_rhythm_preds, all_rhythm_targets, all_rhythm_probs = [], [], []
    all_qa_preds, all_qa_targets = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", ncols=120):
            data          = batch['data'].to(device)
            rhythm_target = batch['rhythm_label']
            qa_target     = batch['qa_label']

            if args.model_type == 'rhythm':
                logits = model(data)
                probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds  = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                rhythm_logits, qa_logits = model(data)
                probs  = F.softmax(rhythm_logits, dim=1)[:, 1].cpu().numpy()
                preds  = torch.argmax(rhythm_logits, dim=1).cpu().numpy()
                qa_preds = torch.argmax(qa_logits, dim=1).cpu().numpy()
                all_qa_preds.extend(qa_preds)
                all_qa_targets.extend(qa_target.numpy())

            all_rhythm_preds.extend(preds)
            all_rhythm_targets.extend(rhythm_target.numpy())
            all_rhythm_probs.extend(probs)

    all_rhythm_targets = np.array(all_rhythm_targets)
    all_rhythm_preds   = np.array(all_rhythm_preds)
    all_rhythm_probs   = np.array(all_rhythm_probs)

    # --- Reports ---
    print("\nRHYTHM CLASSIFICATION PERFORMANCE (AFib vs Normal)")
    print(classification_report(all_rhythm_targets, all_rhythm_preds, target_names=['Normal', 'AFib']))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_rhythm_targets, all_rhythm_preds))

    auroc = auprc = None
    try:
        auroc = roc_auc_score(all_rhythm_targets, all_rhythm_probs)
        auprc = average_precision_score(all_rhythm_targets, all_rhythm_probs)
        print(f"\nAUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
    except ValueError as e:
        print(f"Could not compute AUROC/AUPRC: {e}")

    if args.model_type == 'multitask' and all_qa_targets:
        all_qa_targets = np.array(all_qa_targets)
        all_qa_preds   = np.array(all_qa_preds)
        print("\nQA CLASSIFICATION PERFORMANCE")
        print(classification_report(all_qa_targets, all_qa_preds))

    # --- Save predictions CSV ---
    csv_data = {
        'rh_true':   all_rhythm_targets,
        'rh_pred':   all_rhythm_preds,
        'afib_prob': all_rhythm_probs,
    }
    if args.model_type == 'multitask' and len(all_qa_preds):
        csv_data['qa_true'] = np.array(all_qa_targets)
        csv_data['qa_pred'] = np.array(all_qa_preds)

    results_csv = output_path / "test_predictions.csv"
    pd.DataFrame(csv_data).to_csv(results_csv, index=False)
    print(f"\nPredictions saved to: {results_csv}")

    # --- DeepBeat stratified metrics (by predicted signal quality level) ---
    try:
        # Mirrors organize_results() using already-loaded test_dict (avoids np.load on .pkl)
        preds_db = pd.read_csv(results_csv)
        preds_db['ID'] = test_dict['ID']
        if 'qa_pred' not in preds_db.columns:
            preds_db['qa_pred'] = test_dict['qa_label']

        output_0 = deepbeat_metrics(preds_db, level=0)
        output_1 = deepbeat_metrics(preds_db, level=1)
        output_2 = deepbeat_metrics(preds_db, level=2)

        # Save per-level metrics CSV
        rows = []
        for level, out in [(0, output_0), (1, output_1), (2, output_2)]:
            row = {'qa_level': level}
            row.update({k: float(v) for k, v in out.items()})
            rows.append(row)
        metrics_df = pd.DataFrame(rows)
        metrics_csv = output_path / "deepbeat_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"DeepBeat stratified metrics saved to: {metrics_csv}")

        # Log to W&B
        db_log = {}
        for level, out in [(0, output_0), (1, output_1), (2, output_2)]:
            for metric, val in out.items():
                db_log[f"DeepBeat/QA{level}/{metric}"] = float(val)
                wandb.summary[f"deepbeat_qa{level}_{metric.lower()}"] = float(val)
        wandb.log(db_log)

        db_artifact = wandb.Artifact(
            name=f"{args.file_name}-deepbeat-metrics",
            type="evaluation",
        )
        db_artifact.add_file(str(metrics_csv))
        wandb.log_artifact(db_artifact)
        print("   W&B DeepBeat metrics artifact uploaded.")
    except Exception as e:
        print(f"WARNING: DeepBeat stratified metrics failed: {e}")

    # --- Log to W&B ---
    test_log = {}
    if auroc is not None:
        test_log["Test/AUROC"] = auroc
        test_log["Test/AUPRC"] = auprc
        wandb.summary["test_auroc"] = auroc
        wandb.summary["test_auprc"] = auprc
    rhythm_f1  = f1_score(all_rhythm_targets, all_rhythm_preds, average='macro', zero_division=0)
    rhythm_acc = accuracy_score(all_rhythm_targets, all_rhythm_preds)
    test_log["Test/rhythm_f1"]  = rhythm_f1
    test_log["Test/rhythm_acc"] = rhythm_acc
    wandb.summary["test_rhythm_f1"]  = rhythm_f1
    wandb.summary["test_rhythm_acc"] = rhythm_acc
    if args.model_type == 'multitask' and len(all_qa_preds):
        qa_acc = accuracy_score(np.array(all_qa_targets), np.array(all_qa_preds))
        test_log["Test/qa_acc"] = qa_acc
        wandb.summary["test_qa_acc"] = qa_acc
    wandb.log(test_log)

    csv_artifact = wandb.Artifact(
        name=f"{args.file_name}-test-predictions",
        type="evaluation",
        metadata={"test_data_path": args.test_data_path},
    )
    csv_artifact.add_file(str(results_csv))
    wandb.log_artifact(csv_artifact)
    print("   W&B test-predictions artifact uploaded.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    args = parse_args()
    device = torch.device(args.device)
    args.num_workers = get_optimal_workers(args.num_workers)

    print(f"Model type:   {args.model_type}")
    print(f"Using device: {device}\n")

    # --- Monitor / early-stop metrics ---
    if args.monitor_metric is None:
        args.monitor_metric = default_monitor_metric(args.model_type)
    if args.early_stopping_metric is None:
        args.early_stopping_metric = args.monitor_metric

    monitor_metric  = args.monitor_metric
    monitor_mode    = 'min' if monitor_metric == 'loss' else 'max'
    best_metric_val = float('inf') if monitor_mode == 'min' else -float('inf')
    best_epoch = 0
    print(f"Monitoring: {monitor_metric} (mode: {monitor_mode})")

    # --- History ---
    history = init_history(args.model_type)

    # --- Resume ---
    resume_checkpoint = None
    start_epoch = 1
    if args.resume_from is not None:
        if not Path(args.resume_from).exists():
            print(f"ERROR: Checkpoint not found: {args.resume_from}. Starting fresh.")
        else:
            print("\n" + "=" * 60)
            print("RESUMING TRAINING FROM CHECKPOINT")
            print("=" * 60)
            resume_checkpoint = args.resume_from

    # --- Tuned params ---
    apply_tuned_params(args)

    hyperparams = {
        # Model
        'model_type':         args.model_type,
        'backbone':           args.backbone,
        'freeze_backbone':    args.freeze_backbone,
        'dropout':            args.dropout,
        # Optimiser
        'batch_size':         args.batch_size,
        'epochs':             args.epochs,
        'learning_rate':      args.learning_rate,
        'weight_decay':       args.weight_decay,
        'backbone_lr_scale':  args.backbone_lr_scale,
        'grad_clip':          args.grad_clip,
        # Loss (multitask)
        'qa_loss_weight':     args.qa_loss_weight,
        'rhythm_loss_weight': args.rhythm_loss_weight,
        # Training config
        'monitor_metric':     args.monitor_metric,
        'device':             args.device,
        'seed':               42,
        # Data paths (for provenance)
        'train_data_path':    args.train_data_path,
        'val_data_path':      args.val_data_path,
    }
    history['hyperparameters'] = hyperparams
    if args.notes:
        history['notes'] = args.notes
    print(hyperparams)

    # --- W&B init ---
    print("\nInitialising Weights & Biases...")
    run = setup_wandb(args, hyperparams)
    print(f"  W&B run: {run.url}\n")

    # --- Data versioning ---
    print("Logging data artifacts...")
    log_data_artifacts(args)

    # --- Data loading ---
    print("\nLoading training data...")
    train_dict = load_pickle_file(args.train_data_path)
    data_train = train_dict['data']
    label_train_r, label_train_q = train_dict['rhythm_label'], train_dict['qa_label']
    print(f"Train shape: {data_train.shape}")

    print("Loading validation data...")
    val_dict = load_pickle_file(args.val_data_path)
    data_val = val_dict['data']
    label_val_r, label_val_q = val_dict['rhythm_label'], val_dict['qa_label']
    print(f"Val shape:   {data_val.shape}")

    is_preprocessed = train_dict.get('preprocessed', False)
    if is_preprocessed:
        print("Preprocessed data detected — skipping resampling and normalization.")

    # Log dataset sizes to W&B config
    wandb.config.update({
        "train_samples": len(data_train),
        "val_samples":   len(data_val),
        "preprocessed":  is_preprocessed,
    }, allow_val_change=True)

    train_dataset = DeepBeatDataset(data_train, label_train_q, label_train_r, preprocessed=is_preprocessed)
    val_dataset   = DeepBeatDataset(data_val,   label_val_q,   label_val_r,   preprocessed=is_preprocessed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    # --- Model & optimizer ---
    print("\nBuilding model...")
    freeze_changed = False
    if resume_checkpoint is not None:
        temp_ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        saved_hp = temp_ckpt.get('history', {}).get('hyperparameters', {})
        args.dropout  = saved_hp.get('dropout', args.dropout)
        args.backbone = saved_hp.get('backbone', args.backbone)
        saved_freeze  = saved_hp.get('freeze_backbone', args.freeze_backbone)
        freeze_changed = (saved_freeze != args.freeze_backbone)
        if freeze_changed:
            print(f"  Freeze state changing: {saved_freeze} → {args.freeze_backbone} "
                  f"— optimizer will be reset (not loaded from checkpoint)")

    model = build_model(args, device)
    if args.freeze_backbone:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        backbone_params = list(model.encoder.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]
        backbone_lr     = args.learning_rate * args.backbone_lr_scale
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params,     'lr': args.learning_rate},
        ], weight_decay=args.weight_decay)
        print(f"Differential LR — backbone: {backbone_lr:.2e}, head: {args.learning_rate:.2e}")

    if resume_checkpoint is not None:
        opt_to_load = None if freeze_changed else optimizer
        checkpoint = load_checkpoint(resume_checkpoint, model, opt_to_load, device)
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)

        hist_key = f'val_{monitor_metric}'
        if hist_key in history and len(history[hist_key]) > 0:
            vals = history[hist_key]
            best_metric_val = min(vals) if monitor_mode == 'min' else max(vals)
            best_epoch = vals.index(best_metric_val) + 1

        print(f"\nResuming from epoch {start_epoch}")
        print(f"Best val {monitor_metric} so far: {best_metric_val:.4f} at epoch {best_epoch}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters:      {param_count:,}")
    print(f"Backbone frozen: {'YES' if args.freeze_backbone else 'NO'}")

    # Log model parameter count
    wandb.config.update({"model_parameters": param_count}, allow_val_change=True)

    # Watch model: log gradients and parameters every 50 steps
    wandb.watch(model, log="gradients", log_freq=50)

    # --- Early stopping ---
    early_stopper = None
    if args.early_stopping:
        mode = 'min' if args.early_stopping_metric == 'loss' else 'max'
        early_stopper = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=mode,
            verbose=True,
        )
        if resume_checkpoint is not None:
            restore_early_stopping_state(early_stopper, checkpoint)
        print(f"\n{'='*60}")
        print(f"EARLY STOPPING ENABLED")
        print(f"  metric={args.early_stopping_metric}, patience={args.early_stopping_patience}, mode={mode}")
        print(f"{'='*60}\n")

    output_path = Path(args.output_path) / args.file_name
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    print("Starting training...")
    epoch = start_epoch - 1
    val_m = {}
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_m = run_one_epoch(model, train_loader, optimizer, device, epoch, args, is_training=True)
            val_m   = run_one_epoch(model, val_loader,   optimizer, device, epoch, args, is_training=False)

            log_wandb(train_m, val_m, epoch, args.model_type)
            update_history(history, train_m, val_m, args.model_type)
            print_epoch_summary(epoch, train_m, val_m, args.model_type)

            # Early stopping
            if early_stopper is not None:
                if early_stopper(val_m[args.early_stopping_metric], epoch):
                    print(f"\nEarly stopping at epoch {epoch}")
                    wandb.log({"early_stop_epoch": epoch}, step=epoch)
                    break

            # Save best model
            current_val = val_m[monitor_metric]
            is_best = (current_val < best_metric_val) if monitor_mode == 'min' else (current_val > best_metric_val)
            if is_best:
                best_metric_val = current_val
                best_epoch = epoch
                save_checkpoint(epoch, model, optimizer, val_m, history, args,
                                output_path, None, early_stopper, checkpoint_type='best')
                save_model_artifact(output_path, args.file_name, epoch, monitor_metric, best_metric_val)
                wandb.log({f"best/{monitor_metric}": best_metric_val, "best/epoch": epoch}, step=epoch)
                print(f"   * New best model saved ({monitor_metric}: {best_metric_val:.4f})")

            # Progress checkpoint every 10 epochs (enables resume)
            if epoch % 10 == 0:
                save_checkpoint(epoch, model, optimizer, val_m, history, args,
                                output_path, None, early_stopper, checkpoint_type='progress')
                print(f"   Progress checkpoint saved (epoch {epoch})")

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED (Ctrl+C) — saving checkpoint...")
        save_checkpoint(epoch, model, optimizer, val_m, history, args,
                        output_path, None, early_stopper, checkpoint_type='interrupted')
        resume_path = output_path / f"{args.file_name}_interrupted.pth"
        print(f"Resume with: --resume_from {resume_path}")
        print("=" * 60)

    except Exception as e:
        print(f"\nTRAINING FAILED: {e}")
        try:
            save_checkpoint(epoch, model, optimizer, val_m, history, args,
                            output_path, None, early_stopper, checkpoint_type='error')
            print("Error checkpoint saved.")
        except Exception:
            print("Could not save error checkpoint.")
        wandb.finish(exit_code=1)
        raise

    # --- Final checkpoint & history ---
    save_checkpoint(epoch, model, optimizer, val_m, history, args,
                    output_path, None, early_stopper, checkpoint_type='final')

    with open(output_path / f"{args.file_name}_history.pkl", 'wb') as f:
        pickle.dump(history, f)

    # --- ONNX export ---
    onnx_path = output_path / f"{args.file_name}.onnx"
    try:
        best_ckpt_path = output_path / f"{args.file_name}_best.pth"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt['model_state_dict'])
            print("Loaded best checkpoint for ONNX export.")
        model.eval()
        seq_len = 1250 if args.backbone == 'pulseppg' else 3125
        dummy_input = torch.randn(1, 1, seq_len, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["ppg"],
            output_names=["rhythm_logits"] if args.model_type == 'rhythm' else ["rhythm_logits", "qa_logits"],
            dynamic_axes={"ppg": {0: "batch_size"}},
            opset_version=17,
        )
        print(f"ONNX model saved: {onnx_path}")
        onnx_artifact = wandb.Artifact(
            name=f"{args.file_name}-onnx",
            type="model",
            metadata={"format": "onnx", "opset": 17, "input_shape": [1, seq_len]},
        )
        onnx_artifact.add_file(str(onnx_path))
        wandb.log_artifact(onnx_artifact)
        print("   W&B ONNX artifact uploaded.")
    except Exception as e:
        print(f"WARNING: ONNX export failed: {e}")

    # --- Post-training evaluation ---
    if args.test_data_path is not None:
        run_evaluation(model, args, device, output_path)

    # Summary metrics visible on the W&B run overview page
    wandb.summary["best_epoch"]              = best_epoch
    wandb.summary[f"best_val_{monitor_metric}"] = best_metric_val
    wandb.summary["total_epochs"]            = epoch

    wandb.finish()

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best epoch:        {best_epoch}")
    print(f"Best val {monitor_metric}: {best_metric_val:.4f}")
    if early_stopper is not None and early_stopper.early_stop:
        print(f"Early stopped at epoch {epoch}")
    else:
        print(f"Total epochs: {epoch}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
