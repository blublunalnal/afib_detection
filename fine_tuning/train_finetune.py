import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -- local imports --
sys.path.insert(0, str(Path(__file__).parent))
from fine_tuning_models import DeepBeatDataset, FineTuning_rhythm, FineTuning_multitask

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'pytorch'))
from utils import (
    EarlyStopping, restore_early_stopping_state,
    save_checkpoint, load_checkpoint, load_pickle_file,
    get_optimal_workers,
)


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
    parser.add_argument("--backbone", type = str, default= 'anyppg', help = 'anyppg / pulseppg')
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

    # Device
    parser.add_argument("--device",      type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers", type=int, default=None)

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


def setup_tensorboard(args):
    log_path = Path(args.output_path) / args.file_name
    log_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_path))


def build_model(args, device):
    if args.model_type == 'rhythm':
        return FineTuning_rhythm(dropout=args.dropout, backbone= args.backbone, freeze=args.freeze_backbone).to(device)
    return FineTuning_multitask(dropout=args.dropout, backbone= args.backbone, freeze=args.freeze_backbone).to(device)


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


def log_tensorboard(writer, train_m, val_m, epoch, model_type):
    writer.add_scalars('Loss', {'train': train_m['loss'], 'val': val_m['loss']}, epoch)
    writer.add_scalar('Gradients/train', train_m['grad_norm'], epoch)
    if model_type == 'rhythm':
        writer.add_scalars('F1',       {'train': train_m['f1'],  'val': val_m['f1']},  epoch)
        writer.add_scalars('Accuracy', {'train': train_m['acc'], 'val': val_m['acc']}, epoch)
    else:
        writer.add_scalars('F1/Rhythm',       {'train': train_m['rhythm_f1'],  'val': val_m['rhythm_f1']},  epoch)
        writer.add_scalars('Accuracy/Rhythm', {'train': train_m['rhythm_acc'], 'val': val_m['rhythm_acc']}, epoch)
        writer.add_scalars('Accuracy/QA',     {'train': train_m['qa_acc'],     'val': val_m['qa_acc']},     epoch)
    writer.add_scalar('Val/AUROC', val_m['auroc'], epoch)
    writer.add_scalar('Val/AUPRC', val_m['auprc'], epoch)


def print_epoch_summary(epoch, train_m, val_m, model_type):
    if model_type == 'rhythm':
        print(f"   -> Train Loss: {train_m['loss']:.4f} | F1: {train_m['f1']:.4f} | Acc: {train_m['acc']:.4f}")
        print(f"   -> Val   Loss: {val_m['loss']:.4f} | F1: {val_m['f1']:.4f} | Acc: {val_m['acc']:.4f} | AUROC: {val_m['auroc']:.4f}")
    else:
        print(f"   -> Train Loss: {train_m['loss']:.4f} | Rh F1: {train_m['rhythm_f1']:.4f} | Rh Acc: {train_m['rhythm_acc']:.4f} | QA Acc: {train_m['qa_acc']:.4f}")
        print(f"   -> Val   Loss: {val_m['loss']:.4f} | Rh F1: {val_m['rhythm_f1']:.4f} | Rh Acc: {val_m['rhythm_acc']:.4f} | QA Acc: {val_m['qa_acc']:.4f} | AUROC: {val_m['auroc']:.4f}")


def run_epoch_rhythm(model, dataloader, optimizer, device, epoch, is_training=True,
                     f1_average='macro', max_batches=None, verbose=True, grad_clip=1.0):
    """Single-task epoch runner for FineTuning_rhythm. Labels are integer indices.

    Args:
        max_batches: if set, stop after this many batches (used during Optuna tuning
                     to subsample large datasets and speed up each trial epoch).
        verbose: if False, suppresses the per-batch tqdm bar (used during Optuna tuning).
    """
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
            target = batch['rhythm_label'].to(device)   # (N,) integer indices

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
        'auroc':     roc_auc_score(all_targets, probs)              if not is_training else 0.0,
        'auprc':     average_precision_score(all_targets, probs)    if not is_training else 0.0,
    }
    return metrics


def run_epoch_multitask(model, dataloader, optimizer, device, epoch,
                        qa_weight, rhythm_weight, is_training=True, f1_average='macro',
                        max_batches=None, verbose=True, grad_clip=1.0):
    """Multi-task epoch runner for FineTuning_multitask. Labels are integer indices.

    Args:
        max_batches: if set, stop after this many batches (used during Optuna tuning).
        verbose: if False, suppresses the per-batch tqdm bar (used during Optuna tuning).
    """
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
            data           = batch['data'].to(device)
            rhythm_target  = batch['rhythm_label'].to(device)  # (N,) integer indices
            qa_target      = batch['qa_label'].to(device)       # (N,) integer indices

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
    """Dispatch to the correct epoch runner based on model_type."""
    if args.model_type == 'rhythm':
        return run_epoch_rhythm(model, loader, optimizer, device, epoch,
                                is_training=is_training, grad_clip=args.grad_clip)
    return run_epoch_multitask(
        model, loader, optimizer, device, epoch,
        args.qa_loss_weight, args.rhythm_loss_weight,
        is_training=is_training, grad_clip=args.grad_clip,
    )


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

    monitor_metric = args.monitor_metric
    monitor_mode   = 'min' if monitor_metric == 'loss' else 'max'
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
    history['hyperparameters'] = {
        'batch_size':         args.batch_size,
        'learning_rate':      args.learning_rate,
        'weight_decay':       args.weight_decay,
        'dropout':            args.dropout,
        'freeze_backbone':    args.freeze_backbone,
        'model_type':         args.model_type,
        'backbone':           args.backbone,
        'backbone_lr_scale':  args.backbone_lr_scale,
        'grad_clip':          args.grad_clip,
    }
    print(history['hyperparameters'])

    # --- Data ---
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
        # Pre-load to read saved hyperparameters before building model/optimizer
        temp_ckpt = torch.load(resume_checkpoint, map_location=device)
        saved_hp = temp_ckpt.get('history', {}).get('hyperparameters', {})
        # Restore dropout (affects model construction); freeze_backbone uses CLI value
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
        # Skip loading optimizer state when freeze changed — param groups are incompatible
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

    print(f"Parameters:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"Backbone frozen: {'YES' if args.freeze_backbone else 'NO'}")

    writer = setup_tensorboard(args)

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

            log_tensorboard(writer, train_m, val_m, epoch, args.model_type)
            update_history(history, train_m, val_m, args.model_type)
            print_epoch_summary(epoch, train_m, val_m, args.model_type)

            # Early stopping
            if early_stopper is not None:
                if early_stopper(val_m[args.early_stopping_metric], epoch):
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            # Save best model
            current_val = val_m[monitor_metric]
            is_best = (current_val < best_metric_val) if monitor_mode == 'min' else (current_val > best_metric_val)
            if is_best:
                best_metric_val = current_val
                best_epoch = epoch
                save_checkpoint(epoch, model, optimizer, val_m, history, args,
                                output_path, None, early_stopper, checkpoint_type='best')
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
        raise

    # --- Final checkpoint & history ---
    save_checkpoint(epoch, model, optimizer, val_m, history, args,
                    output_path, None, early_stopper, checkpoint_type='final')

    with open(output_path / f"{args.file_name}_history.pkl", 'wb') as f:
        pickle.dump(history, f)

    writer.close()

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
