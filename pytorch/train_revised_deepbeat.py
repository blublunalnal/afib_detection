import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
import wandb
from tqdm import tqdm

from revised_deepbeat_model import revised_DeepBeatModel
from deepbeat_model import DeepBeatModel
from benchmark_deepbeat import deepbeat_metrics
from utils import (
    DeepBeatDataset, EarlyStopping,
    restore_early_stopping_state, apply_tuned_params,
    save_checkpoint, load_checkpoint, load_pickle_file,
    get_optimal_workers, run_epoch,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train revised DeepBeat model with W&B logging")

    # Data
    parser.add_argument("--train_data_path",             default=r'C:\Users\aoara\develop\deepbeat\data\ori_train.pkl')
    parser.add_argument("--val_data_path",               default=r'C:\Users\aoara\develop\deepbeat\data\ori_val.pkl')
    parser.add_argument("--output_path",                 default=r'C:\Users\aoara\develop\deepbeat\training_output')

    # Experiment
    parser.add_argument("--file_name",        required=True, help="Run name used for file/folder names")
    parser.add_argument("--model",            type=str, default='revised_deepbeat',
                        choices=['revised_deepbeat', 'deepbeat'],
                        help="Model architecture to train (default: revised_deepbeat)")

    # Tuned params
    parser.add_argument("--tuned_params_path", type=str, default=None,
                        help="Path to JSON file with tuned hyperparameters. Overrides individual HP args.")

    # Hyperparameters
    parser.add_argument("--batch_size",           type=int,   default=128)
    parser.add_argument("--epochs",               type=int,   default=100)
    parser.add_argument("--learning_rate",        type=float, default=0.001)
    parser.add_argument("--weight_decay",         type=float, default=0.01)
    parser.add_argument("--qa_loss_weight",        type=float, default=0.2)
    parser.add_argument("--rhythm_loss_weight",    type=float, default=5.0)
    parser.add_argument("--rhythm_class_weights",  type=float, nargs=2, default=None,
                        metavar=('W_NORMAL', 'W_AFIB'),
                        help="Class weights for rhythm loss, e.g. --rhythm_class_weights 1.0 3.0")
    parser.add_argument("--grad_clip",             type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0, set 0 to disable)")

    # Checkpointing
    parser.add_argument("--monitor_metric", type=str, default='rhythm_f1',
                        choices=['rhythm_acc', 'qa_acc', 'loss', 'rhythm_f1'])

    # LR Scheduler
    parser.add_argument("--scheduler",        type=str, default='plateau',
                        choices=['none', 'plateau', 'cosine'],
                        help="LR scheduler type (default: plateau)")
    parser.add_argument("--lr_factor",        type=float, default=0.5,
                        help="Plateau: factor to reduce LR by (default: 0.5)")
    parser.add_argument("--lr_patience",      type=int,   default=7,
                        help="Plateau: epochs with no improvement before reducing LR (default: 7)")
    parser.add_argument("--lr_min",           type=float, default=1e-6,
                        help="Minimum LR for plateau/cosine scheduler (default: 1e-6)")

    # Early stopping
    parser.add_argument("--early_stopping",            action='store_true')
    parser.add_argument("--early_stopping_patience",   type=int,   default=15)
    parser.add_argument("--early_stopping_min_delta",  type=float, default=0.0001)
    parser.add_argument("--early_stopping_metric",     type=str,   default='rhythm_acc',
                        choices=['rhythm_acc', 'qa_acc', 'loss', 'rhythm_f1'])

    # Resume
    parser.add_argument("--resume_from",  type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--resume_epoch", type=int, default=None)

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
                        help="W&B entity (team or username)")
    parser.add_argument("--wandb_tags",    type=str, nargs='*', default=None,
                        help="Optional tags for the W&B run (space-separated)")
    parser.add_argument("--notes",         type=str, default=None,
                        help="Free-text notes saved to W&B and history")
    parser.add_argument("--wandb_run_id",  type=str, default=None,
                        help="W&B run ID to resume a previous run")
    parser.add_argument("--wandb_offline", action='store_true',
                        help="Run W&B in offline mode (sync later with `wandb sync`)")

    return parser.parse_args()



# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def setup_wandb(args, hyperparams: dict):
    mode = "offline" if args.wandb_offline else "online"
    resume_mode = "must" if args.wandb_run_id is not None else None

    tags = list(args.wandb_tags) if args.wandb_tags else []
    tags.append(args.model)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.file_name,
        id=args.wandb_run_id,
        resume=resume_mode,
        config=hyperparams,
        tags=tags,
        notes=args.notes,
        mode=mode,
    )
    print(f"W&B run initialised: {run.url}")
    return run


def log_epoch_to_wandb(train_m: dict, val_m: dict, epoch: int, lr: float = None):
    log_dict = {
        "epoch":              epoch,
        "train/loss":         train_m['loss'],
        "train/rhythm_acc":   train_m['rhythm_acc'],
        "train/qa_acc":       train_m['qa_acc'],
        "train/rhythm_f1":    train_m['rhythm_f1'],
        "train/grad_norm":    train_m['grad_norm'],
        "val/loss":           val_m['loss'],
        "val/rhythm_acc":     val_m['rhythm_acc'],
        "val/qa_acc":         val_m['qa_acc'],
        "val/rhythm_f1":      val_m['rhythm_f1'],
        "val/auroc":          val_m.get('auroc', 0.0),
        "val/auprc":          val_m.get('auprc', 0.0),
    }
    if lr is not None:
        log_dict["train/lr"] = lr
    wandb.log(log_dict, step=epoch)


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def run_evaluation(model, args, device, output_path):
    """Load best checkpoint, run inference on test set, report metrics + log to W&B."""
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("=" * 60)

    best_ckpt_path = output_path / f"{args.file_name}_best.pth"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print("Loaded best checkpoint for evaluation.")
    else:
        print("No best checkpoint found — evaluating with current model weights.")

    model.eval()

    print(f"Loading test data from: {args.test_data_path}")
    test_dict = load_pickle_file(Path(args.test_data_path))
    test_dataset = DeepBeatDataset(
        test_dict['data'],
        test_dict['qa_label'],
        test_dict['rhythm_label'],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    all_rhythm_preds, all_rhythm_targets, all_rhythm_probs = [], [], []
    all_qa_preds, all_qa_targets = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", ncols=120):
            data          = batch['data'].to(device)
            rhythm_target = batch['rhythm_label']
            qa_target     = batch['qa_label']

            qa_logits, rhythm_logits = model(data)

            probs       = F.softmax(rhythm_logits, dim=1)[:, 1].cpu().numpy()
            rhythm_pred = torch.argmax(rhythm_logits, dim=1).cpu().numpy()
            qa_pred     = torch.argmax(qa_logits,     dim=1).cpu().numpy()

            all_rhythm_preds.extend(rhythm_pred)
            all_rhythm_targets.extend(rhythm_target.numpy())
            all_rhythm_probs.extend(probs)
            all_qa_preds.extend(qa_pred)
            all_qa_targets.extend(qa_target.numpy())

    all_rhythm_targets = np.array(all_rhythm_targets)
    all_rhythm_preds   = np.array(all_rhythm_preds)
    all_rhythm_probs   = np.array(all_rhythm_probs)
    all_qa_targets     = np.array(all_qa_targets)
    all_qa_preds       = np.array(all_qa_preds)

    # --- Rhythm report ---
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

    # --- QA report ---
    print("\nQA CLASSIFICATION PERFORMANCE")
    print(classification_report(all_qa_targets, all_qa_preds))

    # --- Save predictions CSV ---
    results_csv = output_path / "test_predictions.csv"
    preds_db = pd.DataFrame({
        'rh_true':   all_rhythm_targets,
        'rh_pred':   all_rhythm_preds,
        'afib_prob': all_rhythm_probs,
        'qa_true':   all_qa_targets,
        'qa_pred':   all_qa_preds,
    })
    if 'ID' in test_dict:
        preds_db['ID'] = test_dict['ID']
    preds_db.to_csv(results_csv, index=False)
    print(f"\nPredictions saved to: {results_csv}")

    # --- DeepBeat stratified metrics (by predicted signal quality level) ---
    try:
        output_0 = deepbeat_metrics(preds_db, level=0)
        output_1 = deepbeat_metrics(preds_db, level=1)
        output_2 = deepbeat_metrics(preds_db, level=2)

        rows = []
        for level, out in [(0, output_0), (1, output_1), (2, output_2)]:
            row = {'qa_level': level}
            row.update({k: float(v) for k, v in out.items()})
            rows.append(row)
        metrics_csv = output_path / "deepbeat_metrics.csv"
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        print(f"DeepBeat stratified metrics saved to: {metrics_csv}")

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
    rhythm_f1  = f1_score(all_rhythm_targets, all_rhythm_preds, average='macro', zero_division=0)
    rhythm_acc = accuracy_score(all_rhythm_targets, all_rhythm_preds)
    qa_acc     = accuracy_score(all_qa_targets, all_qa_preds)

    test_log = {
        "Test/rhythm_f1":  rhythm_f1,
        "Test/rhythm_acc": rhythm_acc,
        "Test/qa_acc":     qa_acc,
    }
    wandb.summary["test_rhythm_f1"]  = rhythm_f1
    wandb.summary["test_rhythm_acc"] = rhythm_acc
    wandb.summary["test_qa_acc"]     = qa_acc
    if auroc is not None:
        test_log["Test/AUROC"] = auroc
        test_log["Test/AUPRC"] = auprc
        wandb.summary["test_auroc"] = auroc
        wandb.summary["test_auprc"] = auprc
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
    print(f"Using device: {device}\n")

    # ---- Checkpoint / resume state ----
    resume_checkpoint = None
    start_epoch = 1
    history = {
        'loss': [], 'val_loss': [],
        'val_rhythm_acc': [], 'val_qa_acc': [], 'val_rhythm_f1': [],
    }

    monitor_metric = args.monitor_metric
    monitor_mode   = 'min' if monitor_metric == 'loss' else 'max'
    best_metric_val = float('inf') if monitor_mode == 'min' else -float('inf')
    best_epoch = 0

    print(f"Checkpointing on: {monitor_metric} (mode: {monitor_mode})")

    if args.resume_from is not None:
        if not Path(args.resume_from).exists():
            print(f"WARNING: Checkpoint not found at {args.resume_from}. Starting fresh.")
        else:
            print("\n" + "=" * 60)
            print("RESUMING TRAINING FROM CHECKPOINT")
            print("=" * 60)
            resume_checkpoint = args.resume_from

    # ---- Hyperparameters ----
    tuned_dropouts = apply_tuned_params(args)
    hyperparams = {
        'batch_size':         args.batch_size,
        'learning_rate':      args.learning_rate,
        'weight_decay':       args.weight_decay,
        'qa_loss_weight':     args.qa_loss_weight,
        'rhythm_loss_weight': args.rhythm_loss_weight,
        'monitor_metric':     args.monitor_metric,
        'dropouts':           tuned_dropouts if tuned_dropouts is not None else 'default',
        'scheduler':               args.scheduler,
        'lr_factor':               args.lr_factor,
        'lr_patience':             args.lr_patience,
        'lr_min':                  args.lr_min,
        'grad_clip':               args.grad_clip,
        'rhythm_class_weights':    args.rhythm_class_weights,
    }
    history['hyperparameters'] = hyperparams
    print(hyperparams)

    # ---- W&B init ----
    wandb_run = setup_wandb(args, hyperparams)

    # ---- Data ----
    print("Loading training data...")
    train_dict = load_pickle_file(Path(args.train_data_path))
    data_train, label_train_r, label_train_q = (
        train_dict['data'], train_dict['rhythm_label'], train_dict['qa_label']
    )
    print(f"Train shape: {data_train.shape}")

    print("Loading validation data...")
    val_dict = load_pickle_file(Path(args.val_data_path))
    data_val, label_val_r, label_val_q = (
        val_dict['data'], val_dict['rhythm_label'], val_dict['qa_label']
    )
    print(f"Val shape: {data_val.shape}")

    # ---- W&B data artifact ----
    data_artifact = wandb.Artifact(
        name=f"{args.file_name}-data",
        type="dataset",
        description="Train and validation pickle files",
        metadata={
            'train_path': args.train_data_path,
            'val_path':   args.val_data_path,
            'train_samples': len(data_train),
            'val_samples':   len(data_val),
        },
    )
    data_artifact.add_file(args.train_data_path, name="train.pkl")
    data_artifact.add_file(args.val_data_path,   name="val.pkl")
    wandb.log_artifact(data_artifact)

    train_dataset = DeepBeatDataset(data_train, label_train_q, label_train_r)
    val_dataset   = DeepBeatDataset(data_val,   label_val_q,   label_val_r)

    # ---- Rhythm class weights ----
    if args.rhythm_class_weights is not None:
        rhythm_class_weights = torch.FloatTensor(args.rhythm_class_weights)
        print(f"Using manual rhythm class weights: Normal={args.rhythm_class_weights[0]}, AFib={args.rhythm_class_weights[1]}")
    else:
        rhythm_class_weights = None
        print("No rhythm class weights applied.")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    # ---- Model & optimizer ----
    print("Creating model...")
    if resume_checkpoint is not None:
        temp_ckpt = torch.load(resume_checkpoint, map_location=device)
        if 'hyperparameters' in temp_ckpt:
            saved_drops = temp_ckpt['hyperparameters'].get('dropouts')
            if saved_drops != 'default':
                tuned_dropouts = saved_drops

    print(f"Model architecture: {args.model}")
    if args.model == 'deepbeat':
        model = DeepBeatModel(dropouts=tuned_dropouts).to(device)
    else:
        model = revised_DeepBeatModel(dropouts=tuned_dropouts).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # ---- LR Scheduler ----
    scheduler = None
    if args.scheduler == 'plateau':
        sched_mode = 'min' if args.monitor_metric == 'loss' else 'max'
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_mode,
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )
        print(f"LR scheduler: ReduceLROnPlateau (mode={sched_mode}, factor={args.lr_factor}, patience={args.lr_patience})")
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min,
        )
        print(f"LR scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min={args.lr_min})")

    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(resume_checkpoint, model, optimizer, device, scheduler=scheduler)
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)

        hist_key = f'val_{monitor_metric}'
        if hist_key in history and len(history[hist_key]) > 0:
            if monitor_mode == 'min':
                best_metric_val = min(history[hist_key])
            else:
                best_metric_val = max(history[hist_key])
            best_epoch = history[hist_key].index(best_metric_val) + 1

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best val {monitor_metric} so far: {best_metric_val:.4f} at epoch {best_epoch}")
    else:
        if tuned_dropouts is not None:
            print("Model initialised with TUNED dropout values")
        else:
            print("Model initialised with DEFAULT dropout values")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    wandb.watch(model, log='gradients', log_freq=50)

    # ---- Early stopping ----
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
        print("EARLY STOPPING ENABLED")
        print(f"  Metric:   {args.early_stopping_metric}")
        print(f"  Patience: {args.early_stopping_patience}")
        print(f"  Mode:     {mode}")
        print(f"{'='*60}\n")

    # ---- Output directory ----
    output_path = Path(args.output_path) / args.file_name
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    print("Starting training...")
    epoch = start_epoch
    val_m = {}   # guard against NameError if training fails before first val epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_m = run_epoch(
                model, train_loader, optimizer, device, epoch,
                args.qa_loss_weight, args.rhythm_loss_weight,
                is_training=True, progress_bar=True, grad_clip=args.grad_clip,
                rhythm_class_weights=rhythm_class_weights,
            )
            val_m = run_epoch(
                model, val_loader, optimizer, device, epoch,
                args.qa_loss_weight, args.rhythm_loss_weight,
                is_training=False, progress_bar=True,
                rhythm_class_weights=rhythm_class_weights,
            )

            # Step LR scheduler
            if scheduler is not None:
                if args.scheduler == 'plateau':
                    scheduler.step(val_m[args.monitor_metric])
                else:
                    scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            log_epoch_to_wandb(train_m, val_m, epoch, lr=current_lr)

            history['loss'].append(train_m['loss'])
            history['val_loss'].append(val_m['loss'])
            history['val_rhythm_acc'].append(val_m['rhythm_acc'])
            history['val_qa_acc'].append(val_m['qa_acc'])
            history['val_rhythm_f1'].append(val_m['rhythm_f1'])

            print(
                f"   -> Train Loss: {train_m['loss']:.4f} | "
                f"Rh Acc: {train_m['rhythm_acc']:.4f} | "
                f"Rh F1: {train_m['rhythm_f1']:.4f} | "
                f"QA Acc: {train_m['qa_acc']:.4f}"
            )
            print(
                f"   -> Val   Loss: {val_m['loss']:.4f} | "
                f"Rh Acc: {val_m['rhythm_acc']:.4f} | "
                f"Rh F1: {val_m['rhythm_f1']:.4f} | "
                f"QA Acc: {val_m['qa_acc']:.4f} | "
                f"AUROC: {val_m.get('auroc', 0):.4f}"
            )

            # Early stopping check
            if early_stopper is not None:
                if early_stopper(val_m[args.early_stopping_metric], epoch):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    wandb.run.summary['early_stopped_epoch'] = epoch
                    break

            # Best model checkpoint
            current_val = val_m[monitor_metric]
            is_best = (
                (monitor_mode == 'min' and current_val < best_metric_val) or
                (monitor_mode == 'max' and current_val > best_metric_val)
            )
            if is_best:
                best_metric_val = current_val
                best_epoch = epoch
                save_checkpoint(
                    epoch, model, optimizer, val_m, history, args,
                    output_path, tuned_dropouts, early_stopper,
                    checkpoint_type='best', scheduler=scheduler,
                )
                print(f"   * New best saved ({monitor_metric}: {best_metric_val:.4f})")
                wandb.run.summary[f'best_val_{monitor_metric}'] = best_metric_val
                wandb.run.summary['best_epoch'] = best_epoch

            # Progress checkpoint every 10 epochs
            if epoch % 10 == 0:
                save_checkpoint(
                    epoch, model, optimizer, val_m, history, args,
                    output_path, tuned_dropouts, early_stopper,
                    checkpoint_type='progress', scheduler=scheduler,
                )
                print(f"   Progress checkpoint saved (epoch {epoch})")

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("TRAINING INTERRUPTED (Ctrl+C)")
        print("=" * 60)
        save_checkpoint(
            epoch, model, optimizer, val_m, history, args,
            output_path, tuned_dropouts, early_stopper,
            checkpoint_type='interrupted', scheduler=scheduler,
        )
        print(f"Interrupted checkpoint saved at epoch {epoch}")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TRAINING FAILED: {e}")
        print("=" * 60)
        try:
            save_checkpoint(
                epoch, model, optimizer, val_m, history, args,
                output_path, tuned_dropouts, early_stopper,
                checkpoint_type='error', scheduler=scheduler,
            )
        except Exception:
            print("Could not save error checkpoint")
        raise

    # ---- Final checkpoint & history ----
    save_checkpoint(
        epoch, model, optimizer, val_m, history, args,
        output_path, tuned_dropouts, early_stopper,
        checkpoint_type='final', scheduler=scheduler,
    )
    with open(output_path / f"{args.file_name}_history.pkl", 'wb') as f:
        pickle.dump(history, f)

    # ---- Test evaluation ----
    if args.test_data_path is not None:
        run_evaluation(model, args, device, output_path)

    # ---- W&B model artifact ----
    best_ckpt_path = output_path / f"{args.file_name}_best.pth"
    if best_ckpt_path.exists():
        model_artifact = wandb.Artifact(
            name=f"{args.file_name}-model",
            type="model",
            description=f"Best {args.model} checkpoint",
            metadata={
                'best_epoch':            best_epoch,
                f'best_val_{monitor_metric}': best_metric_val,
            },
        )
        model_artifact.add_file(str(best_ckpt_path))
        wandb.log_artifact(model_artifact)

    wandb.finish()

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best epoch:          {best_epoch}")
    print(f"Best val {monitor_metric}: {best_metric_val:.4f}")
    if early_stopper is not None and early_stopper.early_stop:
        print(f"Early stopped at epoch {epoch} (patience {args.early_stopping_patience})")
    else:
        print(f"Total epochs: {epoch}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
