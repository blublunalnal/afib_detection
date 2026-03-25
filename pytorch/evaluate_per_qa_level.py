import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve,
)
import wandb
from tqdm import tqdm

from revised_deepbeat_model import revised_DeepBeatModel
from deepbeat_model import DeepBeatModel
from benchmark_deepbeat import deepbeat_metrics
from utils import DeepBeatDataset, load_pickle_file, get_optimal_workers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a DeepBeat checkpoint with per-QA-level optimal thresholds"
    )

    # Required
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to the .pth checkpoint file to evaluate")
    parser.add_argument("--test_data_path",  required=True,
                        help="Path to the test pickle file")
    parser.add_argument("--val_data_path",   required=True,
                        help="Path to the validation pickle file (used to find per-QA thresholds)")

    # Model — inferred from checkpoint if not specified
    parser.add_argument("--model", type=str, default=None,
                        choices=['revised_deepbeat', 'deepbeat'],
                        help="Model architecture. If omitted, inferred from checkpoint metadata.")

    # Output
    parser.add_argument("--output_path", type=str, default=None,
                        help="Directory to save predictions CSV and metrics. "
                             "Defaults to the checkpoint's parent directory.")
    parser.add_argument("--file_name", type=str, default=None,
                        help="Base name for output files. Defaults to the checkpoint stem.")

    # Inference
    parser.add_argument("--batch_size",   type=int, default=128)
    parser.add_argument("--device",       type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers",  type=int, default=None)

    # W&B (optional)
    parser.add_argument("--no_wandb",       action='store_true',
                        help="Disable W&B logging entirely")
    parser.add_argument("--wandb_project",  type=str, default="afib-detection")
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--wandb_tags",     type=str, nargs='*', default=None)
    parser.add_argument("--wandb_offline",  action='store_true')

    return parser.parse_args()


def build_model(model_name: str, dropouts, device):
    if model_name == 'deepbeat':
        return DeepBeatModel(dropouts=dropouts).to(device)
    return revised_DeepBeatModel(dropouts=dropouts).to(device)


def make_loader(data_dict, batch_size, num_workers, device):
    dataset = DeepBeatDataset(
        data_dict['data'],
        data_dict['qa_label'],
        data_dict['rhythm_label'],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
    )


def run_inference(model, loader, device, desc="Inferring"):
    rhythm_targets, rhythm_preds, rhythm_probs = [], [], []
    qa_targets, qa_preds = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, ncols=120):
            data          = batch['data'].to(device)
            rhythm_target = batch['rhythm_label']
            qa_target     = batch['qa_label']

            qa_logits, rhythm_logits = model(data)

            probs       = F.softmax(rhythm_logits, dim=1)[:, 1].cpu().numpy()
            rhythm_pred = torch.argmax(rhythm_logits, dim=1).cpu().numpy()
            qa_pred     = torch.argmax(qa_logits,     dim=1).cpu().numpy()

            rhythm_targets.extend(rhythm_target.numpy())
            rhythm_preds.extend(rhythm_pred)
            rhythm_probs.extend(probs)
            qa_targets.extend(qa_target.numpy())
            qa_preds.extend(qa_pred)

    return (
        np.array(rhythm_targets),
        np.array(rhythm_preds),
        np.array(rhythm_probs),
        np.array(qa_targets),
        np.array(qa_preds),
    )


def find_optimal_threshold_for_qa_level(probs, targets, qa_preds, level):
    """
    Find the F1-maximising threshold using only validation samples
    where the model predicted qa_pred == level.
    Falls back to 0.5 if the subset is empty or has only one class.
    """
    mask = qa_preds == level
    if mask.sum() == 0:
        print(f"  QA={level}: no validation samples predicted at this level — defaulting to 0.5")
        return 0.5

    sub_probs   = probs[mask]
    sub_targets = targets[mask]

    if len(np.unique(sub_targets)) < 2:
        print(f"  QA={level}: only one class in validation subset — defaulting to 0.5")
        return 0.5

    precision, recall, thresholds = precision_recall_curve(sub_targets, sub_probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    opt_thr = float(thresholds[f1[:-1].argmax()])
    print(f"  QA={level}: n={mask.sum():,}  optimal threshold = {opt_thr:.4f}")
    return opt_thr


def apply_per_qa_thresholds(probs, qa_preds, thresholds_per_level):
    """
    For each test sample, apply the threshold that corresponds to its
    predicted QA level.
    """
    preds = np.zeros(len(probs), dtype=int)
    for level, thr in thresholds_per_level.items():
        mask = qa_preds == level
        preds[mask] = (probs[mask] >= thr).astype(int)
    return preds


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    args = parse_args()
    device = torch.device(args.device)
    args.num_workers = get_optimal_workers(args.num_workers)
    print(f"Using device: {device}\n")

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ---- Output paths ----
    output_path = Path(args.output_path) if args.output_path else ckpt_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    file_name = args.file_name or ckpt_path.stem

    # ---- Load checkpoint ----
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_name = args.model
    if model_name is None:
        saved_args = checkpoint.get('args', {})
        if isinstance(saved_args, dict):
            model_name = saved_args.get('model', 'revised_deepbeat')
        elif hasattr(saved_args, 'model'):
            model_name = saved_args.model
        else:
            model_name = 'revised_deepbeat'
        print(f"Model architecture inferred from checkpoint: {model_name}")
    else:
        print(f"Model architecture (from CLI): {model_name}")

    dropouts = None
    saved_hp = checkpoint.get('hyperparameters', {})
    if isinstance(saved_hp, dict):
        saved_drops = saved_hp.get('dropouts')
        if saved_drops and saved_drops != 'default':
            dropouts = saved_drops

    # ---- Build model & load weights ----
    model = build_model(model_name, dropouts, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # ---- W&B init ----
    use_wandb = not args.no_wandb
    if use_wandb:
        mode = "offline" if args.wandb_offline else "online"
        tags = list(args.wandb_tags) if args.wandb_tags else []
        tags += [model_name, "evaluation", "per-qa-threshold"]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"eval_per_qa_{file_name}",
            config={
                'checkpoint': str(ckpt_path),
                'model': model_name,
                'test_data': args.test_data_path,
                'val_data': args.val_data_path,
                'epoch': checkpoint.get('epoch'),
                'threshold_strategy': 'per_qa_level',
            },
            tags=tags,
            mode=mode,
        )

    # ---- Validation inference (full — we need qa_pred to split by level) ----
    print(f"\nLoading validation data: {args.val_data_path}")
    val_dict = load_pickle_file(Path(args.val_data_path))
    val_loader = make_loader(val_dict, args.batch_size, args.num_workers, device)

    (val_rhythm_targets, _, val_rhythm_probs,
     _, val_qa_preds) = run_inference(model, val_loader, device, desc="Val inference")

    # ---- Find per-QA optimal thresholds ----
    print("\nFinding per-QA-level optimal thresholds (F1-maximising on validation set):")
    qa_levels = [0, 1, 2]
    thresholds_per_level = {}
    for level in qa_levels:
        thresholds_per_level[level] = find_optimal_threshold_for_qa_level(
            val_rhythm_probs, val_rhythm_targets, val_qa_preds, level
        )

    # ---- Test inference ----
    print(f"\nLoading test data: {args.test_data_path}")
    test_dict = load_pickle_file(Path(args.test_data_path))
    test_loader = make_loader(test_dict, args.batch_size, args.num_workers, device)

    (all_rhythm_targets, all_rhythm_preds, all_rhythm_probs,
     all_qa_targets, all_qa_preds) = run_inference(model, test_loader, device, desc="Evaluating test set")

    # Default (argmax) predictions
    # Per-QA threshold predictions
    all_rhythm_preds_per_qa = apply_per_qa_thresholds(
        all_rhythm_probs, all_qa_preds, thresholds_per_level
    )

    # ---- Reports ----
    print("\n" + "=" * 60)
    print("PER-QA-LEVEL THRESHOLD EVALUATION ON TEST SET")
    print("=" * 60)

    print("\nRHYTHM CLASSIFICATION @ threshold=0.5 (argmax baseline)")
    print(classification_report(all_rhythm_targets, all_rhythm_preds, target_names=['Normal', 'AFib']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_rhythm_targets, all_rhythm_preds))

    print("\nRHYTHM CLASSIFICATION @ per-QA-level optimal thresholds")
    thr_str = "  |  ".join([f"QA{l}={thresholds_per_level[l]:.4f}" for l in qa_levels])
    print(f"  Thresholds: {thr_str}")
    print(classification_report(all_rhythm_targets, all_rhythm_preds_per_qa, target_names=['Normal', 'AFib']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_rhythm_targets, all_rhythm_preds_per_qa))

    auroc = auprc = None
    try:
        auroc = roc_auc_score(all_rhythm_targets, all_rhythm_probs)
        auprc = average_precision_score(all_rhythm_targets, all_rhythm_probs)
        print(f"\nAUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
    except ValueError as e:
        print(f"Could not compute AUROC/AUPRC: {e}")

    print("\nQA CLASSIFICATION PERFORMANCE")
    print(classification_report(all_qa_targets, all_qa_preds))

    # ---- Save predictions CSV ----
    preds_df = pd.DataFrame({
        'rh_true':         all_rhythm_targets,
        'rh_pred':         all_rhythm_preds,
        'rh_pred_per_qa':  all_rhythm_preds_per_qa,
        'afib_prob':       all_rhythm_probs,
        'qa_true':         all_qa_targets,
        'qa_pred':         all_qa_preds,
    })
    if 'ID' in test_dict:
        preds_df['ID'] = test_dict['ID']

    results_csv = output_path / f"{file_name}_per_qa_predictions.csv"
    preds_df.to_csv(results_csv, index=False)
    print(f"\nPredictions saved to: {results_csv}")

    # Build a version of the DataFrame where 'rh_pred' holds the per-QA predictions
    # so deepbeat_metrics (which reads 'rh_pred') uses the right values.
    preds_df_per_qa = preds_df.copy()
    preds_df_per_qa['rh_pred'] = preds_df_per_qa['rh_pred_per_qa']

    # ---- DeepBeat metrics @ default threshold (0.5 / argmax) ----
    print("\n" + "-" * 60)
    print("DeepBeat metrics @ default threshold (argmax):")
    try:
        rows_default = []
        for level in qa_levels:
            out = deepbeat_metrics(preds_df, level=level)
            row = {'qa_level': level, 'threshold': 0.5}
            row.update({k: float(v) for k, v in out.items()})
            rows_default.append(row)

        default_csv = output_path / f"{file_name}_per_qa_deepbeat_metrics_default.csv"
        pd.DataFrame(rows_default).to_csv(default_csv, index=False)
        print(f"Saved to: {default_csv}")
    except Exception as e:
        print(f"WARNING: DeepBeat metrics (default threshold) failed: {e}")
        rows_default = []

    # ---- DeepBeat metrics @ per-QA optimal thresholds ----
    print("\n" + "-" * 60)
    print("DeepBeat metrics @ per-QA-level optimal thresholds:")
    try:
        rows_per_qa = []
        for level in qa_levels:
            out = deepbeat_metrics(preds_df_per_qa, level=level)
            row = {'qa_level': level, 'threshold': thresholds_per_level[level]}
            row.update({k: float(v) for k, v in out.items()})
            rows_per_qa.append(row)

        per_qa_csv = output_path / f"{file_name}_per_qa_deepbeat_metrics_optimized.csv"
        pd.DataFrame(rows_per_qa).to_csv(per_qa_csv, index=False)
        print(f"Saved to: {per_qa_csv}")
    except Exception as e:
        print(f"WARNING: DeepBeat metrics (per-QA thresholds) failed: {e}")
        rows_per_qa = []

    # ---- Summary ----
    rhythm_f1         = f1_score(all_rhythm_targets, all_rhythm_preds,        average='binary', zero_division=0)
    rhythm_f1_per_qa  = f1_score(all_rhythm_targets, all_rhythm_preds_per_qa, average='binary', zero_division=0)
    rhythm_acc        = accuracy_score(all_rhythm_targets, all_rhythm_preds)
    rhythm_acc_per_qa = accuracy_score(all_rhythm_targets, all_rhythm_preds_per_qa)
    qa_acc            = accuracy_score(all_qa_targets, all_qa_preds)

    # ---- W&B logging ----
    if use_wandb:
        test_log = {
            "Test/rhythm_f1":         rhythm_f1,
            "Test/rhythm_f1_per_qa":  rhythm_f1_per_qa,
            "Test/rhythm_acc":        rhythm_acc,
            "Test/rhythm_acc_per_qa": rhythm_acc_per_qa,
            "Test/qa_acc":            qa_acc,
        }
        for level in qa_levels:
            test_log[f"Test/threshold_qa{level}"] = thresholds_per_level[level]
        if auroc is not None:
            test_log["Test/AUROC"] = auroc
            test_log["Test/AUPRC"] = auprc

        wandb.summary.update({k.replace("Test/", "test_"): v for k, v in test_log.items()})
        wandb.log(test_log)

        # Log DeepBeat metrics
        db_log = {}
        for rows, tag in [(rows_default, "DeepBeat_default"), (rows_per_qa, "DeepBeat_per_qa")]:
            for row in rows:
                lvl = int(row['qa_level'])
                for metric, val in row.items():
                    if metric not in ('qa_level', 'threshold'):
                        db_log[f"{tag}/QA{lvl}/{metric}"] = float(val)
        wandb.log(db_log)

        # Artifacts
        artifact = wandb.Artifact(name=f"{file_name}-per-qa-eval", type="evaluation")
        artifact.add_file(str(results_csv))
        if rows_default:
            artifact.add_file(str(default_csv))
        if rows_per_qa:
            artifact.add_file(str(per_qa_csv))
        wandb.log_artifact(artifact)
        print("\n   W&B artifacts uploaded.")
        wandb.finish()

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Rhythm F1  @ 0.5 (argmax):    {rhythm_f1:.4f}")
    print(f"Rhythm F1  @ per-QA optimal:  {rhythm_f1_per_qa:.4f}")
    print(f"Rhythm Acc @ 0.5 (argmax):    {rhythm_acc:.4f}")
    print(f"Rhythm Acc @ per-QA optimal:  {rhythm_acc_per_qa:.4f}")
    print(f"QA Acc:                        {qa_acc:.4f}")
    if auroc is not None:
        print(f"AUROC:                         {auroc:.4f}")
        print(f"AUPRC:                         {auprc:.4f}")
    print(f"\nPer-QA thresholds used:")
    for level in qa_levels:
        print(f"  QA={level}: {thresholds_per_level[level]:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
