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
    parser = argparse.ArgumentParser(description="Evaluate a saved DeepBeat checkpoint on a test set")

    # Required
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to the .pth checkpoint file to evaluate")
    parser.add_argument("--test_data_path",  required=True,
                        help="Path to the test pickle file")
    parser.add_argument("--val_data_path",   required=True,
                        help="Path to the validation pickle file (used to find optimal threshold)")

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
    rhythm_preds, rhythm_targets, rhythm_probs = [], [], []
    qa_preds, qa_targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, ncols=120):
            data          = batch['data'].to(device)
            rhythm_target = batch['rhythm_label']
            qa_target     = batch['qa_label']

            qa_logits, rhythm_logits = model(data)

            probs       = F.softmax(rhythm_logits, dim=1)[:, 1].cpu().numpy()
            rhythm_pred = torch.argmax(rhythm_logits, dim=1).cpu().numpy()
            qa_pred     = torch.argmax(qa_logits,     dim=1).cpu().numpy()

            rhythm_preds.extend(rhythm_pred)
            rhythm_targets.extend(rhythm_target.numpy())
            rhythm_probs.extend(probs)
            qa_preds.extend(qa_pred)
            qa_targets.extend(qa_target.numpy())

    return (
        np.array(rhythm_targets),
        np.array(rhythm_preds),
        np.array(rhythm_probs),
        np.array(qa_targets),
        np.array(qa_preds),
    )


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

    # Infer model architecture
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

    # Infer dropouts
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
        tags += [model_name, "evaluation"]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"eval_{file_name}",
            config={
                'checkpoint': str(ckpt_path),
                'model': model_name,
                'test_data': args.test_data_path,
                'val_data': args.val_data_path,
                'epoch': checkpoint.get('epoch'),
            },
            tags=tags,
            mode=mode,
        )

    # ---- Test inference ----
    print(f"\nLoading test data: {args.test_data_path}")
    test_dict = load_pickle_file(Path(args.test_data_path))
    test_loader = make_loader(test_dict, args.batch_size, args.num_workers, device)

    (all_rhythm_targets, all_rhythm_preds, all_rhythm_probs,
     all_qa_targets, all_qa_preds) = run_inference(model, test_loader, device, desc="Evaluating test set")

    # ---- Find optimal threshold from validation set ----
    print(f"\nFinding optimal threshold from validation set: {args.val_data_path}")
    val_dict = load_pickle_file(Path(args.val_data_path))
    val_loader = make_loader(val_dict, args.batch_size, args.num_workers, device)

    val_probs, val_targets = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val inference (threshold)", ncols=120):
            rh_logits = model(batch['data'].to(device))[1]
            val_probs.extend(F.softmax(rh_logits, dim=1)[:, 1].cpu().numpy())
            val_targets.extend(batch['rhythm_label'].numpy())
    val_probs   = np.array(val_probs)
    val_targets = np.array(val_targets)

    precision_v, recall_v, thresholds_v = precision_recall_curve(val_targets, val_probs)
    f1_v = 2 * precision_v * recall_v / (precision_v + recall_v + 1e-8)
    opt_threshold = float(thresholds_v[f1_v[:-1].argmax()])
    print(f"Optimal threshold (val F1-maximising): {opt_threshold:.4f}")

    all_rhythm_preds_opt = (all_rhythm_probs >= opt_threshold).astype(int)

    # ---- Reports ----
    print("\n" + "=" * 60)
    print("POST-TRAINING EVALUATION ON TEST SET")
    print("=" * 60)

    print("\nRHYTHM CLASSIFICATION PERFORMANCE @ threshold=0.5 (AFib vs Normal)")
    print(classification_report(all_rhythm_targets, all_rhythm_preds, target_names=['Normal', 'AFib']))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_rhythm_targets, all_rhythm_preds))

    print(f"\nRHYTHM CLASSIFICATION PERFORMANCE @ threshold={opt_threshold:.4f} (optimal from val)")
    print(classification_report(all_rhythm_targets, all_rhythm_preds_opt, target_names=['Normal', 'AFib']))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_rhythm_targets, all_rhythm_preds_opt))

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
    results_csv = output_path / f"{file_name}_test_predictions.csv"
    preds_db = pd.DataFrame({
        'rh_true':     all_rhythm_targets,
        'rh_pred':     all_rhythm_preds,
        'rh_pred_opt': all_rhythm_preds_opt,
        'afib_prob':   all_rhythm_probs,
        'qa_true':     all_qa_targets,
        'qa_pred':     all_qa_preds,
    })
    if 'ID' in test_dict:
        preds_db['ID'] = test_dict['ID']
    preds_db.to_csv(results_csv, index=False)
    print(f"\nPredictions saved to: {results_csv}")

    # ---- Optimized-threshold predictions CSV ----
    # rh_pred is replaced with the opt-threshold predictions so deepbeat_metrics
    # (which reads 'rh_pred') operates on the optimized predictions.
    preds_db_opt = preds_db.copy()
    preds_db_opt['rh_pred'] = preds_db_opt['rh_pred_opt']

    opt_csv = output_path / f"{file_name}_test_predictions_optimized.csv"
    preds_db_opt.to_csv(opt_csv, index=False)
    print(f"Optimized predictions saved to: {opt_csv}")

    # ---- DeepBeat stratified metrics @ default threshold (0.5) ----
    try:
        output_0 = deepbeat_metrics(preds_db, level=0)
        output_1 = deepbeat_metrics(preds_db, level=1)
        output_2 = deepbeat_metrics(preds_db, level=2)

        rows = []
        for level, out in [(0, output_0), (1, output_1), (2, output_2)]:
            row = {'qa_level': level}
            row.update({k: float(v) for k, v in out.items()})
            rows.append(row)
        metrics_csv = output_path / f"{file_name}_deepbeat_metrics.csv"
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        print(f"DeepBeat stratified metrics (thr=0.5) saved to: {metrics_csv}")

        if use_wandb:
            db_log = {}
            for level, out in [(0, output_0), (1, output_1), (2, output_2)]:
                for metric, val in out.items():
                    db_log[f"DeepBeat/QA{level}/{metric}"] = float(val)
                    wandb.summary[f"deepbeat_qa{level}_{metric.lower()}"] = float(val)
            wandb.log(db_log)

            db_artifact = wandb.Artifact(name=f"{file_name}-deepbeat-metrics", type="evaluation")
            db_artifact.add_file(str(metrics_csv))
            wandb.log_artifact(db_artifact)
            print("   W&B DeepBeat metrics artifact uploaded.")
    except Exception as e:
        print(f"WARNING: DeepBeat stratified metrics (thr=0.5) failed: {e}")

    # ---- DeepBeat stratified metrics @ optimal threshold ----
    try:
        print(f"\nDeepBeat metrics using optimized threshold ({opt_threshold:.4f}):")
        opt_output_0 = deepbeat_metrics(preds_db_opt, level=0)
        opt_output_1 = deepbeat_metrics(preds_db_opt, level=1)
        opt_output_2 = deepbeat_metrics(preds_db_opt, level=2)

        opt_rows = []
        for level, out in [(0, opt_output_0), (1, opt_output_1), (2, opt_output_2)]:
            row = {'qa_level': level}
            row.update({k: float(v) for k, v in out.items()})
            opt_rows.append(row)
        opt_metrics_csv = output_path / f"{file_name}_deepbeat_metrics_optimized.csv"
        pd.DataFrame(opt_rows).to_csv(opt_metrics_csv, index=False)
        print(f"DeepBeat stratified metrics (thr={opt_threshold:.4f}) saved to: {opt_metrics_csv}")

        if use_wandb:
            db_opt_log = {}
            for level, out in [(0, opt_output_0), (1, opt_output_1), (2, opt_output_2)]:
                for metric, val in out.items():
                    db_opt_log[f"DeepBeat_opt/QA{level}/{metric}"] = float(val)
                    wandb.summary[f"deepbeat_opt_qa{level}_{metric.lower()}"] = float(val)
            wandb.log(db_opt_log)

            db_opt_artifact = wandb.Artifact(name=f"{file_name}-deepbeat-metrics-opt", type="evaluation")
            db_opt_artifact.add_file(str(opt_metrics_csv))
            wandb.log_artifact(db_opt_artifact)
            print("   W&B DeepBeat opt metrics artifact uploaded.")
    except Exception as e:
        print(f"WARNING: DeepBeat stratified metrics (opt threshold) failed: {e}")

    # ---- W&B summary ----
    rhythm_f1      = f1_score(all_rhythm_targets, all_rhythm_preds,     average='binary', zero_division=0)
    rhythm_f1_opt  = f1_score(all_rhythm_targets, all_rhythm_preds_opt, average='binary', zero_division=0)
    rhythm_acc     = accuracy_score(all_rhythm_targets, all_rhythm_preds)
    rhythm_acc_opt = accuracy_score(all_rhythm_targets, all_rhythm_preds_opt)
    qa_acc         = accuracy_score(all_qa_targets, all_qa_preds)

    if use_wandb:
        test_log = {
            "Test/rhythm_f1":      rhythm_f1,
            "Test/rhythm_f1_opt":  rhythm_f1_opt,
            "Test/rhythm_acc":     rhythm_acc,
            "Test/rhythm_acc_opt": rhythm_acc_opt,
            "Test/opt_threshold":  opt_threshold,
            "Test/qa_acc":         qa_acc,
        }
        wandb.summary["test_rhythm_f1"]      = rhythm_f1
        wandb.summary["test_rhythm_f1_opt"]  = rhythm_f1_opt
        wandb.summary["test_rhythm_acc"]     = rhythm_acc
        wandb.summary["test_rhythm_acc_opt"] = rhythm_acc_opt
        wandb.summary["test_opt_threshold"]  = opt_threshold
        wandb.summary["test_qa_acc"]         = qa_acc
        if auroc is not None:
            test_log["Test/AUROC"] = auroc
            test_log["Test/AUPRC"] = auprc
            wandb.summary["test_auroc"] = auroc
            wandb.summary["test_auprc"] = auprc
        wandb.log(test_log)

        csv_artifact = wandb.Artifact(
            name=f"{file_name}-test-predictions",
            type="evaluation",
            metadata={"test_data_path": args.test_data_path},
        )
        csv_artifact.add_file(str(results_csv))
        csv_artifact.add_file(str(opt_csv))
        wandb.log_artifact(csv_artifact)
        print("   W&B test-predictions artifact uploaded.")
        wandb.finish()

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Rhythm F1  @ 0.5:          {rhythm_f1:.4f}")
    print(f"Rhythm F1  @ opt ({opt_threshold:.4f}):  {rhythm_f1_opt:.4f}")
    print(f"Rhythm Acc @ 0.5:          {rhythm_acc:.4f}")
    print(f"Rhythm Acc @ opt:          {rhythm_acc_opt:.4f}")
    print(f"QA Acc:                    {qa_acc:.4f}")
    if auroc is not None:
        print(f"AUROC:                     {auroc:.4f}")
        print(f"AUPRC:                     {auprc:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
