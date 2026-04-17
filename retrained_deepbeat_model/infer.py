import warnings
warnings.filterwarnings('ignore')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pytorch'))

from revised_deepbeat_model import revised_DeepBeatModel
from utils import load_pickle_file, get_optimal_workers


class InferenceDataset(Dataset):
    """Minimal dataset for inference — labels are optional."""

    def __init__(self, data, qa_labels=None, rhythm_labels=None):
        self.data = torch.FloatTensor(data).permute(0, 2, 1)  # (N,800,1) -> (N,1,800)
        n = len(self.data)
        self.qa_labels     = torch.LongTensor(qa_labels)     if qa_labels     is not None else torch.full((n,), -1, dtype=torch.long)
        self.rhythm_labels = torch.LongTensor(rhythm_labels) if rhythm_labels is not None else torch.full((n,), -1, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data':          self.data[idx],
            'qa_label':      self.qa_labels[idx],
            'rhythm_label':  self.rhythm_labels[idx],
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a DeepBeat checkpoint and per-QA-level thresholds"
    )
    parser.add_argument("--checkpoint_path",
                        default=str(Path(__file__).resolve().parent / "revised_deepbeat_diff_branch_2_best.pth"),
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--thresholds_path",  
                        default=str(Path(__file__).resolve().parent / "model_thresholds.json"),
                        help="Path to the per_qa_thresholds.json produced by evaluate_per_qa_level.py")
    parser.add_argument("--data_path",        required=True,
                        help="Path to the input pickle file")
    parser.add_argument("--model",            type=str, default=None,
                        choices=['revised_deepbeat', 'deepbeat'],
                        help="Model architecture (inferred from checkpoint if omitted)")
    parser.add_argument("--output_path",      type=str, default=None,
                        help="Directory for output CSV (defaults to checkpoint directory)")
    parser.add_argument("--file_name",        type=str, default=None,
                        help="Base name for output files (defaults to checkpoint stem)")
    parser.add_argument("--batch_size",       type=int, default=256)
    parser.add_argument("--device",           type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers",      type=int, default=None)
    return parser.parse_args()


def build_model(model_name, dropouts, device):
    return revised_DeepBeatModel(dropouts=dropouts).to(device)


def run_inference(model, loader, device):
    rhythm_probs, qa_preds = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring", ncols=120):
            data = batch['data'].to(device)
            qa_logits, rhythm_logits = model(data)

            rhythm_probs.extend(F.softmax(rhythm_logits, dim=1)[:, 1].cpu().numpy())
            qa_preds.extend(torch.argmax(qa_logits, dim=1).cpu().numpy())

    return np.array(rhythm_probs), np.array(qa_preds)


def apply_per_qa_thresholds(probs, qa_preds, thresholds):
    preds = np.zeros(len(probs), dtype=int)
    for level, thr in thresholds.items():
        mask = qa_preds == int(level)
        preds[mask] = (probs[mask] >= thr).astype(int)
    return preds


def main():
    args = parse_args()
    device = torch.device(args.device)
    args.num_workers = get_optimal_workers(args.num_workers)

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_path = Path(args.output_path) if args.output_path else ckpt_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    file_name = args.file_name or ckpt_path.stem

    # ---- Load thresholds ----
    with open(args.thresholds_path) as f:
        thresholds = json.load(f)  # keys are strings: "0", "1", "2"
    print("Per-QA thresholds loaded:")
    for k, v in thresholds.items():
        print(f"  QA={k}: {v:.4f}")

    # ---- Load checkpoint ----
    print(f"\nLoading checkpoint: {ckpt_path}")
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

    dropouts = None
    saved_hp = checkpoint.get('hyperparameters', {})
    if isinstance(saved_hp, dict):
        saved_drops = saved_hp.get('dropouts')
        if saved_drops and saved_drops != 'default':
            dropouts = saved_drops

    model = build_model(model_name, dropouts, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Load data ----
    print(f"\nLoading data: {args.data_path}")
    data_dict = load_pickle_file(Path(args.data_path))

    qa_labels     = data_dict.get('qa_label')
    rhythm_labels = data_dict.get('rhythm_label')
    has_labels    = qa_labels is not None and rhythm_labels is not None

    dataset = InferenceDataset(data_dict['data'], qa_labels, rhythm_labels)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    # ---- Inference ----
    rhythm_probs, qa_preds = run_inference(model, loader, device)
    afib_pred = apply_per_qa_thresholds(rhythm_probs, qa_preds, thresholds)

    # ---- Build output DataFrame ----
    out = pd.DataFrame({
        'afib_prob': rhythm_probs,
        'afib_pred': afib_pred,
        'qa_pred':   qa_preds,
    })
    if 'ID' in data_dict:
        out.insert(0, 'ID', data_dict['ID'])
    if has_labels:
        out['rh_true'] = rhythm_labels
        out['qa_true'] = qa_labels

    out_csv = output_path / f"{file_name}_predictions.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nPredictions saved to: {out_csv}")

    # ---- Optional metrics (only when ground-truth labels are present) ----
    if has_labels:
        from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

        thr_str = "  |  ".join([f"QA{k}={v:.4f}" for k, v in thresholds.items()])
        print(f"\n{'='*60}")
        print(f"RHYTHM CLASSIFICATION @ per-QA thresholds ({thr_str})")
        print(classification_report(rhythm_labels, afib_pred, target_names=['Normal', 'AFib']))
        try:
            auroc = roc_auc_score(rhythm_labels, rhythm_probs)
            auprc = average_precision_score(rhythm_labels, rhythm_probs)
            print(f"AUROC: {auroc:.4f}  |  AUPRC: {auprc:.4f}")
        except ValueError as e:
            print(f"Could not compute AUROC/AUPRC: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
