import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'pytorch'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from fine_tuning_models import DeepBeatDataset, FineTuning_rhythm, FineTuning_multitask
from utils import load_pickle_file, load_checkpoint, get_optimal_workers


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AUROC for a fine-tuned model")
    parser.add_argument("--test_data_path",  type=str, required=True,
                        help="Path to test data pickle file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the saved .pth checkpoint")
    parser.add_argument("--output_file",     type=str, required=True,
                        help="Output file path for the ROC curve plot (e.g. roc.png)")
    parser.add_argument("--batch_size",      type=int, default=128)
    parser.add_argument("--device",          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    saved_hp     = checkpoint.get('history', {}).get('hyperparameters', {})
    dropout      = saved_hp.get('dropout', 0.3)
    freeze       = saved_hp.get('freeze_backbone', False)
    backbone     = saved_hp.get('backbone')
    model_type   = saved_hp.get('model_type', 'rhythm')
    preprocessed = checkpoint.get('history', {}).get('preprocessed', False)
    print(f"  dropout={dropout}, freeze_backbone={freeze}, backbone={backbone}, model_type={model_type}")

    # --- Load test data ---
    print(f"\nLoading test data from: {args.test_data_path}")
    test_dict = load_pickle_file(args.test_data_path)

    is_preprocessed = test_dict.get('preprocessed', preprocessed)
    if is_preprocessed:
        print("Preprocessed data detected — skipping resampling and normalization.")

    target_hz = 50 if backbone == 'pulseppg' else 125
    test_dataset = DeepBeatDataset(
        test_dict['data'],
        test_dict['qa_label'],
        test_dict['rhythm_label'],
        preprocessed=is_preprocessed,
        target_hz=target_hz,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=get_optimal_workers(4),
    )

    # --- Build and load model ---
    if model_type == 'multitask':
        model = FineTuning_multitask(dropout=dropout, backbone=backbone, freeze=freeze).to(device)
    else:
        model = FineTuning_rhythm(dropout=dropout, backbone=backbone, freeze=freeze).to(device)
    load_checkpoint(args.checkpoint_path, model, optimizer=None, device=device)
    model.eval()

    # --- Inference ---
    print("\nRunning inference...")
    all_probs, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            data          = batch['data'].to(device)
            rhythm_target = batch['rhythm_label']

            if model_type == 'multitask':
                rhythm_logits, _ = model(data)
            else:
                rhythm_logits = model(data)

            probs = F.softmax(rhythm_logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(rhythm_target.numpy())

    all_targets = np.array(all_targets)
    all_probs   = np.array(all_probs)

    # --- Compute ROC ---
    auroc = roc_auc_score(all_targets, all_probs)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    print(f"\nAUROC: {auroc:.4f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {auroc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — AFib Detection')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    print(f"ROC curve saved to: {output_file}")


if __name__ == "__main__":
    main()
