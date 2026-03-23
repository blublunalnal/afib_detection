import sys
from pathlib import Path

# Must be set before local imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'pytorch'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

from fine_tuning_models import DeepBeatDataset, FineTuning_rhythm, FineTuning_multitask
from utils import load_pickle_file, load_checkpoint, get_optimal_workers
from benchmark_deepbeat import deepbeat_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FineTuning_rhythm Model Performance")
    parser.add_argument("--test_data_path",  type=str, required=True,
                        help="Path to test data pickle file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the saved .pth checkpoint")
    parser.add_argument("--batch_size",      type=int, default=128)
    parser.add_argument("--device",          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--output_dir",      type=str, default="evaluation_results",
                        help="Directory to save results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoint to read saved hyperparameters ---
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    # Hyperparameters are stored under history['hyperparameters']
    saved_hp = checkpoint.get('history', {}).get('hyperparameters', {})
    dropout        = saved_hp.get('dropout', 0.3)
    freeze         = saved_hp.get('freeze_backbone', False)
    backbone       = saved_hp.get('backbone')
    model_type     = saved_hp.get('model_type', 'rhythm')
    preprocessed   = checkpoint.get('history', {}).get('preprocessed', False)
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
    all_preds, all_targets, all_probs = [], [], []
    all_qa_preds, all_qa_targets = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            data          = batch['data'].to(device)
            rhythm_target = batch['rhythm_label']
            qa_target     = batch['qa_label']

            if model_type == 'multitask':
                rhythm_logits, qa_logits = model(data)
                qa_preds = torch.argmax(qa_logits, dim=1).cpu().numpy()
                all_qa_preds.extend(qa_preds)
                all_qa_targets.extend(qa_target.numpy())
            else:
                rhythm_logits = model(data)

            probs = F.softmax(rhythm_logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(rhythm_logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(rhythm_target.numpy())
            all_probs.extend(probs)

    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)
    all_probs   = np.array(all_probs)

    # --- Reports ---
    print("\n" + "=" * 50)
    print("RHYTHM CLASSIFICATION PERFORMANCE (AFib vs Normal)")
    print("=" * 50)
    print(classification_report(all_targets, all_preds, target_names=['Normal', 'AFib']))

    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_targets, all_preds))

    try:
        auroc = roc_auc_score(all_targets, all_probs)
        auprc = average_precision_score(all_targets, all_probs)
        print(f"\nAUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
    except ValueError as e:
        print(f"Could not compute AUROC/AUPRC: {e}")

    if model_type == 'multitask' and all_qa_targets:
        all_qa_targets = np.array(all_qa_targets)
        all_qa_preds   = np.array(all_qa_preds)
        print("\nQA CLASSIFICATION PERFORMANCE")
        print(classification_report(all_qa_targets, all_qa_preds))

    # --- Save predictions ---
    csv_data = {
        'rh_true': all_targets,
        'rh_pred': all_preds,
        'afib_prob': all_probs,
    }
    if model_type == 'multitask' and len(all_qa_preds):
        csv_data['qa_true'] = np.array(all_qa_targets)
        csv_data['qa_pred'] = np.array(all_qa_preds)

    results_csv = output_path / "test_predictions.csv"
    pd.DataFrame(csv_data).to_csv(results_csv, index=False)
    print(f"\nDetailed predictions saved to: {results_csv}")

    # --- DeepBeat stratified metrics (by QA signal quality level) ---
    try:
        preds_db = pd.read_csv(results_csv)
        preds_db['ID'] = test_dict['ID']
        if 'qa_pred' not in preds_db.columns:
            preds_db['qa_pred'] = test_dict['qa_label']

        output_0 = deepbeat_metrics(preds_db, level=0)
        output_1 = deepbeat_metrics(preds_db, level=1)
        output_2 = deepbeat_metrics(preds_db, level=2)

        rows = []
        for level, out in [(0, output_0), (1, output_1), (2, output_2)]:
            row = {'qa_level': level}
            row.update({k: float(v) for k, v in out.items()})
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        print("\n" + "=" * 50)
        print("DEEPBEAT STRATIFIED METRICS (by QA level)")
        print("=" * 50)
        print(metrics_df.to_string(index=False))

        metrics_csv = output_path / "deepbeat_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"\nDeepBeat metrics saved to: {metrics_csv}")
    except Exception as e:
        print(f"WARNING: DeepBeat stratified metrics failed: {e}")


if __name__ == "__main__":
    main()
