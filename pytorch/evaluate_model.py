import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Import local modules
from deepbeat_model import DeepBeatModel
from utils import (DeepBeatDataset, load_pickle_file, load_checkpoint, get_optimal_workers)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepBeat Model Performance")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data pickle file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved .pth checkpoint")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading test data from: {args.test_data_path}")
    test_data_dict = load_pickle_file(args.test_data_path)
    
    # Prepare Dataset
    test_dataset = DeepBeatDataset(
        test_data_dict['data'], 
        test_data_dict['qa_label'], 
        test_data_dict['rhythm']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=get_optimal_workers(4)
    )

    # 1. Peek into checkpoint to get hyperparameters (specifically dropouts)
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    tuned_dropouts = None
    if 'hyperparameters' in checkpoint:
        saved_dropouts = checkpoint['hyperparameters'].get('dropouts')
        if saved_dropouts != 'default':
            tuned_dropouts = saved_dropouts
            print("Detected custom dropout values in checkpoint.")

    # 2. Initialize Model with correct dropout
    model = DeepBeatModel(dropouts=tuned_dropouts).to(device)
    
    # 3. Load weights
    # Note: load_checkpoint in utils.py expects model and optimizer. 
    # Since we are just evaluating, we can pass None for optimizer.
    load_checkpoint(args.checkpoint_path, model, optimizer=None, device=device)
    model.eval()

    # 4. Run Evaluation
    print("\nRunning inference...")
    all_qa_preds, all_qa_true = [], []
    all_rh_preds, all_rh_true = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            data = batch['data'].to(device)
            qa_logits, rh_logits = model(data)
            
            # Get predictions
            qa_pred = torch.argmax(qa_logits, dim=1).cpu().numpy()
            rh_pred = torch.argmax(rh_logits, dim=1).cpu().numpy()
            
            # Get ground truth
            qa_true = torch.argmax(batch['qa_label'], dim=1).numpy()
            rh_true = torch.argmax(batch['rhythm_label'], dim=1).numpy()
            
            all_qa_preds.extend(qa_pred)
            all_qa_true.extend(qa_true)
            all_rh_preds.extend(rh_pred)
            all_rh_true.extend(rh_true)

    # 5. Generate and Print Reports
    print("\n" + "="*30)
    print("RHYTHM BRANCH PERFORMANCE (AFib vs Normal)")
    print("="*30)
    print(classification_report(all_rh_true, all_rh_preds, target_names=['Normal', 'AFib']))
    
    print("\n" + "="*30)
    print("QUALITY BRANCH PERFORMANCE (QA)")
    print("="*30)
    # Adjust target_names based on your specific labels (e.g., Good, Bad, Average)
    print(classification_report(all_qa_true, all_qa_preds))

    # 6. Save results to CSV for further analysis
    results_df = pd.DataFrame({
        'rh_true': all_rh_true,
        'rh_pred': all_rh_preds,
        'qa_true': all_qa_true,
        'qa_pred': all_qa_preds
    })
    
    results_csv = output_path / "test_predictions.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nDetailed predictions saved to: {results_csv}")

if __name__ == "__main__":
    main()