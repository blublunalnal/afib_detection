import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepbeat_model import DeepBeatModel 
from utils import (DeepBeatDataset, EarlyStopping, 
                   setup_tensorboard, restore_early_stopping_state, 
                   apply_tuned_params, save_checkpoint, load_checkpoint, load_pickle_file, get_optimal_workers, run_epoch)


def parser_args():
    parser = argparse.ArgumentParser()
    
    # data path
    parser.add_argument("--orig_data_path", default=r'C:\Users\aoara\develop\deepbeat\data\original_data')
    parser.add_argument("--val_data_path", default= '/content/ori_val.pkl')
    parser.add_argument("--db_orig_replaced_path", default="/content/ori_train_clean_updated.pkl")
    parser.add_argument("--db_orig_replaced_vsm_path", default="/content/db_orig_replaced_vsm.pkl")
    parser.add_argument("--output_path", default=r'C:\Users\aoara\develop\deepbeat\training_output')
    parser.add_argument("--tuned_params_path", type=str, default=None, 
                        help="Path to JSON file with tuned hyperparameters. If provided, overrides other hyperparameter args.")
    
    # metric to save best model
    parser.add_argument("--monitor_metric", type= str, default= 'rhythm_f1', choices=['rhythm_acc', 'qa_acc', 'loss', 'rhythm_f1'])
   

    # experiment config
    parser.add_argument("--file_name", required=True, help="name the file (model name)")
    valid_choices = ['db_orig', 'db_orig_replaced', 'db_orig_replaced_vsm']
    # db_orig: original db training data (cleaned)
    # db_orig_replaced: db_orig substituted with relabled data 
    # db_orig_replaced_vsm: db_orig_replaced + new vsm data
    parser.add_argument("--training_choice", choices=valid_choices, required=True, help=str(valid_choices))
   
    # hyperparameters
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization)")
    parser.add_argument("--qa_loss_weight", type=float, default=0.2)
    parser.add_argument("--rhythm_loss_weight", type=float, default=5.0)
    
    # Early stopping
    parser.add_argument("--early_stopping", action='store_true', 
                        help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Number of epochs with no improvement before stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001,
                        help="Minimum change in monitored metric to qualify as improvement")
    parser.add_argument("--early_stopping_metric", type=str, default='rhythm_acc',
                        choices=['rhythm_acc', 'qa_acc', 'loss', 'rhythm_f1'],
                        help="Metric to monitor for early stopping")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Specific epoch to resume from (if checkpoint contains multiple)")
    
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers", type=int, default=None) 
    
    
    return parser.parse_args()



def load_training_data(args):
    """

    Args:
        args 

    Returns:
         np.array: unshuffled training data
    """

    print("=" * 60)
    print(f"TRAINING CHOICE: {args.training_choice}")
    print("=" * 60)
    
    path_dict = {
        'db_orig': args.orig_data_path,
        'db_orig_replaced': args.db_orig_replaced_path,
        'db_orig_replaced_vsm': args.db_orig_replaced_vsm_path
    }
    data_to_shuffle = load_pickle_file(path_dict[args.training_choice])
    
    
    return data_to_shuffle

# training part starts

def compute_accuracy(qa_logits, rhythm_logits, targets, device):
    """Compute accuracy using raw logits"""
    # QA
    qa_pred = torch.argmax(qa_logits, dim=1)
    qa_true = torch.argmax(targets['qa_label'].to(device), dim=1)
    qa_acc = (qa_pred == qa_true).float().mean()
    
    # Rhythm
    rhythm_pred = torch.argmax(rhythm_logits, dim=1)
    rhythm_true = torch.argmax(targets['rhythm_label'].to(device), dim=1)
    rhythm_acc = (rhythm_pred == rhythm_true).float().mean()
    
    return qa_acc.item(), rhythm_acc.item()


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    args = parser_args()
    device = torch.device(args.device)
    args.num_workers = get_optimal_workers(args.num_workers)
    print(f"Using device: {device}\n")
    
    # Check if resuming from checkpoint
    # Check if resuming from checkpoint
    resume_checkpoint = None
    start_epoch = 1
    # Save history with hyperparameters
    history = {'loss': [], 'val_loss': [], 'val_rhythm_acc': [], 'val_qa_acc': [], 'val_rhythm_f1': []}
    

    # Determine monitoring metric and mode
    monitor_metric = args.monitor_metric
    if not monitor_metric: 
        monitor_metric = 'rhythm_acc' # Default fallback
        
    monitor_mode = 'min' if monitor_metric == 'loss' else 'max'
    best_metric_val = float('inf') if monitor_mode == 'min' else -float('inf')
    best_epoch = 0
    
    print(f"Model checkpointing monitoring: {monitor_metric} (mode: {monitor_mode})")
    
    if args.resume_from is not None:
        if not Path(args.resume_from).exists():
            print(f"ERROR: Checkpoint file not found: {args.resume_from}")
            print("Starting training from scratch instead.\n")
        else:
            print("\n" + "="*60)
            print("RESUMING TRAINING FROM CHECKPOINT")
            print("="*60)
            # will load the full checkpoint after creating model
            resume_checkpoint = args.resume_from
    
    # Load tuned parameters if provided 
    tuned_dropouts = apply_tuned_params(args)
    history['hyperparameters'] = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'qa_loss_weight': args.qa_loss_weight,
        'rhythm_loss_weight': args.rhythm_loss_weight,
        'dropouts': tuned_dropouts if tuned_dropouts is not None else 'default'
    }
    print(history['hyperparameters'])
    
    # 1. Load Data
    print("Loading training data...")
    train_data_dict = load_training_data(args)
    data_train, label_train_r, label_train_q = train_data_dict['data'], train_data_dict['rhythm'], train_data_dict['qa_label']
    print(f"Train Shape: {data_train.shape}")

    print("Loading validation data...")
    db_val = load_pickle_file(Path(args.val_data_path))
    data_val, label_val_r, label_val_q = db_val['data'], db_val['rhythm'], db_val['qa_label']
    print(f"Val Shape: {data_val.shape}")
    
    # 2. Dataset & Loader
    train_dataset = DeepBeatDataset(data_train, label_train_q, label_train_r)
    val_dataset = DeepBeatDataset(data_val, label_val_q, label_val_r)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), 
                              persistent_workers=True if args.num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), 
                            persistent_workers=True if args.num_workers > 0 else False)
    
    # 3. Model & Optimizer
    print("Creating model...")
    
    # Load checkpoint if resuming
    if resume_checkpoint is not None:
        # Load checkpoint to get hyperparameters
        temp_checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Get dropouts from checkpoint
        if 'hyperparameters' in temp_checkpoint:
            saved_dropouts = temp_checkpoint['hyperparameters'].get('dropouts')
            if saved_dropouts != 'default':
                tuned_dropouts = saved_dropouts
        
        # Create model with saved dropouts
        if tuned_dropouts is not None:
            model = DeepBeatModel(dropouts=tuned_dropouts).to(device)
        else:
            model = DeepBeatModel().to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Now load the full checkpoint
        checkpoint = load_checkpoint(resume_checkpoint, model, optimizer, device)
        
        # Restore training state
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        best_val_rhythm_acc = checkpoint.get('val_metrics', {}).get('rhythm_acc', 0.0)
        best_epoch = checkpoint.get('epoch', 0)
        
        # Find the actual best epoch from history
        # Find the actual best epoch from history based on current monitor metric
        hist_key = f'val_{monitor_metric}' # keys in history are prefixed with 'val_'
        if hist_key in history and len(history[hist_key]) > 0:
            if monitor_mode == 'min':
                best_metric_val = min(history[hist_key])
            else:
                best_metric_val = max(history[hist_key])
            best_epoch = history[hist_key].index(best_metric_val) + 1
            
        print(f"\n✓ Training will resume from epoch {start_epoch}")
        print(f"  Best val {monitor_metric} so far: {best_metric_val:.4f} at epoch {best_epoch}")
        
    else:
        # Fresh training - create model normally
        if tuned_dropouts is not None:
            model = DeepBeatModel(dropouts=tuned_dropouts).to(device)
            print("Model initialized with TUNED dropout values")
        else:
            model = DeepBeatModel().to(device)
            print("Model initialized with DEFAULT dropout values")
        
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    writer = setup_tensorboard(args)
    
    # Initialize early stopping if enabled
    early_stopper = None
    if args.early_stopping:
        # Determine mode based on metric
        mode = 'min' if args.early_stopping_metric == 'loss' else 'max'
        early_stopper = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=mode,
            verbose=True
        )
        
        # Restore early stopping state if resuming
        if resume_checkpoint is not None:
            restore_early_stopping_state(early_stopper, checkpoint)
        
        print(f"\n{'='*60}")
        print(f"EARLY STOPPING ENABLED")
        print(f"{'='*60}")
        print(f"Monitoring metric:  {args.early_stopping_metric}")
        print(f"Patience:           {args.early_stopping_patience} epochs")
        print(f"Min delta:          {args.early_stopping_min_delta}")
        print(f"Mode:               {mode} (better = {'lower' if mode == 'min' else 'higher'})")
        print(f"{'='*60}\n")
    
    # 4. Training Loop
    output_path = Path(args.output_path) / args.file_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Starting training...")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # Train
            train_m = run_epoch(model, train_loader, optimizer, device, epoch, 
                                args.qa_loss_weight, args.rhythm_loss_weight, is_training=True, progress_bar= True)
            
            # Validate
            val_m = run_epoch(model, val_loader, optimizer, device, epoch, 
                              args.qa_loss_weight, args.rhythm_loss_weight, is_training=False, progress_bar= True)
            
            # Logging
            writer.add_scalars('Loss', {'train': train_m['loss'], 'val': val_m['loss']}, epoch)
            writer.add_scalars('Accuracy/Rhythm', {'train': train_m['rhythm_acc'], 'val': val_m['rhythm_acc']}, epoch)
            writer.add_scalars('Accuracy/QA', {'train': train_m['qa_acc'], 'val': val_m['qa_acc']}, epoch)
            writer.add_scalars('F1_macro/Rhythm', {'train': train_m['rhythm_f1'], 'val': val_m['rhythm_f1']}, epoch)
            
            history['loss'].append(train_m['loss'])
            history['val_loss'].append(val_m['loss'])
            history['val_rhythm_acc'].append(val_m['rhythm_acc'])
            history['val_qa_acc'].append(val_m['qa_acc'])
            history['val_rhythm_f1'].append(val_m['rhythm_f1'])
            
            # Print summary of epoch
            print(f"   -> Train Loss: {train_m['loss']:.4f} | Rh Acc: {train_m['rhythm_acc']:.4f} | Rh F1: {train_m['rhythm_f1']:.2f}| QA Acc: {train_m['qa_acc']:.4f}")
            print(f"   -> Val   Loss: {val_m['loss']:.4f} | Rh Acc: {val_m['rhythm_acc']:.4f} |Rh f1: {val_m['rhythm_f1']: .2f} | QA Acc: {val_m['qa_acc']:.4f}")
            
            # Check early stopping
            if early_stopper is not None:
                metric_value = val_m[args.early_stopping_metric]
                if early_stopper(metric_value, epoch):
                    print(f"\nStopping training early at epoch {epoch}")
                    break
            
            # Save Best (Dynamic Metric)
            current_val = val_m[monitor_metric]
            is_best = False
            
            if monitor_mode == 'min':
                if current_val < best_metric_val:
                    is_best = True
            else: # max
                if current_val > best_metric_val:
                    is_best = True
            
            if is_best:
                best_metric_val = current_val
                best_epoch = epoch
                
                save_checkpoint(epoch, model, optimizer, val_m, history, args, 
                               output_path, tuned_dropouts, early_stopper, 
                               checkpoint_type='best')
                
                print(f"   ⭐ New Best Model Saved! ({monitor_metric}: {best_metric_val:.4f})")
            
            # Save progress checkpoint every 10 epochs (for resume capability)
            if epoch % 10 == 0:
                save_checkpoint(epoch, model, optimizer, val_m, history, args, 
                               output_path, tuned_dropouts, early_stopper, 
                               checkpoint_type='progress')
                print(f"   💾 Progress checkpoint saved (epoch {epoch})")
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⚠️  TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print("="*60)
        print("Saving current progress before exiting...")
        
        # Save interrupted checkpoint
        save_checkpoint(epoch, model, optimizer, val_m, history, args, 
                       output_path, tuned_dropouts, early_stopper, 
                       checkpoint_type='interrupted')
        print(f"✓ Interrupted checkpoint saved at epoch {epoch}")
        print(f"\nTo resume training, use:")
        print(f"  --resume_from {output_path / f'{args.file_name}_interrupted.pth'}")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n\n" + "="*60)
        print(f"❌ TRAINING FAILED WITH ERROR")
        print("="*60)
        print(f"Error: {str(e)}")
        print("\nSaving current progress before exiting...")
        
        # Save error checkpoint
        try:
            save_checkpoint(epoch, model, optimizer, val_m, history, args, 
                           output_path, tuned_dropouts, early_stopper, 
                           checkpoint_type='error')
            print(f"✓ Error checkpoint saved at epoch {epoch}")
            print(f"\nTo resume training, use:")
            print(f"  --resume_from {output_path / f'{args.file_name}_error.pth'}")
        except:
            print("Failed to save error checkpoint")
        
        print("="*60 + "\n")
        raise  # Re-raise the exception
    
    # Save Final
    #torch.save(model.state_dict(), output_path / f"{args.file_name}_final.pth")
    save_checkpoint(epoch, model, optimizer, val_m, history, args, 
                           output_path, tuned_dropouts, early_stopper, 
                           checkpoint_type='final')
    
  
    
    with open(output_path / f"{args.file_name}_history.pkl", 'wb') as f:
        pickle.dump(history, f)
        
    writer.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Epoch:          {best_epoch}")
    print(f"Best Val {monitor_metric}: {best_metric_val:.4f}")
    if early_stopper is not None and early_stopper.early_stop:
        print(f"Early Stopped:       Yes (after {epoch} epochs)")
        print(f"Patience:            {args.early_stopping_patience}")
    else:
        print(f"Total Epochs:        {epoch}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
