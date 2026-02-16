import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import os
import multiprocessing as mp


# Add tqdm for progress bars
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
import torch.nn.functional as F

class DeepBeatDataset(Dataset):
    """PyTorch Dataset for DeepBeat data"""
    
    def __init__(self, data, qa_labels, rhythm_labels):
        
        # Input is (N, 800, 1) -> Permute to (N, 1, 800)
        self.data = torch.FloatTensor(data).permute(0, 2, 1)
        self.qa_labels = torch.FloatTensor(qa_labels)
        self.rhythm_labels = torch.FloatTensor(rhythm_labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'qa_label': self.qa_labels[idx],
            'rhythm_label': self.rhythm_labels[idx]
        }


def compute_f1_score(logits, targets, device, average='binary'):
    """
    Compute F1 score from logits and one-hot encoded targets
    
    Args:
        logits: Raw model outputs (N, num_classes)
        targets: One-hot encoded labels (N, num_classes)
        device: torch device
        average: F1 averaging method ('macro', 'micro', 'weighted', 'binary')
    
    Returns:
        F1 score (float)
    """
    # Convert logits to predictions
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    
    # Convert one-hot targets to class indices
    true = torch.argmax(targets.to(device), dim=1).cpu().numpy()
    
    # Calculate F1 score
    # For binary classification (rhythm: 2 classes), can use 'binary' or 'macro'
    # For multi-class (QA: 3 classes), use 'macro', 'micro', or 'weighted'
    f1 = f1_score(true, pred, average=average, zero_division=0)
    
    return f1


def calculate_auroc(logits, targets):
    if isinstance(logits, list):
        if len(logits) == 0:
            return 0.0
        # Stack all tensors into single tensor
        logits = torch.cat(logits, dim=0)
       
    
    probs = F.softmax(logits, dim=1)
    positive_probs = probs[:, 1].detach().cpu().numpy()
    auc = roc_auc_score(targets.cpu().numpy(), positive_probs)
    
    return auc

def calculate_auprc(logits, targets):
    if isinstance(logits, list):
        if len(logits) == 0:
            return 0.0
        # Stack all tensors into single tensor
        logits = torch.cat(logits, dim=0)
    
    logits = torch.tensor(logits)
    probs = F.softmax(logits, dim=1)
    positive_probs = probs[:, 1].detach().cpu().numpy()
    auprc = average_precision_score(targets.cpu().numpy(), positive_probs)
    return auprc
    

def compute_loss(qa_logits, rhythm_logits, targets, device, qa_weight=0.2, rhythm_weight=5.0):
    """
    Args:
        qa_logits: Raw outputs from model (N, 3)
        rhythm_logits: Raw outputs from model (N, 2)
        targets: Dictionary containing 'qa_label' and 'rhythm_label' (One-hot)
    """
    qa_target = targets['qa_label'].to(device)
    rhythm_target = targets['rhythm_label'].to(device)
    
    # 1. QA Loss: CrossEntropyLoss expects class indices, not one-hot
    # Convert one-hot (N, 3) -> indices (N,)
    qa_target_indices = torch.argmax(qa_target, dim=1)
    qa_loss = nn.CrossEntropyLoss()(qa_logits, qa_target_indices)
    
    # 2. Rhythm Loss: BCEWithLogitsLoss is more stable than Sigmoid + BCELoss
    # this is treating Afib and Normal Signals as two independent labels
    # Takes raw logits (N, 2) and one-hot targets (N, 2)
    #rhythm_loss = nn.BCEWithLogitsLoss()(rhythm_logits, rhythm_target)
    rhythm_target_indices = torch.argmax(rhythm_target, dim=1)
    rhythm_loss = nn.CrossEntropyLoss()(rhythm_logits, rhythm_target_indices)
    
    total_loss = qa_weight * qa_loss + rhythm_weight * rhythm_loss 
    
    return total_loss, qa_loss, rhythm_loss


def run_epoch(model, dataloader, optimizer, device, epoch, qa_weight, rhythm_weight, 
                       is_training=True, f1_average='macro', progress_bar=False):
    model.train() if is_training else model.eval()
    desc_str = f"Epoch {epoch} [{'TRAIN' if is_training else 'VAL'}]"
    
    running_loss = 0.0
    running_qa_loss, running_rhythm_loss = 0.0, 0.0
    all_qa_preds, all_qa_targets = [], []
    all_rhythm_preds, all_rhythm_targets = [], []
    total_grad_norm = 0.0
    all_rhythm_logits = []
    
    # Correctly wrap the dataloader for the progress bar
    iterable = tqdm(dataloader, desc=desc_str, leave=True, ncols=120) if progress_bar else dataloader
    
    with torch.set_grad_enabled(is_training):
        for batch in iterable:
            data = batch['data'].to(device)
            
            if is_training:
                optimizer.zero_grad()
            
            qa_logits, rhythm_logits = model(data)
            
            loss, qa_loss, rhythm_loss = compute_loss(
                qa_logits, rhythm_logits, batch, device, qa_weight, rhythm_weight
            )
            
            if is_training:
                loss.backward()
                
                # Track the norm without clipping
                # Setting max_norm to float('inf') means the gradients are never scaled for monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm= float('inf')
                )
                total_grad_norm += grad_norm.item()
                optimizer.step()
            
            running_loss += loss.item()
            running_qa_loss += qa_loss.item()
            running_rhythm_loss += rhythm_loss.item()
            
            # Get predictions
            qa_pred = torch.argmax(qa_logits, dim=1).detach().cpu().numpy()
            rhythm_pred = torch.argmax(rhythm_logits, dim=1).detach().cpu().numpy()
            
            # --- Target Handling ---
            # Use argmax ONLY if your labels are one-hot encoded. 
            # If they are already class indices, just use .cpu().numpy()
            qa_true = torch.argmax(batch['qa_label'], dim=1).detach().cpu().numpy()
            rhythm_true = torch.argmax(batch['rhythm_label'], dim=1).detach().cpu().numpy()
            
            all_qa_preds.extend(qa_pred)
            all_qa_targets.extend(qa_true)
            all_rhythm_preds.extend(rhythm_pred)
            all_rhythm_targets.extend(rhythm_true)
            all_rhythm_logits.append(rhythm_logits.detach().cpu())

            # Update progress bar in real-time
            if progress_bar:
                iterable.set_postfix({'loss': f"{loss.item():.4f}"})

    # Compute final metrics
    metrics = {
        'loss': running_loss / len(dataloader),
        'rhythm_loss': running_rhythm_loss /  len(dataloader),
        'qa_loss': running_qa_loss / len(dataloader),
        'rhythm_f1': f1_score(all_rhythm_targets, all_rhythm_preds, average=f1_average, zero_division=0),
        'qa_acc': accuracy_score(all_qa_targets, all_qa_preds),
        'rhythm_acc': accuracy_score(all_rhythm_targets, all_rhythm_preds),
        'grad_norm': total_grad_norm / len(dataloader) if is_training else 0.0,
        'auroc': calculate_auroc(all_rhythm_logits, all_rhythm_targets) if not is_training else 0.0,
        'auprc': calculate_auprc(all_rhythm_logits, all_rhythm_targets) if not is_training else 0.0 
    }
    
    return metrics


def compute_loss_single_task(logits, targets, device, branch = 'rhythm'):
    """
    calculate logits for single-task
       
    """
    if branch == 'rhythm':
        target = targets['rhythm_label'].to(device)
    else: 
        target = targets['qa_label'].to(device)
    
    
    target_indices = torch.argmax(target, dim=1)
    loss = nn.CrossEntropyLoss()(logits, target_indices)
    

    return loss

def run_epoch_single_task(model, dataloader, optimizer, device, epoch, qa_weight, rhythm_weight, 
                       is_training=True, f1_average='macro', progress_bar=False):
    model.train() if is_training else model.eval()
    desc_str = f"Epoch {epoch} [{'TRAIN' if is_training else 'VAL'}]"
    
    running_loss = 0.0
    running_qa_loss, running_rhythm_loss = 0.0, 0.0
    all_qa_preds, all_qa_targets = [], []
    all_rhythm_preds, all_rhythm_targets = [], []
    total_grad_norm = 0.0
    all_rhythm_logits = []
    
    # Correctly wrap the dataloader for the progress bar
    iterable = tqdm(dataloader, desc=desc_str, leave=True, ncols=120) if progress_bar else dataloader
    
    with torch.set_grad_enabled(is_training):
        for batch in iterable:
            data = batch['data'].to(device)
            
            if is_training:
                optimizer.zero_grad()
            
            qa_logits, rhythm_logits = model(data)
            
            loss, qa_loss, rhythm_loss = compute_loss(
                qa_logits, rhythm_logits, batch, device, qa_weight, rhythm_weight
            )
            
            if is_training:
                loss.backward()
                
                # Track the norm without clipping
                # Setting max_norm to float('inf') means the gradients are never scaled for monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm= float('inf')
                )
                total_grad_norm += grad_norm.item()
                optimizer.step()
            
            running_loss += loss.item()
            running_qa_loss += qa_loss.item()
            running_rhythm_loss += rhythm_loss.item()
            
            # Get predictions
            qa_pred = torch.argmax(qa_logits, dim=1).detach().cpu().numpy()
            rhythm_pred = torch.argmax(rhythm_logits, dim=1).detach().cpu().numpy()
            
            # --- Target Handling ---
            # Use argmax ONLY if your labels are one-hot encoded. 
            # If they are already class indices, just use .cpu().numpy()
            qa_true = torch.argmax(batch['qa_label'], dim=1).detach().cpu().numpy()
            rhythm_true = torch.argmax(batch['rhythm_label'], dim=1).detach().cpu().numpy()
            
            all_qa_preds.extend(qa_pred)
            all_qa_targets.extend(qa_true)
            all_rhythm_preds.extend(rhythm_pred)
            all_rhythm_targets.extend(rhythm_true)
            all_rhythm_logits.append(rhythm_logits.detach().cpu())

            # Update progress bar in real-time
            if progress_bar:
                iterable.set_postfix({'loss': f"{loss.item():.4f}"})

    # Compute final metrics
    metrics = {
        'loss': running_loss / len(dataloader),
        'rhythm_loss': running_rhythm_loss /  len(dataloader),
        'qa_loss': running_qa_loss / len(dataloader),
        'rhythm_f1': f1_score(all_rhythm_targets, all_rhythm_preds, average=f1_average, zero_division=0),
        'qa_acc': accuracy_score(all_qa_targets, all_qa_preds),
        'rhythm_acc': accuracy_score(all_rhythm_targets, all_rhythm_preds),
        'grad_norm': total_grad_norm / len(dataloader) if is_training else 0.0,
        'auroc': calculate_auroc(all_rhythm_logits, all_rhythm_targets) if not is_training else 0.0,
        'auprc': calculate_auprc(all_rhythm_logits, all_rhythm_targets) if not is_training else 0.0 
    }
    
    return metrics



def get_optimal_workers(user_specified=None):
    """
    Determine optimal number of DataLoader workers
    
    Args:
        user_specified: User-specified number of workers (None for auto-detect)
    
    Returns:
        Optimal number of workers
    """
    if user_specified is not None:
        print(f"Using user-specified num_workers: {user_specified}")
        return user_specified
    
    # Auto-detect
    try:
        cpu_count = mp.cpu_count()
    except NotImplementedError:
        cpu_count = os.cpu_count() or 4
    
    # Rule of thumb: Use CPU count - 2, but at least 2, max 8
    # Leave 1-2 cores for main process and system
    optimal = max(2, min(cpu_count - 2, 8))
    
    # On Windows, high worker counts can be problematic
    if os.name == 'nt':  # Windows
        optimal = min(optimal, 4)
    
    print(f"Auto-detected num_workers: {optimal} (CPU cores: {cpu_count})")
    return optimal

# -- Early Stopping utils for training
class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            mode (str): 'max' for metrics where higher is better (accuracy),
                       'min' for metrics where lower is better (loss)
            verbose (bool): Print messages when early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, current_score, epoch):
        """
        Check if training should stop
        
        Args:
            current_score (float): Current epoch's validation metric
            epoch (int): Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        # Check if there's improvement
        if self.mode == 'max':
            improved = current_score > (self.best_score + self.min_delta)
        else:  # mode == 'min'
            improved = current_score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"Validation improved! Resetting patience counter.")
        else:
            self.counter += 1
            if self.verbose:
                print(f" No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n Early stopping triggered! No improvement for {self.patience} epochs.")
                    print(f"   Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        
        return False
    
def restore_early_stopping_state(early_stopper, checkpoint):
    """Restore early stopping state from checkpoint"""
    if early_stopper is not None and 'early_stopping_state' in checkpoint:
        es_state = checkpoint['early_stopping_state']
        early_stopper.counter = es_state['counter']
        early_stopper.best_score = es_state['best_score']
        early_stopper.best_epoch = es_state['best_epoch']
        early_stopper.patience = es_state['patience']
        early_stopper.min_delta = es_state['min_delta']
        early_stopper.mode = es_state['mode']
        
        print(f"\n✓ Early stopping state restored:")
        print(f"  - Counter: {early_stopper.counter}/{early_stopper.patience}")
        print(f"  - Best score: {early_stopper.best_score:.4f} at epoch {early_stopper.best_epoch}")
    

def setup_tensorboard(args):
    log_path = Path(args.output_path) / Path(args.file_name)
    log_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_path))


# --- loading tuned parameters from optuna
def apply_tuned_params(args):
    """Apply tuned parameters from JSON file if provided.
       Return tuned dropouts and override other hyperparameters in args
    """
    if args.tuned_params_path is None:
        print("No tuned parameters file provided. Using command-line arguments.")
        return None
    
    if not Path(args.tuned_params_path).exists():
        print(f"WARNING: Tuned params file not found at {args.tuned_params_path}")
        print("Using command-line arguments instead.")
        return None
    
    def load_tuned_params(json_path):
        """Load tuned hyperparameters from JSON file"""
        with open(json_path, 'r') as f:
            params = json.load(f)
        return params
    
    print(f"Loading tuned parameters from: {args.tuned_params_path}")
    tuned_params = load_tuned_params(args.tuned_params_path)
    
    # Extract ready_to_use parameters
    if 'ready_to_use' in tuned_params:
        ready_params = tuned_params['ready_to_use']
        dropouts = ready_params.get('dropouts', {})
        
        # Override args with tuned values
        args.batch_size = int(ready_params.get('batch_size', args.batch_size))
        args.learning_rate = ready_params.get('lr', args.learning_rate)
        args.weight_decay = ready_params.get('weight_decay', args.weight_decay)
        args.qa_loss_weight = ready_params.get('qa_weight', args.qa_loss_weight)
        args.rhythm_loss_weight = ready_params.get('rhythm_weight', args.rhythm_loss_weight)
        
        print("\n" + "="*60)
        print("LOADED TUNED HYPERPARAMETERS")
        print("="*60)
        print(f"Batch size:      {args.batch_size}")
        print(f"Learning rate:   {args.learning_rate:.6f}")
        print(f"Weight decay:    {args.weight_decay:.6f}")
        print(f"QA weight:       {args.qa_loss_weight:.6f}")
        print(f"Rhythm weight:   {args.rhythm_loss_weight:.6f}")
        print(f"\nDropout values:")
        for key, val in dropouts.items():
            print(f"  {key}: {val:.6f}")
        print("="*60 + "\n")
        
        return dropouts
    else:
        print("WARNING: 'ready_to_use' key not found in tuned params file")
        return None
    
# --- checkpoint saving and loading

def save_checkpoint(epoch, model, optimizer, val_metrics, history, args, 
                   output_path, tuned_dropouts, early_stopper=None, 
                   checkpoint_type='progress'):
    """
    Save a comprehensive training checkpoint
    
    Args:
        checkpoint_type: 'progress' for regular saves, 'best' for best model, 'final' for end
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'history': history,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'qa_loss_weight': args.qa_loss_weight,
            'rhythm_loss_weight': args.rhythm_loss_weight,
            'dropouts': tuned_dropouts if tuned_dropouts is not None else 'default'
        },
        'args': vars(args),  # Save all arguments for full reproducibility
    }
    
    # Save early stopping state if it exists
    if early_stopper is not None:
        checkpoint['early_stopping_state'] = {
            'counter': early_stopper.counter,
            'best_score': early_stopper.best_score,
            'best_epoch': early_stopper.best_epoch,
            'patience': early_stopper.patience,
            'min_delta': early_stopper.min_delta,
            'mode': early_stopper.mode
        }
    
    filename = f"{args.file_name}_{checkpoint_type}.pth"
    checkpoint_path = output_path / filename
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load a training checkpoint
    
    Returns:
        checkpoint dict with all saved information
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint= torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if optimizer provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  - Resumed from epoch: {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        print(f"  - Last val rhythm acc: {checkpoint['val_metrics'].get('rhythm_acc', 'N/A'):.4f}")
    
    return checkpoint

def load_pickle_file(path):
    with open(path, 'rb') as file:
        return pickle.load(file)



# --- (deprecated) Data Loading tools ---

def remove_nan_data(data_dict):
    """Remove samples containing NaN values"""
    no_nan_mask = ~np.isnan(data_dict['data']).any(axis=(1, 2))
    for k in data_dict.keys():
        data_dict[k] = data_dict[k][no_nan_mask]
    return data_dict

def load_original_data(data_path, file_name):
    data = np.load(Path(data_path) / file_name, allow_pickle=True)
    output = {}
    output['data'] = data['signal']
    output['qa_label'] = data['qa_label']
    output['rhythm'] = data['rhythm']
    
    params = pd.DataFrame(data['parameters'])
    params.rename(index=str, columns={0: 'timestamp', 1: 'stream', 2: 'ID'}, inplace=True)
    output['ID'] = np.array(params['ID'].to_list())
    output = remove_nan_data(output)
    return output

def load_relabeled_data(data_path):
    def load_from_mat(dir_path, file_name):
        file_mat = loadmat(Path(dir_path) / file_name)
        return file_mat.get(file_name[:-4])
    
    combined = {}
    combined['data'] = load_from_mat(data_path, 'db_vsm_combined_data.mat')
    combined['qa_label'] = load_from_mat(data_path, 'db_vsm_combined_label_q.mat')
    combined['rhythm'] = load_from_mat(data_path, 'db_vsm_combined_label_r.mat')
    combined['ID'] = load_from_mat(data_path, 'db_vsm_combined_sub_id.mat').flatten()
    combined['data'] = combined['data'].reshape(combined['data'].shape[0], combined['data'].shape[1], 1)
    
    num_classes_rhythm = 2
    num_classes_qa = 3
    combined['rhythm'] = np.eye(num_classes_rhythm)[combined['rhythm'].flatten().astype(int)]
    combined['qa_label'] = np.eye(num_classes_qa)[combined['qa_label'].flatten().astype(int)]
    
    relabeled_db = {}
    relabeled_vsm = {}
    db_mask = (combined['ID'] < 1000).flatten()
    vsm_mask = (combined['ID'] >= 1000).flatten()
    
    relabeled_db['data'] = combined['data'][db_mask, :]
    relabeled_db['qa_label'] = combined['qa_label'][db_mask, :]
    relabeled_db['rhythm'] = combined['rhythm'][db_mask, :]
    relabeled_db['ID'] = combined['ID'][db_mask].flatten()
    
    relabeled_vsm['data'] = combined['data'][vsm_mask, :]
    relabeled_vsm['qa_label'] = combined['qa_label'][vsm_mask, :]
    relabeled_vsm['rhythm'] = combined['rhythm'][vsm_mask, :]
    relabeled_vsm['ID'] = combined['ID'][vsm_mask].flatten()
    
    return combined, relabeled_db, relabeled_vsm

def replace_updated_subjects_db(db_train, relabeled_db):
    subjects_to_replace = np.unique(relabeled_db['ID'])
    mask_keep = ~np.isin(db_train['ID'], subjects_to_replace)
    
    db_train['data'] = db_train['data'][mask_keep]
    db_train['rhythm'] = db_train['rhythm'][mask_keep]
    db_train['qa_label'] = db_train['qa_label'][mask_keep]
    db_train['ID'] = db_train['ID'][mask_keep]
    
    db_train['data'] = np.concatenate([db_train['data'], relabeled_db['data']], axis=0)
    db_train['rhythm'] = np.concatenate([db_train['rhythm'], relabeled_db['rhythm']], axis=0)
    db_train['qa_label'] = np.concatenate([db_train['qa_label'], relabeled_db['qa_label']], axis=0)
    db_train['ID'] = np.concatenate([db_train['ID'], relabeled_db['ID']], axis=0)
    return db_train

def attach_VSM(db_data, relabeled_vsm):
    db_data['data'] = np.concatenate([db_data['data'], relabeled_vsm['data']], axis=0)
    db_data['rhythm'] = np.concatenate([db_data['rhythm'], relabeled_vsm['rhythm']], axis=0)
    db_data['qa_label'] = np.concatenate([db_data['qa_label'], relabeled_vsm['qa_label']], axis=0)
    db_data['ID'] = np.concatenate([db_data['ID'], relabeled_vsm['ID']], axis=0)
    return db_data

def shuffle_data(db_train):
    data_train = db_train['data']
    label_train_r = db_train['rhythm']
    label_train_q = db_train['qa_label']
    idx = np.random.permutation(len(label_train_r))
    data_train = data_train[idx, :]
    label_train_r = label_train_r[idx]
    label_train_q = label_train_q[idx]
    return data_train, label_train_r, label_train_q

