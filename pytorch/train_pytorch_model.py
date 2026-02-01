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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepbeat_model import DeepBeatModel 

class DeepBeatDataset(Dataset):
    """PyTorch Dataset for DeepBeat data"""
    
    def __init__(self, data, qa_labels, rhythm_labels):
        """
        Args:
            data: Signal data (N, 800, 1)
            qa_labels: QA labels one-hot encoded (N, 3)
            rhythm_labels: Rhythm labels one-hot encoded (N, 2)
        """
        # CRITICAL FIX: PyTorch Conv1d needs (Batch, Channels, Length)
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

# --- Data Loading Utilities (Kept from original) ---

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

def load_substituted_relabeled_data(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

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
    
    idx = np.random.permutation(range(len(label_train_r)))
    data_train = data_train[idx, :]
    label_train_r = label_train_r[idx]
    label_train_q = label_train_q[idx]
    return data_train, label_train_r, label_train_q

def load_training_data(args):
    print("=" * 60)
    print(f"TRAINING CHOICE: {args.training_choice}")
    print("=" * 60)
    
    if args.training_choice in ["db_orig_replaced", "db_orig_replaced_w_vsm"]:
        data_to_shuffle = load_substituted_relabeled_data(args.db_orig_replaced_path)
        if args.training_choice == "db_orig_replaced_w_vsm":
            _, _, relabeled_vsm = load_relabeled_data(args.relabled_path)
            return attach_VSM(data_to_shuffle, relabeled_vsm)
        return data_to_shuffle
    
    if args.training_choice == "db_orig":
        return load_original_data(args.orig_data_path, 'train.npz')
    
    if args.training_choice in ["db_relabel", "db_relabel_w_vsm"]:
        db_train = load_original_data(args.orig_data_path, 'train.npz')
        _, relabeled_db, relabeled_vsm = load_relabeled_data(args.relabled_path)
        data_to_shuffle = replace_updated_subjects_db(db_train, relabeled_db)
        if args.training_choice == "db_relabel_w_vsm":
            return attach_VSM(data_to_shuffle, relabeled_vsm)
    return data_to_shuffle

# --- Training Logic ---

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
    # Takes raw logits (N, 2) and one-hot targets (N, 2)
    rhythm_loss = nn.BCEWithLogitsLoss()(rhythm_logits, rhythm_target)
    
    total_loss = qa_weight * qa_loss + rhythm_weight * rhythm_loss 
    
    return total_loss, qa_loss, rhythm_loss

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

def run_epoch(model, dataloader, optimizer, device, epoch, qa_weight, rhythm_weight, is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()
        
    metrics = {
        'loss': 0.0, 'qa_loss': 0.0, 'rhythm_loss': 0.0,
        'qa_acc': 0.0, 'rhythm_acc': 0.0
    }
    
    num_batches = 0
    
    # Use torch.set_grad_enabled to handle train/eval modes conveniently
    with torch.set_grad_enabled(is_training):
        for batch in dataloader:
            data = batch['data'].to(device)
            
            if is_training:
                optimizer.zero_grad()
            
            # Forward pass: Unpack tuple (qa, rhythm)
            qa_logits, rhythm_logits = model(data)
            
            # Compute loss
            loss, qa_loss, rhythm_loss = compute_loss(
                qa_logits, rhythm_logits, batch, device, qa_weight, rhythm_weight
            )
            
            if is_training:
                loss.backward()
                optimizer.step()
            
            # Compute accuracy
            qa_acc, rhythm_acc = compute_accuracy(qa_logits, rhythm_logits, batch, device)
            
            metrics['loss'] += loss.item()
            metrics['qa_loss'] += qa_loss.item()
            metrics['rhythm_loss'] += rhythm_loss.item()
            metrics['qa_acc'] += qa_acc
            metrics['rhythm_acc'] += rhythm_acc
            num_batches += 1
    
    # Average metrics
    return {k: v / num_batches for k, v in metrics.items()}

def setup_tensorboard(args):
    log_path = Path(args.output_path) / Path(args.file_name)
    log_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_path))

def parser_args():
    parser = argparse.ArgumentParser()
    
    # data path
    parser.add_argument("--orig_data_path", default=r'C:\Users\aoara\develop\deepbeat\data\original_data')
    parser.add_argument("--relabled_path", default=r'C:\Users\aoara\develop\deepbeat\data\relabeled_data')
    parser.add_argument("--output_path", default=r'C:\Users\aoara\develop\deepbeat\training_output')

    # experiment config
    parser.add_argument("--file_name", required=True, help="name the file (model name)")
    valid_choices = ['db_orig', 'db_relabel', 'db_relabel_w_vsm', 'db_orig_replaced', 'db_orig_replaced_vsm']
    parser.add_argument("--training_choice", choices=valid_choices, required=True, help=str(valid_choices))
    parser.add_argument("--db_orig_replaced_path", default=r"C:\Users\aoara\develop\deepbeat\output\replace_relabeled.pkl")
    
    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization)")
    parser.add_argument("--qa_loss_weight", type=float, default=0.2)
    parser.add_argument("--rhythm_loss_weight", type=float, default=5.0)
    
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    args = parser_args()
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # 1. Load Data
    print("Loading training data...")
    data_to_shuffle = load_training_data(args)
    data_train, label_train_r, label_train_q = shuffle_data(data_to_shuffle)
    print(f"Train Shape: {data_train.shape}")

    print("Loading validation data...")
    db_val = load_original_data(args.orig_data_path, 'validate.npz')
    data_val, label_val_r, label_val_q = db_val['data'], db_val['rhythm'], db_val['qa_label']
    print(f"Val Shape: {data_val.shape}")
    
    # 2. Dataset & Loader
    train_dataset = DeepBeatDataset(data_train, label_train_q, label_train_r)
    val_dataset = DeepBeatDataset(data_val, label_val_q, label_val_r)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    
    # 3. Model & Optimizer
    print("Creating model...")
    model = DeepBeatModel().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = setup_tensorboard(args)
    
    # 4. Training Loop
    history = {'loss': [], 'val_loss': [], 'val_rhythm_acc': [], 'val_qa_acc': []}
    best_val_rhythm_acc = 0.0
    best_epoch = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # Train
        train_m = run_epoch(model, train_loader, optimizer, device, epoch, 
                            args.qa_loss_weight, args.rhythm_loss_weight, is_training=True)
        
        # Validate
        val_m = run_epoch(model, val_loader, optimizer, device, epoch, 
                          args.qa_loss_weight, args.rhythm_loss_weight, is_training=False)
        
        # Logging
        writer.add_scalars('Loss', {'train': train_m['loss'], 'val': val_m['loss']}, epoch)
        writer.add_scalars('Accuracy/Rhythm', {'train': train_m['rhythm_acc'], 'val': val_m['rhythm_acc']}, epoch)
        writer.add_scalars('Accuracy/QA', {'train': train_m['qa_acc'], 'val': val_m['qa_acc']}, epoch)
        
        history['loss'].append(train_m['loss'])
        history['val_loss'].append(val_m['loss'])
        history['val_rhythm_acc'].append(val_m['rhythm_acc'])
        history['val_qa_acc'].append(val_m['qa_acc'])
        
        # Save Best
        if val_m['rhythm_acc'] > best_val_rhythm_acc:
            best_val_rhythm_acc = val_m['rhythm_acc']
            best_epoch = epoch + 1
            
            output_path = Path(args.output_path) / args.file_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_m
            }, output_path / f"{args.file_name}_best.pth")
            
            print(f"Epoch {epoch+1}: New Best Rhythm Acc: {best_val_rhythm_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss {train_m['loss']:.4f} | Val Loss {val_m['loss']:.4f} | Val Rhythm Acc {val_m['rhythm_acc']:.4f}")

    # Save Final
    output_path = Path(args.output_path) / args.file_name
    torch.save(model.state_dict(), output_path / f"{args.file_name}_final.pth")
    
    with open(output_path / f"{args.file_name}_history.pkl", 'wb') as f:
        pickle.dump(history, f)
        
    writer.close()
    print(f"\nTraining Complete. Best Epoch: {best_epoch} with Acc: {best_val_rhythm_acc:.4f}")

if __name__ == "__main__":
    main()