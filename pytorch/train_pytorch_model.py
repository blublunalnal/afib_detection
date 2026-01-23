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
import h5py as h5
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
        self.data = torch.FloatTensor(data)
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


def parser_args():
    parser = argparse.ArgumentParser()

    # repo and model path
    parser.add_argument("--db_repo", default=r'C:\Users\aoara\develop\deepbeat')
    parser.add_argument("--db_h5_file", default=r"C:\Users\aoara\develop\deepbeat\deepbeat.h5")
    
    # data path
    parser.add_argument("--orig_data_path", default=r'C:\Users\aoara\develop\deepbeat\data\original_data')
    parser.add_argument("--relabled_path", default=r'C:\Users\aoara\develop\deepbeat\data\relabeled_data')
    
    # output path
    parser.add_argument("--output_path", default=r'C:\Users\aoara\develop\deepbeat\training_output')

    # experiment config
    parser.add_argument("--file_name", required=True, help="name the file (model name)")
    valid_choices = ['db_orig', 'db_relabel', 'db_relabel_w_vsm', 'db_orig_replaced', 'db_orig_replaced_vsm']
    parser.add_argument("--training_choice", choices=valid_choices, required=True, 
                       help="training data choice: " + str(valid_choices))
    parser.add_argument("--db_orig_replaced_path", 
                       default=r"C:\Users\aoara\develop\deepbeat\output\replace_relabeled.pkl")
    
    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    # loss weights for multi-task learning
    parser.add_argument("--qa_loss_weight", type=float, default=0.2,
                       help="Weight for QA (quality assessment) loss (default: 0.2)")
    parser.add_argument("--rhythm_loss_weight", type=float, default=5.0,
                       help="Weight for rhythm classification loss (default: 5.0)")
    
    # device
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    return args


def setup_tensorboard(args):
    """Setup TensorBoard logging"""
    log_path = Path(args.output_path) / Path(args.file_name)
    log_path.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_path))
    return writer


def remove_nan_data(data_dict):
    """Remove samples containing NaN values"""
    no_nan_mask = ~np.isnan(data_dict['data']).any(axis=(1, 2))
    
    for k in data_dict.keys():
        data_dict[k] = data_dict[k][no_nan_mask]
    
    return data_dict


def load_original_data(data_path, file_name):
    """Load original training/validation data"""
    data = np.load(Path(data_path) / file_name, allow_pickle=True)
    output = {}
    output['data'] = data['signal']
    output['qa_label'] = data['qa_label']
    output['rhythm'] = data['rhythm']
    
    params = pd.DataFrame(data['parameters'])
    params.rename(index=str, columns={0: 'timestamp', 1: 'stream', 2: 'ID'}, inplace=True)
    output['ID'] = np.array(params['ID'].to_list())
    
    # Remove NaN data
    output = remove_nan_data(output)
    
    return output


def load_relabeled_data(data_path):
    """Load relabeled data from MATLAB files"""
    
    def load_from_mat(dir_path, file_name):
        file_mat = loadmat(Path(dir_path) / file_name)
        file = file_mat.get(file_name[:-4])
        return file
    
    combined = {}
    combined['data'] = load_from_mat(data_path, 'db_vsm_combined_data.mat')
    combined['qa_label'] = load_from_mat(data_path, 'db_vsm_combined_label_q.mat')
    combined['rhythm'] = load_from_mat(data_path, 'db_vsm_combined_label_r.mat')
    combined['ID'] = load_from_mat(data_path, 'db_vsm_combined_sub_id.mat').flatten()
    
    # Reshape to match original data format
    combined['data'] = combined['data'].reshape(combined['data'].shape[0], combined['data'].shape[1], 1)
    
    # One-hot encoding
    num_classes_rhythm = 2
    num_classes_qa = 3
    combined['rhythm'] = np.eye(num_classes_rhythm)[combined['rhythm'].flatten().astype(int)]
    combined['qa_label'] = np.eye(num_classes_qa)[combined['qa_label'].flatten().astype(int)]
    
    relabeled_db = {}
    relabeled_vsm = {}
    
    # VSM index starts from 1000
    db_mask = (combined['ID'] < 1000).flatten()
    vsm_mask = (combined['ID'] >= 1000).flatten()
    
    # Separate DB data
    relabeled_db['data'] = combined['data'][db_mask, :]
    relabeled_db['qa_label'] = combined['qa_label'][db_mask, :]
    relabeled_db['rhythm'] = combined['rhythm'][db_mask, :]
    relabeled_db['ID'] = combined['ID'][db_mask].flatten()
    
    # Separate VSM data
    relabeled_vsm['data'] = combined['data'][vsm_mask, :]
    relabeled_vsm['qa_label'] = combined['qa_label'][vsm_mask, :]
    relabeled_vsm['rhythm'] = combined['rhythm'][vsm_mask, :]
    relabeled_vsm['ID'] = combined['ID'][vsm_mask].flatten()
    
    return combined, relabeled_db, relabeled_vsm


def replace_updated_subjects_db(db_train, relabeled_db):
    """Replace old data with relabeled data for specific subjects"""
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
    """Load saved substituted original data"""
    with open(path, 'rb') as file:
        orig_sub_relabel = pickle.load(file)
    return orig_sub_relabel


def attach_VSM(db_data, relabeled_vsm):
    """Attach VSM data to existing dataset"""
    db_data['data'] = np.concatenate([db_data['data'], relabeled_vsm['data']], axis=0)
    db_data['rhythm'] = np.concatenate([db_data['rhythm'], relabeled_vsm['rhythm']], axis=0)
    db_data['qa_label'] = np.concatenate([db_data['qa_label'], relabeled_vsm['qa_label']], axis=0)
    db_data['ID'] = np.concatenate([db_data['ID'], relabeled_vsm['ID']], axis=0)
    return db_data


def shuffle_data(db_train):
    """Shuffle training data"""
    data_train = db_train['data']
    label_train_r = db_train['rhythm']
    label_train_q = db_train['qa_label']
    
    idx = np.random.permutation(range(len(label_train_r)))
    data_train = data_train[idx, :]
    label_train_r = label_train_r[idx]
    label_train_q = label_train_q[idx]
    
    return data_train, label_train_r, label_train_q


def load_training_data(args):
    """Load training data based on specified choice"""
    print("=" * 60)
    print(f"TRAINING CHOICE: {args.training_choice}")
    print("=" * 60)
    
    # db_orig_replaced: replace relabeled data, keep unrelabeled data
    if args.training_choice in ["db_orig_replaced", "db_orig_replaced_w_vsm"]:
        data_to_shuffle = load_substituted_relabeled_data(args.db_orig_replaced_path)
        
        if args.training_choice == "db_orig_replaced_w_vsm":
            _, _, relabeled_vsm = load_relabeled_data(args.relabled_path)
            return attach_VSM(data_to_shuffle, relabeled_vsm)
        
        return data_to_shuffle
    
    # Handle db_orig
    if args.training_choice == "db_orig":
        return load_original_data(args.orig_data_path, 'train.npz')
    
    # db_relabel: keep ONLY relabeled data
    if args.training_choice in ["db_relabel", "db_relabel_w_vsm"]:
        db_train = load_original_data(args.orig_data_path, 'train.npz')
        _, relabeled_db, relabeled_vsm = load_relabeled_data(args.relabled_path)
        data_to_shuffle = replace_updated_subjects_db(db_train, relabeled_db)
        
        if args.training_choice == "db_relabel_w_vsm":
            return attach_VSM(data_to_shuffle, relabeled_vsm)
    
    return data_to_shuffle


def compute_loss(model, outputs, targets, device, qa_weight=0.2, rhythm_weight=5.0):
    """
    Compute multi-task loss with weights.
    
    Args:
        model: DeepBeat model
        outputs: Model outputs dictionary
        targets: Ground truth targets dictionary
        device: Device tensors are on
        qa_weight: Weight for QA loss (default: 0.2)
        rhythm_weight: Weight for rhythm loss (default: 5.0)
    
    Returns:
        total_loss, qa_loss, rhythm_loss
    """
    qa_target = targets['qa_label'].to(device)
    rhythm_target = targets['rhythm_label'].to(device)
    
    # Categorical cross-entropy for QA (3 classes)
    qa_loss = nn.CrossEntropyLoss()(outputs['qa_output'], qa_target)
    
    # Binary cross-entropy for rhythm (2 classes) 
    rhythm_loss = nn.BCELoss()(outputs['rhythm_output'], rhythm_target)
    
    # Add L2 regularization
    l2_reg = model.get_l2_regularization()
    
    # Weighted loss
    total_loss = qa_weight * qa_loss + rhythm_weight * rhythm_loss + l2_reg
    
    return total_loss, qa_loss, rhythm_loss


def compute_accuracy(outputs, targets):
    """Compute accuracy for both outputs"""
    # QA accuracy
    qa_pred = torch.argmax(outputs['qa_output'], dim=1)
    qa_true = torch.argmax(targets['qa_label'], dim=1)
    qa_acc = (qa_pred == qa_true).float().mean()
    
    # Rhythm accuracy
    rhythm_pred = torch.argmax(outputs['rhythm_output'], dim=1)
    rhythm_true = torch.argmax(targets['rhythm_label'], dim=1)
    rhythm_acc = (rhythm_pred == rhythm_true).float().mean()
    
    return qa_acc.item(), rhythm_acc.item()


def train_epoch(model, dataloader, optimizer, device, epoch, qa_weight, rhythm_weight):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_qa_loss = 0
    total_rhythm_loss = 0
    total_qa_acc = 0
    total_rhythm_acc = 0
    num_batches = 0
    
    for batch in dataloader:
        data = batch['data'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Compute loss
        loss, qa_loss, rhythm_loss = compute_loss(model, outputs, batch, device, qa_weight, rhythm_weight)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        qa_acc, rhythm_acc = compute_accuracy(outputs, batch)
        
        # Accumulate metrics
        total_loss += loss.item()
        total_qa_loss += qa_loss.item()
        total_rhythm_loss += rhythm_loss.item()
        total_qa_acc += qa_acc
        total_rhythm_acc += rhythm_acc
        num_batches += 1
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_qa_loss = total_qa_loss / num_batches
    avg_rhythm_loss = total_rhythm_loss / num_batches
    avg_qa_acc = total_qa_acc / num_batches
    avg_rhythm_acc = total_rhythm_acc / num_batches
    
    return {
        'loss': avg_loss,
        'qa_loss': avg_qa_loss,
        'rhythm_loss': avg_rhythm_loss,
        'qa_accuracy': avg_qa_acc,
        'rhythm_accuracy': avg_rhythm_acc
    }


def validate_epoch(model, dataloader, device, qa_weight, rhythm_weight):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    total_qa_loss = 0
    total_rhythm_loss = 0
    total_qa_acc = 0
    total_rhythm_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            data = batch['data'].to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss
            loss, qa_loss, rhythm_loss = compute_loss(model, outputs, batch, device, qa_weight, rhythm_weight)
            
            # Compute accuracy
            qa_acc, rhythm_acc = compute_accuracy(outputs, batch)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_qa_loss += qa_loss.item()
            total_rhythm_loss += rhythm_loss.item()
            total_qa_acc += qa_acc
            total_rhythm_acc += rhythm_acc
            num_batches += 1
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_qa_loss = total_qa_loss / num_batches
    avg_rhythm_loss = total_rhythm_loss / num_batches
    avg_qa_acc = total_qa_acc / num_batches
    avg_rhythm_acc = total_rhythm_acc / num_batches
    
    return {
        'loss': avg_loss,
        'qa_loss': avg_qa_loss,
        'rhythm_loss': avg_rhythm_loss,
        'qa_accuracy': avg_qa_acc,
        'rhythm_accuracy': avg_rhythm_acc
    }


def main():
    # Check GPU status
    print("PYTORCH GPU STATUS")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60 + "\n")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Parse arguments
    args = parser_args()
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # Load training data
    print("Loading training data...")
    data_to_shuffle = load_training_data(args)
    data_train, label_train_r, label_train_q = shuffle_data(data_to_shuffle)
    print(f"Training data shape: {data_train.shape}")
    print(f"Training QA labels shape: {label_train_q.shape}")
    print(f"Training rhythm labels shape: {label_train_r.shape}\n")
    
    # Load validation data
    print("Loading validation data...")
    db_val = load_original_data(args.orig_data_path, 'validate.npz')
    data_val, label_val_r, label_val_q = db_val['data'], db_val['rhythm'], db_val['qa_label']
    print(f"Validation data shape: {data_val.shape}")
    print(f"Validation QA labels shape: {label_val_q.shape}")
    print(f"Validation rhythm labels shape: {label_val_r.shape}\n")
    
    # Create datasets
    train_dataset = DeepBeatDataset(data_train, label_train_q, label_train_r)
    val_dataset = DeepBeatDataset(data_val, label_val_q, label_val_r)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("Creating model...")
    model = DeepBeatModel().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Create optimizer (using original config from Keras model)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Setup TensorBoard
    writer = setup_tensorboard(args)
    
    # Training history
    history = {
        'loss': [],
        'qa_output_loss': [],
        'rhythm_output_loss': [],
        'qa_output_accuracy': [],
        'rhythm_output_accuracy': [],
        'val_loss': [],
        'val_qa_output_loss': [],
        'val_rhythm_output_loss': [],
        'val_qa_output_accuracy': [],
        'val_rhythm_output_accuracy': []
    }
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    print(f"Loss weights - QA: {args.qa_loss_weight}, Rhythm: {args.rhythm_loss_weight}")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, 
                                    args.qa_loss_weight, args.rhythm_loss_weight)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device,
                                     args.qa_loss_weight, args.rhythm_loss_weight)
        
        # Store history
        history['loss'].append(train_metrics['loss'])
        history['qa_output_loss'].append(train_metrics['qa_loss'])
        history['rhythm_output_loss'].append(train_metrics['rhythm_loss'])
        history['qa_output_accuracy'].append(train_metrics['qa_accuracy'])
        history['rhythm_output_accuracy'].append(train_metrics['rhythm_accuracy'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_qa_output_loss'].append(val_metrics['qa_loss'])
        history['val_rhythm_output_loss'].append(val_metrics['rhythm_loss'])
        history['val_qa_output_accuracy'].append(val_metrics['qa_accuracy'])
        history['val_rhythm_output_accuracy'].append(val_metrics['rhythm_accuracy'])
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('QA_Loss/train', train_metrics['qa_loss'], epoch)
        writer.add_scalar('QA_Loss/val', val_metrics['qa_loss'], epoch)
        writer.add_scalar('Rhythm_Loss/train', train_metrics['rhythm_loss'], epoch)
        writer.add_scalar('Rhythm_Loss/val', val_metrics['rhythm_loss'], epoch)
        writer.add_scalar('QA_Accuracy/train', train_metrics['qa_accuracy'], epoch)
        writer.add_scalar('QA_Accuracy/val', val_metrics['qa_accuracy'], epoch)
        writer.add_scalar('Rhythm_Accuracy/train', train_metrics['rhythm_accuracy'], epoch)
        writer.add_scalar('Rhythm_Accuracy/val', val_metrics['rhythm_accuracy'], epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"QA Acc: {train_metrics['qa_accuracy']:.4f}, "
              f"Rhythm Acc: {train_metrics['rhythm_accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"QA Acc: {val_metrics['qa_accuracy']:.4f}, "
              f"Rhythm Acc: {val_metrics['rhythm_accuracy']:.4f}")
    
    print("=" * 60)
    print("Training complete!\n")
    
    # Save model and history
    print("Saving model and history...")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_dir = output_path / Path(args.file_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, model_dir / (args.file_name + '.pth'))
    
    # Save history separately
    all_history = {
        'model_name': args.file_name + '.pth',
        'training_data': args.training_choice,
        'date': datetime.now().isoformat(),
        'qa_loss_weight': args.qa_loss_weight,
        'rhythm_loss_weight': args.rhythm_loss_weight,
        'history': history
    }
    
    with open(model_dir / (args.file_name + '_history.pkl'), 'wb') as file:
        pickle.dump(all_history, file)
    
    writer.close()
    print(f"Model and history saved to {model_dir}")


if __name__ == "__main__":
    main()
