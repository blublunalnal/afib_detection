"""
Optuna Hyperparameter Tuning for DeepBeat PyTorch Model

This script integrates with your existing train_pytorch_model.py to tune:
1. Learning rate
2. Batch size  
3. Loss weights (QA vs Rhythm)
4. L2 regularization weights (optional - see discussion below)

Usage:
    python optuna_tune_deepbeat.py \
        --training_choice db_orig_replaced \
        --n_trials 50 \
        --study_name deepbeat_tuning
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import optuna
from optuna.trial import Trial, TrialState

# Import from your existing training script
from train_pytorch_model import (
    DeepBeatDataset,
    load_original_data,
    load_relabeled_data,
    load_substituted_relabeled_data,
    replace_updated_subjects_db,
    attach_VSM,
    shuffle_data,
    compute_accuracy
)

# Import model
from deepbeat_model import DeepBeatModel


def parser_args():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Tuning for DeepBeat')
    
    # Data paths (same as original training script)
    parser.add_argument("--orig_data_path", default=r'C:\Users\aoara\develop\deepbeat\data\original_data')
    parser.add_argument("--relabled_path", default=r'C:\Users\aoara\develop\deepbeat\data\relabeled_data')
    parser.add_argument("--db_orig_replaced_path", default=r"C:\Users\aoara\develop\deepbeat\output\replace_relabeled.pkl")
    
    # Output path
    parser.add_argument("--output_path", default=r'C:\Users\aoara\develop\deepbeat\optuna_studies')
    
    # Training data choice
    valid_choices = ['db_orig', 'db_relabel', 'db_relabel_w_vsm', 'db_orig_replaced', 'db_orig_replaced_w_vsm']
    parser.add_argument("--training_choice", choices=valid_choices, required=True,
                       help="Training data choice")
    
    # Optuna parameters
    parser.add_argument("--study_name", type=str, default="deepbeat_study",
                       help="Name of the Optuna study")
    parser.add_argument("--n_trials", type=int, default=50,
                       help="Number of trials to run")
    parser.add_argument("--n_epochs", type=int, default=30,
                       help="Number of epochs per trial (use fewer for faster tuning)")
    
    # What to tune
    parser.add_argument("--tune_l2", action='store_true',
                       help="Whether to tune L2 regularization weights (see recommendations)")
    parser.add_argument("--tune_architecture", action='store_true',
                       help="Whether to tune architecture (filters, kernels, etc.)")
    
    # Device
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    return args


def load_training_data(args):
    """Load training data based on choice (same as original script)"""
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


def compute_loss_tunable(model, outputs, targets, device, qa_weight, rhythm_weight, l2_scale=1.0):
    """
    Compute loss with tunable L2 scale
    
    Args:
        l2_scale: Multiplier for L2 regularization (1.0 = use original weights)
    """
    qa_target = targets['qa_label'].to(device)
    rhythm_target = targets['rhythm_label'].to(device)
    
    qa_loss = nn.CrossEntropyLoss()(outputs['qa_output'], qa_target)
    rhythm_loss = nn.BCELoss()(outputs['rhythm_output'], rhythm_target)
    
    # L2 regularization with optional scaling
    l2_reg = model.get_l2_regularization() * l2_scale
    
    total_loss = qa_weight * qa_loss + rhythm_weight * rhythm_loss + l2_reg
    
    return total_loss, qa_loss, rhythm_loss


def objective(trial: Trial, args, train_loader, val_loader, device):
    """
    Objective function for Optuna optimization
    
    This function is called for each trial and returns the metric to optimize.
    """
    
    # =====================================================================
    # HYPERPARAMETER SEARCH SPACE
    # =====================================================================
    
    # 1. Learning rate (IMPORTANT - always tune this)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # 2. Batch size (IMPORTANT - affects training dynamics)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    # 3. Loss weights (VERY IMPORTANT for multi-task learning with relabeled data)
    qa_loss_weight = trial.suggest_float('qa_loss_weight', 0.05, 2.0)
    rhythm_loss_weight = trial.suggest_float('rhythm_loss_weight', 0.5, 15.0)
    
    # 4. L2 regularization scale (optional - see discussion below)
    if args.tune_l2:
        # Instead of tuning each L2 weight individually, tune a global scale factor
        # This maintains relative ratios while allowing overall strength adjustment
        l2_scale = trial.suggest_float('l2_scale', 0.1, 5.0)
    else:
        l2_scale = 1.0  # Use original L2 weights
    
    # 5. Architecture hyperparameters (optional - usually not needed for retraining)
    # Skip this unless you have reason to believe architecture needs changing
    
    # =====================================================================
    # RECREATE DATALOADERS WITH TRIAL BATCH SIZE
    # =====================================================================
    
    # Get datasets from existing loaders
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # Create new dataloaders with trial batch size
    trial_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    trial_val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # =====================================================================
    # CREATE MODEL
    # =====================================================================
    
    model = DeepBeatModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    
    best_f1 = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_qa_acc = 0
        train_rhythm_acc = 0
        num_train_batches = 0
        
        for batch in trial_train_loader:
            data = batch['data'].to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            loss, qa_loss, rhythm_loss = compute_loss_tunable(
                model, outputs, batch, device,
                qa_loss_weight, rhythm_loss_weight, l2_scale
            )
            
            loss.backward()
            optimizer.step()
            
            qa_acc, rhythm_acc = compute_accuracy(outputs, batch)
            
            train_loss += loss.item()
            train_qa_acc += qa_acc
            train_rhythm_acc += rhythm_acc
            num_train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_qa_acc = 0
        val_rhythm_acc = 0
        num_val_batches = 0
        
        all_rhythm_preds = []
        all_rhythm_true = []
        
        with torch.no_grad():
            for batch in trial_val_loader:
                data = batch['data'].to(device)
                outputs = model(data)
                
                loss, _, _ = compute_loss_tunable(
                    model, outputs, batch, device,
                    qa_loss_weight, rhythm_loss_weight, l2_scale
                )
                
                qa_acc, rhythm_acc = compute_accuracy(outputs, batch)
                
                val_loss += loss.item()
                val_qa_acc += qa_acc
                val_rhythm_acc += rhythm_acc
                num_val_batches += 1
                
                # Collect predictions for F1 calculation
                rhythm_pred = torch.argmax(outputs['rhythm_output'], dim=1).cpu().numpy()
                rhythm_true = torch.argmax(batch['rhythm_label'], dim=1).numpy()
                all_rhythm_preds.extend(rhythm_pred)
                all_rhythm_true.extend(rhythm_true)
        
        # Calculate metrics
        avg_val_loss = val_loss / num_val_batches
        avg_val_rhythm_acc = val_rhythm_acc / num_val_batches
        
        # Calculate F1 score (our optimization target)
        val_f1 = f1_score(all_rhythm_true, all_rhythm_preds, average='weighted')
        
        # Report intermediate value for pruning
        trial.report(val_f1, epoch)
        
        # Handle pruning (stop unpromising trials early)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Trial {trial.number} - Early stopping at epoch {epoch}")
                break
        
        # Log progress
        if epoch % 5 == 0:
            print(f"Trial {trial.number}, Epoch {epoch}/{args.n_epochs}: "
                  f"Val Loss={avg_val_loss:.4f}, Val Rhythm Acc={avg_val_rhythm_acc:.4f}, "
                  f"Val F1={val_f1:.4f}")
    
    return best_f1


def main():
    # Parse arguments
    args = parser_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # =====================================================================
    # LOAD DATA (only once, reused across all trials)
    # =====================================================================
    
    print("Loading training data...")
    data_to_shuffle = load_training_data(args)
    data_train, label_train_r, label_train_q = shuffle_data(data_to_shuffle)
    print(f"Training data shape: {data_train.shape}")
    
    print("Loading validation data...")
    db_val = load_original_data(args.orig_data_path, 'validate.npz')
    data_val, label_val_r, label_val_q = db_val['data'], db_val['rhythm'], db_val['qa_label']
    print(f"Validation data shape: {data_val.shape}\n")
    
    # Create datasets (will be reused with different batch sizes)
    train_dataset = DeepBeatDataset(data_train, label_train_q, label_train_r)
    val_dataset = DeepBeatDataset(data_val, label_val_q, label_val_r)
    
    # Create initial dataloaders (batch size will be changed per trial)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # =====================================================================
    # CREATE OPTUNA STUDY
    # =====================================================================
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    study_path = output_path / f"{args.study_name}.db"
    
    # Create study with pruning
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f'sqlite:///{study_path}',  # Save progress to database
        load_if_exists=True,  # Resume if interrupted
        direction='maximize',  # Maximize F1 score
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # Don't prune first 5 trials
            n_warmup_steps=5      # Evaluate at least 5 epochs
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # =====================================================================
    # RUN OPTIMIZATION
    # =====================================================================
    
    print("=" * 60)
    print("STARTING HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Tune L2 weights: {args.tune_l2}")
    print(f"Training choice: {args.training_choice}")
    print("=" * 60 + "\n")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, train_loader, val_loader, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
        callbacks=[
            # Save after each trial
            lambda study, trial: save_intermediate_results(study, output_path, args.study_name)
        ]
    )
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_value:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    best_params_path = output_path / f"{args.study_name}_best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_trial': study.best_trial.number,
            'best_f1_score': study.best_value,
            'best_params': study.best_params,
            'training_choice': args.training_choice,
            'date': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  - {best_params_path}")
    print(f"  - {study_path}")
    
    # Generate visualizations
    generate_visualizations(study, output_path, args.study_name)
    
    # Print command to use best hyperparameters
    print("\n" + "=" * 60)
    print("TO TRAIN WITH BEST HYPERPARAMETERS:")
    print("=" * 60)
    print(f"python train_pytorch_model.py \\")
    print(f"    --file_name {args.study_name}_final \\")
    print(f"    --training_choice {args.training_choice} \\")
    print(f"    --epochs 150 \\")
    print(f"    --batch_size {study.best_params['batch_size']} \\")
    print(f"    --learning_rate {study.best_params['learning_rate']:.6f} \\")
    print(f"    --qa_loss_weight {study.best_params['qa_loss_weight']:.4f} \\")
    print(f"    --rhythm_loss_weight {study.best_params['rhythm_loss_weight']:.4f}")


def save_intermediate_results(study, output_path, study_name):
    """Save intermediate results after each trial"""
    if len(study.trials) % 5 == 0:  # Save every 5 trials
        intermediate_path = output_path / f"{study_name}_intermediate.json"
        with open(intermediate_path, 'w') as f:
            json.dump({
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number,
                'best_value': study.best_value,
                'best_params': study.best_params
            }, f, indent=2)


def generate_visualizations(study, output_path, study_name):
    """Generate Optuna visualizations"""
    try:
        import optuna.visualization as vis
        
        print("\nGenerating visualizations...")
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(output_path / f"{study_name}_optimization_history.html")
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(output_path / f"{study_name}_param_importances.html")
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(output_path / f"{study_name}_parallel_coordinate.html")
        
        # Slice plot
        fig = vis.plot_slice(study)
        fig.write_html(output_path / f"{study_name}_slice.html")
        
        print("Visualizations saved:")
        print(f"  - {study_name}_optimization_history.html")
        print(f"  - {study_name}_param_importances.html")
        print(f"  - {study_name}_parallel_coordinate.html")
        print(f"  - {study_name}_slice.html")
        
    except ImportError:
        print("\nInstall plotly for visualizations: pip install plotly")


if __name__ == '__main__':
    main()
