import optuna
from optuna.trial import TrialState
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path

# Import your existing code
# Ensure deepbeat_model.py and train_pytorch_model.py are in the same folder
from deepbeat_model import DeepBeatModel
from train_pytorch_model import (
    load_training_data, 
    shuffle_data, 
    DeepBeatDataset, 
    run_epoch, 
    load_original_data
)


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
    
    args = parser.parse_args()
    return args

def get_data(device):
    """Load data once to avoid reloading every trial"""
    args = parser_args()
    print("Loading Data for Tuning...")
    
    # Load Train
    data_to_shuffle = load_training_data(args)
    data_train, label_train_r, label_train_q = shuffle_data(data_to_shuffle)
    
    # Load Val
    db_val = load_original_data(args.orig_data_path, 'validate.npz')
    data_val, label_val_r, label_val_q = db_val['data'], db_val['rhythm'], db_val['qa_label']

    # Create Datasets
    # Moving data to device immediately if it fits in VRAM can speed up tuning
    # otherwise keep it on CPU and move in loop (like in your original script)
    train_dataset = DeepBeatDataset(data_train, label_train_q, label_train_r)
    val_dataset = DeepBeatDataset(data_val, label_val_q, label_val_r)
    
    return train_dataset, val_dataset

def objective(trial):
    # 1. Define Hyperparameters to Tune
    
    # Dropouts (Independent for each layer allows architecture discovery)
    dropouts = {
        'do57': trial.suggest_float("do57", 0.0, 0.6),
        'do58': trial.suggest_float("do58", 0.2, 0.7),
        'do59': trial.suggest_float("do59", 0.2, 0.7),
        'do60': trial.suggest_float("do60", 0.3, 0.8), # QA Branch
        'do61': trial.suggest_float("do61", 0.2, 0.6), # Rhythm Branch
        'do62': trial.suggest_float("do62", 0.2, 0.6),
        'do63': trial.suggest_float("do63", 0.0, 0.5),
    }

    # Optimization Params
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # Loss Weights
    # If Rhythm is primary, we might want to tune how much QA matters
    qa_weight = trial.suggest_float("qa_weight", 0.1, 1.0)
    rhythm_weight = trial.suggest_float("rhythm_weight", 1.0, 8.0)

    # 2. Setup Model & Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with suggested dropouts
    model = DeepBeatModel(dropouts=dropouts).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Use the global datasets loaded outside
    train_loader = DataLoader(TRAIN_DS, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(VAL_DS, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Training Loop (Shortened for Tuning)
    # We don't need 100 epochs to know if params are bad. 15-20 is usually enough for pruning.
    n_epochs = 20 
    
    for epoch in range(1, n_epochs + 1):
        # We reuse your existing run_epoch function
        _ = run_epoch(model, train_loader, optimizer, device, epoch, 
                      qa_weight, rhythm_weight, is_training=True)
        
        val_metrics = run_epoch(model, val_loader, optimizer, device, epoch, 
                                qa_weight, rhythm_weight, is_training=False)
        
        # Metric to optimize: Rhythm Accuracy
        accuracy = val_metrics['rhythm_acc']

        # 4. Reporting & Pruning
        # Report the intermediate value to Optuna
        trial.report(accuracy, epoch)

        # Handle Pruning (Stop unpromising trials early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    # Load data once globally
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_DS, VAL_DS = get_data(device)

    # Create Study
    study = optuna.create_study(
        direction="maximize", # We want higher accuracy
        sampler=optuna.samplers.TPESampler(seed=42), # Bayesian sampler
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5) # Stop if worse than median
    )

    print("Starting optimization...")
    # n_trials: How many different combinations to try
    study.optimize(objective, n_trials=50, timeout=None) 

    # Print Results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n" + "="*40)
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Rhythm Acc): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")