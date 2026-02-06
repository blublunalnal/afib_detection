import optuna
from optuna.trial import TrialState
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import json
import pickle
from datetime import datetime
import os
import multiprocessing as mp

# Import your existing code
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
    parser.add_argument("--n_trials", type=int, default=30,
                       help="Number of trials (reduced from 50 for faster completion)")
    parser.add_argument("--n_epochs", type=int, default=5,
                       help="Epochs per trial (reduced from 20 for faster tuning)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of data loading workers (auto-detect if not specified)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume existing study if it exists")
    
    args = parser.parse_args()
    return args


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


def get_data(args, device):
    """Load data once to avoid reloading every trial"""
    print("Loading Data for Tuning...")
    
    # Load Train
    data_to_shuffle = load_training_data(args)
    data_train, label_train_r, label_train_q = shuffle_data(data_to_shuffle)
    
    # Load Val
    db_val = load_original_data(args.orig_data_path, 'validate.npz')
    data_val, label_val_r, label_val_q = db_val['data'], db_val['rhythm'], db_val['qa_label']

    # Create Datasets
    train_dataset = DeepBeatDataset(data_train, label_train_q, label_train_r)
    val_dataset = DeepBeatDataset(data_val, label_val_q, label_val_r)
    
    return train_dataset, val_dataset


class EarlyStopping:
    """Early stopping to terminate training when validation doesn't improve"""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


def objective(trial):
    """Objective function for Optuna optimization"""
    
    # 1. Define Hyperparameters to Tune
    
   
    # - Shared bottleneck: do57 (early) -> do58, do59 (late)
    # - QA branch: do60 (single dropout, non-primary task)
    # - Rhythm branch: do61, do62 (early) -> do63 (late, keep features)
    
    # Early shared feature extraction (do57)
    early_shared_dropout = trial.suggest_float("early_shared_dropout", 0.0, 0.3)
    
    # Late shared feature extraction (do58, do59)
    late_shared_dropout = trial.suggest_float("late_shared_dropout", 0.2, 0.7)
    
    # QA branch (do60) - can be higher since rhythm is primary
    qa_dropout = trial.suggest_float("qa_dropout", 0.3, 0.8)
    
    # Early rhythm branch (do61, do62)
    early_rhythm_dropout = trial.suggest_float("early_rhythm_dropout", 0.2, 0.6)
    
    # Late rhythm branch (do63) - typically low to preserve final features
    late_rhythm_dropout = trial.suggest_float("late_rhythm_dropout", 0.0, 0.3)
    
    # Map to original dropout dict structure
    dropouts = {
        'do57': early_shared_dropout,      # After conv1d_84 (shared bottleneck start)
        'do58': late_shared_dropout,       # After conv1d_85 (shared bottleneck mid)
        'do59': late_shared_dropout,       # After conv1d_86 (shared features)
        'do60': qa_dropout,                # QA branch
        'do61': early_rhythm_dropout,      # After conv1d_88 (rhythm start)
        'do62': early_rhythm_dropout,      # After conv1d_89 (rhythm mid)
        'do63': late_rhythm_dropout,       # After conv1d_90 (rhythm late)
    }

    # Optimization Params
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # Loss Weights - allow QA to vary more since rhythm is primary
    qa_weight = trial.suggest_float("qa_weight", 0.1, 1.5)
    rhythm_weight = trial.suggest_float("rhythm_weight", 1.0, 10.0)

    # 2. Setup Model & Data
    device = DEVICE  # Use global device
    
    # Initialize model with suggested dropouts
    model = DeepBeatModel(dropouts=dropouts).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Use the global datasets loaded outside
    # OPTIMIZED: Auto-detected optimal num_workers for faster data loading
    train_loader = DataLoader(TRAIN_DS, batch_size=batch_size, shuffle=True, 
                             num_workers=NUM_WORKERS, 
                             pin_memory=True if device.type=='cuda' else False,
                             persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(VAL_DS, batch_size=batch_size, shuffle=False, 
                           num_workers=NUM_WORKERS, 
                           pin_memory=True if device.type=='cuda' else False,
                           persistent_workers=True if NUM_WORKERS > 0 else False)

    # 3. Training Loop with Early Stopping
    n_epochs = ARGS.n_epochs  # Use from args (default 5)
    early_stopping = EarlyStopping(patience=2, min_delta=0.001)
    
    best_accuracy = 0.0
    
    for epoch in range(1, n_epochs + 1):
        # Training
        _ = run_epoch(model, train_loader, optimizer, device, epoch, 
                      qa_weight, rhythm_weight, is_training=True)
        
        # Validation
        val_metrics = run_epoch(model, val_loader, optimizer, device, epoch, 
                                qa_weight, rhythm_weight, is_training=False)
        
        # Metric to optimize: Rhythm Accuracy
        accuracy = val_metrics['rhythm_acc']
        best_accuracy = max(best_accuracy, accuracy)

        # 4. Reporting & Pruning
        trial.report(accuracy, epoch)

        # Handle Pruning (Stop unpromising trials early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping if validation plateaus
        if early_stopping(accuracy):
            print(f"  Early stopping at epoch {epoch}")
            break

    return best_accuracy


def save_study_results(study, output_path, study_name):
    """Save study results to files"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save best parameters as JSON
    best_params_file = output_dir / f"{study_name}_best_params_{timestamp}.json"
    
    # Expand grouped dropouts to individual values for easy use
    best_trial = study.best_trial
    expanded_dropouts = {
        'do57': best_trial.params.get('early_shared_dropout'),
        'do58': best_trial.params.get('late_shared_dropout'),
        'do59': best_trial.params.get('late_shared_dropout'),
        'do60': best_trial.params.get('qa_dropout'),
        'do61': best_trial.params.get('early_rhythm_dropout'),
        'do62': best_trial.params.get('early_rhythm_dropout'),
        'do63': best_trial.params.get('late_rhythm_dropout'),
    }
    
    best_params = {
        'best_value': study.best_trial.value,
        'best_trial_number': study.best_trial.number,
        'grouped_params': study.best_trial.params,
        'expanded_dropouts': expanded_dropouts,
        'ready_to_use': {
            'dropouts': expanded_dropouts,
            'lr': best_trial.params.get('lr'),
            'weight_decay': best_trial.params.get('weight_decay'),
            'batch_size': best_trial.params.get('batch_size'),
            'qa_weight': best_trial.params.get('qa_weight'),
            'rhythm_weight': best_trial.params.get('rhythm_weight'),
        }
    }
    
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n✓ Best parameters saved to: {best_params_file}")
    
    # 2. Save all trials data
    trials_file = output_dir / f"{study_name}_all_trials_{timestamp}.json"
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        })
    
    with open(trials_file, 'w') as f:
        json.dump(trials_data, f, indent=2)
    print(f"✓ All trials saved to: {trials_file}")
    
    # 3. Save study object as pickle
    study_file = output_dir / f"{study_name}_study_{timestamp}.pkl"
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    print(f"✓ Study object saved to: {study_file}")
    
    # 4. Save summary statistics
    summary_file = output_dir / f"{study_name}_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Optuna Study Summary: {study_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Study Statistics:\n")
        f.write(f"  Total trials: {len(study.trials)}\n")
        f.write(f"  Completed trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}\n")
        f.write(f"  Pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}\n\n")
        
        f.write(f"Best Trial:\n")
        f.write(f"  Trial number: {study.best_trial.number}\n")
        f.write(f"  Rhythm Accuracy: {study.best_trial.value:.4f}\n\n")
        
        f.write(f"Best Parameters (Grouped):\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nExpanded Dropout Values:\n")
        for key, value in expanded_dropouts.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Top 5 Trials:\n")
        f.write("="*60 + "\n")
        
        # Sort trials by value
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
        
        for i, trial in enumerate(sorted_trials, 1):
            f.write(f"\n{i}. Trial {trial.number} - Accuracy: {trial.value:.4f}\n")
            for key, value in trial.params.items():
                f.write(f"   {key}: {value}\n")
    
    print(f"✓ Summary saved to: {summary_file}")


if __name__ == "__main__":
    # Parse arguments
    ARGS = parser_args()
    
    # Setup output directory
    output_dir = Path(ARGS.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Determine optimal number of workers
    NUM_WORKERS = get_optimal_workers(ARGS.num_workers)
    
    # Load data once globally
    print("\n" + "="*60)
    TRAIN_DS, VAL_DS = get_data(ARGS, DEVICE)
    print(f"Train samples: {len(TRAIN_DS)}, Val samples: {len(VAL_DS)}")
    print("="*60 + "\n")

    # Create or load study with SQLite persistence (Optuna only supports database backends)
    storage_name = f"sqlite:///{output_dir / ARGS.study_name}.db"
    
    if ARGS.resume:
        print(f"Attempting to resume study: {ARGS.study_name}")
        try:
            study = optuna.load_study(
                study_name=ARGS.study_name,
                storage=storage_name
            )
            print(f"✓ Resumed study with {len(study.trials)} existing trials")
        except KeyError:
            print("No existing study found, creating new one...")
            study = optuna.create_study(
                study_name=ARGS.study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=3,      # num trials finish before pruning starts (STUDY-level)
                    n_warmup_steps=2,        # epochs each trial gets before it can be pruned (TRIAL-level)
                    interval_steps=1         # Check every epoch
                ),
                storage=storage_name,
                load_if_exists=True
            )
    else:
        study = optuna.create_study(
            study_name=ARGS.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3,
                n_warmup_steps=2,
                interval_steps=1
            ),
            storage=storage_name,
            load_if_exists=False
        )

    print(f"\nOptimization Configuration:")
    print(f"  Study name: {ARGS.study_name}")
    print(f"  Number of trials: {ARGS.n_trials}")
    print(f"  Epochs per trial: {ARGS.n_epochs}")
    print(f"  DataLoader workers: {NUM_WORKERS}")
    print(f"  Storage: {storage_name}")
    print(f"  Training data: {ARGS.training_choice}")
    print(f"\nDropout Strategy: Architecture-Aware (5 groups)")
    print(f"  - Early shared (do57)")
    print(f"  - Late shared (do58, do59)")
    print(f"  - QA branch (do60)")
    print(f"  - Early rhythm (do61, do62)")
    print(f"  - Late rhythm (do63)")
    print("\nStarting optimization...\n")
    
    # Optimize with callbacks for progress tracking
    try:
        study.optimize(
            objective, 
            n_trials=ARGS.n_trials,
            timeout=None,
            callbacks=[
                # Save intermediate results every 5 trials
                lambda study, trial: save_study_results(study, ARGS.output_path, ARGS.study_name) 
                if trial.number % 5 == 0 and trial.number > 0 else None
            ]
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user. Saving current results...")
    
    # Final results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    if complete_trials:
        print(f"\nBest trial:")
        trial = study.best_trial
        print(f"  Value (Rhythm Acc): {trial.value:.4f}")
        print(f"  Params (grouped):")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    
    # Save final results
    print("\n" + "="*60)
    save_study_results(study, ARGS.output_path, ARGS.study_name)
    print("="*60)
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"✓ To resume this study, use: --resume --study_name {ARGS.study_name}")
