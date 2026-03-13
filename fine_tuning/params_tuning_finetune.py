import warnings
warnings.filterwarnings('ignore')

import sys
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
from tqdm import tqdm

# -- local imports --
sys.path.insert(0, str(Path(__file__).parent))
from fine_tuning_models import DeepBeatDataset, FineTuning_rhythm, FineTuning_multitask
from train_finetune import run_epoch_rhythm, run_epoch_multitask

sys.path.insert(0, str(Path(__file__).parent.parent / 'pytorch'))
from utils import get_optimal_workers, load_pickle_file


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE = {
    "lr":           {"low": 1e-6, "high": 5e-4, "log": True},
    "weight_decay": {"low": 1e-3, "high": 0.5,  "log": True},
    "dropout":      {"low": 0.3,  "high": 0.8},
    "batch_size":   {"choices": [64, 128, 256]},
    # multitask-only (ignored for rhythm model)
    "qa_weight":    {"low": 0.1,  "high": 2.0},
    "rhythm_weight":{"low": 1.0,  "high": 10.0},
}


def load_search_space(raw) -> dict:
    """
    Parse the --search_space argument.

    Accepts:
      - None            → returns DEFAULT_SEARCH_SPACE
      - a file path     → reads JSON from that file, merges over defaults
      - a JSON string   → parses inline, merges over defaults

    Each key maps to either:
      {"low": float, "high": float}            # suggest_float, linear
      {"low": float, "high": float, "log": True}  # suggest_float, log-uniform
      {"choices": [v1, v2, ...]}               # suggest_categorical
    """
    if raw is None:
        return DEFAULT_SEARCH_SPACE.copy()

    path = Path(raw)
    if path.exists():
        with open(path) as f:
            user_space = json.load(f)
    else:
        try:
            user_space = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--search_space is neither a valid file path nor valid JSON: {e}"
            )

    merged = DEFAULT_SEARCH_SPACE.copy()
    merged.update(user_space)
    return merged


def suggest_from_spec(trial, name: str, spec: dict):
    """Dispatch to the correct Optuna suggest_* call based on spec keys."""
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    low  = spec["low"]
    high = spec["high"]
    log  = spec.get("log", False)
    return trial.suggest_float(name, low, high, log=log)


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for AnyPPG fine-tuning')

    # Data
    parser.add_argument("--train_data_path", required=True, help="Path to training pickle file")
    parser.add_argument("--val_data_path",   required=True, help="Path to validation pickle file")
    parser.add_argument("--output_path",     required=True, help="Root directory for outputs")

    # Model
    parser.add_argument("--model_type", required=True, choices=['rhythm', 'multitask'])
    parser.add_argument("--backbone", type=str, default='anyppg', help='anyppg / pulseppg')
    parser.add_argument("--freeze_backbone", action='store_true',
                        help="Freeze AnyPPG encoder during tuning")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
                        help="Backbone LR = lr * backbone_lr_scale when backbone is unfrozen "
                             "(same as train_finetune.py default)")

    # Search space
    parser.add_argument(
        "--search_space", type=str, default=None,
        help=(
            "Custom hyperparameter search space as a JSON string or path to a JSON file. "
            "Keys not specified fall back to defaults. "
            "Example (inline): "
            '\'{"lr": {"low": 1e-5, "high": 1e-3, "log": true}, '
            '"batch_size": {"choices": [32, 64]}}\' '
            "Example (file): path/to/search_space.json"
        ),
    )

    # Optuna settings
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name (defaults to finetune_{model_type}_study)")
    parser.add_argument("--n_trials",    type=int, default=30,  help="Number of trials")
    parser.add_argument("--n_epochs",    type=int, default=5,   help="Epochs per trial")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Max training batches per epoch (subsample large datasets for faster tuning). "
                             "e.g. 500 limits each epoch to 500 batches instead of the full ~11k")
    parser.add_argument("--resume",      action='store_true',  help="Resume existing study")

    # Device
    parser.add_argument("--device",      type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--num_workers", type=int, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets(args):
    """Load data once — reused across all trials to avoid repeated disk I/O."""
    print("Loading data...")

    train_dict = load_pickle_file(args.train_data_path)
    data_train  = train_dict['data']
    label_r_tr  = train_dict['rhythm_label']
    label_q_tr  = train_dict['qa_label']
    print(f"  Train shape: {data_train.shape}")

    val_dict   = load_pickle_file(args.val_data_path)
    data_val   = val_dict['data']
    label_r_val = val_dict['rhythm_label']
    label_q_val = val_dict['qa_label']
    print(f"  Val shape:   {data_val.shape}")

    is_preprocessed = train_dict.get('preprocessed', False)
    if is_preprocessed:
        print("  Preprocessed data detected — skipping resampling and normalization.")

    train_ds = DeepBeatDataset(data_train, label_q_tr,  label_r_tr,  preprocessed=is_preprocessed)
    val_ds   = DeepBeatDataset(data_val,   label_q_val, label_r_val, preprocessed=is_preprocessed)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

class EarlyStoppingOptuna:
    """Lightweight early stopping for within-trial use."""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def __call__(self, score):
        if self.best is None or score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class ProgressBarCallback:
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.pbar     = None

    def __call__(self, study, trial):
        if self.pbar is None:
            self.pbar = tqdm(total=self.n_trials, desc="Optuna trials")
        self.pbar.update(1)
        best = study.best_value if study.best_trial else 0.0
        self.pbar.set_postfix({
            'best': f"{best:.4f}",
            'last': f"{trial.value:.4f}" if trial.value is not None else "N/A",
        })

    def close(self):
        if self.pbar is not None:
            self.pbar.close()


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial):
    global DEVICE, TRAIN_DS, VAL_DS, NUM_WORKERS, ARGS, SEARCH_SPACE

    # --- Suggest hyperparameters from SEARCH_SPACE ---
    lr           = suggest_from_spec(trial, "lr",           SEARCH_SPACE["lr"])
    weight_decay = suggest_from_spec(trial, "weight_decay", SEARCH_SPACE["weight_decay"])
    dropout      = suggest_from_spec(trial, "dropout",      SEARCH_SPACE["dropout"])
    batch_size   = suggest_from_spec(trial, "batch_size",   SEARCH_SPACE["batch_size"])

    qa_weight     = 1.0
    rhythm_weight = 1.0
    if ARGS.model_type == 'multitask':
        qa_weight     = suggest_from_spec(trial, "qa_weight",     SEARCH_SPACE["qa_weight"])
        rhythm_weight = suggest_from_spec(trial, "rhythm_weight", SEARCH_SPACE["rhythm_weight"])

    # --- Model ---
    if ARGS.model_type == 'rhythm':
        model = FineTuning_rhythm(dropout=dropout, backbone=ARGS.backbone, freeze=ARGS.freeze_backbone).to(DEVICE)
    else:
        model = FineTuning_multitask(dropout=dropout, backbone=ARGS.backbone, freeze=ARGS.freeze_backbone).to(DEVICE)

    if ARGS.freeze_backbone:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        backbone_params = list(model.encoder.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]
        backbone_lr     = lr * ARGS.backbone_lr_scale
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params,     'lr': lr},
        ], weight_decay=weight_decay)

    # --- DataLoaders ---
    pin = (DEVICE.type == 'cuda')
    train_loader = DataLoader(TRAIN_DS, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader   = DataLoader(VAL_DS,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=(NUM_WORKERS > 0))

    # --- Training loop ---
    # Patience=2: overfitting onset was at epoch 2, no need to wait longer
    early_stopper = EarlyStoppingOptuna(patience=2, min_delta=0.001)
    best_val = 0.0

    epoch_bar = tqdm(range(1, ARGS.n_epochs + 1), desc=f"Trial {trial.number}", leave=False)
    for epoch in epoch_bar:
        if ARGS.model_type == 'rhythm':
            _       = run_epoch_rhythm(model, train_loader, optimizer, DEVICE, epoch,
                                       is_training=True,  max_batches=ARGS.max_batches, verbose=False)
            val_m   = run_epoch_rhythm(model, val_loader,   optimizer, DEVICE, epoch,
                                       is_training=False, verbose=False)
            metric  = val_m['f1']
        else:
            _       = run_epoch_multitask(model, train_loader, optimizer, DEVICE, epoch,
                                          qa_weight, rhythm_weight, is_training=True,
                                          max_batches=ARGS.max_batches, verbose=False)
            val_m   = run_epoch_multitask(model, val_loader,   optimizer, DEVICE, epoch,
                                          qa_weight, rhythm_weight, is_training=False,
                                          verbose=False)
            metric  = val_m['rhythm_f1']

        best_val = max(best_val, metric)
        epoch_bar.set_postfix({'val_f1': f"{metric:.4f}", 'best': f"{best_val:.4f}"})

        # Report to Optuna for pruning
        trial.report(metric, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if early_stopper(metric):
            break

    return best_val


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_study_results(study, output_path, study_name, model_type):
    output_dir = Path(output_path) / study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best = study.best_trial

    # Build ready_to_use block matching apply_tuned_params() in train_finetune.py
    ready = {
        'lr':           best.params['lr'],
        'weight_decay': best.params['weight_decay'],
        'dropout':      best.params['dropout'],
        'batch_size':   best.params['batch_size'],
    }
    if model_type == 'multitask':
        ready['qa_weight']     = best.params.get('qa_weight', 1.0)
        ready['rhythm_weight'] = best.params.get('rhythm_weight', 1.0)

    best_params = {
        'best_value':        best.value,
        'best_trial_number': best.number,
        'metric_optimized':  'f1' if model_type == 'rhythm' else 'rhythm_f1',
        'model_type':        model_type,
        'all_params':        best.params,
        'ready_to_use':      ready,
    }

    # 1. Best params JSON
    params_file = output_dir / f"{study_name}_best_params_{timestamp}.json"
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"  Best params  -> {params_file}")

    # 2. All trials JSON
    trials_file = output_dir / f"{study_name}_all_trials_{timestamp}.json"
    trials_data = [
        {
            'number':            t.number,
            'value':             t.value,
            'params':            t.params,
            'state':             t.state.name,
            'datetime_start':    t.datetime_start.isoformat() if t.datetime_start else None,
            'datetime_complete': t.datetime_complete.isoformat() if t.datetime_complete else None,
        }
        for t in study.trials
    ]
    with open(trials_file, 'w') as f:
        json.dump(trials_data, f, indent=2)
    print(f"  All trials   -> {trials_file}")

    # 3. Study pickle
    study_file = output_dir / f"{study_name}_study_{timestamp}.pkl"
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    print(f"  Study object -> {study_file}")

    # 4. Summary text
    summary_file = output_dir / f"{study_name}_summary_{timestamp}.txt"
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]
    top5      = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Optuna Study: {study_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model type:        {model_type}\n")
        f.write(f"Total trials:      {len(study.trials)}\n")
        f.write(f"Completed trials:  {len(completed)}\n")
        f.write(f"Pruned trials:     {len(pruned)}\n\n")
        f.write(f"Best trial:        #{best.number}\n")
        f.write(f"Best F1:           {best.value:.4f}\n\n")
        f.write("Best parameters:\n")
        for k, v in best.params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTop 5 trials:\n")
        for i, t in enumerate(top5, 1):
            f.write(f"\n{i}. Trial #{t.number} — F1: {t.value:.4f}\n")
            for k, v in t.params.items():
                f.write(f"   {k}: {v}\n")
    print(f"  Summary      -> {summary_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ARGS = parse_args()

    if ARGS.study_name is None:
        ARGS.study_name = f"finetune_{ARGS.model_type}_study"

    DEVICE       = torch.device(ARGS.device)
    NUM_WORKERS  = get_optimal_workers(ARGS.num_workers)
    SEARCH_SPACE = load_search_space(ARGS.search_space)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    print(f"Device:          {DEVICE}")
    print(f"Model type:      {ARGS.model_type}")
    print(f"Backbone frozen: {'YES' if ARGS.freeze_backbone else 'NO'}")

    # Load data once globally — shared across all trials
    print("\n" + "=" * 60)
    TRAIN_DS, VAL_DS = load_datasets(ARGS)
    print(f"Train samples: {len(TRAIN_DS)} | Val samples: {len(VAL_DS)}")
    print("=" * 60 + "\n")

    # SQLite storage for persistence and resume
    output_dir   = Path(ARGS.output_path) / ARGS.study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{output_dir / ARGS.study_name}.db"

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2, interval_steps=1)

    if ARGS.resume:
        print(f"Resuming study: {ARGS.study_name}")
        try:
            study = optuna.load_study(study_name=ARGS.study_name, storage=storage_name)
            print(f"  Resumed with {len(study.trials)} existing trials")
        except KeyError:
            print("  No existing study found — creating new one.")
            study = optuna.create_study(
                study_name=ARGS.study_name, direction='maximize',
                sampler=sampler, pruner=pruner,
                storage=storage_name, load_if_exists=True,
            )
    else:
        study = optuna.create_study(
            study_name=ARGS.study_name, direction='maximize',
            sampler=sampler, pruner=pruner,
            storage=storage_name, load_if_exists=False,
        )

    print(f"Optimization config:")
    print(f"  Study name:    {ARGS.study_name}")
    print(f"  Metric:        {'f1' if ARGS.model_type == 'rhythm' else 'rhythm_f1'} (maximize)")
    print(f"  Trials:        {ARGS.n_trials}")
    print(f"  Epochs/trial:  {ARGS.n_epochs}")
    print(f"  Storage:       {storage_name}")
    print(f"\nSearch space:")
    active_keys = list(SEARCH_SPACE.keys())
    if ARGS.model_type == 'rhythm':
        active_keys = [k for k in active_keys if k not in ('qa_weight', 'rhythm_weight')]
    for k in active_keys:
        spec = SEARCH_SPACE[k]
        if "choices" in spec:
            print(f"  {k:<14} choices={spec['choices']}")
        else:
            scale = "log-uniform" if spec.get("log") else "uniform"
            print(f"  {k:<14} [{spec['low']}, {spec['high']}]  {scale}")
    print("\nStarting optimization...\n")

    progress_cb = ProgressBarCallback(ARGS.n_trials)

    def periodic_save(study, trial):
        if trial.number > 0 and trial.number % 5 == 0:
            save_study_results(study, ARGS.output_path, ARGS.study_name, ARGS.model_type)

    try:
        study.optimize(
            objective,
            n_trials=ARGS.n_trials,
            callbacks=[progress_cb, periodic_save],
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted — saving results...")
    finally:
        progress_cb.close()

    # Final summary
    completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    pruned    = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Finished trials:  {len(study.trials)}")
    print(f"Completed:        {len(completed)}")
    print(f"Pruned:           {len(pruned)}")

    if completed:
        best = study.best_trial
        print(f"\nBest trial: #{best.number}")
        print(f"Best F1:    {best.value:.4f}")
        print(f"Params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

    print(f"\nSaving final results...")
    save_study_results(study, ARGS.output_path, ARGS.study_name, ARGS.model_type)

    print(f"\nTo use best params in training:")
    print(f"  --tuned_params_path {output_dir}/<best_params_file>.json")
    print(f"To resume this study:")
    print(f"  --resume --study_name {ARGS.study_name}")
