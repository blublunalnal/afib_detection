"""
Preprocess DeepBeat data for AnyPPG fine-tuning.

Applies resampling (32Hz -> 125Hz) and z-score normalization once,
then saves the result to disk. Subsequent training/tuning runs load
the cached file instead of reprocessing.

Usage:
    python preprocess.py --input_path data/train.pkl --output_path data/train_preprocessed.pkl
    python preprocess.py --input_path data/val.pkl   --output_path data/val_preprocessed.pkl
"""

import sys
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'pytorch'))
from utils import load_pickle_file


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess DeepBeat data for AnyPPG fine-tuning")
    parser.add_argument("--input_path",  required=True, help="Path to raw pickle file")
    parser.add_argument("--output_path", required=True, help="Path to save preprocessed pickle file")
    return parser.parse_args()


def preprocess(data):
    """
    Args:
        data: np.ndarray of shape (N, L, C) @ 32Hz

    Returns:
        np.ndarray of shape (N, C, L') @ 125Hz, z-score normalized
    """
    # 1. Upsample 32Hz -> 125Hz along time axis
    print("  Resampling 32Hz -> 125Hz ...")
    data_resampled = ss.resample_poly(data, 125, 32, axis=1)
    print(f"  Shape after resampling: {data_resampled.shape}")

    # 2. Convert to tensor and permute to (N, C, L)
    x = torch.FloatTensor(data_resampled).permute(0, 2, 1)

    # 3. Z-score normalization along the time axis
    print("  Applying z-score normalization ...")
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std(dim=-1, keepdim=True)
    x    = (x - mean) / (std + 1e-8)

    return x.numpy()  # save as numpy, DataLoader handles tensor conversion


def main():
    args = parse_args()

    input_path  = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    data_dict = load_pickle_file(input_path)

    data           = data_dict['data']
    rhythm_labels  = data_dict['rhythm_label']
    qa_labels      = data_dict['qa_label']
    print(f"  Raw data shape: {data.shape}  ({len(rhythm_labels)} samples)")

    print("Preprocessing ...")
    t0 = time.time()
    data_processed = preprocess(data)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Processed shape: {data_processed.shape}")

    # Build output dict — preserves all original keys, replaces 'data'
    out_dict = {k: v for k, v in data_dict.items()}
    out_dict['data'] = data_processed
    out_dict['preprocessed'] = True
    out_dict['original_shape'] = data.shape
    out_dict['processed_shape'] = data_processed.shape

    print(f"Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(out_dict, f)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  Saved ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == '__main__':
    main()
