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
    parser.add_argument("--chunk_size",  type=int, default=512,
                        help="Samples processed per chunk (lower = less peak RAM, default: 512)")
    return parser.parse_args()


def preprocess(data, chunk_size=512):
    """
    Args:
        data: np.ndarray of shape (N, L, C) @ 32Hz
        chunk_size: number of samples processed at a time to limit peak RAM

    Returns:
        np.ndarray of shape (N, C, L') @ 125Hz, z-score normalized
    """
    N        = data.shape[0]
    target_L = int(data.shape[1] * (125 / 32))  # 3125
    C        = data.shape[2]

    out = np.empty((N, C, target_L), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        chunk = ss.resample_poly(data[start:end], 125, 32, axis=1)  # (chunk, L', C)
        x     = torch.FloatTensor(chunk).permute(0, 2, 1)           # (chunk, C, L')

        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        out[start:end] = ((x - mean) / (std + 1e-8)).numpy()

        del chunk, x

        if (start // chunk_size) % 10 == 0:
            print(f"  Processed {end}/{N} samples...")

    return out


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

    print(f"Preprocessing (chunk_size={args.chunk_size}) ...")
    t0 = time.time()
    data_processed = preprocess(data, chunk_size=args.chunk_size)
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
