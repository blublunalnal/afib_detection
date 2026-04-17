"""
One-time script to push the retrained revised_DeepBeatModel to HuggingFace Hub.

Steps:
  1. Load the .pth checkpoint and the per-QA thresholds JSON
  2. Build DeepBeatHubModel with those dropouts + thresholds
  3. Load the checkpoint weights
  4. Call push_to_hub() — uploads model.safetensors + config.json

Run:
    python push_to_hub.py \
        --repo_id        your-username/revised-deepbeat \
        --checkpoint_path revised_deepbeat_diff_branch_2_best.pth \
        --thresholds_path path/to/best_per_qa_thresholds.json \
        [--private]
"""

import argparse
import json
from pathlib import Path

import torch

from hub_model import DeepBeatHubModel


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",          required=True,
                        help="HuggingFace repo id, e.g. your-username/revised-deepbeat")
    parser.add_argument("--checkpoint_path",
                        default=str(here / "revised_deepbeat_diff_branch_2_best.pth"),
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--thresholds_path",  required=True,
                        help="Path to *_per_qa_thresholds.json from evaluate_per_qa_level.py")
    parser.add_argument("--private",          action="store_true",
                        help="Create the Hub repo as private")
    parser.add_argument("--device",           default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Load thresholds ----
    with open(args.thresholds_path) as f:
        thresholds = json.load(f)
    print("Thresholds:", thresholds)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    dropouts = None
    saved_hp = ckpt.get('hyperparameters', {})
    if isinstance(saved_hp, dict):
        saved_drops = saved_hp.get('dropouts')
        if saved_drops and saved_drops != 'default':
            dropouts = saved_drops
    print("Dropouts:", dropouts or "using model defaults")

    # ---- Build hub model ----
    model = DeepBeatHubModel(dropouts=dropouts, thresholds=thresholds)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Push ----
    print(f"\nPushing to: https://huggingface.co/{args.repo_id}")
    model.push_to_hub(
        repo_id=args.repo_id,
        private=args.private,
        commit_message="Upload revised_DeepBeatModel with per-QA thresholds",
    )
    print("Done.")
    print(f"\nLoad later with:")
    print(f"  from hub_model import DeepBeatHubModel")
    print(f"  model = DeepBeatHubModel.from_pretrained('{args.repo_id}')")
    print(f"  preds = model.predict(signals)  # signals: np.ndarray (N, 800, 1)")


if __name__ == "__main__":
    main()
