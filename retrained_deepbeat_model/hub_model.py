"""
HuggingFace Hub-compatible wrapper for revised_DeepBeatModel.

Bundles per-QA thresholds and dropout config inside config.json so
weights + configuration are always pushed and loaded together.

Usage — load from Hub:
    from hub_model import DeepBeatHubModel
    model = DeepBeatHubModel.from_pretrained("your-username/revised-deepbeat")
    predictions = model.predict(signals)   # signals: np.ndarray (N, 800, 1)

Usage — push to Hub (see push_to_hub.py):
    model.push_to_hub("your-username/revised-deepbeat")
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pytorch'))
from revised_deepbeat_model import revised_DeepBeatModel


DEFAULT_DROPOUTS = {
    'do57': 0.118, 'do58': 0.545, 'do59': 0.568,
    'do60': 0.3,   'do61': 0.374, 'do62': 0.414, 'do63': 0.017,
}

DEFAULT_THRESHOLDS = {"0": 0.5, "1": 0.5, "2": 0.5}


class DeepBeatHubModel(revised_DeepBeatModel, PyTorchModelHubMixin):
    """
    revised_DeepBeatModel with HuggingFace Hub support.

    All __init__ args are serialised to config.json on push_to_hub()
    and restored automatically on from_pretrained().
    """

    def __init__(
        self,
        dropouts: dict = None,
        thresholds: dict = None,
    ):
        dropouts   = dropouts   or DEFAULT_DROPOUTS
        thresholds = thresholds or DEFAULT_THRESHOLDS

        revised_DeepBeatModel.__init__(self, dropouts=dropouts)

        # Store as plain dicts so they round-trip through config.json
        self.dropouts   = dropouts
        self.thresholds = {str(k): float(v) for k, v in thresholds.items()}

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        signals: np.ndarray,
        batch_size: int = 128,
        device: str = None,
    ) -> dict:
        """
        Run end-to-end inference with per-QA thresholds applied.

        Args:
            signals:    numpy array of shape (N, 800, 1)
            batch_size: number of samples per forward pass
            device:     'cuda' | 'cpu' | None (auto-detect)

        Returns:
            dict with keys:
                afib_prob        – AFib probability per sample (N,)
                afib_pred_per_qa – binary AFib prediction using per-QA thresholds (N,)
                afib_pred_default– binary AFib prediction at threshold=0.5 (N,)
                qa_pred          – predicted QA level 0/1/2 (N,)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device).eval()

        # (N, 800, 1) → (N, 1, 800)
        x = torch.FloatTensor(signals).permute(0, 2, 1)

        all_probs, all_rh_preds, all_qa_preds = [], [], []

        for i in range(0, len(x), batch_size):
            batch = x[i: i + batch_size].to(device)
            qa_logits, rh_logits = self(batch)

            probs   = F.softmax(rh_logits, dim=1)[:, 1].cpu().numpy()
            rh_pred = torch.argmax(rh_logits, dim=1).cpu().numpy()
            qa_pred = torch.argmax(qa_logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_rh_preds.extend(rh_pred)
            all_qa_preds.extend(qa_pred)

        probs   = np.array(all_probs)
        rh_pred = np.array(all_rh_preds)
        qa_pred = np.array(all_qa_preds)

        per_qa_pred = self._apply_thresholds(probs, qa_pred)

        return {
            'afib_prob':         probs,
            'afib_pred_per_qa':  per_qa_pred,
            'afib_pred_default': rh_pred,
            'qa_pred':           qa_pred,
        }

    def _apply_thresholds(self, probs: np.ndarray, qa_preds: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(probs), dtype=int)
        for level_str, thr in self.thresholds.items():
            mask = qa_preds == int(level_str)
            preds[mask] = (probs[mask] >= thr).astype(int)
        return preds
