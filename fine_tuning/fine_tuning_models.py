import torch
import torch.nn as nn
from resnet1d import Net1D
from pathlib import Path
import numpy as np
import scipy.signal as ss
from torch.utils.data import Dataset


class DeepBeatDataset(Dataset):
    """PyTorch Dataset for DeepBeat data adapted for AnyPPG.

    Accepts either:
      - Raw data (N, L, C) @ 32Hz  — resampling + normalization applied on init
      - Preprocessed data (N, C, L) @ 125Hz — pass preprocessed=True to skip processing
    """

    def __init__(self, data, qa_labels, rhythm_labels, preprocessed=False):
        if preprocessed:
            # Data already resampled and normalized by preprocess.py: shape (N, C, L)
            self.data = torch.FloatTensor(data)
        else:
            # 1. Upsample from 32Hz to 125Hz using polyphase filter (data shape: N, L, C)
            data_resampled = ss.resample_poly(data, 125, 32, axis=1)

            # Convert to tensor and permute to (N, C, L) -> (N, 1, 3125)
            self.data = torch.FloatTensor(data_resampled).permute(0, 2, 1)

            # 2. Z-score normalization along the time axis (dim=-1)
            mean = self.data.mean(dim=-1, keepdim=True)
            std  = self.data.std(dim=-1, keepdim=True)
            self.data = (self.data - mean) / (std + 1e-8)

        # Labels are integer class indices, not one-hot
        self.qa_labels = torch.LongTensor(qa_labels)
        self.rhythm_labels = torch.LongTensor(rhythm_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            # Returns shape (1, 3125)
            'data': self.data[idx],
            'qa_label': self.qa_labels[idx],
            'rhythm_label': self.rhythm_labels[idx]
        }


class BackboneBuilder(nn.Module):
    def __init__(self, backbone=None, freeze=False):
        super(BackboneBuilder, self).__init__()  # Fix: was super(FineTuning_rhythm, ...)
        self.encoder = None
        self.output_size = 0
        self._freeze = freeze
        if backbone == 'anyppg':
            encoder, output_size = self._load_anyppg()
            self.encoder = encoder
            self.output_size = output_size
        self._configure_freeze()

    def _configure_freeze(self):
        if self.encoder is None:
            return
        for param in self.encoder.parameters():
            param.requires_grad = not self._freeze

    def _load_anyppg(self):
        anyppg_cfg = {
            "in_channels": 1,
            "base_filters": 64,
            "ratio": 1.0,
            "filter_list": [64, 160, 160, 400, 400, 512],
            "m_blocks_list": [2, 2, 2, 3, 3, 1],
            "kernel_size": 3,
            "stride": 2,
            "groups_width": 16,
            "use_bn": True,
            "use_do": True,
            "verbose": False,
        }
        anyppg = Net1D(**anyppg_cfg)
        ckpt_path = Path(__file__).parent / "anyppg_ckpt.pth"
        state_dict = torch.load(ckpt_path, map_location="cpu")
        anyppg.load_state_dict(state_dict)
        return anyppg, 512  # Fix: was return self.anyppg (method ref)

    def forward(self, x):  # Fix: was def foward (typo)
        return self.encoder(x)

    def get_output_size(self):
        return self.output_size


class FineTuning_rhythm(nn.Module):
    def __init__(self, dropout=0.3, backbone='anyppg', freeze=False):
        super(FineTuning_rhythm, self).__init__()

        self.encoder = BackboneBuilder(backbone=backbone, freeze=freeze)  

        encoder_out = self.encoder.get_output_size()  # 512

        # Rhythm classification head:
        # Encoder outputs (N, 512) — already globally pooled
        self.rhythm_branch = nn.Sequential(
            nn.Linear(encoder_out, 175),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(175, 2),          
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.rhythm_branch(x)
        return x


class FineTuning_multitask(nn.Module):
    def __init__(self, dropout=0.3, backbone='anyppg', freeze=False):
        super(FineTuning_multitask, self).__init__()  

        self.encoder = BackboneBuilder(backbone=backbone, freeze=freeze)

        encoder_out = self.encoder.get_output_size()  # 512

        # Rhythm classification head:
        # Encoder outputs (N, 512) — already globally pooled
        self.rhythm_branch = nn.Sequential(
            nn.Linear(encoder_out, 175),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(175, 2),
        )

        self.qa_branch = nn.Sequential(
            nn.Linear(encoder_out, 175),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(175, 3),
        )

    def forward(self, x):  # Fix: was foward (typo)
        shared = self.encoder(x)
        rhythm_out = self.rhythm_branch(shared)
        qa_out = self.qa_branch(shared)  
        return rhythm_out, qa_out
        
    