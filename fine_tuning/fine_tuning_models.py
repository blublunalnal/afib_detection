import torch
import torch.nn as nn
from resnet1d import Net1D
from pathlib import Path
import numpy as np
import scipy.signal as ss
from torch.utils.data import Dataset
from ResNet1D_Net import Net

class DeepBeatDataset(Dataset):
    """PyTorch Dataset for DeepBeat data adapted for AnyPPG.

    Accepts either:
      - Raw data (N, L, C) @ 32Hz  — resampling + normalization applied on init
      - Preprocessed data (N, C, L) @ 125Hz — pass preprocessed=True to skip processing
    """

    def __init__(self, data, qa_labels, rhythm_labels, preprocessed=False, chunk_size=512):
        if preprocessed:
            # Data already resampled and normalized by preprocess.py: shape (N, C, L)
            self.data = torch.FloatTensor(data)
        else:
            # Process in chunks to avoid allocating the full resampled array at once.
            # Peak RAM per chunk: chunk_size * 800 * 1 (raw) + chunk_size * 3125 * 1 (resampled)
            N = data.shape[0]
            target_L = int(data.shape[1] * (125 / 32))  # 3125
            C = data.shape[2]

            self.data = torch.empty(N, C, target_L, dtype=torch.float32)

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk = ss.resample_poly(data[start:end], 125, 32, axis=1)  # (chunk, L', C)
                chunk_t = torch.FloatTensor(chunk).permute(0, 2, 1)         # (chunk, C, L')

                # Z-score normalization per sample along time axis
                mean = chunk_t.mean(dim=-1, keepdim=True)
                std  = chunk_t.std(dim=-1, keepdim=True)
                self.data[start:end] = (chunk_t - mean) / (std + 1e-8)

                del chunk, chunk_t  # free intermediates immediately

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
        elif backbone == 'pulseppg':
            encoder, output_size = self._load_pulseppg()
        else:
            raise ValueError("specify a backbone: anyppg / pulseppg")
        
        self.encoder = encoder
        self.output_size = output_size
        self._configure_freeze()

    def _configure_freeze(self):
        if self.encoder is None:
            return
        for param in self.encoder.parameters():
            param.requires_grad = not self._freeze
    
    def _load_pulseppg(self):
        
        # 1. Initialize the model structure
        model = Net(
            in_channels=1,
            base_filters=128,
            kernel_size=11,
            stride=2,
            groups=1,
            n_block=12,
            finalpool='max'
        )

        weights_path = Path(__file__).parent / "pulseppg_ckpt.pkl"
        checkpoint = torch.load(weights_path, map_location="cpu")

        # Extract the actual model weights using the "trained_net" key
        state_dict = checkpoint["net"]

        # Apply the weights to the model
        model.load_state_dict(state_dict, strict=True)
        return model, 512
        

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
        print(f'using backbone: {backbone}')

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
        print(f'using backbone: {backbone}')
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
        
    