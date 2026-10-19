
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

LANDMARK_DIM = 2 * 21 * 3

def landmark_dim() -> int:
    return LANDMARK_DIM

class GestureCNNLSTM(nn.Module):

    def __init__(
        self,
        num_classes: int,
        input_dim: int = LANDMARK_DIM,
        conv_channels: Tuple[int, ...] = (128, 256, 256),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, conv_channels[0]),
            nn.LayerNorm(conv_channels[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        convs = []
        c_in = conv_channels[0]
        for c_out in conv_channels:
            convs.append(
                nn.Conv1d(c_in, c_out, kernel_size=3, padding=1, bias=False)
            )
            convs.append(nn.BatchNorm1d(c_out))
            convs.append(nn.GELU())
            convs.append(nn.Dropout(dropout))
            c_in = c_out
        self.conv_stack = nn.Sequential(*convs)

        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, t, _ = x.shape
        h = self.in_proj(x)
        h = h.transpose(1, 2)
        h = self.conv_stack(h)
        h = h.transpose(1, 2)
        out, _ = self.lstm(h)
        pooled, _ = out.max(dim=1)
        logits = self.head(pooled)
        return logits

class GestureCNNLSTMCompact(nn.Module):

    def __init__(
        self,
        num_classes: int,
        input_dim: int = LANDMARK_DIM,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        c0, c1, c2 = 128, 256, 256
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, c0),
            nn.LayerNorm(c0),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(c0, c1, 3, padding=1, bias=False),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            c2, 256, 2, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x).transpose(1, 2)
        h = self.conv(h).transpose(1, 2)
        out, _ = self.lstm(h)
        pooled, _ = out.max(dim=1)
        return self.head(pooled)

def build_gesture_model(
    num_classes: int,
    input_dim: int = LANDMARK_DIM,
    state: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:

    if state is not None and any(k.startswith("conv.") for k in state):
        return GestureCNNLSTMCompact(num_classes=num_classes, input_dim=input_dim)
    return GestureCNNLSTM(num_classes=num_classes, input_dim=input_dim)
