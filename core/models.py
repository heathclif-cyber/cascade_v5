"""
core/models.py — Cascade v5 Model Definitions

AttentionLSTM: LSTM dengan self-attention + 3 output heads
  - momentum_strength  : seberapa kuat momentum saat ini
  - exhaustion_score   : seberapa dekat kondisi exhaustion/reversal
  - residual_correction: koreksi error LGBM (dilatih dari OOF residuals)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class TemporalAttention(nn.Module):
    """Self-attention over time steps — model belajar timestep mana yang paling relevan."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq_len)
        weights = F.softmax(scores, dim=1)                # (batch, seq_len)
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        return context, weights


class AttentionLSTM(nn.Module):
    """
    LSTM dengan temporal attention + 3 specialized output heads.

    Input : (batch, seq_len, n_features) — 20 trajectory features × 32 bars
    Output: tuple(momentum_strength, exhaustion_score, residual_correction)
            masing-masing shape (batch, 1)

    Training:
      - momentum_strength  : regression target (0–1)
      - exhaustion_score   : binary classification (0/1)
      - residual_correction: regression target (OOF residual dari LGBM)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)

        # Head 1: Momentum Strength (0–1, regression)
        self.momentum_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Head 2: Exhaustion Score (0–1, binary)
        self.exhaustion_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Head 3: Residual Correction (regression, bisa negatif/positif)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),   # output -1 to 1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        context, _  = self.attention(lstm_out)  # (batch, hidden)
        context     = self.dropout(context)

        momentum   = self.momentum_head(context)    # (batch, 1)
        exhaustion = self.exhaustion_head(context)  # (batch, 1)
        residual   = self.residual_head(context)    # (batch, 1)

        return momentum, exhaustion, residual


def load_attention_lstm(path: Path, input_size: int, device: str = "cpu") -> AttentionLSTM:
    from config import LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT
    model = AttentionLSTM(
        input_size  = input_size,
        hidden_size = LSTM_HIDDEN,
        num_layers  = LSTM_LAYERS,
        dropout     = LSTM_DROPOUT,
    )
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_momentum_expert(
    model: AttentionLSTM,
    X_seq: np.ndarray,          # (N, seq_len, n_features)
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inference CPU untuk seluruh dataset.
    Returns: momentum_strength, exhaustion_score, residual_correction — masing-masing (N,)
    """
    model.eval()
    device = torch.device("cpu")
    model  = model.to(device)

    all_mom, all_exh, all_res = [], [], []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            batch = torch.tensor(X_seq[i:i+batch_size], dtype=torch.float32).to(device)
            m, e, r = model(batch)
            all_mom.append(m.squeeze(-1).numpy())
            all_exh.append(e.squeeze(-1).numpy())
            all_res.append(r.squeeze(-1).numpy())

    return (
        np.concatenate(all_mom),
        np.concatenate(all_exh),
        np.concatenate(all_res),
    )
