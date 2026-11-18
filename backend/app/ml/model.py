from __future__ import annotations

import torch
import torch.nn as nn


class BanglaGRUModel(nn.Module):
    """Placeholder BiGRU-CTC model definition for future training phases."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, num_classes: int = 100):
        super().__init__()
        self.input_dim = input_dim
        self.feature_head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_head(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        return self.classifier(x)
