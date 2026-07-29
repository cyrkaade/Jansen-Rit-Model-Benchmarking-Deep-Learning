"""Auditable extraction of the paper's EEG Transformer.

The computation matches ``deepjr.transformer.EEGTransformer`` at the published
configuration. Positional encoding is registered as a buffer so checkpoints
and device transfers are reliable; this does not change its numerical values.
"""

from __future__ import annotations

import torch
from torch import nn


class PublishedEEGTransformer(nn.Module):
    """Transformer-encoder-style regressor used by the upstream notebook."""

    def __init__(
        self,
        *,
        num_channels: int,
        num_timepoints: int,
        output_dim: int = 1,
        embed_dim: int = 256,
        num_heads: int = 8,
        intermediate_dim: int = 1024,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.input_proj = nn.Linear(num_channels, embed_dim)
        self.active_feature_gate = nn.Parameter(torch.ones(embed_dim))
        self.register_buffer(
            "positional_encoding",
            _positional_encoding(num_timepoints, embed_dim),
            persistent=True,
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return regression outputs for EEG shaped ``(batch, channels, time)``."""

        if eeg.ndim != 3:
            raise ValueError("eeg must have shape (batch, channels, time)")
        if eeg.shape[-1] != self.positional_encoding.shape[0]:
            raise ValueError("eeg time dimension does not match configured length")

        projected = self.input_proj(eeg.transpose(1, 2))
        projected = projected * torch.sigmoid(self.active_feature_gate)[None, None, :]
        projected = projected + self.positional_encoding[None, :, :]
        projected = projected.permute(1, 0, 2)

        attended, _ = self.multihead_attn(projected, projected, projected)
        attended = self.norm1(attended)
        feed_forward = self.ffn(attended)
        encoded = self.norm2(feed_forward + attended)
        return self.output_layer(encoded.mean(dim=0))


def _positional_encoding(num_timepoints: int, embed_dim: int) -> torch.Tensor:
    positions = torch.arange(num_timepoints, dtype=torch.float32)[:, None]
    even_dimensions = torch.arange(0, embed_dim, 2, dtype=torch.float32)
    divisors = torch.pow(10000.0, even_dimensions / embed_dim)
    angles = positions / divisors[None, :]
    encoding = torch.zeros((num_timepoints, embed_dim), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding
