"""LAB 07, section 9: sinusoidal positional encoding for batch-first tensors."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def positional_encoding(
    seq_len: int,
    d_model: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return PE with shape (seq_len, d_model); support odd feature counts."""
    if seq_len < 0 or d_model <= 0:
        raise ValueError("seq_len must be nonnegative and d_model must be positive")
    if not dtype.is_floating_point:
        raise TypeError("positional encoding requires a floating-point dtype")
    # Compute trigonometric functions in at least float32, including under AMP.
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    positions = torch.arange(seq_len, device=device, dtype=compute_dtype)[:, None]
    even_indices = torch.arange(0, d_model, 2, device=device, dtype=compute_dtype)
    angles = positions * torch.exp(-math.log(10000.0) * even_indices / d_model)
    pe = torch.empty(seq_len, d_model, device=device, dtype=compute_dtype)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])
    return pe.to(dtype=dtype)


class PositionalEncoding(nn.Module):
    """Add PE to x of shape (batch, tokens, d_model)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", positional_encoding(max_len, d_model).unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected (batch, tokens, {self.d_model}), got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("input embeddings must be floating point")
        if x.shape[1] > self.max_len:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_len={self.max_len}")
        return self.dropout(x + self.pe[:, : x.shape[1]].to(device=x.device, dtype=x.dtype))
