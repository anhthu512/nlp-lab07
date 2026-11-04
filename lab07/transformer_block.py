"""LAB 07, sections 17-18: explicit multi-head attention and post-norm block."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    """Batch-first self-attention; True in a mask means a blocked position."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by positive n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_shapes: dict[str, tuple[int, ...]] = {}

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] == 0:
            raise ValueError(f"expected nonempty (batch, tokens, {self.d_model}) input")
        batch, length, _ = x.shape
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, length) or key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be boolean with shape (batch, tokens)")
            key_padding_mask = key_padding_mask.to(x.device)
        if attn_mask is not None:
            if attn_mask.shape != (length, length) or attn_mask.dtype != torch.bool:
                raise ValueError("attn_mask must be boolean with shape (tokens, tokens)")
            attn_mask = attn_mask.to(x.device)

        def split_heads(projection: nn.Linear) -> Tensor:
            return projection(x).reshape(batch, length, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = split_heads(self.q_proj), split_heads(self.k_proj), split_heads(self.v_proj)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        blocked = torch.zeros((batch, 1, length, length), dtype=torch.bool, device=x.device)
        if key_padding_mask is not None:
            blocked = blocked | key_padding_mask[:, None, None, :]
        if attn_mask is not None:
            blocked = blocked | attn_mask[None, None, :, :]
        # Fully masked rows stay finite in both forward and backward passes.
        all_blocked = blocked.all(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(blocked, float("-inf"))
        masked_scores = masked_scores.masked_fill(all_blocked, 0.0)
        weights = torch.softmax(masked_scores, dim=-1).masked_fill(blocked, 0.0)
        if key_padding_mask is not None:
            weights = weights.masked_fill(key_padding_mask[:, None, :, None], 0.0)
        context = self.dropout(weights) @ v
        merged = context.transpose(1, 2).contiguous().reshape(batch, length, self.d_model)
        output = self.out_proj(merged)
        # Also remove output-projection bias from positions with no available keys.
        output = output.masked_fill(all_blocked[:, 0], 0.0)
        if key_padding_mask is not None:
            output = output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        self.last_shapes = {
            "X": tuple(x.shape), "Q": tuple(q.shape), "K": tuple(k.shape),
            "V": tuple(v.shape), "Attention scores": tuple(scores.shape),
            "Attention weights": tuple(weights.shape),
            "Attention output (per head)": tuple(context.shape),
            "Concatenated heads": tuple(merged.shape), "Attention output": tuple(output.shape),
        }
        return output, weights if return_attention else None


class TransformerBlock(nn.Module):
    """Attention -> Add & LayerNorm -> FFN -> Add & LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.residual_dropout1 = nn.Dropout(dropout)
        self.residual_dropout2 = nn.Dropout(dropout)
        self.last_shapes: dict[str, tuple[int, ...]] = {}

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        return_attention: bool = False,
        attn_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        attention_output, weights = self.attention(
            x, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            return_attention=return_attention,
        )
        hidden = self.norm1(x + self.residual_dropout1(attention_output))
        ffn_hidden = self.activation(self.ffn_in(hidden))
        ffn_output = self.ffn_out(self.ffn_dropout(ffn_hidden))
        output = self.norm2(hidden + self.residual_dropout2(ffn_output))
        if key_padding_mask is not None:
            output = output.masked_fill(key_padding_mask.to(x.device).unsqueeze(-1), 0.0)
        self.last_shapes = {
            **self.attention.last_shapes,
            "Residual + LayerNorm 1": tuple(hidden.shape),
            "FFN hidden": tuple(ffn_hidden.shape), "FFN": tuple(ffn_output.shape),
            "Output": tuple(output.shape),
        }
        if return_attention:
            return output, weights
        return output
