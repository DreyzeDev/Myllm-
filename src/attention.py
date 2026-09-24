from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


KVCache = tuple[torch.Tensor, torch.Tensor]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[-2]
        positions = torch.arange(position_offset, position_offset + seq_len, device=x.device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(device=x.device))
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos = embedding.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = embedding.sin().to(dtype=x.dtype)[None, None, :, :]
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta)
        self.dropout = dropout
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        query = self.rope(query, past_length)
        key = self.rope(key, past_length)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None
        key_length = key.shape[-2]
        if past_length == 0:
            attention_mask = None
            is_causal = True
        elif seq_len == 1:
            attention_mask = None
            is_causal = False
        else:
            query_positions = past_length + torch.arange(seq_len, device=hidden_states.device)
            key_positions = torch.arange(key_length, device=hidden_states.device)
            attention_mask = key_positions[None, :] <= query_positions[:, None]
            is_causal = False

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.residual_dropout(self.out_proj(attended)), present

