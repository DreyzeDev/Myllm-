from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.attention import CausalSelfAttention, KVCache
from src.config import ModelConfig
from src.layers import RMSNorm, SwiGLU


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[tuple[KVCache, ...]] = None


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(
            config.hidden_size, config.num_heads, config.rope_theta, config.dropout
        )
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        attention_output, present = self.attention(self.attn_norm(hidden_states), past_key_value, use_cache)
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, present


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_module)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[KVCache, ...]] = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if past_length + input_ids.shape[1] > self.config.context_length:
            raise ValueError("input plus KV-cache exceeds configured context_length")
        hidden_states = self.embedding_dropout(self.token_embeddings(input_ids))
        presents: list[KVCache] = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, present = layer(hidden_states, past, use_cache)
            if present is not None:
                presents.append(present)
        logits = self.lm_head(self.final_norm(hidden_states))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1))
        return ModelOutput(logits, loss, tuple(presents) if use_cache else None)

