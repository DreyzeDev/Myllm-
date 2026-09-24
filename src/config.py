from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 1024
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 1408
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("the attention head dimension must be even for RoPE")
        if min(self.context_length, self.num_layers, self.num_heads, self.intermediate_size) <= 0:
            raise ValueError("context_length, num_layers, num_heads and intermediate_size must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        config = cls(**values)
        config.validate()
        return config


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def model_config_dict(config: ModelConfig) -> dict[str, Any]:
    return asdict(config)
