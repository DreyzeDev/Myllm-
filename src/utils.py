from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import ModelConfig
from src.model import DecoderOnlyTransformer


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_model_checkpoint(checkpoint_dir: str | Path, device: torch.device | str = "cpu") -> DecoderOnlyTransformer:
    path = Path(checkpoint_dir)
    config_path = path / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint is missing {config_path}")
    values: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    model = DecoderOnlyTransformer(ModelConfig.from_dict(values))
    state = torch.load(path / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model

