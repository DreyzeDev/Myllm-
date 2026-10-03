from __future__ import annotations

import math
from pathlib import Path

from src.config import ModelConfig, load_config
from src.model import DecoderOnlyTransformer
from src.train import PROMPT_EVAL_PROMPTS


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARAMETERS = 109_529_856


def _expected_parameter_count(config: ModelConfig) -> int:
    # Tied token embeddings / LM head count once; all Linear layers are biasless.
    h = config.hidden_size
    per_layer = 4 * h * h + 3 * h * config.intermediate_size + 2 * h
    return config.vocab_size * h + config.num_layers * per_layer + h


def test_v2_config_is_valid_and_matches_target_architecture() -> None:
    values = load_config(ROOT / "configs" / "model_v2.yaml")
    config = ModelConfig.from_dict(values["model"])

    assert config.vocab_size == 32_000
    assert config.context_length == 1_024
    assert config.hidden_size == 768
    assert config.num_layers == 12
    assert config.num_heads == 12
    assert config.hidden_size // config.num_heads == 64
    assert config.intermediate_size == 2_048
    assert config.tie_embeddings is True
    assert _expected_parameter_count(config) == EXPECTED_PARAMETERS
    assert "checkpoint" not in values["training"]
    assert values["data"]["processed_dir"] == "data/processed/full_v1"
    assert values["training"]["max_steps"] == values["training"]["schedule_steps"]
    assert values["training"]["learning_rate"] == 2e-4
    assert values["training"]["gradient_accumulation_steps"] == 16
    effective_tokens_per_step = (
        values["training"]["batch_size"]
        * values["training"]["gradient_accumulation_steps"]
        * config.context_length
    )
    assert effective_tokens_per_step == 16_384
    assert values["training"]["max_steps"] == math.ceil(2_077_200_374 / effective_tokens_per_step)
    assert values["training"]["prompt_evaluation_prompts"] == [
        "Москва — столица",
        "Столица Франции —",
        "Земля вращается вокруг",
        "Солнце — это",
        "Солнечная система состоит из",
        "Вода при нормальном атмосферном давлении",
        "Python — это",
        "2 + 2 =",
        "7 * 8 =",
    ]
    assert PROMPT_EVAL_PROMPTS != tuple(values["training"]["prompt_evaluation_prompts"])


def test_v2_model_parameter_count_and_tied_weights() -> None:
    values = load_config(ROOT / "configs" / "model_v2.yaml")
    model = DecoderOnlyTransformer(ModelConfig.from_dict(values["model"]))

    assert model.num_parameters() == EXPECTED_PARAMETERS
    assert model.lm_head.weight is model.token_embeddings.weight
    assert model.config.context_length == 1_024
