import json

import torch

from src.config import ModelConfig, model_config_dict
from src.model import DecoderOnlyTransformer
from src.utils import load_model_checkpoint


def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=97,
        context_length=32,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
    )


def test_model_shapes_and_parameter_count() -> None:
    model = DecoderOnlyTransformer(small_config())
    assert model.num_parameters() > 0
    for batch, sequence in ((1, 5), (3, 9)):
        result = model(torch.randint(0, 97, (batch, sequence)))
        assert result.logits.shape == (batch, sequence, 97)


def test_embeddings_are_randomly_initialized_and_tied() -> None:
    torch.manual_seed(2)
    first = DecoderOnlyTransformer(small_config())
    torch.manual_seed(3)
    second = DecoderOnlyTransformer(small_config())
    assert first.lm_head.weight.data_ptr() == first.token_embeddings.weight.data_ptr()
    assert not torch.equal(first.token_embeddings.weight, second.token_embeddings.weight)


def test_model_checkpoint_round_trip(tmp_path) -> None:
    config = small_config()
    model = DecoderOnlyTransformer(config).eval()
    checkpoint = tmp_path / "step_00001"
    checkpoint.mkdir()
    torch.save(model.state_dict(), checkpoint / "model.pt")
    (checkpoint / "model_config.json").write_text(json.dumps(model_config_dict(config)), encoding="utf-8")
    inputs = torch.randint(0, config.vocab_size, (2, 7))
    with torch.no_grad():
        expected = model(inputs).logits
        restored = load_model_checkpoint(checkpoint)
        actual = restored(inputs).logits
    assert torch.equal(expected, actual)
