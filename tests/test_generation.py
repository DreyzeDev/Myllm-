import torch

from src.config import ModelConfig
from src.generate import generate_tokens
from src.model import DecoderOnlyTransformer


def test_generation_uses_kv_cache_and_slides_past_context_limit() -> None:
    torch.manual_seed(23)
    config = ModelConfig(
        vocab_size=31,
        context_length=8,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
    )
    model = DecoderOnlyTransformer(config).eval()
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    generated = generate_tokens(
        model,
        prompt,
        max_new_tokens=4,
        temperature=0,
        top_k=None,
        repetition_penalty=1.0,
    )

    assert generated[: config.context_length] == prompt[-config.context_length :]
    assert len(generated) == config.context_length + 4
    assert all(0 <= token_id < config.vocab_size for token_id in generated)
