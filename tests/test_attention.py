import torch

from src.attention import CausalSelfAttention, RotaryEmbedding


def test_rope_applies_expected_rotation_and_position_offset() -> None:
    rope = RotaryEmbedding(head_dim=4, theta=10000.0)
    values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    at_zero = rope(values, position_offset=0)
    at_one = rope(values, position_offset=1)

    assert torch.equal(at_zero, values)
    expected = torch.tensor(
        [[[[
            torch.cos(torch.tensor(1.0)) - 3.0 * torch.sin(torch.tensor(1.0)),
            2.0 * torch.cos(torch.tensor(0.01)) - 4.0 * torch.sin(torch.tensor(0.01)),
            3.0 * torch.cos(torch.tensor(1.0)) + torch.sin(torch.tensor(1.0)),
            4.0 * torch.cos(torch.tensor(0.01)) + 2.0 * torch.sin(torch.tensor(0.01)),
        ]]]]
    )
    assert torch.allclose(at_one, expected, atol=1e-6, rtol=1e-6)


def test_future_tokens_do_not_change_past_attention_outputs() -> None:
    torch.manual_seed(7)
    attention = CausalSelfAttention(hidden_size=32, num_heads=4).eval()
    first = torch.randn(2, 6, 32)
    second = first.clone()
    second[:, 5, :] += 50.0
    with torch.no_grad():
        output_a, _ = attention(first)
        output_b, _ = attention(second)
    assert torch.allclose(output_a[:, :5], output_b[:, :5], atol=1e-6, rtol=1e-6)


def test_kv_cache_matches_full_sequence() -> None:
    torch.manual_seed(11)
    attention = CausalSelfAttention(hidden_size=32, num_heads=4).eval()
    values = torch.randn(1, 5, 32)
    with torch.no_grad():
        full, _ = attention(values)
        prefix, cache = attention(values[:, :4], use_cache=True)
        last, _ = attention(values[:, 4:], past_key_value=cache, use_cache=True)
    assert torch.allclose(full[:, 4:], last, atol=1e-6, rtol=1e-6)
    assert cache is not None
    assert prefix.shape[1] == 4


def test_full_model_cached_logits_match_uncached_logits() -> None:
    from src.config import ModelConfig
    from src.model import DecoderOnlyTransformer

    torch.manual_seed(17)
    model = DecoderOnlyTransformer(
        ModelConfig(
            vocab_size=41,
            context_length=16,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            intermediate_size=64,
        )
    ).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))
    with torch.no_grad():
        full_logits = model(input_ids).logits
        prefix = model(input_ids[:, :3], use_cache=True)
        cache = prefix.past_key_values
        assert cache is not None
        for position in range(3, input_ids.shape[1]):
            output = model(input_ids[:, position : position + 1], past_key_values=cache, use_cache=True)
            assert torch.allclose(output.logits, full_logits[:, position : position + 1], atol=1e-5, rtol=1e-5)
            cache = output.past_key_values
