import torch

from src.attention import CausalSelfAttention


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

