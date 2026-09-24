import torch

from src.config import ModelConfig
from src.model import DecoderOnlyTransformer
from src.train import build_optimizer, train_step


def test_toy_training_step_updates_weights_and_loss() -> None:
    torch.manual_seed(13)
    config = ModelConfig(
        vocab_size=16,
        context_length=8,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
    )
    model = DecoderOnlyTransformer(config)
    optimizer = build_optimizer(model, learning_rate=0.03, weight_decay=0.0)
    inputs = torch.zeros((4, 8), dtype=torch.long)
    labels = torch.zeros_like(inputs)
    before = float(model(inputs, labels=labels).loss.item())
    train_step(model, [(inputs, labels)], optimizer, torch.device("cpu"), gradient_clip_norm=1.0)
    after = float(model(inputs, labels=labels).loss.item())
    assert torch.isfinite(torch.tensor(after))
    assert after < before
