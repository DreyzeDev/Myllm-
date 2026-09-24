from __future__ import annotations

import platform
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig
from src.model import DecoderOnlyTransformer
from src.train import build_optimizer, build_scheduler, save_checkpoint, train_step
from src.utils import count_parameters, load_model_checkpoint, select_device, set_seed


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA BF16 supported: {torch.cuda.is_bf16_supported()}")

    set_seed(1234)
    config = ModelConfig(
        vocab_size=300,
        context_length=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
    )
    device = select_device()
    model = DecoderOnlyTransformer(config).to(device)
    print(f"Sanity model parameters: {count_parameters(model):,}")
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, warmup_steps=1, max_steps=2, min_learning_rate_ratio=0.1)
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 16), device=device)
    initial = model(input_ids, labels=labels)
    if initial.loss is None or not torch.isfinite(initial.loss):
        raise RuntimeError("Forward/loss check failed")
    print(f"Initial loss: {initial.loss.item():.4f}")
    optimizer.zero_grad(set_to_none=True)
    initial.loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Backward pass did not produce gradients")
    optimizer.step()
    scheduler.step()
    after = model(input_ids, labels=labels)
    if after.loss is None or not torch.isfinite(after.loss):
        raise RuntimeError("Post-update forward check failed")
    print(f"Post-update loss: {after.loss.item():.4f}")

    with tempfile.TemporaryDirectory(prefix="my-llm-sanity-") as temporary:
        checkpoint = save_checkpoint(temporary, 1, model, optimizer, scheduler, None)
        restored = load_model_checkpoint(checkpoint, device)
        model.eval()
        with torch.no_grad():
            reference = model(input_ids).logits
            loaded = restored(input_ids).logits
        if not torch.allclose(reference, loaded, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Checkpoint reload output mismatch")
        required = {"model.pt", "optimizer.pt", "scheduler.pt", "training_state.json"}
        if not required.issubset({item.name for item in checkpoint.iterdir()}):
            raise RuntimeError("Checkpoint is missing expected files")
    print("Checkpoint save/load: OK")
    print("SANITY CHECK: PASS")


if __name__ == "__main__":
    main()

