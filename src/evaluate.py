from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.dataset import TokenBlockDataset
from src.utils import load_model_checkpoint, select_device


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    precision: str = "fp32",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_index, (input_ids, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda" and precision in {"bf16", "fp16"}:
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            autocast = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            autocast = nullcontext()
        with autocast:
            output = model(input_ids, labels=labels)
        token_count = labels.numel()
        total_loss += float(output.loss.item()) * token_count
        total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("Validation dataset has no batches")
    loss = total_loss / total_tokens
    return {"loss": loss, "perplexity": math.exp(min(loss, 20.0))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on prepared validation blocks")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    data_dir = Path(args.data)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset = TokenBlockDataset(data_dir / "validation.bin", metadata["validation_blocks"], metadata["block_length"])
    if not len(dataset):
        raise ValueError("Validation set is empty. Add more records or prepare data with a non-zero validation split.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = select_device()
    precision = "fp32"
    if device.type == "cuda":
        precision = "bf16" if args.precision == "auto" and torch.cuda.is_bf16_supported() else args.precision
        if precision == "auto":
            precision = "fp16"
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            precision = "fp16"
    model = load_model_checkpoint(args.checkpoint, device)
    result = evaluate_model(model, loader, device, args.max_batches, precision)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

