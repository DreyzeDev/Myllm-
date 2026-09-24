from __future__ import annotations

import argparse
import json
import math
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
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_index, (input_ids, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
    args = parser.parse_args()
    data_dir = Path(args.data)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset = TokenBlockDataset(data_dir / "validation.bin", metadata["validation_blocks"], metadata["block_length"])
    if not len(dataset):
        raise ValueError("Validation set is empty. Add more records or prepare data with a non-zero validation split.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = select_device()
    model = load_model_checkpoint(args.checkpoint, device)
    result = evaluate_model(model, loader, device, args.max_batches)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

