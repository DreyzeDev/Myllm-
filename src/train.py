from __future__ import annotations

import argparse
import json
import logging
import math
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ModelConfig, load_config, model_config_dict
from src.dataset import TokenBlockDataset
from src.evaluate import evaluate_model
from src.model import DecoderOnlyTransformer
from src.utils import configure_logging, count_parameters, select_device, set_seed


def resolve_precision(requested: str, device: torch.device) -> str:
    if device.type != "cuda" or requested == "fp32":
        return "fp32"
    if requested == "bf16":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "fp16":
        return "fp16"
    if requested != "auto":
        raise ValueError("precision must be one of auto, bf16, fp16, fp32")
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_learning_rate_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_step(
    model: DecoderOnlyTransformer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "fp32",
    gradient_clip_norm: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_losses: list[float] = []
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("train_step received no batches")
    for input_ids, labels in batch_list:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, precision):
            output = model(input_ids, labels=labels)
            if not torch.isfinite(output.loss):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite training loss detected; optimizer step was skipped")
            loss = output.loss / len(batch_list)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        batch_losses.append(float(loss.detach().item()) * len(batch_list))
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not torch.isfinite(gradient_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite gradient norm detected; optimizer step was skipped")
    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return sum(batch_losses) / len(batch_losses)


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_validation_loss: float | None,
    scaler: torch.amp.GradScaler | None = None,
    tokens_trained: int = 0,
) -> Path:
    target = Path(output_dir) / f"step_{step:05d}"
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    torch.save(optimizer.state_dict(), target / "optimizer.pt")
    torch.save(scheduler.state_dict(), target / "scheduler.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), target / "scaler.pt")
    rng_state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, target / "rng.pt")
    (target / "model_config.json").write_text(
        json.dumps(model_config_dict(model.config), indent=2) + "\n", encoding="utf-8"
    )
    state = {
        "step": step,
        "best_validation_loss": best_validation_loss,
        "tokens_trained": tokens_trained,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "training_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return target


def _load_training_state(
    checkpoint_dir: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float | None, int]:
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True), strict=True)
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=device, weights_only=True))
    scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", map_location=device, weights_only=True))
    scaler_path = checkpoint_dir / "scaler.pt"
    if scaler is not None and scaler_path.exists():
        scaler.load_state_dict(torch.load(scaler_path, map_location="cpu", weights_only=True))
    state = json.loads((checkpoint_dir / "training_state.json").read_text(encoding="utf-8"))
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=True)
        torch.set_rng_state(rng_state["cpu"])
        if device.type == "cuda" and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    return int(state["step"]), state.get("best_validation_loss"), int(state.get("tokens_trained", 0))


def run_training(config_path: str | Path, resume_from: str | Path | None = None) -> None:
    config = load_config(config_path)
    model_config = ModelConfig.from_dict(config["model"])
    training = config["training"]
    data_options = config["data"]
    set_seed(int(training["seed"]))
    device = select_device()
    precision = resolve_precision(str(training["precision"]), device)
    logging.info("device=%s precision=%s", device, precision)

    model = DecoderOnlyTransformer(model_config).to(device)
    logging.info("trainable_parameters=%s", f"{count_parameters(model):,}")
    optimizer = build_optimizer(model, float(training["learning_rate"]), float(training["weight_decay"]))
    scheduler = build_scheduler(
        optimizer,
        int(training["warmup_steps"]),
        int(training["max_steps"]),
        float(training["min_learning_rate_ratio"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))

    data_dir = Path(data_options["processed_dir"])
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata["block_length"]) != model_config.context_length + 1:
        raise ValueError("Prepared data context_length does not match model.context_length in the config")
    if "vocab_size" in metadata and int(metadata["vocab_size"]) != model_config.vocab_size:
        raise ValueError("Prepared data tokenizer vocabulary does not match model.vocab_size in the config")
    train_data = TokenBlockDataset(data_dir / "train.bin", int(metadata["train_blocks"]), int(metadata["block_length"]))
    if not len(train_data):
        raise ValueError("Prepared training set is empty. Add more text or reduce context_length.")
    train_loader = DataLoader(
        train_data,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    validation_loader = None
    if int(metadata["validation_blocks"]) > 0:
        validation_data = TokenBlockDataset(
            data_dir / "validation.bin", int(metadata["validation_blocks"]), int(metadata["block_length"])
        )
        validation_loader = DataLoader(validation_data, batch_size=int(training["batch_size"]), shuffle=False)

    start_step = 0
    best_validation_loss: float | None = None
    tokens_trained = 0
    if resume_from is not None:
        start_step, best_validation_loss, tokens_trained = _load_training_state(
            Path(resume_from), model, optimizer, scheduler, device, scaler
        )
        logging.info("resumed_from=%s step=%d", resume_from, start_step)

    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_metrics_path = output_dir / "initial_metrics.json"
    if start_step == 0 and not initial_metrics_path.exists():
        initial_metrics = evaluate_model(model, train_loader, device, 1, precision)
        initial_metrics["step"] = 0
        if validation_loader is not None:
            initial_metrics["validation_loss"] = evaluate_model(model, validation_loader, device, 1, precision)["loss"]
        initial_metrics_path.write_text(json.dumps(initial_metrics, indent=2) + "\n", encoding="utf-8")
        logging.info("initial_train_loss=%.4f", initial_metrics["loss"])
    baseline_checkpoint = output_dir / "step_00000"
    if start_step == 0 and not baseline_checkpoint.exists():
        save_checkpoint(output_dir, 0, model, optimizer, scheduler, None, scaler, tokens_trained)
        logging.info("untrained_baseline_saved=%s", baseline_checkpoint)
    log_path = output_dir / "training_log.jsonl"
    log_mode = "a" if start_step else "w"
    data_iterator = iter(train_loader)

    def next_micro_batches(count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        nonlocal data_iterator
        result = []
        for _ in range(count):
            try:
                result.append(next(data_iterator))
            except StopIteration:
                data_iterator = iter(train_loader)
                result.append(next(data_iterator))
        return result

    last_saved = start_step
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        for step in range(start_step + 1, int(training["max_steps"]) + 1):
            batches = next_micro_batches(int(training["gradient_accumulation_steps"]))
            tokens_this_step = sum(int(labels.numel()) for _, labels in batches)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                loss = train_step(
                    model,
                    batches,
                    optimizer,
                    device,
                    precision,
                    float(training["gradient_clip_norm"]),
                    scaler,
                )
            except torch.cuda.OutOfMemoryError:
                logging.exception("CUDA OOM at step=%d; no checkpoint was overwritten", step)
                raise
            if not math.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            scheduler.step()
            tokens_trained += tokens_this_step
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "tokens_trained": tokens_trained,
            }
            if device.type == "cuda":
                record["cuda_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
                record["cuda_reserved_memory_bytes"] = int(torch.cuda.memory_reserved(device))
            if step % int(training["log_interval"]) == 0:
                logging.info("step=%d train_loss=%.4f lr=%.3g", step, loss, record["learning_rate"])
            if validation_loader is not None and step % int(training["evaluation_interval"]) == 0:
                metrics = evaluate_model(
                    model, validation_loader, device, int(training["evaluation_batches"]), precision
                )
                record["validation_loss"] = metrics["loss"]
                record["perplexity"] = metrics["perplexity"]
                logging.info("step=%d validation_loss=%.4f perplexity=%.2f", step, metrics["loss"], metrics["perplexity"])
                if best_validation_loss is None or metrics["loss"] < best_validation_loss:
                    best_validation_loss = metrics["loss"]
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            if step % int(training["save_interval"]) == 0:
                saved = save_checkpoint(
                    output_dir, step, model, optimizer, scheduler, best_validation_loss, scaler, tokens_trained
                )
                logging.info("checkpoint_saved=%s", saved)
                last_saved = step
        final_step = int(training["max_steps"])
        if final_step > start_step and last_saved != final_step:
            saved = save_checkpoint(
                output_dir, final_step, model, optimizer, scheduler, best_validation_loss, scaler, tokens_trained
            )
            logging.info("checkpoint_saved=%s", saved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the from-scratch decoder-only model")
    parser.add_argument("--config", default="configs/model_v1.yaml")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    configure_logging()
    run_training(args.config, args.resume_from)


if __name__ == "__main__":
    main()
