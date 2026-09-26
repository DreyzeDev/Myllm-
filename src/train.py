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
from src.generate import generate_tokens
from src.model import DecoderOnlyTransformer
from src.tokenizer import ByteBPETokenizer
from src.utils import configure_logging, count_parameters, select_device, set_seed


PROMPT_EVAL_PROMPTS = (
    "Москва — столица",
    "Земля вращается вокруг",
    "Солнечная система состоит из",
    "Вода при нормальном атмосферном давлении",
    "Python — это",
    "Столица Франции —",
    "2 + 2 =",
)


def evaluate_fixed_prompts(
    model: DecoderOnlyTransformer,
    tokenizer: ByteBPETokenizer,
    step: int,
    output_path: str | Path,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    try:
        model.eval()
        eos_token_id = tokenizer.token_id("<eos>")
        with output_file.open("a", encoding="utf-8") as log_file:
            for prompt in PROMPT_EVAL_PROMPTS:
                prompt_ids = tokenizer.encode(prompt, add_bos=True)
                output_ids = generate_tokens(
                    model,
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    seed=seed,
                    eos_token_id=eos_token_id,
                )
                continuation = tokenizer.decode(output_ids[len(prompt_ids) :])
                record = {
                    "step": step,
                    "prompt": prompt,
                    "continuation": continuation,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "seed": seed,
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                logging.info("prompt_eval step=%d prompt=%r continuation=%r", step, prompt, continuation)
    finally:
        if was_training:
            model.train()
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


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
    load_scheduler: bool = True,
) -> tuple[int, float | None, int]:
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True), strict=True)
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=device, weights_only=True))
    if load_scheduler:
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


def run_training(
    config_path: str | Path,
    resume_from: str | Path | None = None,
    reset_scheduler: bool = False,
) -> None:
    config = load_config(config_path)
    model_config = ModelConfig.from_dict(config["model"])
    training = config["training"]
    data_options = config["data"]
    if reset_scheduler and resume_from is None:
        raise ValueError("--reset-scheduler requires --resume-from")
    schedule_steps = int(training.get("schedule_steps", training["max_steps"]))
    if schedule_steps <= 0:
        raise ValueError("schedule_steps must be positive")
    validation_increase_patience = int(training.get("validation_increase_patience", 0))
    if validation_increase_patience < 0:
        raise ValueError("validation_increase_patience must not be negative")
    prompt_evaluation_interval = int(training.get("prompt_evaluation_interval", 0))
    if prompt_evaluation_interval < 0:
        raise ValueError("prompt_evaluation_interval must not be negative")
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
        schedule_steps,
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
            Path(resume_from), model, optimizer, scheduler, device, scaler, load_scheduler=not reset_scheduler
        )
        logging.info("resumed_from=%s step=%d", resume_from, start_step)
    if reset_scheduler:
        reset_at_step = training.get("reset_scheduler_at_step")
        if reset_at_step is not None and start_step != int(reset_at_step):
            raise ValueError(
                f"Scheduler reset is only allowed at step {int(reset_at_step)}, got checkpoint step {start_step}"
            )
        base_learning_rate = float(training["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = base_learning_rate
            group["initial_lr"] = base_learning_rate
        scheduler = build_scheduler(
            optimizer,
            int(training["warmup_steps"]),
            schedule_steps,
            float(training["min_learning_rate_ratio"]),
        )
        logging.info(
            "scheduler_reset=True schedule_steps=%d warmup_steps=%d base_lr=%.3g min_lr=%.3g",
            schedule_steps,
            int(training["warmup_steps"]),
            base_learning_rate,
            base_learning_rate * float(training["min_learning_rate_ratio"]),
        )
    else:
        logging.info(
            "scheduler_resume=checkpoint schedule_steps=%d warmup_steps=%d",
            schedule_steps,
            int(training["warmup_steps"]),
        )

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
    validation_history_by_step: dict[int, float] = {}
    if start_step and validation_increase_patience and log_path.exists():
        with log_path.open(encoding="utf-8") as existing_log:
            for line in existing_log:
                try:
                    prior_record = json.loads(line)
                    prior_step = int(prior_record["step"])
                    prior_loss = float(prior_record["validation_loss"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if prior_step <= start_step:
                    validation_history_by_step[prior_step] = prior_loss
    validation_history = [validation_history_by_step[step] for step in sorted(validation_history_by_step)]
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
    completed_step = start_step
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
            completed_step = step
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
            stop_for_validation_rise = False
            if validation_loader is not None and step % int(training["evaluation_interval"]) == 0:
                metrics = evaluate_model(
                    model, validation_loader, device, int(training["evaluation_batches"]), precision
                )
                if not math.isfinite(float(metrics["loss"])) or not math.isfinite(float(metrics["perplexity"])):
                    raise FloatingPointError(f"Non-finite validation metric at step {step}")
                record["validation_loss"] = metrics["loss"]
                record["perplexity"] = metrics["perplexity"]
                validation_history.append(float(metrics["loss"]))
                increase_streak = 0
                for index in range(len(validation_history) - 1, 0, -1):
                    if validation_history[index] > validation_history[index - 1]:
                        increase_streak += 1
                    else:
                        break
                if validation_increase_patience:
                    record["validation_increase_streak"] = increase_streak
                    stop_for_validation_rise = increase_streak >= validation_increase_patience
                logging.info("step=%d validation_loss=%.4f perplexity=%.2f", step, metrics["loss"], metrics["perplexity"])
                if best_validation_loss is None or metrics["loss"] < best_validation_loss:
                    best_validation_loss = metrics["loss"]
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            if step % int(training["save_interval"]) == 0 or stop_for_validation_rise:
                saved = save_checkpoint(
                    output_dir, step, model, optimizer, scheduler, best_validation_loss, scaler, tokens_trained
                )
                logging.info("checkpoint_saved=%s", saved)
                last_saved = step
            if prompt_evaluation_interval and step % prompt_evaluation_interval == 0:
                tokenizer = ByteBPETokenizer.load(training.get("prompt_evaluation_tokenizer", "tokenizer/tokenizer.json"))
                evaluate_fixed_prompts(
                    model,
                    tokenizer,
                    step,
                    output_dir / training.get("prompt_evaluation_log", "prompt_evaluations.jsonl"),
                    int(training.get("prompt_max_new_tokens", 64)),
                    float(training.get("prompt_temperature", 0.7)),
                    int(training.get("prompt_top_k", 50)),
                    float(training.get("prompt_top_p", 0.9)),
                    float(training.get("prompt_repetition_penalty", 1.15)),
                    int(training.get("prompt_seed", 42)),
                )
            if stop_for_validation_rise:
                logging.warning(
                    "stopping_after_validation_increase_streak=%d step=%d; checkpoint_saved=%d",
                    validation_increase_patience,
                    step,
                    last_saved,
                )
                break
        if completed_step > start_step and last_saved != completed_step:
            saved = save_checkpoint(
                output_dir, completed_step, model, optimizer, scheduler, best_validation_loss, scaler, tokens_trained
            )
            logging.info("checkpoint_saved=%s", saved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the from-scratch decoder-only model")
    parser.add_argument("--config", default="configs/model_v1.yaml")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="restore model/optimizer state but start a fresh schedule from the configured learning rate",
    )
    args = parser.parse_args()
    configure_logging()
    run_training(args.config, args.resume_from, reset_scheduler=args.reset_scheduler)


if __name__ == "__main__":
    main()
