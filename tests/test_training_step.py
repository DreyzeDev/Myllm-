import json

import numpy as np
import torch
import yaml

from src.config import ModelConfig
from src.model import DecoderOnlyTransformer
from src.train import build_optimizer, run_training, train_step


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


def test_training_pipeline_saves_and_resumes_checkpoint(tmp_path) -> None:
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    block_length = 9
    train_blocks = np.asarray(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9, 10]],
        dtype=np.uint16,
    )
    validation_blocks = np.asarray([[3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.uint16)
    train_blocks.tofile(data_dir / "train.bin")
    validation_blocks.tofile(data_dir / "validation.bin")
    (data_dir / "metadata.json").write_text(
        json.dumps(
            {
                "block_length": block_length,
                "train_blocks": len(train_blocks),
                "validation_blocks": len(validation_blocks),
                "vocab_size": 16,
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config = {
        "model": {
            "vocab_size": 16,
            "context_length": block_length - 1,
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 32,
        },
        "data": {"processed_dir": str(data_dir)},
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.01,
            "min_learning_rate_ratio": 0.1,
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "max_steps": 1,
            "gradient_clip_norm": 1.0,
            "save_interval": 1,
            "evaluation_interval": 1,
            "evaluation_batches": 1,
            "log_interval": 1,
            "num_workers": 0,
            "precision": "fp32",
            "seed": 5,
            "output_dir": str(tmp_path / "checkpoints"),
        },
    }

    def write_config() -> None:
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    write_config()
    run_training(config_path)
    first_checkpoint = tmp_path / "checkpoints" / "step_00001"
    assert {"model.pt", "optimizer.pt", "scheduler.pt", "training_state.json"}.issubset(
        {path.name for path in first_checkpoint.iterdir()}
    )

    config["training"]["max_steps"] = 2
    write_config()
    run_training(config_path, resume_from=first_checkpoint)

    second_checkpoint = tmp_path / "checkpoints" / "step_00002"
    state = json.loads((second_checkpoint / "training_state.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (tmp_path / "checkpoints" / "training_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert state["step"] == 2
    assert [record["step"] for record in records] == [1, 2]
    assert "validation_loss" in records[-1]
    assert "perplexity" in records[-1]

    config["training"]["max_steps"] = 3
    config["training"]["schedule_steps"] = 2
    config["training"]["reset_scheduler_at_step"] = 2
    config["training"]["learning_rate"] = 0.005
    config["training"]["warmup_steps"] = 0
    write_config()
    run_training(config_path, resume_from=second_checkpoint, reset_scheduler=True)

    third_checkpoint = tmp_path / "checkpoints" / "step_00003"
    state = json.loads((third_checkpoint / "training_state.json").read_text(encoding="utf-8"))
    scheduler = torch.load(third_checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    assert state["step"] == 3
    assert scheduler["last_epoch"] == 1
    assert scheduler["base_lrs"] == [0.005, 0.005]
