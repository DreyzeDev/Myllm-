from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tokenizer import ByteBPETokenizer


def find_input_files(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted([*source.rglob("*.txt"), *source.rglob("*.jsonl")])
    else:
        raise FileNotFoundError(f"Input path does not exist: {source}")
    if any(file.suffix.lower() not in {".txt", ".jsonl"} for file in files):
        raise ValueError("Only .txt and .jsonl input files are supported")
    if not files:
        raise ValueError(f"No .txt or .jsonl files found under {source}")
    return files


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def read_records(files: Iterable[Path], text_key: str = "text", deduplicate: bool = True) -> tuple[list[str], int]:
    records: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for file in files:
        if file.suffix.lower() == ".txt":
            contents = file.read_text(encoding="utf-8-sig")
            candidates = re.split(r"\n\s*\n", contents)
            if len(candidates) == 1:
                lines = [line for line in contents.splitlines() if line.strip()]
                if len(lines) > 1:
                    candidates = lines
        else:
            candidates = []
            with file.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {file}:{line_number}: {exc}") from exc
                    if isinstance(row, str):
                        candidates.append(row)
                    elif isinstance(row, dict) and isinstance(row.get(text_key), str):
                        candidates.append(row[text_key])
                    else:
                        raise ValueError(f"Expected a string or an object with string key {text_key!r} in {file}:{line_number}")
        for candidate in candidates:
            text = _normalize(candidate)
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if deduplicate and digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            records.append(text)
    return records, duplicates


def split_records(records: list[str], validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0 or not records:
        return records, []
    if len(records) == 1:
        words = records[0].split()
        if len(words) < 2:
            return records, []
        validation_count = max(1, round(len(words) * validation_fraction))
        validation_count = min(validation_count, len(words) - 1)
        split_at = len(words) - validation_count
        return [" ".join(words[:split_at])], [" ".join(words[split_at:])]
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    validation_count = max(1, round(len(records) * validation_fraction))
    validation_count = min(validation_count, len(records) - 1)
    validation_ids = set(order[:validation_count])
    return (
        [record for index, record in enumerate(records) if index not in validation_ids],
        [record for index, record in enumerate(records) if index in validation_ids],
    )


def _write_blocks(records: list[str], tokenizer: ByteBPETokenizer, path: Path, block_length: int) -> tuple[int, int]:
    eos_id = tokenizer.token_id("<eos>")
    pending: list[int] = []
    block_count = 0
    token_count = 0
    with path.open("wb") as output:
        for record in records:
            ids = tokenizer.encode(record)
            ids.append(eos_id)
            token_count += len(ids)
            pending.extend(ids)
            while len(pending) >= block_length:
                block = np.asarray(pending[:block_length], dtype=np.uint16)
                output.write(block.tobytes())
                pending = pending[block_length:]
                block_count += 1
    return block_count, token_count


def prepare_dataset(
    input_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    context_length: int,
    validation_fraction: float = 0.01,
    seed: int = 42,
    text_key: str = "text",
    deduplicate: bool = True,
) -> dict[str, object]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    files = find_input_files(input_path)
    records, duplicate_count = read_records(files, text_key, deduplicate)
    if not records:
        raise ValueError("No non-empty text records were found in the input files")
    train_records, validation_records = split_records(records, validation_fraction, seed)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size > 65536:
        raise ValueError("The uint16 dataset format supports vocabularies up to 65,536 tokens")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    block_length = context_length + 1
    train_blocks, train_tokens = _write_blocks(train_records, tokenizer, destination / "train.bin", block_length)
    val_blocks, val_tokens = _write_blocks(validation_records, tokenizer, destination / "validation.bin", block_length)
    metadata: dict[str, object] = {
        "format": "uint16_token_blocks",
        "context_length": context_length,
        "block_length": block_length,
        "train_blocks": train_blocks,
        "validation_blocks": val_blocks,
        "train_tokens_before_chunking": train_tokens,
        "validation_tokens_before_chunking": val_tokens,
        "records": len(records),
        "duplicates_removed": duplicate_count,
        "tokenizer": str(Path(tokenizer_path)),
        "vocab_size": tokenizer.vocab_size,
        "seed": seed,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, block_count: int, block_length: int) -> None:
        self.path = Path(path)
        self.block_count = block_count
        self.block_length = block_length
        expected_size = block_count * block_length * np.dtype(np.uint16).itemsize
        if not self.path.exists() or self.path.stat().st_size != expected_size:
            raise ValueError(f"Token file size does not match metadata: {self.path}")
        self._tokens = np.memmap(self.path, dtype=np.uint16, mode="r", shape=(block_count, block_length)) if block_count else None

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._tokens is None:
            raise IndexError("dataset contains no full token blocks")
        block = np.asarray(self._tokens[index], dtype=np.int64).copy()
        values = torch.from_numpy(block)
        return values[:-1], values[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare .txt/.jsonl pretraining data into next-token blocks")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--tokenizer", required=True, help="Path to trained tokenizer.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--keep-duplicates", action="store_true")
    args = parser.parse_args()
    metadata = prepare_dataset(
        args.input,
        args.tokenizer,
        args.output,
        args.context_length,
        args.validation_fraction,
        args.seed,
        args.text_key,
        deduplicate=not args.keep_duplicates,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
