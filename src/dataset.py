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


def iter_record_texts(files: Iterable[Path], text_key: str = "text") -> Iterator[str]:
    """Read records one at a time so tokenizer/data preparation need not hold a corpus in RAM."""
    for text, _ in iter_record_rows(files, text_key):
        yield text


def iter_record_rows(files: Iterable[Path], text_key: str = "text") -> Iterator[tuple[str, str | None]]:
    """Stream normalized text with an optional source language label."""
    for file in files:
        if file.suffix.lower() == ".txt":
            contents = file.read_text(encoding="utf-8-sig")
            candidates = re.split(r"\n\s*\n", contents)
            if len(candidates) == 1:
                lines = [line for line in contents.splitlines() if line.strip()]
                if len(lines) > 1:
                    candidates = lines
            for candidate in candidates:
                text = _normalize(candidate)
                if text:
                    yield text, None
            continue
        with file.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {file}:{line_number}: {exc}") from exc
                if isinstance(row, str):
                    candidate = row
                    language = None
                elif isinstance(row, dict) and isinstance(row.get(text_key), str):
                    candidate = row[text_key]
                    language = row.get("language") if isinstance(row.get("language"), str) else None
                else:
                    raise ValueError(f"Expected a string or an object with string key {text_key!r} in {file}:{line_number}")
                text = _normalize(candidate)
                if text:
                    yield text, language


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
    writer = _TokenBlockWriter(tokenizer, path, block_length)
    for record in records:
        writer.write(record)
    return writer.close()


class _TokenBlockWriter:
    def __init__(self, tokenizer: ByteBPETokenizer, path: Path, block_length: int) -> None:
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.token_id("<eos>")
        self.path = path
        self.block_length = block_length
        self.pending: list[int] = []
        self.block_count = 0
        self.token_count = 0
        self.output = path.open("wb")

    def write(self, record: str) -> int:
        ids = self.tokenizer.encode(record)
        ids.append(self.eos_id)
        self.token_count += len(ids)
        written_tokens = len(ids)
        self.pending.extend(ids)
        while len(self.pending) >= self.block_length:
            block = np.asarray(self.pending[:self.block_length], dtype=np.uint16)
            self.output.write(block.tobytes())
            del self.pending[:self.block_length]
            self.block_count += 1
        return written_tokens

    def close(self) -> tuple[int, int]:
        self.output.flush()
        self.output.close()
        return self.block_count, self.token_count


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
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    files = find_input_files(input_path)
    exact_hashes: set[bytes] = set()
    split_hashes: list[int] = []
    duplicate_count = 0
    record_count = 0
    seed_bytes = str(seed).encode("ascii") + b"\0"
    for text in iter_record_texts(files, text_key):
        exact_hash = hashlib.sha256(text.encode("utf-8")).digest()
        if deduplicate and exact_hash in exact_hashes:
            duplicate_count += 1
            continue
        exact_hashes.add(exact_hash)
        split_hash = int.from_bytes(hashlib.sha256(seed_bytes + text.encode("utf-8")).digest()[:8], "big")
        split_hashes.append(split_hash)
        record_count += 1
    if not record_count:
        raise ValueError("No non-empty text records were found in the input files")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size > 65536:
        raise ValueError("The uint16 dataset format supports vocabularies up to 65,536 tokens")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    block_length = context_length + 1
    train_writer = _TokenBlockWriter(tokenizer, destination / "train.bin", block_length)
    validation_writer = _TokenBlockWriter(tokenizer, destination / "validation.bin", block_length)
    tokens_by_language: dict[str, dict[str, int]] = {"train": {}, "validation": {}}
    words_by_language: dict[str, dict[str, int]] = {"train": {}, "validation": {}}
    word_pattern = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[А-Яа-яЁё]+|\d+", re.UNICODE)
    validation_hashes: set[int] = set()
    if 0.0 < validation_fraction < 1.0 and record_count > 1:
        validation_count = min(max(1, round(record_count * validation_fraction)), record_count - 1)
        validation_hashes = set(sorted(split_hashes)[:validation_count])
    if record_count == 1 and 0.0 < validation_fraction < 1.0:
        only_record, language_hint = next(iter(iter_record_rows(files, text_key)))
        language = language_hint or "unknown"
        train_records, validation_records = split_records([only_record], validation_fraction, seed)
        for record in train_records:
            count = train_writer.write(record)
            tokens_by_language["train"][language] = tokens_by_language["train"].get(language, 0) + count
            words_by_language["train"][language] = words_by_language["train"].get(language, 0) + len(word_pattern.findall(record))
        for record in validation_records:
            count = validation_writer.write(record)
            tokens_by_language["validation"][language] = tokens_by_language["validation"].get(language, 0) + count
            words_by_language["validation"][language] = words_by_language["validation"].get(language, 0) + len(word_pattern.findall(record))
    else:
        seen_hashes: set[bytes] = set()
        for text, language_hint in iter_record_rows(files, text_key):
            exact_hash = hashlib.sha256(text.encode("utf-8")).digest()
            if deduplicate and exact_hash in seen_hashes:
                continue
            seen_hashes.add(exact_hash)
            split_hash = int.from_bytes(hashlib.sha256(seed_bytes + text.encode("utf-8")).digest()[:8], "big")
            split = "validation" if split_hash in validation_hashes else "train"
            writer = validation_writer if split == "validation" else train_writer
            count = writer.write(text)
            language = language_hint or "unknown"
            tokens_by_language[split][language] = tokens_by_language[split].get(language, 0) + count
            words_by_language[split][language] = words_by_language[split].get(language, 0) + len(word_pattern.findall(text))
    train_blocks, train_tokens = train_writer.close()
    val_blocks, val_tokens = validation_writer.close()
    metadata: dict[str, object] = {
        "format": "uint16_token_blocks",
        "context_length": context_length,
        "block_length": block_length,
        "train_blocks": train_blocks,
        "validation_blocks": val_blocks,
        "train_tokens_before_chunking": train_tokens,
        "validation_tokens_before_chunking": val_tokens,
        "estimated_total_tokens": train_tokens + val_tokens,
        "language_tokens": {
            language: tokens_by_language["train"].get(language, 0) + tokens_by_language["validation"].get(language, 0)
            for language in sorted(set(tokens_by_language["train"]) | set(tokens_by_language["validation"]))
        },
        "language_token_share_percent": {
            language: round(
                (tokens_by_language["train"].get(language, 0) + tokens_by_language["validation"].get(language, 0))
                * 100 / max(1, train_tokens + val_tokens),
                4,
            )
            for language in sorted(set(tokens_by_language["train"]) | set(tokens_by_language["validation"]))
        },
        "language_words": {
            language: words_by_language["train"].get(language, 0) + words_by_language["validation"].get(language, 0)
            for language in sorted(set(words_by_language["train"]) | set(words_by_language["validation"]))
        },
        "average_tokens_per_word": {
            language: round(
                (tokens_by_language["train"].get(language, 0) + tokens_by_language["validation"].get(language, 0))
                / max(1, words_by_language["train"].get(language, 0) + words_by_language["validation"].get(language, 0)),
                4,
            )
            for language in sorted(set(tokens_by_language["train"]) | set(tokens_by_language["validation"]))
        },
        "records": record_count,
        "duplicates_removed": duplicate_count,
        "tokenizer": str(Path(tokenizer_path)),
        "vocab_size": tokenizer.vocab_size,
        "seed": seed,
        "split_strategy": "document_sha256_seeded_exact_fraction",
        "validation_documents": min(max(1, round(record_count * validation_fraction)), record_count - 1)
        if 0.0 < validation_fraction < 1.0 and record_count > 1
        else 0,
        "deduplicate": deduplicate,
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
