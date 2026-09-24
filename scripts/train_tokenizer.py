from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import find_input_files, read_records
from src.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train byte-level BPE from your own text files")
    parser.add_argument("--input", default="data/raw", help="A .txt/.jsonl file or directory")
    parser.add_argument("--output", default="tokenizer/tokenizer.json")
    parser.add_argument("--config", default="configs/model_v1.yaml")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=None)
    parser.add_argument("--no-sync-model-vocab", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    target_vocab = args.vocab_size or int(config["tokenizer"]["vocab_size"])
    min_frequency = args.min_frequency or int(config["tokenizer"]["min_frequency"])
    files = find_input_files(args.input)
    records, _ = read_records(files, text_key=config.get("tokenizer", {}).get("text_key", "text"))
    if not records:
        raise ValueError("No non-empty text records were found")
    with tempfile.TemporaryDirectory(prefix="my-llm-tokenizer-") as temporary:
        corpus = Path(temporary) / "corpus.txt"
        corpus.write_text("\n\n".join(records), encoding="utf-8")
        tokenizer = ByteBPETokenizer.train([corpus], args.output, target_vocab, min_frequency)
    print(f"Tokenizer saved: {args.output}")
    print(f"Actual vocabulary size: {tokenizer.vocab_size:,} (requested {target_vocab:,})")
    if not args.no_sync_model_vocab:
        config["model"]["vocab_size"] = tokenizer.vocab_size
        config["tokenizer"]["vocab_size"] = target_vocab
        with Path(args.config).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        print(f"Updated model.vocab_size in {args.config} to {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
