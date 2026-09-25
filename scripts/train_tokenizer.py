from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import find_input_files, iter_record_texts
from src.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train byte-level BPE from your own text files")
    parser.add_argument("--input", default="data/raw", help="A .txt/.jsonl file or directory")
    parser.add_argument("--output", default="tokenizer/tokenizer.json")
    parser.add_argument("--config", default="configs/model_v1.yaml")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=None)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Deterministically keep every Nth document; useful when full-corpus BPE statistics need too much RAM",
    )
    parser.add_argument("--no-sync-model-vocab", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    target_vocab = args.vocab_size or int(config["tokenizer"]["vocab_size"])
    min_frequency = args.min_frequency or int(config["tokenizer"]["min_frequency"])
    if not 0.0 < args.sample_fraction <= 1.0:
        raise ValueError("--sample-fraction must be greater than 0 and at most 1")
    sample_stride = max(1, round(1 / args.sample_fraction))
    files = find_input_files(args.input)
    visited_count = 0
    record_count = 0
    with tempfile.TemporaryDirectory(prefix="my-llm-tokenizer-") as temporary:
        corpus = Path(temporary) / "corpus.txt"
        with corpus.open("w", encoding="utf-8", newline="\n") as output:
            for record in iter_record_texts(files, text_key=config.get("tokenizer", {}).get("text_key", "text")):
                visited_count += 1
                if (visited_count - 1) % sample_stride:
                    continue
                output.write(record)
                output.write("\n\n")
                record_count += 1
        if not record_count:
            raise ValueError("No non-empty text records were found")
        tokenizer = ByteBPETokenizer.train([corpus], args.output, target_vocab, min_frequency)
    print(f"Tokenizer saved: {args.output}")
    print(f"Documents visited: {visited_count:,}")
    print(f"Tokenizer training documents streamed: {record_count:,} (every {sample_stride}th document)")
    print(f"Actual vocabulary size: {tokenizer.vocab_size:,} (requested {target_vocab:,})")
    if not args.no_sync_model_vocab:
        config["model"]["vocab_size"] = tokenizer.vocab_size
        config["tokenizer"]["vocab_size"] = target_vocab
        with Path(args.config).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        print(f"Updated model.vocab_size in {args.config} to {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
