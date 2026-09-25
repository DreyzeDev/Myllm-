from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[А-Яа-яЁё]+|\d+", re.UNICODE)
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
LONG_REPEAT_RE = re.compile(r"([A-Za-zА-Яа-яЁё0-9])\1{31,}")
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]{1,64}@[\w.-]+\.[A-Za-z]{2,}(?!\w)")
SECRET_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|password|пароль|секретный\s+ключ)"
    r"\s*[:=]\s*[^\s,;]{8,}"
)
PHONE_CONTEXT_RE = re.compile(
    r"(?i)(?:тел(?:ефон)?|phone|mobile|call|контакт|звонить|номер)\s*[:#-]?\s*"
    r"\+?\d[\d\s().-]{8,}\d"
)
TRAILING_BOILERPLATE = re.compile(
    r"(?im)^\s*(?:оставить комментарий|читать дальше|читайте также|похожие статьи|"
    r"поделиться|все права защищены|all rights reserved|subscribe to our newsletter)\b.*$"
)
GUTENBERG_START_RE = re.compile(r"(?im)^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*\s*$")
GUTENBERG_END_RE = re.compile(r"(?im)^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*\s*$")
_BYTE_BIT_LOOKUP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._hidden_depth += 1
        elif tag.lower() in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def strip_html(text: str) -> str:
    if not HTML_TAG_RE.search(text):
        return html.unescape(text)
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    return html.unescape("".join(parser.parts))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n"))
    text = strip_html(text)
    start = GUTENBERG_START_RE.search(text)
    if start:
        text = text[start.end():]
    end = GUTENBERG_END_RE.search(text)
    if end:
        text = text[:end.start()]
    for match in list(TRAILING_BOILERPLATE.finditer(text)):
        if match.start() >= int(len(text) * 0.75):
            text = text[:match.start()]
            break
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return unicodedata.normalize("NFC", text).strip()


def language_of(text: str) -> str:
    cyrillic = sum("\u0400" <= char <= "\u04ff" for char in text)
    latin = sum(("a" <= char.lower() <= "z") for char in text)
    letters = cyrillic + latin
    if letters == 0:
        return "other"
    if cyrillic / letters >= 0.25:
        return "ru"
    if latin / letters >= 0.80:
        return "en"
    return "other"


def _simhash(text: str, max_shingles: int = 4_096) -> int:
    words = [word.lower() for word in WORD_RE.findall(text)]
    if len(words) < 5:
        return 0
    count = len(words) - 4
    stride = max(1, count // max_shingles)
    fingerprints = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(" ".join(words[index:index + 5]).encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            for index in range(0, count, stride)
        ),
        dtype=np.uint64,
        count=(count + stride - 1) // stride,
    )
    byte_matrix = fingerprints.view(np.uint8).reshape(-1, 8)
    ones = _BYTE_BIT_LOOKUP[byte_matrix].sum(axis=0, dtype=np.int64).reshape(-1)
    weights = 2 * ones - len(fingerprints)
    return sum(1 << bit for bit, weight in enumerate(weights) if weight >= 0)


def _sensitive_reason(text: str) -> str | None:
    if EMAIL_RE.search(text):
        return "email"
    if PHONE_CONTEXT_RE.search(text):
        return "phone_with_contact_context"
    if SECRET_RE.search(text):
        return "credential_pattern"
    return None


def _noise_reason(text: str, words: list[str]) -> str | None:
    if LONG_REPEAT_RE.search(text):
        return "repeated_character_noise"
    if len(words) >= 100 and len({word.casefold() for word in words}) / len(words) < 0.04:
        return "low_lexical_diversity"
    if len(URL_RE.findall(text)) > max(5, len(words) // 100):
        return "link_spam"
    return None


def _input_files(input_path: Path) -> tuple[list[Path], Path]:
    if input_path.is_file():
        files = [input_path]
        base_dir = input_path.parent
    else:
        base_dir = input_path
        files = sorted([*input_path.rglob("*.txt"), *input_path.rglob("*.jsonl")])
    if not files:
        raise FileNotFoundError(f"No .txt or .jsonl files found in {input_path}")
    manifest_path = base_dir / "raw_manifest.json"
    if manifest_path.exists():
        files = [path for path in files if path.name != "raw_manifest.json"]
    return files, base_dir


def _iter_rows(files: list[Path], base_dir: Path, counters: Counter[str]) -> Iterator[dict[str, str]]:
    for path in files:
        if path.suffix.lower() == ".txt":
            try:
                content = path.read_bytes().decode("utf-8-sig", errors="strict")
            except UnicodeDecodeError:
                counters["bad_records_removed"] += 1
                continue
            yield {
                "text": content,
                "source_id": _source_id(path),
                "source_path": path.relative_to(base_dir).as_posix(),
            }
            continue
        try:
            with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        counters["bad_records_removed"] += 1
                        continue
                    if isinstance(value, str):
                        yield {"text": value, "source_id": path.stem, "source_path": path.relative_to(base_dir).as_posix()}
                    elif isinstance(value, dict) and isinstance(value.get("text"), str):
                        row = {
                            "text": value["text"],
                            "source_id": str(value.get("source_id", path.stem)),
                            "source_path": str(value.get("source_path", path.relative_to(base_dir).as_posix())),
                        }
                        for key in ("source_url", "title", "language_hint"):
                            if isinstance(value.get(key), str):
                                row[key] = value[key]
                        yield row
                    else:
                        counters["bad_records_removed"] += 1
        except UnicodeDecodeError:
            counters["bad_records_removed"] += 1


def _read_rows(input_path: Path) -> tuple[list[dict[str, str]], int, int]:
    files, base_dir = _input_files(input_path)
    counters: Counter[str] = Counter()
    rows = list(_iter_rows(files, base_dir, counters))
    return rows, int(counters.get("bad_records_removed", 0)), sum(path.stat().st_size for path in files)


def _source_id(path: Path) -> str:
    parts = path.parts
    if "rsd" in parts:
        return "rsd_public_domain"
    if path.name.startswith("project_gutenberg_1228"):
        return "gutenberg_origin_species_1228"
    if path.name.startswith("project_gutenberg_944"):
        return "gutenberg_voyage_beagle_944"
    return path.stem


def _prepare_row(
    row: dict[str, str],
    min_chars: int = 300,
    min_words: int = 50,
    allowed_languages: tuple[str, ...] = ("ru", "en"),
) -> dict[str, Any]:
    text = normalize_text(row["text"])
    if not text:
        return {"reason": "empty_removed"}
    if "\ufffd" in text or sum(ord(char) < 9 or 13 < ord(char) < 32 for char in text) > max(2, len(text) // 1000):
        return {"reason": "corrupt_removed"}
    words = WORD_RE.findall(text)
    if len(text) < min_chars or len(words) < min_words:
        return {"reason": "too_short_removed"}
    noise_reason = _noise_reason(text, words)
    if noise_reason:
        return {"reason": f"noise_{noise_reason}_removed"}
    lines = text.splitlines()
    if len(lines) > 20 and len(set(lines)) <= max(1, len(lines) // 5):
        return {"reason": "repetitive_noise_removed"}
    sensitive_reason = _sensitive_reason(text)
    if sensitive_reason:
        return {"reason": f"sensitive_{sensitive_reason}_removed"}
    language = language_of(text)
    if language not in set(allowed_languages):
        return {"reason": "language_filtered_removed"}
    record = {
        "text": text,
        "source_id": row.get("source_id", "unknown"),
        "source_path": row.get("source_path", ""),
        "language": language,
    }
    for key in ("source_url", "title"):
        if row.get(key):
            record[key] = row[key]
    return {
        "record": record,
        "exact_hash": hashlib.sha256(text.encode("utf-8")).digest(),
        "fingerprint": _simhash(text),
        "language": language,
        "word_count": len(words),
    }


def _prepare_row_worker(
    task: tuple[dict[str, str], int, int, tuple[str, ...]],
) -> dict[str, Any]:
    return _prepare_row(*task)


class _NearDuplicateIndex:
    """SimHash lookup using three bands, avoiding a full quadratic scan."""

    _BANDS = ((0, 22), (22, 21), (43, 21))

    def __init__(self, hamming_distance: int) -> None:
        self.hamming_distance = hamming_distance
        self.fingerprints: list[int] = []
        self.buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in self._BANDS]

    def _keys(self, fingerprint: int) -> list[int]:
        return [
            (fingerprint >> shift) & ((1 << width) - 1)
            for shift, width in self._BANDS
        ]

    def is_duplicate(self, fingerprint: int) -> bool:
        candidates: set[int] = set()
        for band, key in enumerate(self._keys(fingerprint)):
            candidates.update(self.buckets[band].get(key, ()))
        return any(
            (fingerprint ^ self.fingerprints[index]).bit_count() <= self.hamming_distance
            for index in candidates
        )

    def add(self, fingerprint: int) -> None:
        index = len(self.fingerprints)
        self.fingerprints.append(fingerprint)
        for band, key in enumerate(self._keys(fingerprint)):
            self.buckets[band][key].append(index)


class _CorpusCleaner:
    def __init__(self, near_duplicate_distance: int = 2, allowed_languages: tuple[str, ...] = ("ru", "en")) -> None:
        self.counters: Counter[str] = Counter()
        self.exact_hashes: set[bytes] = set()
        self.near_duplicates = _NearDuplicateIndex(near_duplicate_distance)
        self.allowed_languages = set(allowed_languages)

    def process_prepared(self, prepared: dict[str, Any]) -> dict[str, str] | None:
        reason = prepared.get("reason")
        if reason:
            self.counters[str(reason)] += 1
            return None
        exact_hash = prepared["exact_hash"]
        if exact_hash in self.exact_hashes:
            self.counters["exact_duplicates_removed"] += 1
            return None
        fingerprint = int(prepared["fingerprint"])
        if self.near_duplicates.is_duplicate(fingerprint):
            self.counters["near_duplicates_removed"] += 1
            return None
        self.exact_hashes.add(exact_hash)
        self.near_duplicates.add(fingerprint)
        self.counters["documents_kept"] += 1
        language = str(prepared["language"])
        self.counters[f"words_{language}"] += int(prepared["word_count"])
        return prepared["record"]

    def process(self, row: dict[str, str], min_chars: int = 300, min_words: int = 50) -> dict[str, str] | None:
        prepared = _prepare_row(row, min_chars, min_words, tuple(sorted(self.allowed_languages)))
        return self.process_prepared(prepared)


def clean_records(
    rows: Iterable[dict[str, str]],
    min_chars: int = 300,
    min_words: int = 50,
    near_duplicate_distance: int = 2,
    allowed_languages: tuple[str, ...] = ("ru", "en"),
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    accepted: list[dict[str, str]] = []
    cleaner = _CorpusCleaner(near_duplicate_distance, allowed_languages)
    for row in rows:
        record = cleaner.process(row, min_chars, min_words)
        if record is not None:
            accepted.append(record)
    return accepted, dict(cleaner.counters)


def clean_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    allowed_languages: tuple[str, ...] = ("ru", "en"),
    workers: int = 8,
) -> dict[str, Any]:
    source = Path(input_dir)
    destination = Path(output_dir)
    files, base_dir = _input_files(source)
    raw_bytes = sum(path.stat().st_size for path in files)
    cleaner = _CorpusCleaner(allowed_languages=allowed_languages)
    destination.mkdir(parents=True, exist_ok=True)
    corpus_path = destination / "corpus.jsonl"
    clean_text_bytes = 0
    source_documents: Counter[str] = Counter()
    workers = max(1, min(int(workers), os.cpu_count() or 1))
    batch_size = 64
    rows_seen = 0
    allowed = tuple(sorted(set(allowed_languages)))
    with ProcessPoolExecutor(max_workers=workers) as executor, corpus_path.open("w", encoding="utf-8", newline="\n") as output:
        batch: list[dict[str, str]] = []

        def flush_batch() -> None:
            nonlocal clean_text_bytes
            tasks = ((row, 300, 50, allowed) for row in batch)
            for prepared in executor.map(_prepare_row_worker, tasks, chunksize=4):
                record = cleaner.process_prepared(prepared)
                if record is None:
                    continue
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                clean_text_bytes += len(record["text"].encode("utf-8"))
                source_documents[record["source_id"]] += 1
            batch.clear()

        for row in _iter_rows(files, base_dir, cleaner.counters):
            batch.append(row)
            rows_seen += 1
            if len(batch) >= batch_size:
                flush_batch()
            if rows_seen and rows_seen % 100_000 == 0:
                print(f"cleaner: rows={rows_seen:,} kept={cleaner.counters.get('documents_kept', 0):,}", flush=True)
        if batch:
            flush_batch()
    counters = cleaner.counters
    if not counters.get("documents_kept"):
        raise ValueError("No usable documents remain after filtering")
    language_words = {lang: int(counters.get(f"words_{lang}", 0)) for lang in ("ru", "en", "other")}
    total_words = sum(language_words.values())
    exact_duplicates = int(counters.get("exact_duplicates_removed", 0))
    near_duplicates = int(counters.get("near_duplicates_removed", 0))
    sensitive_removed = sum(value for key, value in counters.items() if key.startswith("sensitive_"))
    stats: dict[str, Any] = {
        "documents": int(counters["documents_kept"]),
        "raw_bytes": raw_bytes,
        "clean_text_bytes": clean_text_bytes,
        "jsonl_bytes": corpus_path.stat().st_size,
        "words": total_words,
        "language_words": language_words,
        "language_share_percent": {
            lang: round(count * 100 / total_words, 4) if total_words else 0.0
            for lang, count in language_words.items()
        },
        "filters": dict(counters),
        "exact_duplicates_removed": exact_duplicates,
        "near_duplicates_removed": near_duplicates,
        "sensitive_records_removed": sensitive_removed,
        "source_documents": dict(source_documents),
        "min_chars": 300,
        "min_words": 50,
        "near_duplicate_hamming_distance": 2,
        "corpus_file": str(corpus_path),
    }
    (destination / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw MyLLM V1 corpus and report basic safety/language statistics")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/cleaned")
    parser.add_argument("--allow-other-languages", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Parallel document cleaners (default: 8)")
    args = parser.parse_args()
    allowed_languages = ("ru", "en", "other") if args.allow_other_languages else ("ru", "en")
    stats = clean_dataset(args.input, args.output, allowed_languages, args.workers)
    print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
