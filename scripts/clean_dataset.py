from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


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
    weights = [0] * 64
    for index in range(0, count, stride):
        shingle = " ".join(words[index:index + 5]).encode("utf-8")
        fingerprint = int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (fingerprint >> bit) & 1 else -1
    value = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            value |= 1 << bit
    return value


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


def _read_rows(input_path: Path) -> tuple[list[dict[str, str]], int, int]:
    rows: list[dict[str, str]] = []
    bad_records = 0
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
    for path in files:
        try:
            content = path.read_bytes().decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            bad_records += 1
            continue
        if path.suffix.lower() == ".txt":
            source_id = _source_id(path)
            rows.append({
                "text": content,
                "source_id": source_id,
                "source_path": path.relative_to(base_dir).as_posix(),
            })
            continue
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                bad_records += 1
                continue
            if isinstance(value, str):
                rows.append({"text": value, "source_id": path.stem, "source_path": path.relative_to(base_dir).as_posix()})
            elif isinstance(value, dict) and isinstance(value.get("text"), str):
                rows.append({
                    "text": value["text"],
                    "source_id": str(value.get("source_id", path.stem)),
                    "source_path": str(value.get("source_path", path.relative_to(base_dir).as_posix())),
                })
            else:
                bad_records += 1
    return rows, bad_records, sum(path.stat().st_size for path in files)


def _source_id(path: Path) -> str:
    parts = path.parts
    if "rsd" in parts:
        return "rsd_public_domain"
    if path.name.startswith("project_gutenberg_1228"):
        return "gutenberg_origin_species_1228"
    if path.name.startswith("project_gutenberg_944"):
        return "gutenberg_voyage_beagle_944"
    return path.stem


def clean_records(
    rows: Iterable[dict[str, str]],
    min_chars: int = 300,
    min_words: int = 50,
    near_duplicate_distance: int = 2,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    accepted: list[dict[str, str]] = []
    exact_hashes: set[str] = set()
    fingerprints: list[int] = []
    counters: Counter[str] = Counter()
    for row in rows:
        text = normalize_text(row["text"])
        if not text:
            counters["empty_removed"] += 1
            continue
        if "\ufffd" in text or sum(ord(char) < 9 or 13 < ord(char) < 32 for char in text) > max(2, len(text) // 1000):
            counters["corrupt_removed"] += 1
            continue
        words = WORD_RE.findall(text)
        if len(text) < min_chars or len(words) < min_words:
            counters["too_short_removed"] += 1
            continue
        noise_reason = _noise_reason(text, words)
        if noise_reason:
            counters[f"noise_{noise_reason}_removed"] += 1
            continue
        if len(set(text.splitlines())) <= max(1, len(text.splitlines()) // 5) and len(text.splitlines()) > 20:
            counters["repetitive_noise_removed"] += 1
            continue
        reason = _sensitive_reason(text)
        if reason:
            counters[f"sensitive_{reason}_removed"] += 1
            continue
        exact_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if exact_hash in exact_hashes:
            counters["exact_duplicates_removed"] += 1
            continue
        fingerprint = _simhash(text)
        if any((fingerprint ^ prior).bit_count() <= near_duplicate_distance for prior in fingerprints):
            counters["near_duplicates_removed"] += 1
            continue
        exact_hashes.add(exact_hash)
        fingerprints.append(fingerprint)
        accepted.append({
            "text": text,
            "source_id": row.get("source_id", "unknown"),
            "source_path": row.get("source_path", ""),
            "language": language_of(text),
        })
        counters["documents_kept"] += 1
        counters[f"words_{accepted[-1]['language']}"] += len(words)
    return accepted, dict(counters)


def clean_dataset(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(input_dir)
    destination = Path(output_dir)
    rows, bad_records, raw_bytes = _read_rows(source)
    documents, counters = clean_records(rows)
    counters["bad_records_removed"] = bad_records
    if not documents:
        raise ValueError("No usable documents remain after filtering")
    destination.mkdir(parents=True, exist_ok=True)
    corpus_path = destination / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in documents:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    language_words = {lang: int(counters.get(f"words_{lang}", 0)) for lang in ("ru", "en", "other")}
    total_words = sum(language_words.values())
    clean_text_bytes = sum(len(record["text"].encode("utf-8")) for record in documents)
    source_documents = dict(Counter(record["source_id"] for record in documents))
    exact_duplicates = int(counters.get("exact_duplicates_removed", 0))
    near_duplicates = int(counters.get("near_duplicates_removed", 0))
    sensitive_removed = sum(value for key, value in counters.items() if key.startswith("sensitive_"))
    stats: dict[str, Any] = {
        "documents": len(documents),
        "raw_bytes": raw_bytes,
        "clean_text_bytes": clean_text_bytes,
        "jsonl_bytes": corpus_path.stat().st_size,
        "words": total_words,
        "language_words": language_words,
        "language_share_percent": {
            lang: round(count * 100 / total_words, 4) if total_words else 0.0
            for lang, count in language_words.items()
        },
        "filters": counters,
        "exact_duplicates_removed": exact_duplicates,
        "near_duplicates_removed": near_duplicates,
        "sensitive_records_removed": sensitive_removed,
        "source_documents": source_documents,
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
    args = parser.parse_args()
    stats = clean_dataset(args.input, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
