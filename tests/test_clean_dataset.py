from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_dataset import clean_dataset, clean_records, normalize_text


def _long_text(prefix: str = "Это связный русский текст для проверки очистки корпуса.") -> str:
    return " ".join([prefix] * 18)


def test_normalize_unicode_html_and_gutenberg_wrapper() -> None:
    raw = "\ufeff*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n<h1>Семья\u0301</h1>\n" + _long_text() + "\n*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***"
    cleaned = normalize_text(raw)
    assert "*** START" not in cleaned
    assert "*** END" not in cleaned
    assert "<h1>" not in cleaned
    assert "Семья́" in cleaned
    assert "\r" not in cleaned


def test_clean_records_drops_exact_and_near_duplicates() -> None:
    original = " ".join(f"token{index}" for index in range(300))
    near = original + " token299 token298"
    accepted, stats = clean_records([
        {"text": original, "source_id": "a"},
        {"text": original, "source_id": "duplicate"},
        {"text": near, "source_id": "near"},
    ])
    assert len(accepted) == 1
    assert stats["exact_duplicates_removed"] == 1
    assert stats["near_duplicates_removed"] == 1


def test_sensitive_pattern_filter_and_language_classification() -> None:
    english = "This is a sufficiently long English text with a contact email " + "words " * 60 + "contact: person@example.org"
    accepted, stats = clean_records([
        {"text": english, "source_id": "source"},
        {"text": _long_text(), "source_id": "source"},
    ])
    assert len(accepted) == 1
    assert accepted[0]["language"] == "ru"
    assert stats["sensitive_email_removed"] == 1


def test_repetitive_machine_noise_is_removed() -> None:
    spam = " ".join(["buy cheap offer now"] * 100)
    repeated_char = _long_text() + " " + "a" * 40
    punctuation_run = _long_text() + " " + "." * 40
    accepted, stats = clean_records([
        {"text": spam, "source_id": "spam"},
        {"text": repeated_char, "source_id": "machine"},
        {"text": punctuation_run, "source_id": "book"},
    ])
    assert [row["source_id"] for row in accepted] == ["book"]
    assert stats["noise_low_lexical_diversity_removed"] == 1
    assert stats["noise_repeated_character_noise_removed"] == 1


def test_clean_dataset_writes_jsonl_and_statistics(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    clean_dir = tmp_path / "cleaned"
    raw_dir.mkdir()
    text = _long_text()
    (raw_dir / "sample.txt").write_text(text, encoding="utf-8")
    stats = clean_dataset(raw_dir, clean_dir)
    records = [json.loads(line) for line in (clean_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stats["documents"] == 1
    assert records[0]["source_id"] == "sample"
    assert records[0]["source_path"] == "sample.txt"
    assert records[0]["language"] == "ru"
    assert json.loads((clean_dir / "stats.json").read_text(encoding="utf-8"))["jsonl_bytes"] > 0
