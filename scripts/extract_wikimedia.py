from __future__ import annotations

import argparse
import bz2
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import mwparserfromhell


ROOT = Path(__file__).resolve().parents[1]
SKIP_TAGS = {"ref", "references", "gallery", "timeline", "score", "math", "chem"}
SKIP_LINK_PREFIXES = {"category:", "file:", "image:", "media:", "wikipedia:", "портал:"}
WIKI_SOURCES = {
    "ruwiki": ("ruwiki_cc_by_sa", "ru", "https://ru.wikipedia.org/wiki/"),
    "simplewiki": ("simplewiki_cc_by_sa", "en", "https://simple.wikipedia.org/wiki/"),
    "enwikibooks": ("enwikibooks_cc_by_sa", "en", "https://en.wikibooks.org/wiki/"),
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return child.text or ""
    return ""


def _plain_text(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)
    for tag in list(code.filter_tags(recursive=True)):
        name = str(tag.tag).strip().lower()
        if name in SKIP_TAGS:
            code.remove(tag)
    for link in list(code.filter_wikilinks(recursive=True)):
        target = str(link.title).strip().lower()
        if any(target.startswith(prefix) for prefix in SKIP_LINK_PREFIXES):
            code.remove(link)
    text = code.strip_code(normalize=True, collapse=False)
    text = re.sub(r"(?m)^\s*(?:\[\d+\]|↑\s*\d+)\s*$", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _plain_text_worker(wikitext: str) -> tuple[str, bool]:
    try:
        return _plain_text(wikitext), False
    except Exception:
        return "", True


def _flush_batch(
    batch: list[tuple[str, str, str]],
    executor: ProcessPoolExecutor,
    output: Any,
    counters: Counter[str],
    source_id: str,
    language: str,
) -> int:
    output_bytes = 0
    texts = executor.map(_plain_text_worker, (item[2] for item in batch), chunksize=8)
    for (title, page_url, _), (text, parser_error) in zip(batch, texts):
        if parser_error:
            counters["parser_errors_skipped"] += 1
            text = ""
        if len(text) < 300:
            counters["short_or_empty_skipped"] += 1
        else:
            record = {
                "text": title + "\n" + text,
                "source_id": source_id,
                "source_path": title,
                "source_url": page_url,
                "language_hint": language,
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            output.write(line)
            output_bytes += len(line.encode("utf-8"))
            counters["documents_written"] += 1
    return output_bytes


def extract_dump(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    workers: int = 8,
) -> dict[str, object]:
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace an existing extracted corpus: {destination}; pass --overwrite to replace it")
    if source.suffix.lower() != ".bz2":
        raise ValueError(f"Expected a .bz2 Wikimedia XML dump: {source}")
    project = source.name.split("-", 1)[0]
    if project not in WIKI_SOURCES:
        raise ValueError(f"Unsupported Wikimedia project filename: {source.name}")
    source_id, language, wiki_base = WIKI_SOURCES[project]
    workers = max(1, min(int(workers), os.cpu_count() or 1))
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    output_bytes = 0
    root: ET.Element | None = None
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor, bz2.open(source, "rb") as compressed, temporary.open("w", encoding="utf-8", newline="\n") as output:
            batch: list[tuple[str, str, str]] = []
            for event, element in ET.iterparse(compressed, events=("start", "end")):
                if root is None:
                    if event == "start":
                        root = element
                    continue
                if event != "end" or _local_name(element.tag) != "page":
                    continue
                counters["pages_seen"] += 1
                title = _child_text(element, "title").strip()
                namespace = _child_text(element, "ns").strip()
                if namespace != "0":
                    counters["non_article_namespace_skipped"] += 1
                else:
                    redirect = any(_local_name(child.tag) == "redirect" for child in element)
                    if redirect:
                        counters["redirects_skipped"] += 1
                    else:
                        revision = next((child for child in element if _local_name(child.tag) == "revision"), None)
                        wikitext = ""
                        if revision is not None:
                            wikitext = next(
                                ((child.text or "") for child in revision if _local_name(child.tag) == "text"),
                                "",
                            )
                        page_url = wiki_base + quote(title.replace(" ", "_"), safe="()!,._-~")
                        batch.append((title, page_url, wikitext))
                        if len(batch) >= 128:
                            output_bytes += _flush_batch(batch, executor, output, counters, source_id, language)
                            batch.clear()
                element.clear()
                if root is not None:
                    root.clear()
                if counters["pages_seen"] % 25_000 == 0:
                    print(
                        f"{project}: pages={counters['pages_seen']:,} documents={counters['documents_written']:,}",
                        flush=True,
                    )
            if batch:
                output_bytes += _flush_batch(batch, executor, output, counters, source_id, language)
            output.flush()
    except Exception:
        raise
    temporary.replace(destination)
    stats: dict[str, object] = {
        "source_id": source_id,
        "project": project,
        "language": language,
        "dump": str(source),
        "output": str(destination),
        "output_bytes": output_bytes,
        **dict(counters),
    }
    stats_path = destination.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = source.parent / "wikimedia_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("files", []):
            if entry.get("source_id") == source_id:
                entry["extracted_documents"] = int(counters["documents_written"])
                entry["extracted_jsonl_bytes"] = output_bytes
                entry["extract_stats"] = stats_path.name
                break
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a pinned Wikimedia XML dump to plain-text JSONL")
    parser.add_argument("--input", required=True, help="A downloaded .bz2 pages-articles multistream dump")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Parallel wikitext parsers (default: 8)")
    args = parser.parse_args()
    print(json.dumps(extract_dump(args.input, args.output, overwrite=args.overwrite, workers=args.workers), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
