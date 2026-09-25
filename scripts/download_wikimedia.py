from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "20260901"
SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "ruwiki_cc_by_sa",
        "project": "ruwiki",
        "language": "ru",
        "name": "Russian Wikipedia, articles and primary meta-pages",
        "wiki_url": "https://ru.wikipedia.org/",
        "bytes": 6_184_856_270,
        "sha1": "304a841954f69d0ea25d0ff6f46b313dfde19d57",
    },
    {
        "id": "simplewiki_cc_by_sa",
        "project": "simplewiki",
        "language": "en",
        "name": "Simple English Wikipedia, articles and primary meta-pages",
        "wiki_url": "https://simple.wikipedia.org/",
        "bytes": 385_846_687,
        "sha1": "50152ec5f40669eef3dc7b251d1a025fe3c0bb55",
    },
    {
        "id": "enwikibooks_cc_by_sa",
        "project": "enwikibooks",
        "language": "en",
        "name": "English Wikibooks, current main-namespace articles",
        "wiki_url": "https://en.wikibooks.org/",
        "bytes": 207_339_557,
        "sha1": "b77914248dbf39f4641956b79358cde5affa1dd0",
    },
)
CHUNK_BYTES = 4 * 1024 * 1024
PROGRESS_BYTES = 128 * 1024 * 1024


def _hash_file(path: Path) -> tuple[str, str]:
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            sha1.update(chunk)
            sha256.update(chunk)
    return sha1.hexdigest(), sha256.hexdigest()


def _download(source: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    project = str(source["project"])
    filename = f"{project}-{SNAPSHOT}-pages-articles-multistream.xml.bz2"
    url = f"https://dumps.wikimedia.org/{project}/{SNAPSHOT}/{filename}"
    target = output_dir / filename
    partial = target.with_suffix(target.suffix + ".part")
    expected_bytes = int(source["bytes"])
    expected_sha1 = str(source["sha1"])

    if target.exists():
        if target.stat().st_size == expected_bytes:
            actual_sha1, actual_sha256 = _hash_file(target)
            if actual_sha1 == expected_sha1:
                print(f"Verified existing {filename} ({expected_bytes:,} bytes)", flush=True)
                return _file_record(source, target, url, actual_sha1, actual_sha256)
        raise ValueError(f"Existing file failed its pinned size/checksum; leaving it untouched: {target}")

    output_dir.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise ValueError(f"Partial download is larger than expected; leaving it untouched: {partial}")
    retries = 0
    last_reported = offset // PROGRESS_BYTES
    while offset < expected_bytes:
        headers = {"User-Agent": "MyLLM-V1-open-corpus-downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status == 206:
                    mode = "ab"
                else:
                    offset = 0
                    mode = "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(CHUNK_BYTES):
                        output.write(chunk)
                        offset += len(chunk)
                        report = offset // PROGRESS_BYTES
                        if report > last_reported:
                            last_reported = report
                            print(f"{project}: {offset / (1024**3):.2f} / {expected_bytes / (1024**3):.2f} GiB", flush=True)
            retries = 0
        except (OSError, urllib.error.URLError) as exc:
            retries += 1
            if retries > 6:
                raise RuntimeError(f"Download stopped after repeated network errors; partial data is kept at {partial}: {exc}") from exc
            offset = partial.stat().st_size if partial.exists() else 0
            delay = min(30, 2 ** retries)
            print(f"{project}: network interruption at {offset:,} bytes; retrying in {delay}s", flush=True)
            time.sleep(delay)

    if partial.stat().st_size != expected_bytes:
        raise ValueError(f"Size check failed for {filename}: expected {expected_bytes}, got {partial.stat().st_size}")
    actual_sha1, actual_sha256 = _hash_file(partial)
    if actual_sha1 != expected_sha1:
        raise ValueError(f"Wikimedia SHA-1 check failed for {filename}: expected {expected_sha1}, got {actual_sha1}")
    os.replace(partial, target)
    print(f"Verified {filename} ({expected_bytes:,} bytes; SHA-1 {actual_sha1})", flush=True)
    return _file_record(source, target, url, actual_sha1, actual_sha256)


def _file_record(
    source: dict[str, Any], path: Path, url: str, sha1: str, sha256: str
) -> dict[str, Any]:
    return {
        "source_id": source["id"],
        "name": source["name"],
        "project": source["project"],
        "language": source["language"],
        "snapshot": SNAPSHOT,
        "url": url,
        "wiki_url": source["wiki_url"],
        "license": "CC BY-SA 4.0; eligible text contributions may also be available under GFDL",
        "license_url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en",
        "bytes": path.stat().st_size,
        "sha1_official": sha1,
        "sha256": sha256,
        "path": path.name,
        "extracted_documents": None,
    }


def download_wikimedia(
    output_dir: str | Path,
    max_download_gb: float = 8.0,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    selected_sources = [source for source in SOURCES if source_ids is None or source["id"] in source_ids]
    unknown_ids = set(source_ids or ()) - {source["id"] for source in SOURCES}
    if unknown_ids:
        raise ValueError(f"Unknown source IDs: {', '.join(sorted(unknown_ids))}")
    if not selected_sources:
        raise ValueError("No Wikimedia sources were selected")
    expected_bytes = sum(int(source["bytes"]) for source in selected_sources)
    if expected_bytes > max_download_gb * 1024**3:
        raise ValueError(
            f"Pinned Wikimedia selection is {expected_bytes / (1024**3):.2f} GiB, above the configured size limit"
        )
    destination = Path(output_dir)
    records = [_download(source, destination) for source in selected_sources]
    manifest_path = destination / "wikimedia_manifest.json"
    existing_records: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_records = {
            str(record["source_id"]): record
            for record in existing.get("files", [])
            if isinstance(record, dict) and record.get("source_id")
        }
    for record in records:
        previous = existing_records.get(str(record["source_id"]), {})
        for key in ("extracted_documents", "extracted_jsonl_bytes", "extract_stats"):
            if previous.get(key) is not None:
                record[key] = previous[key]
        stats_path = destination / f"{record['project']}.stats.json"
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            record["extracted_documents"] = int(stats.get("documents_written", 0))
            record["extracted_jsonl_bytes"] = int(stats.get("output_bytes", 0))
            record["extract_stats"] = stats_path.name
            record["extraction_counts"] = {
                key: int(stats.get(key, 0))
                for key in (
                    "pages_seen",
                    "redirects_skipped",
                    "non_article_namespace_skipped",
                    "short_or_empty_skipped",
                    "parser_errors_skipped",
                )
            }
        existing_records[str(record["source_id"])] = record
    records = [
        existing_records[source["id"]]
        for source in SOURCES
        if source["id"] in existing_records
    ]
    manifest: dict[str, Any] = {
        "dataset": "MyLLM V1 Wikimedia open-text selection",
        "snapshot": SNAPSHOT,
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "extraction": "scripts/extract_wikimedia.py; main namespace, current revisions, plain-text wikitext conversion",
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wikimedia manifest written: wikimedia_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned, checksum-verified Wikimedia text dumps")
    parser.add_argument("--output", default="data/raw/wikimedia")
    parser.add_argument("--max-download-gb", type=float, default=8.0)
    parser.add_argument("--source-ids", nargs="*", default=None, help="Optionally download only selected manifest IDs")
    args = parser.parse_args()
    download_wikimedia(ROOT / args.output, args.max_download_gb, args.source_ids)


if __name__ == "__main__":
    main()
