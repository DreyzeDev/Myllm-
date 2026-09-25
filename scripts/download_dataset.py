from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RSD_REPOSITORY = "nevmenandr/RSD"
RSD_COMMIT = "476876e8a5c430fb0f6e147bf1519c9b43699261"
RSD_PREFIXES = (
    "author/fiction/period/18/corpus/",
    "author/fiction/period/19-1/corpus/",
    "author/fiction/period/19-2/nonbrevia/corpus/",
    "author/fiction/period/19-20/corpus/",
    "author/nonfiction/journalism/corpus/",
    "author/nonfiction/science/combined/corpus/",
)
RSD_DOCUMENTS = 161
RSD_BYTES = 109_615_661
GUTENBERG = (
    {
        "id": "gutenberg_origin_species_1228",
        "filename": "project_gutenberg_1228_origin_of_species.txt",
        "url": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        "bytes": 970_612,
        "sha256": "ededa9c0bf8761efed092c303b46c1c92de956838cba6249a33bedfd6d7363b4",
    },
    {
        "id": "gutenberg_voyage_beagle_944",
        "filename": "project_gutenberg_944_voyage_of_beagle.txt",
        "url": "https://www.gutenberg.org/cache/epub/944/pg944.txt",
        "bytes": 1_227_345,
        "sha256": "e13d57170a6dd9f7d97ddbc14cbde961dffbd95e8861ed199e5f00a525e7e52a",
    },
)


def _get_bytes(url: str, retries: int = 4) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MyLLM-V1-dataset-preparer/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_error}")


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _write_verified(path: Path, data: bytes, expected_bytes: int, expected_hash: str, hash_kind: str) -> None:
    actual_hash = hashlib.sha256(data).hexdigest() if hash_kind == "sha256" else _git_blob_sha(data)
    if len(data) != expected_bytes or actual_hash != expected_hash:
        raise ValueError(
            f"Integrity check failed for {path}: expected {expected_bytes} bytes/{expected_hash}, "
            f"got {len(data)} bytes/{actual_hash}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        existing_hash = hashlib.sha256(existing).hexdigest() if hash_kind == "sha256" else _git_blob_sha(existing)
        if len(existing) == expected_bytes and existing_hash == expected_hash:
            return
    path.write_bytes(data)


def _rsd_entries() -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{RSD_REPOSITORY}/git/trees/{RSD_COMMIT}?recursive=1"
    tree = json.loads(_get_bytes(url).decode("utf-8"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated RSD tree; refusing an incomplete download")
    entries = [
        item
        for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and item.get("path", "").endswith(".txt")
        and any(item["path"].startswith(prefix) for prefix in RSD_PREFIXES)
    ]
    entries.sort(key=lambda item: item["path"])
    total_bytes = sum(int(item["size"]) for item in entries)
    if len(entries) != RSD_DOCUMENTS or total_bytes != RSD_BYTES:
        raise RuntimeError(
            f"Pinned RSD selection changed: expected {RSD_DOCUMENTS} files/{RSD_BYTES} bytes, "
            f"found {len(entries)} files/{total_bytes} bytes"
        )
    return entries


def _download_rsd_file(raw_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    rel_path = str(entry["path"])
    target = raw_dir / "rsd" / rel_path
    expected_bytes = int(entry["size"])
    expected_sha = str(entry["sha"])
    if target.exists() and target.stat().st_size == expected_bytes and _git_blob_sha(target.read_bytes()) == expected_sha:
        data = target.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/{RSD_REPOSITORY}/{RSD_COMMIT}/{rel_path}"
        data = _get_bytes(url)
        _write_verified(target, data, expected_bytes, expected_sha, "git-blob")
    return {
        "source_id": "rsd_public_domain",
        "path": str(target.relative_to(raw_dir)),
        "source_path": rel_path,
        "bytes": len(data),
        "git_blob_sha": expected_sha,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def download_dataset(
    raw_dir: Path,
    max_download_mb: float,
    skip_rsd: bool = False,
    skip_gutenberg: bool = False,
    workers: int = 12,
) -> dict[str, Any]:
    expected_total = (0 if skip_rsd else RSD_BYTES) + (
        0 if skip_gutenberg else sum(int(item["bytes"]) for item in GUTENBERG)
    )
    if expected_total > max_download_mb * 1024 * 1024:
        raise ValueError(f"Expected download is {expected_total / (1024 * 1024):.1f} MiB, above the configured size limit")

    raw_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    if not skip_rsd:
        entries = _rsd_entries()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_download_rsd_file, raw_dir, entry): entry for entry in entries}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                files.append(result)
                print(f"RSD verified {index:03d}/{RSD_DOCUMENTS}", flush=True)

    if not skip_gutenberg:
        for source in GUTENBERG:
            target = raw_dir / source["filename"]
            if target.exists() and target.stat().st_size == source["bytes"] and hashlib.sha256(target.read_bytes()).hexdigest() == source["sha256"]:
                data = target.read_bytes()
            else:
                data = _get_bytes(source["url"])
                _write_verified(target, data, int(source["bytes"]), str(source["sha256"]), "sha256")
            files.append({
                "source_id": source["id"],
                "path": str(target.relative_to(raw_dir)),
                "url": source["url"],
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
            print(f"Verified {source['filename']} ({len(data):,} bytes)", flush=True)

    manifest = {
        "dataset": "MyLLM V1 curated starter corpus",
        "rsd_commit": RSD_COMMIT,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }
    (raw_dir / "raw_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned, license-documented MyLLM V1 starter corpus")
    parser.add_argument("--output", default="data/raw", help="Local raw-data directory")
    parser.add_argument("--max-download-mb", type=float, default=128.0, help="Abort if the planned download exceeds this size")
    parser.add_argument("--workers", type=int, default=12, help="Parallel RSD downloads")
    parser.add_argument("--skip-rsd", action="store_true")
    parser.add_argument("--skip-gutenberg", action="store_true")
    args = parser.parse_args()
    manifest = download_dataset(ROOT / args.output, args.max_download_mb, args.skip_rsd, args.skip_gutenberg, args.workers)
    print(f"Downloaded/verified {manifest['file_count']} files ({manifest['total_bytes']:,} bytes)")
    print("Raw manifest written: raw_manifest.json")


if __name__ == "__main__":
    main()
