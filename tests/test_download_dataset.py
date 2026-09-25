from __future__ import annotations

import hashlib

import pytest

from scripts.download_dataset import _git_blob_sha, _write_verified, download_dataset


def test_git_blob_sha_matches_git_object_format() -> None:
    assert _git_blob_sha(b"abc") == "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"


def test_verified_writer_checks_size_and_checksum(tmp_path) -> None:
    content = b"small trusted source"
    target = tmp_path / "raw.txt"
    _write_verified(target, content, len(content), hashlib.sha256(content).hexdigest(), "sha256")
    assert target.read_bytes() == content
    with pytest.raises(ValueError, match="Integrity check failed"):
        _write_verified(target, content, len(content) + 1, hashlib.sha256(content).hexdigest(), "sha256")


def test_download_size_guard_rejects_before_network(tmp_path) -> None:
    with pytest.raises(ValueError, match="above the configured size limit"):
        download_dataset(tmp_path, max_download_mb=1.0)
    assert list(tmp_path.iterdir()) == []
