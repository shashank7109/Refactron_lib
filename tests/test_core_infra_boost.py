"""Coverage boosts for core infra modules."""

from __future__ import annotations

import json
from pathlib import Path

import libcst as cst
import pytest

from refactron.core.analysis_result import FileAnalysisError
from refactron.core.cache import ASTCache
from refactron.core.credentials import (
    RefactronCredentials,
    credentials_path,
    delete_credentials,
    load_credentials,
    save_credentials,
)
from refactron.core.incremental import IncrementalAnalysisTracker
from refactron.core.parallel import ParallelProcessor


def test_credentials_roundtrip_and_bad_inputs(tmp_path: Path) -> None:
    cred_path = tmp_path / "credentials.json"
    creds = RefactronCredentials(
        api_base_url="https://api.example.test",
        access_token="tok",
        token_type="Bearer",
    )
    save_credentials(creds, cred_path)
    loaded = load_credentials(cred_path)
    assert loaded is not None
    assert loaded.api_base_url == "https://api.example.test"
    assert delete_credentials(cred_path) is True
    assert delete_credentials(cred_path) is False

    cred_path.write_text("not-json", encoding="utf-8")
    assert load_credentials(cred_path) is None
    cred_path.write_text(json.dumps(["bad"]), encoding="utf-8")
    assert load_credentials(cred_path) is None
    cred_path.write_text(json.dumps({"api_base_url": "", "access_token": "x"}), encoding="utf-8")
    assert load_credentials(cred_path) is None
    assert isinstance(credentials_path(), Path)


def test_ast_cache_get_put_corrupt_and_clear(tmp_path: Path) -> None:
    cache = ASTCache(cache_dir=tmp_path / "cache")
    path = tmp_path / "a.py"
    content = "x = 1\n"
    module = cst.parse_module(content)

    assert cache.get(path, content) is None
    cache.put(path, content, module, metadata={"k": 1})
    got = cache.get(path, content)
    assert got is not None
    assert got[1]["k"] == 1

    # Corrupt cache file should be handled and removed.
    bad_hash = cache._compute_hash("broken")
    bad_file = cache._get_cache_path(bad_hash)
    bad_file.write_bytes(b"\x80not-a-pickle")
    assert cache.get(path, "broken") is None
    assert not bad_file.exists()

    stats = cache.get_stats()
    assert stats["enabled"] is True
    cache.clear()
    assert cache.get_stats()["cache_file_count"] == 0


def test_parallel_processor_sequential_and_thread_modes(tmp_path: Path) -> None:
    files = [tmp_path / "ok.py", tmp_path / "bad.py"]
    for f in files:
        f.write_text("x=1\n", encoding="utf-8")

    def process_func(p: Path):
        if p.name == "bad.py":
            raise ValueError("boom")
        return None, None

    p_seq = ParallelProcessor(max_workers=1, use_processes=False, enabled=True)
    _, errors = p_seq.process_files(files, process_func)
    assert p_seq.enabled is False
    assert len(errors) == 1
    assert isinstance(errors[0], FileAnalysisError)

    p_thr = ParallelProcessor(max_workers=2, use_processes=False, enabled=True)
    results, errors = p_thr.process_files(files, lambda p: (None, None))
    assert results == []
    assert errors == []
    assert p_thr.get_config()["max_workers"] == 2


def test_incremental_tracker_changed_files_and_cleanup(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    tracker = IncrementalAnalysisTracker(state_file=state_file, enabled=True)
    file1 = tmp_path / "f1.py"
    file2 = tmp_path / "f2.py"
    file1.write_text("x=1\n", encoding="utf-8")
    file2.write_text("y=2\n", encoding="utf-8")

    assert tracker.has_file_changed(file1) is True
    tracker.update_file_state(file1)
    assert tracker.has_file_changed(file1) is False
    changed = tracker.get_changed_files([file1, file2])
    assert file2 in changed
    tracker.update_file_state(file2)
    tracker.save()
    assert state_file.exists()

    tracker.remove_file_state(file2)
    tracker.cleanup_missing_files({file1})
    stats = tracker.get_stats()
    assert stats["tracked_files"] >= 1

    tracker.clear()
    assert tracker.get_stats()["tracked_files"] == 0
