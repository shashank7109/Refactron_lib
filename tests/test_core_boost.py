"""Tests for core/memory_profiler.py, core/cache.py, core/repositories.py"""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from refactron.core.cache import ASTCache
from refactron.core.memory_profiler import (
    MemoryProfiler,
    MemorySnapshot,
    estimate_file_size_mb,
    stream_large_file,
)

# ──────────────────────────── MemorySnapshot ──────────────────────────────────


class TestMemorySnapshot:
    def test_str(self):
        s = MemorySnapshot(rss_mb=100.0, vms_mb=200.0, percent=50.0, available_mb=1000.0)
        assert "RSS" in str(s) and "100.00" in str(s)


# ──────────────────────────── MemoryProfiler ──────────────────────────────────


class TestMemoryProfiler:
    def test_init_with_psutil(self):
        mock_psutil = MagicMock()
        mock_proc = MagicMock()
        mock_psutil.Process.return_value = mock_proc
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            mp = MemoryProfiler()
        assert mp._psutil_available

    def test_init_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            mp = MemoryProfiler()
        assert not mp._psutil_available

    def test_get_current_memory_fallback(self):
        mp = MemoryProfiler()
        mp._psutil_available = False
        snap = mp.get_current_memory()
        assert snap.rss_mb == 0.0

    def test_get_current_memory_with_psutil(self):
        mp = MemoryProfiler()
        mp._psutil_available = True
        mem_info = MagicMock()
        mem_info.rss = 100 * 1024 * 1024
        mem_info.vms = 200 * 1024 * 1024
        virtual_mem = MagicMock()
        virtual_mem.percent = 50.0
        virtual_mem.available = 1024 * 1024 * 1024
        mp._process = MagicMock()
        mp._process.memory_info.return_value = mem_info
        mp._psutil = MagicMock()
        mp._psutil.virtual_memory.return_value = virtual_mem
        snap = mp.get_current_memory()
        assert snap.rss_mb == pytest.approx(100.0)

    def test_get_current_memory_psutil_exception(self):
        mp = MemoryProfiler()
        mp._psutil_available = True
        mp._process = MagicMock()
        mp._process.memory_info.side_effect = Exception("read error")
        snap = mp.get_current_memory()
        assert snap.rss_mb == 0.0

    def test_snapshot_disabled(self):
        mp = MemoryProfiler(enabled=False)
        snap = mp.snapshot("label")
        assert snap.rss_mb == 0.0

    def test_snapshot_stores(self):
        mp = MemoryProfiler()
        mp.get_current_memory = MagicMock(return_value=MemorySnapshot(1.0, 2.0, 50.0, 100.0))
        mp.snapshot("test")
        assert "test" in mp._snapshots

    def test_compare_missing_label(self):
        mp = MemoryProfiler()
        result = mp.compare("a", "b")
        assert result == {}

    def test_compare_snapshots(self):
        mp = MemoryProfiler()
        mp._snapshots["start"] = MemorySnapshot(100.0, 200.0, 50.0, 500.0)
        mp._snapshots["end"] = MemorySnapshot(110.0, 210.0, 52.0, 490.0)
        diff = mp.compare("start", "end")
        assert diff["rss_mb_diff"] == pytest.approx(10.0)

    def test_profile_function_disabled(self):
        mp = MemoryProfiler(enabled=False)
        result, info = mp.profile_function(lambda: 42)
        assert result == 42 and info == {}

    def test_profile_function_enabled(self):
        mp = MemoryProfiler()
        mp.get_current_memory = MagicMock(return_value=MemorySnapshot(1.0, 2.0, 50.0, 100.0))
        result, diff = mp.profile_function(lambda: "hello")
        assert result == "hello"

    def test_optimize_for_large_file_above_threshold(self):
        mp = MemoryProfiler()
        assert mp.optimize_for_large_files(10.0) is True

    def test_optimize_for_large_file_below_threshold(self):
        mp = MemoryProfiler()
        assert mp.optimize_for_large_files(1.0) is False

    def test_optimize_for_large_file_custom_threshold(self):
        mp = MemoryProfiler()
        assert mp.optimize_for_large_files(3.0, threshold_mb=2.0) is True

    def test_check_memory_pressure_no_psutil(self):
        mp = MemoryProfiler()
        mp._psutil_available = False
        assert mp.check_memory_pressure() is False

    def test_check_memory_pressure_high(self):
        mp = MemoryProfiler(pressure_threshold_percent=60.0)
        mp._psutil_available = True
        mp.get_current_memory = MagicMock(return_value=MemorySnapshot(1.0, 2.0, 90.0, 1000.0))
        assert mp.check_memory_pressure() is True

    def test_check_memory_pressure_low_available_mb(self):
        mp = MemoryProfiler(pressure_threshold_available_mb=1000.0)
        mp._psutil_available = True
        mp.get_current_memory = MagicMock(return_value=MemorySnapshot(1.0, 2.0, 20.0, 100.0))
        assert mp.check_memory_pressure() is True

    def test_suggest_gc_disabled(self):
        mp = MemoryProfiler(enabled=False)
        mp.suggest_gc()  # No error

    def test_suggest_gc_enabled_no_pressure(self):
        mp = MemoryProfiler()
        mp._psutil_available = False
        mp.suggest_gc()  # No error

    def test_get_stats(self):
        mp = MemoryProfiler()
        mp.get_current_memory = MagicMock(return_value=MemorySnapshot(1.0, 2.0, 50.0, 100.0))
        stats = mp.get_stats()
        assert "enabled" in stats

    def test_clear_snapshots(self):
        mp = MemoryProfiler()
        mp._snapshots["x"] = MagicMock()
        mp.clear_snapshots()
        assert len(mp._snapshots) == 0


class TestStreamLargeFile:
    def test_streams_chunks(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_text("a" * 100)
        chunks = list(stream_large_file(str(f), chunk_size=10))
        assert "".join(chunks) == "a" * 100


class TestEstimateFileSizeMb:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "f.py"
        f.write_bytes(b"x" * 1024)
        mb = estimate_file_size_mb(str(f))
        assert mb > 0

    def test_missing_file(self):
        result = estimate_file_size_mb("/nonexistent/path.py")
        assert result == 0.0


# ──────────────────────────── ASTCache ────────────────────────────────────────


class TestASTCache:
    def test_disabled_get_returns_none(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=False)
        assert cache.get(Path("a.py"), "x=1") is None

    def test_disabled_put_does_nothing(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=False)
        cache.put(Path("a.py"), "x=1", MagicMock())  # No error

    def test_put_and_get_memory_hit(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=True)
        mock_module = MagicMock()
        cache.put(Path("a.py"), "x=1", mock_module, metadata={"key": "val"})
        result = cache.get(Path("a.py"), "x=1")
        assert result is not None
        assert cache.stats["memory_hits"] == 1

    def test_disk_hit(self, tmp_path):
        import libcst as cst

        cache = ASTCache(cache_dir=tmp_path, enabled=True)
        # Use a real cst.Module so pickle round-trip works
        real_module = cst.parse_module("x = 1\n")
        cache.put(Path("a.py"), "x=1", real_module)
        # Clear memory cache to force disk hit
        cache._memory_cache.clear()
        result = cache.get(Path("a.py"), "x=1")
        assert result is not None
        assert cache.stats["disk_hits"] == 1

    def test_cache_miss(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=True)
        result = cache.get(Path("a.py"), "missing content")
        assert result is None
        assert cache.stats["misses"] == 1

    def test_corrupted_disk_cache_removed(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=True)
        content_hash = cache._compute_hash("x=1")
        cache_path = cache._get_cache_path(content_hash)
        cache_path.write_bytes(b"not valid pickle")
        result = cache.get(Path("a.py"), "x=1")
        assert result is None
        assert not cache_path.exists()

    def test_clear(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=True)
        cache.put(Path("a.py"), "x=1", MagicMock())
        cache.clear()
        assert len(cache._memory_cache) == 0
        assert cache.stats["hits"] == 0

    def test_get_stats(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=True)
        cache.put(Path("a.py"), "x=1", MagicMock())
        cache.get(Path("a.py"), "x=1")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate_percent"] == 100.0

    def test_cleanup_when_size_exceeds_limit(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=True, max_cache_size_mb=1)
        # Add many fake cache files
        for i in range(5):
            cf = cache.cache_dir / f"fake{i}.cache"
            cf.write_bytes(b"x" * (300 * 1024))  # 300KB each = 1.5MB total
        cache._cleanup_if_needed()  # Should remove some files

    def test_put_write_error_handled(self, tmp_path):
        cache = ASTCache(cache_dir=tmp_path, enabled=True)
        with patch("builtins.open", side_effect=OSError("disk full")):
            cache.put(Path("a.py"), "x=1", MagicMock())  # Should not raise

    def test_init_with_none_cache_dir(self):
        cache = ASTCache(cache_dir=None, enabled=True)
        assert "refactron_ast_cache" in str(cache.cache_dir)


# ──────────────────────────── core/repositories.py ────────────────────────────


class TestRepositoryFromDict:
    def test_from_dict_full(self):
        from refactron.core.repositories import Repository

        data = {
            "id": 1,
            "name": "repo",
            "full_name": "user/repo",
            "description": "desc",
            "private": False,
            "html_url": "https://github.com/user/repo",
            "clone_url": "https://github.com/user/repo.git",
            "ssh_url": "git@github.com:user/repo.git",
            "default_branch": "main",
            "language": "Python",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        repo = Repository.from_dict(data)
        assert repo.name == "repo"

    def test_from_dict_optional_fields(self):
        from refactron.core.repositories import Repository

        data = {
            "id": 2,
            "name": "r",
            "full_name": "u/r",
            "private": True,
            "html_url": "https://github.com/u/r",
            "clone_url": "https://github.com/u/r.git",
            "ssh_url": "git@github.com:u/r.git",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        repo = Repository.from_dict(data)
        assert repo.description is None
        assert repo.default_branch == "main"


class TestListRepositories:
    def test_not_authenticated(self):
        from refactron.core.repositories import list_repositories

        with patch("refactron.core.repositories.load_credentials", return_value=None):
            with pytest.raises(RuntimeError, match="Not authenticated"):
                list_repositories("https://api.test")

    def test_no_access_token(self):
        from refactron.core.repositories import list_repositories

        creds = MagicMock()
        creds.access_token = None
        with patch("refactron.core.repositories.load_credentials", return_value=creds):
            with pytest.raises(RuntimeError, match="No access token"):
                list_repositories("https://api.test")

    def test_expired_token(self):
        from datetime import datetime, timedelta, timezone

        from refactron.core.repositories import list_repositories

        creds = MagicMock()
        creds.access_token = "token"
        creds.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        with patch("refactron.core.repositories.load_credentials", return_value=creds):
            with pytest.raises(RuntimeError, match="expired"):
                list_repositories("https://api.test")

    def test_success_list_response(self):
        import io
        from urllib.response import addinfourl

        from refactron.core.repositories import list_repositories

        creds = MagicMock()
        creds.access_token = "token"
        creds.expires_at = None

        repo_data = [
            {
                "id": 1,
                "name": "r",
                "full_name": "u/r",
                "private": False,
                "html_url": "h",
                "clone_url": "c",
                "ssh_url": "s",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        ]
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(repo_data).encode()

        with patch("refactron.core.repositories.load_credentials", return_value=creds), patch(
            "refactron.core.repositories.urlopen", return_value=mock_resp
        ):
            repos = list_repositories("https://api.test")
        assert len(repos) == 1

    def test_dict_wrapper_response(self):
        from refactron.core.repositories import list_repositories

        creds = MagicMock()
        creds.access_token = "token"
        creds.expires_at = None

        repo_data = {
            "repositories": [
                {
                    "id": 1,
                    "name": "r",
                    "full_name": "u/r",
                    "private": False,
                    "html_url": "h",
                    "clone_url": "c",
                    "ssh_url": "s",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(repo_data).encode()

        with patch("refactron.core.repositories.load_credentials", return_value=creds), patch(
            "refactron.core.repositories.urlopen", return_value=mock_resp
        ):
            repos = list_repositories("https://api.test")
        assert len(repos) == 1

    def test_http_401_raises(self):
        from urllib.error import HTTPError

        from refactron.core.repositories import list_repositories

        creds = MagicMock()
        creds.access_token = "token"
        creds.expires_at = None

        err = HTTPError("url", 401, "Unauthorized", {}, None)
        err.read = lambda: b'{"message": "bad token"}'

        with patch("refactron.core.repositories.load_credentials", return_value=creds), patch(
            "refactron.core.repositories.urlopen", side_effect=err
        ):
            with pytest.raises(RuntimeError, match="401"):
                list_repositories("https://api.test")

    def test_http_403_raises(self):
        from urllib.error import HTTPError

        from refactron.core.repositories import list_repositories

        creds = MagicMock()
        creds.access_token = "token"
        creds.expires_at = None

        err = HTTPError("url", 403, "Forbidden", {}, None)

        with patch("refactron.core.repositories.load_credentials", return_value=creds), patch(
            "refactron.core.repositories.urlopen", side_effect=err
        ):
            with pytest.raises(RuntimeError, match="GitHub access denied"):
                list_repositories("https://api.test")

    def test_url_error_raises(self):
        from urllib.error import URLError

        from refactron.core.repositories import list_repositories

        creds = MagicMock()
        creds.access_token = "token"
        creds.expires_at = None

        with patch("refactron.core.repositories.load_credentials", return_value=creds), patch(
            "refactron.core.repositories.urlopen", side_effect=URLError("network down")
        ):
            with pytest.raises(RuntimeError, match="Network error"):
                list_repositories("https://api.test")

    def test_invalid_json_raises(self):
        from refactron.core.repositories import list_repositories

        creds = MagicMock()
        creds.access_token = "token"
        creds.expires_at = None

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"not json"

        with patch("refactron.core.repositories.load_credentials", return_value=creds), patch(
            "refactron.core.repositories.urlopen", return_value=mock_resp
        ):
            with pytest.raises(RuntimeError, match="Invalid JSON"):
                list_repositories("https://api.test")
