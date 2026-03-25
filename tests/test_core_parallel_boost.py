"""Tests for core/parallel.py"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from refactron.core.analysis_result import FileAnalysisError
from refactron.core.models import FileMetrics
from refactron.core.parallel import ParallelProcessor


def make_metrics(path):
    m = MagicMock(spec=FileMetrics)
    m.file_path = path
    return m


def make_error(path):
    return FileAnalysisError(
        file_path=path, error_message="fail", error_type="Error", recovery_suggestion="check file"
    )


def success_func(p):
    return make_metrics(p), None


def error_func(p):
    return None, make_error(p)


def raises_func(p):
    raise RuntimeError("unexpected")


class TestParallelProcessorInit:
    def test_default_workers_capped_at_8(self):
        with patch("multiprocessing.cpu_count", return_value=16):
            pp = ParallelProcessor()
        assert pp.max_workers <= 8

    def test_explicit_workers(self):
        pp = ParallelProcessor(max_workers=4)
        assert pp.max_workers == 4

    def test_max_workers_zero_forced_to_1(self):
        pp = ParallelProcessor(max_workers=0)
        assert pp.max_workers == 1

    def test_single_worker_disables_parallel(self):
        pp = ParallelProcessor(max_workers=1)
        assert pp.enabled is False

    def test_get_config(self):
        pp = ParallelProcessor(max_workers=2, use_processes=False, enabled=True)
        cfg = pp.get_config()
        assert cfg["max_workers"] == 2
        assert cfg["use_processes"] is False


class TestSequentialProcessing:
    def test_empty_files(self):
        pp = ParallelProcessor(enabled=False)
        results, errors = pp.process_files([], success_func)
        assert results == [] and errors == []

    def test_single_file_success(self):
        pp = ParallelProcessor(enabled=False)
        files = [Path("a.py")]
        results, errors = pp.process_files(files, success_func)
        assert len(results) == 1 and len(errors) == 0

    def test_single_file_error(self):
        pp = ParallelProcessor(enabled=False)
        results, errors = pp.process_files([Path("a.py")], error_func)
        assert len(results) == 0 and len(errors) == 1

    def test_single_file_exception(self):
        pp = ParallelProcessor(enabled=False)
        results, errors = pp.process_files([Path("a.py")], raises_func)
        assert len(errors) == 1

    def test_progress_callback(self):
        pp = ParallelProcessor(enabled=False)
        calls = []
        pp.process_files(
            [Path("a.py"), Path("b.py")],
            success_func,
            progress_callback=lambda c, t: calls.append((c, t)),
        )
        assert calls == [(1, 2), (2, 2)]


class TestThreadedProcessing:
    def test_two_files_threads(self):
        pp = ParallelProcessor(max_workers=2, use_processes=False, enabled=True)
        files = [Path("a.py"), Path("b.py")]
        results, errors = pp.process_files(files, success_func)
        assert len(results) == 2

    def test_thread_error_handling(self):
        pp = ParallelProcessor(max_workers=2, use_processes=False, enabled=True)
        results, errors = pp.process_files([Path("a.py"), Path("b.py")], raises_func)
        assert len(errors) == 2

    def test_thread_progress_callback(self):
        pp = ParallelProcessor(max_workers=2, use_processes=False, enabled=True)
        calls = []
        pp.process_files(
            [Path("a.py"), Path("b.py")],
            success_func,
            progress_callback=lambda c, t: calls.append(c),
        )
        assert len(calls) == 2

    def test_single_file_goes_sequential(self):
        pp = ParallelProcessor(max_workers=4, use_processes=False, enabled=True)
        results, errors = pp.process_files([Path("a.py")], success_func)
        assert len(results) == 1


class TestProcessPoolProcessing:
    def test_process_pool_falls_back_on_exception(self):
        pp = ParallelProcessor(max_workers=2, use_processes=True, enabled=True)
        with patch(
            "refactron.core.parallel.ProcessPoolExecutor", side_effect=Exception("spawn fail")
        ):
            results, errors = pp.process_files([Path("a.py")], success_func)
        assert len(results) == 1

    def test_process_pool_success(self):
        pp = ParallelProcessor(max_workers=2, use_processes=True, enabled=True)
        mock_future = MagicMock()
        mock_future.result.return_value = (make_metrics(Path("a.py")), None)
        mock_exec = MagicMock()
        mock_exec.__enter__ = lambda s: s
        mock_exec.__exit__ = MagicMock(return_value=False)
        mock_exec.submit = lambda f, p: mock_future
        with patch("refactron.core.parallel.ProcessPoolExecutor", return_value=mock_exec), patch(
            "refactron.core.parallel.as_completed", return_value=[mock_future]
        ):
            pp._process_parallel_processes([Path("a.py")], success_func)
