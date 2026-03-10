"""Tests for performance optimization features."""

import tempfile
import time
from pathlib import Path

from refactron import Refactron
from refactron.core.cache import ASTCache
from refactron.core.config import RefactronConfig
from refactron.core.incremental import IncrementalAnalysisTracker
from refactron.core.memory_profiler import MemoryProfiler, MemorySnapshot
from refactron.core.parallel import ParallelProcessor


class TestASTCache:
    """Tests for AST caching."""

    def test_cache_initialization(self):
        """Test cache can be initialized."""
        cache = ASTCache(enabled=True)
        assert cache.enabled
        assert cache.cache_dir.exists()

    def test_cache_disabled(self):
        """Test cache can be disabled."""
        cache = ASTCache(enabled=False)
        assert not cache.enabled

    def test_cache_put_and_get(self):
        """Test storing and retrieving from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ASTCache(cache_dir=Path(tmpdir), enabled=True)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def foo(): pass")

            content = test_file.read_text()

            # First get should miss
            result = cache.get(test_file, content)
            assert result is None
            assert cache.stats["misses"] == 1

            # Put in cache
            import libcst as cst

            module = cst.parse_module(content)
            cache.put(test_file, content, module, {"test": "metadata"})

            # Second get should hit
            result = cache.get(test_file, content)
            assert result is not None
            assert cache.stats["hits"] == 1

            cached_module, cached_metadata = result
            assert cached_metadata["test"] == "metadata"

    def test_cache_invalidation_on_content_change(self):
        """Test cache is invalidated when content changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ASTCache(cache_dir=Path(tmpdir), enabled=True)

            test_file = Path(tmpdir) / "test.py"
            content1 = "def foo(): pass"
            content2 = "def bar(): pass"

            test_file.write_text(content1)

            # Cache first content
            import libcst as cst

            module1 = cst.parse_module(content1)
            cache.put(test_file, content1, module1)

            # Get with different content should miss
            result = cache.get(test_file, content2)
            assert result is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ASTCache(enabled=True)
        stats = cache.get_stats()

        assert "enabled" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate_percent" in stats
        assert stats["enabled"] is True

    def test_cache_clear(self):
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ASTCache(cache_dir=Path(tmpdir), enabled=True)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def foo(): pass")
            content = test_file.read_text()

            import libcst as cst

            module = cst.parse_module(content)
            cache.put(test_file, content, module)

            # Clear cache
            cache.clear()

            # Get should miss after clear
            result = cache.get(test_file, content)
            assert result is None


class TestIncrementalAnalysis:
    """Tests for incremental analysis tracking."""

    def test_tracker_initialization(self):
        """Test tracker can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            tracker = IncrementalAnalysisTracker(state_file=state_file, enabled=True)
            assert tracker.enabled

    def test_tracker_disabled(self):
        """Test tracker can be disabled."""
        tracker = IncrementalAnalysisTracker(enabled=False)
        assert not tracker.enabled

    def test_new_file_detected(self):
        """Test that new files are detected as changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            tracker = IncrementalAnalysisTracker(state_file=state_file, enabled=True)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def foo(): pass")

            # New file should be detected as changed
            assert tracker.has_file_changed(test_file)

    def test_unchanged_file_not_detected(self):
        """Test that unchanged files are not detected as changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            tracker = IncrementalAnalysisTracker(state_file=state_file, enabled=True)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def foo(): pass")

            # Update state
            tracker.update_file_state(test_file)

            # File should not be detected as changed
            assert not tracker.has_file_changed(test_file)

    def test_modified_file_detected(self):
        """Test that modified files are detected as changed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            tracker = IncrementalAnalysisTracker(state_file=state_file, enabled=True)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def foo(): pass")

            # Update state
            tracker.update_file_state(test_file)

            # Modify file
            time.sleep(0.1)  # Ensure mtime changes
            test_file.write_text("def bar(): pass")

            # File should be detected as changed
            assert tracker.has_file_changed(test_file)

    def test_get_changed_files(self):
        """Test filtering changed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            tracker = IncrementalAnalysisTracker(state_file=state_file, enabled=True)

            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"
            file1.write_text("def foo(): pass")
            file2.write_text("def bar(): pass")

            # Update state for file1
            tracker.update_file_state(file1)

            # Get changed files (only file2 should be changed)
            changed = tracker.get_changed_files([file1, file2])
            assert len(changed) == 1
            assert file2 in changed

    def test_state_persistence(self):
        """Test that state is persisted across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def foo(): pass")

            # First tracker instance
            tracker1 = IncrementalAnalysisTracker(state_file=state_file, enabled=True)
            tracker1.update_file_state(test_file)
            tracker1.save()

            # Second tracker instance
            tracker2 = IncrementalAnalysisTracker(state_file=state_file, enabled=True)

            # File should not be detected as changed
            assert not tracker2.has_file_changed(test_file)


class TestParallelProcessing:
    """Tests for parallel processing."""

    def test_processor_initialization(self):
        """Test processor can be initialized."""
        processor = ParallelProcessor(max_workers=4, enabled=True)
        assert processor.enabled
        assert processor.max_workers == 4

    def test_processor_disabled(self):
        """Test processor can be disabled."""
        processor = ParallelProcessor(enabled=False)
        assert not processor.enabled

    def test_processor_auto_workers(self):
        """Test automatic worker count selection."""
        processor = ParallelProcessor(max_workers=None, enabled=True)
        assert processor.max_workers >= 1
        assert processor.max_workers <= 8

    def test_sequential_processing(self):
        """Test sequential file processing."""
        processor = ParallelProcessor(enabled=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(3):
                file_path = Path(tmpdir) / f"test{i}.py"
                file_path.write_text(f"def func{i}(): pass")
                files.append(file_path)

            def process_func(file_path):
                # Simulate processing
                return None, None

            results, errors = processor.process_files(files, process_func)
            assert len(results) == 0  # All return None
            assert len(errors) == 0


class TestMemoryProfiler:
    """Tests for memory profiling."""

    def test_profiler_initialization(self):
        """Test profiler can be initialized."""
        profiler = MemoryProfiler(enabled=True)
        assert profiler.enabled

    def test_profiler_disabled(self):
        """Test profiler can be disabled."""
        profiler = MemoryProfiler(enabled=False)
        assert not profiler.enabled

    def test_get_current_memory(self):
        """Test getting current memory snapshot."""
        profiler = MemoryProfiler(enabled=True)
        snapshot = profiler.get_current_memory()

        assert isinstance(snapshot, MemorySnapshot)
        # Values might be 0 if psutil is not available
        assert snapshot.rss_mb >= 0

    def test_snapshot_labeling(self):
        """Test taking labeled snapshots."""
        profiler = MemoryProfiler(enabled=True)

        profiler.snapshot("test_label")
        assert "test_label" in profiler._snapshots

    def test_profile_function(self):
        """Test profiling a function."""
        profiler = MemoryProfiler(enabled=True)

        def test_func(x, y):
            return x + y

        result, diff = profiler.profile_function(test_func, 1, 2)
        assert result == 3
        assert isinstance(diff, dict)

    def test_memory_pressure_check(self):
        """Test checking memory pressure."""
        profiler = MemoryProfiler(enabled=True)

        # Should return a boolean
        pressure = profiler.check_memory_pressure()
        assert isinstance(pressure, bool)


class TestPerformanceIntegration:
    """Integration tests for performance features."""

    def test_refactron_with_performance_features(self):
        """Test Refactron with all performance features enabled."""
        config = RefactronConfig(
            enable_ast_cache=True,
            enable_incremental_analysis=True,
            enable_parallel_processing=True,
            enable_memory_profiling=True,
        )

        refactron = Refactron(config)

        # Verify components are initialized
        assert refactron.ast_cache.enabled
        assert refactron.incremental_tracker.enabled
        assert refactron.parallel_processor.enabled
        assert refactron.memory_profiler.enabled

    def test_refactron_analyze_with_caching(self):
        """Test analysis with caching components initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(
                """
def simple_function():
    '''A simple function.'''
    return 42
"""
            )

            config = RefactronConfig(
                enable_ast_cache=True,
                enable_incremental_analysis=False,
                enable_parallel_processing=False,
            )
            refactron = Refactron(config)

            # First analysis
            result1 = refactron.analyze(test_file)
            assert result1.total_files == 1

            # Second analysis
            result2 = refactron.analyze(test_file)
            assert result2.total_files == 1

            # Check cache stats - cache component exists
            stats = refactron.get_performance_stats()
            assert "ast_cache" in stats
            assert stats["ast_cache"]["enabled"] is True

    def test_refactron_incremental_analysis(self):
        """Test incremental analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"
            file1.write_text("def foo(): pass")
            file2.write_text("def bar(): pass")

            config = RefactronConfig(
                enable_ast_cache=False,
                enable_incremental_analysis=True,
                enable_parallel_processing=False,
            )
            refactron = Refactron(config)

            # First analysis - both files analyzed
            result1 = refactron.analyze(tmpdir)
            assert result1.total_files == 2

            # Second analysis - no files changed, should skip both
            result2 = refactron.analyze(tmpdir)
            assert result2.total_files == 0

            # Modify one file
            time.sleep(0.1)
            file1.write_text("def baz(): pass")

            # Third analysis - only modified file analyzed
            result3 = refactron.analyze(tmpdir)
            assert result3.total_files == 1

    def test_performance_stats(self):
        """Test getting performance statistics."""
        config = RefactronConfig(
            enable_ast_cache=True,
            enable_incremental_analysis=True,
            enable_parallel_processing=True,
            enable_memory_profiling=True,
        )

        refactron = Refactron(config)
        stats = refactron.get_performance_stats()

        assert "ast_cache" in stats
        assert "incremental_analysis" in stats
        assert "parallel_processing" in stats
        assert "memory_profiler" in stats

    def test_clear_caches(self):
        """Test clearing all caches."""
        config = RefactronConfig(
            enable_ast_cache=True,
            enable_incremental_analysis=True,
        )

        refactron = Refactron(config)

        # Clear caches should not raise errors
        refactron.clear_caches()

        # Verify caches are cleared
        stats = refactron.get_performance_stats()
        assert stats["ast_cache"]["cache_file_count"] == 0
        assert stats["incremental_analysis"]["tracked_files"] == 0
