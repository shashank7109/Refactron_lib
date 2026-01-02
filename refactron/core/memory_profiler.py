"""Memory profiling and optimization utilities."""

import gc
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage percentage
    available_mb: float  # Available memory in MB
    
    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"RSS: {self.rss_mb:.2f} MB, "
            f"VMS: {self.vms_mb:.2f} MB, "
            f"Usage: {self.percent:.1f}%, "
            f"Available: {self.available_mb:.2f} MB"
        )


class MemoryProfiler:
    """
    Memory profiling and optimization utilities.
    
    Helps track and optimize memory usage for large codebases.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the memory profiler.
        
        Args:
            enabled: Whether memory profiling is enabled.
        """
        self.enabled = enabled
        self._snapshots: Dict[str, MemorySnapshot] = {}
        
        # Try to import psutil for accurate memory tracking
        self._psutil_available = False
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._psutil_available = True
            logger.debug("Memory profiler initialized with psutil")
        except ImportError:
            logger.debug("Memory profiler initialized without psutil (limited functionality)")
    
    def get_current_memory(self) -> MemorySnapshot:
        """
        Get current memory usage snapshot.
        
        Returns:
            MemorySnapshot with current memory usage.
        """
        if self._psutil_available:
            try:
                mem_info = self._process.memory_info()
                virtual_mem = self._psutil.virtual_memory()
                
                return MemorySnapshot(
                    rss_mb=mem_info.rss / 1024 / 1024,
                    vms_mb=mem_info.vms / 1024 / 1024,
                    percent=virtual_mem.percent,
                    available_mb=virtual_mem.available / 1024 / 1024,
                )
            except Exception as e:
                logger.warning(f"Failed to get memory info: {e}")
        
        # Fallback: return basic info without psutil
        return MemorySnapshot(
            rss_mb=0.0,
            vms_mb=0.0,
            percent=0.0,
            available_mb=0.0,
        )
    
    def snapshot(self, label: str) -> MemorySnapshot:
        """
        Take a memory snapshot with a label.
        
        Args:
            label: Label for this snapshot.
            
        Returns:
            MemorySnapshot with current memory usage.
        """
        if not self.enabled:
            return MemorySnapshot(0.0, 0.0, 0.0, 0.0)
        
        snapshot = self.get_current_memory()
        self._snapshots[label] = snapshot
        logger.debug(f"Memory snapshot '{label}': {snapshot}")
        return snapshot
    
    def compare(self, start_label: str, end_label: str) -> Dict[str, float]:
        """
        Compare two memory snapshots.
        
        Args:
            start_label: Label of the starting snapshot.
            end_label: Label of the ending snapshot.
            
        Returns:
            Dictionary with memory differences.
        """
        if start_label not in self._snapshots or end_label not in self._snapshots:
            logger.warning(f"Cannot compare: missing snapshot(s)")
            return {}
        
        start = self._snapshots[start_label]
        end = self._snapshots[end_label]
        
        return {
            "rss_mb_diff": end.rss_mb - start.rss_mb,
            "vms_mb_diff": end.vms_mb - start.vms_mb,
            "percent_diff": end.percent - start.percent,
        }
    
    def profile_function(
        self,
        func: Callable[..., T],
        *args: Any,
        label: Optional[str] = None,
        **kwargs: Any
    ) -> tuple[T, Dict[str, Any]]:
        """
        Profile memory usage of a function call.
        
        Args:
            func: Function to profile.
            *args: Positional arguments for the function.
            label: Optional label for logging.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            Tuple of (function result, profiling info).
        """
        if not self.enabled:
            result = func(*args, **kwargs)
            return result, {}
        
        func_name = label or func.__name__
        
        # Take snapshot before
        self.snapshot(f"{func_name}_start")
        
        # Run garbage collection for accurate measurement
        gc.collect()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Take snapshot after
        self.snapshot(f"{func_name}_end")
        
        # Get differences
        diff = self.compare(f"{func_name}_start", f"{func_name}_end")
        
        if diff:
            logger.info(
                f"Memory profile for '{func_name}': "
                f"RSS +{diff['rss_mb_diff']:.2f} MB"
            )
        
        return result, diff
    
    def optimize_for_large_files(self, file_size_mb: float) -> bool:
        """
        Determine if special optimization is needed for a large file.
        
        Args:
            file_size_mb: File size in megabytes.
            
        Returns:
            True if optimization is recommended.
        """
        # Threshold for "large" files
        large_file_threshold_mb = 5.0
        
        if file_size_mb > large_file_threshold_mb:
            logger.info(f"Large file detected ({file_size_mb:.2f} MB), enabling optimizations")
            return True
        
        return False
    
    def check_memory_pressure(self) -> bool:
        """
        Check if the system is under memory pressure.
        
        Returns:
            True if memory pressure is high (>80% usage).
        """
        if not self._psutil_available:
            return False
        
        snapshot = self.get_current_memory()
        
        # Consider memory pressure high if >80% used or <500MB available
        high_pressure = snapshot.percent > 80.0 or snapshot.available_mb < 500.0
        
        if high_pressure:
            logger.warning(f"High memory pressure detected: {snapshot}")
        
        return high_pressure
    
    def suggest_gc(self) -> None:
        """Suggest garbage collection if memory pressure is high."""
        if not self.enabled:
            return
        
        if self.check_memory_pressure():
            logger.info("Running garbage collection due to memory pressure")
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory profiling statistics.
        
        Returns:
            Dictionary containing statistics.
        """
        current = self.get_current_memory()
        
        return {
            "enabled": self.enabled,
            "psutil_available": self._psutil_available,
            "current_rss_mb": current.rss_mb,
            "current_vms_mb": current.vms_mb,
            "current_percent": current.percent,
            "available_mb": current.available_mb,
            "snapshots_count": len(self._snapshots),
        }
    
    def clear_snapshots(self) -> None:
        """Clear all stored snapshots."""
        self._snapshots.clear()


# Memory optimization decorators and utilities

def stream_large_file(file_path: str, chunk_size: int = 8192):
    """
    Stream a large file in chunks instead of reading all at once.
    
    Args:
        file_path: Path to the file.
        chunk_size: Size of each chunk in bytes.
        
    Yields:
        Chunks of file content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def estimate_file_size_mb(file_path: str) -> float:
    """
    Estimate file size in megabytes.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        File size in MB.
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / 1024 / 1024
    except Exception as e:
        logger.warning(f"Failed to get file size for {file_path}: {e}")
        return 0.0
