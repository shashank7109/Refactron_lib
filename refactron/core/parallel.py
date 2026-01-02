"""Parallel processing utilities for performance optimization."""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from refactron.core.analysis_result import FileAnalysisError
from refactron.core.models import FileMetrics

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    Parallel processing manager for analyzing multiple files concurrently.
    
    Supports both multiprocessing and threading based on the task type.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = True,
        enabled: bool = True,
    ):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes/threads. 
                        If None, uses CPU count.
            use_processes: If True, uses multiprocessing; if False, uses threading.
            enabled: Whether parallel processing is enabled.
        """
        self.enabled = enabled
        self.use_processes = use_processes
        
        if max_workers is None:
            # Use CPU count, but cap at reasonable limits
            cpu_count = multiprocessing.cpu_count()
            # Don't use more than 8 workers to avoid resource exhaustion
            self.max_workers = min(cpu_count, 8)
        else:
            self.max_workers = max(1, max_workers)
        
        # Disable parallel processing if only 1 worker
        if self.max_workers == 1:
            self.enabled = False
        
        logger.debug(
            f"Parallel processor initialized: "
            f"enabled={self.enabled}, workers={self.max_workers}, "
            f"mode={'processes' if use_processes else 'threads'}"
        )
    
    def process_files(
        self,
        files: List[Path],
        process_func: Callable[[Path], Tuple[Optional[FileMetrics], Optional[FileAnalysisError]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[FileMetrics], List[FileAnalysisError]]:
        """
        Process multiple files in parallel.
        
        Args:
            files: List of file paths to process.
            process_func: Function to process a single file. Should return 
                         (FileMetrics, None) on success or (None, FileAnalysisError) on error.
            progress_callback: Optional callback for progress updates (completed, total).
            
        Returns:
            Tuple of (successful results, failed files).
        """
        if not self.enabled or len(files) <= 1:
            # Process sequentially if disabled or only one file
            return self._process_sequential(files, process_func, progress_callback)
        
        # Process in parallel
        if self.use_processes:
            return self._process_parallel_processes(files, process_func, progress_callback)
        else:
            return self._process_parallel_threads(files, process_func, progress_callback)
    
    def _process_sequential(
        self,
        files: List[Path],
        process_func: Callable[[Path], Tuple[Optional[FileMetrics], Optional[FileAnalysisError]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[FileMetrics], List[FileAnalysisError]]:
        """Process files sequentially."""
        results: List[FileMetrics] = []
        errors: List[FileAnalysisError] = []
        
        for i, file_path in enumerate(files):
            try:
                result, error = process_func(file_path)
                if result is not None:
                    results.append(result)
                if error is not None:
                    errors.append(error)
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
                errors.append(
                    FileAnalysisError(
                        file_path=file_path,
                        error_message=str(e),
                        error_type=e.__class__.__name__,
                        recovery_suggestion="Check the file for syntax errors or encoding issues",
                    )
                )
            
            if progress_callback:
                progress_callback(i + 1, len(files))
        
        return results, errors
    
    def _process_parallel_threads(
        self,
        files: List[Path],
        process_func: Callable[[Path], Tuple[Optional[FileMetrics], Optional[FileAnalysisError]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[FileMetrics], List[FileAnalysisError]]:
        """Process files in parallel using threads."""
        results: List[FileMetrics] = []
        errors: List[FileAnalysisError] = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_func, file_path): file_path for file_path in files}
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1
                
                try:
                    result, error = future.result()
                    if result is not None:
                        results.append(result)
                    if error is not None:
                        errors.append(error)
                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
                    errors.append(
                        FileAnalysisError(
                            file_path=file_path,
                            error_message=str(e),
                            error_type=e.__class__.__name__,
                            recovery_suggestion="Check the file for syntax errors or encoding issues",
                        )
                    )
                
                if progress_callback:
                    progress_callback(completed, len(files))
        
        return results, errors
    
    def _process_parallel_processes(
        self,
        files: List[Path],
        process_func: Callable[[Path], Tuple[Optional[FileMetrics], Optional[FileAnalysisError]]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[FileMetrics], List[FileAnalysisError]]:
        """
        Process files in parallel using processes.
        
        Note: process_func must be picklable (top-level function or callable class).
        """
        results: List[FileMetrics] = []
        errors: List[FileAnalysisError] = []
        completed = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(process_func, file_path): file_path for file_path in files}
                
                # Process results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    completed += 1
                    
                    try:
                        result, error = future.result()
                        if result is not None:
                            results.append(result)
                        if error is not None:
                            errors.append(error)
                    except Exception as e:
                        logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
                        errors.append(
                            FileAnalysisError(
                                file_path=file_path,
                                error_message=str(e),
                                error_type=e.__class__.__name__,
                                recovery_suggestion="Check the file for syntax errors or encoding issues",
                            )
                        )
                    
                    if progress_callback:
                        progress_callback(completed, len(files))
        except Exception as e:
            logger.error(f"Failed to create process pool: {e}", exc_info=True)
            # Fall back to sequential processing
            logger.info("Falling back to sequential processing")
            return self._process_sequential(files, process_func, progress_callback)
        
        return results, errors
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dictionary containing configuration details.
        """
        return {
            "enabled": self.enabled,
            "max_workers": self.max_workers,
            "use_processes": self.use_processes,
        }
