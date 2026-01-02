"""AST caching layer for performance optimization."""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import libcst as cst

logger = logging.getLogger(__name__)


class ASTCache:
    """
    Cache for parsed AST trees to avoid re-parsing identical files.
    
    Uses file content hashing to determine cache validity.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True, max_cache_size_mb: int = 100):
        """
        Initialize the AST cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, uses temporary directory.
            enabled: Whether caching is enabled.
            max_cache_size_mb: Maximum cache size in megabytes.
        """
        self.enabled = enabled
        self.max_cache_size_mb = max_cache_size_mb
        
        if cache_dir is None:
            import tempfile
            self.cache_dir = Path(tempfile.gettempdir()) / "refactron_ast_cache"
        else:
            self.cache_dir = Path(cache_dir)
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for the current session
        self._memory_cache: Dict[str, Tuple[cst.Module, Dict[str, Any]]] = {}
        
        # Track cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
        }
    
    def _compute_hash(self, content: str) -> str:
        """
        Compute SHA-256 hash of file content.
        
        Args:
            content: File content to hash.
            
        Returns:
            Hexadecimal hash string.
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, content_hash: str) -> Path:
        """
        Get the cache file path for a given content hash.
        
        Args:
            content_hash: Hash of the file content.
            
        Returns:
            Path to the cache file.
        """
        return self.cache_dir / f"{content_hash}.cache"
    
    def get(self, file_path: Path, content: str) -> Optional[Tuple[cst.Module, Dict[str, Any]]]:
        """
        Get cached AST and metadata for a file.
        
        Args:
            file_path: Path to the file.
            content: Current content of the file.
            
        Returns:
            Tuple of (AST module, metadata) if cached, None otherwise.
        """
        if not self.enabled:
            return None
        
        content_hash = self._compute_hash(content)
        
        # Check memory cache first
        if content_hash in self._memory_cache:
            self.stats["hits"] += 1
            self.stats["memory_hits"] += 1
            logger.debug(f"AST cache hit (memory) for {file_path}")
            return self._memory_cache[content_hash]
        
        # Check disk cache
        cache_path = self._get_cache_path(content_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Store in memory cache for faster subsequent access
                self._memory_cache[content_hash] = cached_data
                
                self.stats["hits"] += 1
                self.stats["disk_hits"] += 1
                logger.debug(f"AST cache hit (disk) for {file_path}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache for {file_path}: {e}")
                # If cache is corrupted, remove it
                try:
                    cache_path.unlink()
                except Exception:
                    pass
        
        self.stats["misses"] += 1
        return None
    
    def put(self, file_path: Path, content: str, ast_module: cst.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store AST and metadata in cache.
        
        Args:
            file_path: Path to the file.
            content: Content of the file.
            ast_module: Parsed AST module.
            metadata: Optional metadata to cache alongside the AST.
        """
        if not self.enabled:
            return
        
        content_hash = self._compute_hash(content)
        
        if metadata is None:
            metadata = {}
        
        cached_data = (ast_module, metadata)
        
        # Store in memory cache
        self._memory_cache[content_hash] = cached_data
        
        # Store on disk
        cache_path = self._get_cache_path(content_hash)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Cached AST for {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cache AST for {file_path}: {e}")
        
        # Check cache size and cleanup if needed
        self._cleanup_if_needed()
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds the maximum size."""
        if not self.enabled:
            return
        
        try:
            total_size_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
            max_size_bytes = self.max_cache_size_mb * 1024 * 1024
            
            if total_size_bytes > max_size_bytes:
                logger.info(f"Cache size ({total_size_bytes / 1024 / 1024:.2f} MB) exceeds limit, cleaning up...")
                
                # Sort files by modification time (oldest first)
                cache_files = sorted(self.cache_dir.glob("*.cache"), key=lambda p: p.stat().st_mtime)
                
                # Remove oldest files until we're under the limit
                for cache_file in cache_files:
                    if total_size_bytes <= max_size_bytes * 0.8:  # Clean to 80% of limit
                        break
                    
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    total_size_bytes -= file_size
                    logger.debug(f"Removed cache file: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup cache: {e}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        # Clear memory cache
        self._memory_cache.clear()
        
        # Clear disk cache
        if self.enabled and self.cache_dir.exists():
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                logger.info("AST cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
        
        # Reset statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics.
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0.0
        
        cache_size_bytes = 0
        cache_file_count = 0
        
        if self.enabled and self.cache_dir.exists():
            try:
                cache_files = list(self.cache_dir.glob("*.cache"))
                cache_file_count = len(cache_files)
                cache_size_bytes = sum(f.stat().st_size for f in cache_files)
            except Exception:
                pass
        
        return {
            "enabled": self.enabled,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size_mb": round(cache_size_bytes / 1024 / 1024, 2),
            "cache_file_count": cache_file_count,
            "max_cache_size_mb": self.max_cache_size_mb,
        }
