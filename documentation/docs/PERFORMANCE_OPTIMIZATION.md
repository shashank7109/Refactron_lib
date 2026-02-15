# Performance Optimization Guide

This guide covers Refactron's performance optimization features designed to handle large codebases efficiently.

## Overview

Refactron includes several performance optimization features:

1. **AST Caching** - Avoids re-parsing identical files
2. **Incremental Analysis** - Only analyzes changed files since last run
3. **Parallel Processing** - Analyzes multiple files concurrently
4. **Memory Profiling** - Tracks and optimizes memory usage for large codebases

These features work together to provide significant performance improvements, especially on large projects with 100k+ lines of code.

## Quick Start

### Enable All Optimizations

```python
from refactron import Refactron
from refactron.core.config import RefactronConfig

# Create configuration with all optimizations enabled
config = RefactronConfig(
    enable_ast_cache=True,
    enable_incremental_analysis=True,
    enable_parallel_processing=True,
    max_parallel_workers=4,
)

# Initialize Refactron
refactron = Refactron(config)

# Analyze your codebase
result = refactron.analyze("path/to/your/project")

# Get performance statistics
stats = refactron.get_performance_stats()
print(stats)
```

### Using YAML Configuration

Create a `.refactron.yaml` file:

```yaml
# Performance optimization settings
enable_ast_cache: true
max_ast_cache_size_mb: 100
ast_cache_dir: null  # Uses temp directory by default

enable_incremental_analysis: true
incremental_state_file: null  # Uses temp directory by default

enable_parallel_processing: true
max_parallel_workers: 4
use_multiprocessing: true

enable_memory_profiling: false
memory_optimization_threshold_mb: 5.0
```

Then use it:

```python
from pathlib import Path
from refactron import Refactron
from refactron.core.config import RefactronConfig

config = RefactronConfig.from_file(Path(".refactron.yaml"))
refactron = Refactron(config)
```

## Feature Details

### 1. AST Caching

**Purpose**: The AST cache stores parsed Abstract Syntax Trees to avoid re-parsing files that haven't changed.

**Benefits**:
- 5-10x faster on repeated analysis
- Reduces CPU usage
- Especially effective for large files

**Configuration**:
```python
config = RefactronConfig(
    enable_ast_cache=True,          # Enable caching
    max_ast_cache_size_mb=100,      # Maximum cache size in MB
    ast_cache_dir=Path("/path"),    # Optional: custom cache directory
)
```

**How It Works**:
1. Computes SHA-256 hash of file content
2. Checks if AST is cached for this hash
3. Returns cached AST if available, otherwise parses and caches
4. Automatically cleans up old cache entries when size limit is reached

**Cache Statistics**:
```python
stats = refactron.get_performance_stats()
print(f"Cache hit rate: {stats['ast_cache']['hit_rate_percent']}%")
print(f"Cache size: {stats['ast_cache']['cache_size_mb']} MB")
```

**Clear Cache**:
```python
refactron.ast_cache.clear()
# Or clear all caches
refactron.clear_caches()
```

### 2. Incremental Analysis

**Purpose**: Only analyzes files that have changed since the last run, skipping unchanged files.

**Benefits**:
- Up to 90% reduction in analysis time on subsequent runs
- Ideal for CI/CD pipelines
- Perfect for iterative development

**Configuration**:
```python
config = RefactronConfig(
    enable_incremental_analysis=True,
    incremental_state_file=Path("state.json"),  # Optional: custom state file
)
```

**How It Works**:
1. Tracks file modification time and size
2. On each run, compares current state with saved state
3. Only analyzes files with changed modification time or size
4. Updates state after successful analysis

**Usage Example**:
```python
# First run - analyzes all files
result1 = refactron.analyze("project/")
print(f"Analyzed {result1.total_files} files")

# Second run (no changes) - skips all files
result2 = refactron.analyze("project/")
print(f"Analyzed {result2.total_files} files")  # 0 files

# Modify a file
Path("project/file.py").write_text("# Updated")

# Third run - only analyzes changed file
result3 = refactron.analyze("project/")
print(f"Analyzed {result3.total_files} files")  # 1 file
```

**Force Full Analysis**:
```python
# Clear incremental state to force full analysis
refactron.incremental_tracker.clear()
result = refactron.analyze("project/")
```

### 3. Parallel Processing

**Purpose**: Analyzes multiple files concurrently using multiprocessing or threading.

**Benefits**:
- 2-4x speedup on multi-core systems
- Near-linear scaling up to 4-8 workers
- Automatic fallback to sequential processing on errors

**Configuration**:
```python
config = RefactronConfig(
    enable_parallel_processing=True,
    max_parallel_workers=4,           # Number of worker processes/threads
    use_multiprocessing=True,         # True = processes, False = threads
)
```

**How It Works**:
1. Distributes files across worker processes/threads
2. Each worker analyzes files independently
3. Results are aggregated in the main process
4. Automatic load balancing

**Choosing Workers**:
```python
import multiprocessing

# Auto-detect CPU count (capped at 8)
config = RefactronConfig(max_parallel_workers=None)

# Explicit worker count
config = RefactronConfig(max_parallel_workers=4)

# Based on CPU count
cpu_count = multiprocessing.cpu_count()
config = RefactronConfig(max_parallel_workers=cpu_count)
```

**When to Use**:
- ✅ Large codebases (1000+ files)
- ✅ Multi-core systems
- ✅ I/O-bound operations
- ❌ Small codebases (&lt;10 files)
- ❌ Single-core systems
- ❌ Memory-constrained environments

### 4. Memory Profiling

**Purpose**: Tracks memory usage and provides optimization suggestions for large codebases.

**Benefits**:
- Identifies memory bottlenecks
- Prevents out-of-memory errors
- Optimizes for large files (100k+ lines)

**Configuration**:
```python
config = RefactronConfig(
    enable_memory_profiling=True,
    memory_optimization_threshold_mb=5.0,  # Threshold for "large" files
)
```

**Features**:
- Memory snapshots at key points
- Memory pressure detection
- Automatic garbage collection suggestions
- Memory usage tracking per operation

**Usage Example**:
```python
# Enable profiling
refactron.memory_profiler.enabled = True

# Take snapshots
refactron.memory_profiler.snapshot("before_analysis")
result = refactron.analyze("large_project/")
refactron.memory_profiler.snapshot("after_analysis")

# Compare snapshots
diff = refactron.memory_profiler.compare("before_analysis", "after_analysis")
print(f"Memory used: {diff['rss_mb_diff']:.2f} MB")

# Check memory pressure
if refactron.memory_profiler.check_memory_pressure():
    print("High memory pressure detected!")
    refactron.memory_profiler.suggest_gc()
```

**Memory Statistics**:
```python
stats = refactron.get_performance_stats()
mem_stats = stats['memory_profiler']
print(f"Current RSS: {mem_stats['current_rss_mb']} MB")
print(f"Available: {mem_stats['available_mb']} MB")
```

## Performance Benchmarks

Run the benchmark suite to measure performance:

```bash
python benchmarks/performance_benchmark.py
```

Expected results with all optimizations:
- Small file (100 lines): ~0.01s
- Medium file (500 lines): ~0.05s
- Large file (2000 lines): ~0.5s
- 100k LOC codebase: ~30s

### Benchmark History

Results are tracked over time in `benchmarks/benchmark_history.json`:

```python
import json
from pathlib import Path

history = json.loads(Path("benchmarks/benchmark_history.json").read_text())
latest = history[-1]
print(f"Timestamp: {latest['timestamp']}")
print(f"Results: {latest['results']}")
```

## Best Practices

### For Small Projects (&lt;1000 files)

```python
config = RefactronConfig(
    enable_ast_cache=True,              # Quick cache hits
    enable_incremental_analysis=True,   # Skip unchanged files
    enable_parallel_processing=False,   # Overhead not worth it
)
```

### For Medium Projects (1000-10000 files)

```python
config = RefactronConfig(
    enable_ast_cache=True,
    enable_incremental_analysis=True,
    enable_parallel_processing=True,
    max_parallel_workers=4,
)
```

### For Large Projects (10000+ files)

```python
config = RefactronConfig(
    enable_ast_cache=True,
    max_ast_cache_size_mb=200,          # Larger cache
    enable_incremental_analysis=True,
    enable_parallel_processing=True,
    max_parallel_workers=8,             # More workers
    enable_memory_profiling=True,       # Track memory
)
```

### For CI/CD Pipelines

```python
config = RefactronConfig(
    enable_ast_cache=False,             # Fresh analysis each run
    enable_incremental_analysis=True,   # Compare with previous commit
    enable_parallel_processing=True,
    max_parallel_workers=4,
)
```

## Troubleshooting

### Performance Not Improving

**Symptoms**: Analysis time doesn't improve with optimizations enabled.

**Solutions**:
1. Check if optimizations are actually enabled:
   ```python
   stats = refactron.get_performance_stats()
   print(stats)
   ```

2. Verify cache is being used:
   ```python
   cache_stats = stats['ast_cache']
   print(f"Hit rate: {cache_stats['hit_rate_percent']}%")
   ```

3. Check incremental analysis is working:
   ```python
   # Should be fewer files on second run
   result1 = refactron.analyze("project/")
   result2 = refactron.analyze("project/")
   print(f"First: {result1.total_files}, Second: {result2.total_files}")
   ```

### High Memory Usage

**Symptoms**: Process uses too much memory or crashes.

**Solutions**:
1. Reduce cache size:
   ```python
   config.max_ast_cache_size_mb = 50
   ```

2. Lower parallel workers:
   ```python
   config.max_parallel_workers = 2
   ```

3. Enable memory profiling to find bottlenecks:
   ```python
   config.enable_memory_profiling = True
   ```

4. Clear caches periodically:
   ```python
   refactron.clear_caches()
   ```

### Parallel Processing Slower

**Symptoms**: Parallel processing is slower than sequential.

**Possible Causes**:
- Too few files (overhead > benefit)
- Small files (parsing is fast anyway)
- I/O bottleneck (disk read limited)
- Too many workers (context switching)

**Solutions**:
1. Disable parallel processing for small codebases:
   ```python
   config.enable_parallel_processing = False
   ```

2. Reduce worker count:
   ```python
   config.max_parallel_workers = 2
   ```

3. Use threading instead of multiprocessing:
   ```python
   config.use_multiprocessing = False
   ```

### Cache Not Persisting

**Symptoms**: Cache hits always 0, even after multiple runs.

**Solutions**:
1. Check cache directory permissions
2. Verify cache is enabled:
   ```python
   assert refactron.ast_cache.enabled
   ```
3. Check cache directory exists:
   ```python
   print(refactron.ast_cache.cache_dir)
   assert refactron.ast_cache.cache_dir.exists()
   ```

## Advanced Usage

### Custom Cache Directory

```python
from pathlib import Path

config = RefactronConfig(
    enable_ast_cache=True,
    ast_cache_dir=Path("/var/cache/refactron"),
)
```

### Custom State File Location

```python
config = RefactronConfig(
    enable_incremental_analysis=True,
    incremental_state_file=Path(".refactron-state.json"),
)
```

### Programmatic Cache Management

```python
# Clear specific cache
refactron.ast_cache.clear()

# Get detailed cache stats
stats = refactron.ast_cache.get_stats()
print(f"Cache files: {stats['cache_file_count']}")
print(f"Total size: {stats['cache_size_mb']} MB")

# Manual cache cleanup
refactron.ast_cache._cleanup_if_needed()
```

### Memory Profiling Specific Functions

```python
def my_analysis_function():
    # Your code here
    pass

result, mem_diff = refactron.memory_profiler.profile_function(
    my_analysis_function,
    label="custom_analysis"
)
print(f"Memory used: {mem_diff['rss_mb_diff']} MB")
```

## API Reference

See the module documentation for detailed API reference:

- `refactron.core.cache.ASTCache`
- `refactron.core.incremental.IncrementalAnalysisTracker`
- `refactron.core.parallel.ParallelProcessor`
- `refactron.core.memory_profiler.MemoryProfiler`

## See Also

- [Benchmarks README](../benchmarks/README.md)
- [Configuration Guide](CONFIGURATION.md)
- [Architecture Documentation](../ARCHITECTURE.md)
