#!/usr/bin/env python3
"""
Performance benchmarking script for Refactron.

This script measures the performance of various Refactron operations
to help identify bottlenecks and track performance over time.
"""

import json
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from refactron import Refactron
from refactron.core.config import RefactronConfig


# Performance tracking file
BENCHMARK_HISTORY_FILE = Path(__file__).parent / "benchmark_history.json"


def create_test_file(size: str = "small") -> Path:
    """Create a test Python file of specified size."""
    sizes = {
        "small": 100,  # 100 lines
        "medium": 500,  # 500 lines
        "large": 2000,  # 2000 lines
    }

    lines = sizes.get(size, 100)

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)

    # Generate code
    temp_file.write("# Auto-generated test file\n")
    temp_file.write("import os\nimport sys\n\n")

    for i in range(lines // 10):
        temp_file.write(
            f"""
def function_{i}(param1, param2, param3):
    '''Docstring for function_{i}.'''
    result = 0
    for j in range(10):
        if j % 2 == 0:
            result += j * param1
        else:
            result -= j * param2
    return result + param3

"""
        )

    temp_file.close()
    return Path(temp_file.name)


def benchmark_operation(name: str, operation, iterations: int = 5) -> Dict[str, Any]:
    """Benchmark an operation multiple times and return statistics."""
    times = []

    for _ in range(iterations):
        start = time.time()
        operation()
        end = time.time()
        times.append(end - start)

    return {
        "name": name,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "iterations": iterations,
    }


def benchmark_analysis(refactron: Refactron, file_path: Path) -> Dict[str, Any]:
    """Benchmark the analysis operation."""
    return benchmark_operation("analyze", lambda: refactron.analyze(file_path))


def benchmark_refactoring(refactron: Refactron, file_path: Path) -> Dict[str, Any]:
    """Benchmark the refactoring operation."""
    return benchmark_operation("refactor", lambda: refactron.refactor(file_path, preview=True))


def benchmark_with_caching(refactron: Refactron, file_path: Path) -> Dict[str, Any]:
    """Benchmark analysis with caching enabled."""
    # First run to populate cache
    refactron.analyze(file_path)
    
    # Second run should use cache
    return benchmark_operation("analyze_with_cache", lambda: refactron.analyze(file_path))


def benchmark_parallel_processing(config: RefactronConfig, files: List[Path]) -> Dict[str, Any]:
    """Benchmark parallel vs sequential processing."""
    results = {}
    
    # Sequential processing
    config_seq = RefactronConfig(
        enabled_analyzers=config.enabled_analyzers,
        enable_parallel_processing=False,
    )
    refactron_seq = Refactron(config_seq)
    
    start = time.time()
    refactron_seq.analyze(files[0].parent)
    sequential_time = time.time() - start
    
    results["sequential_time"] = sequential_time
    
    # Parallel processing
    config_par = RefactronConfig(
        enabled_analyzers=config.enabled_analyzers,
        enable_parallel_processing=True,
        max_parallel_workers=4,
    )
    refactron_par = Refactron(config_par)
    
    start = time.time()
    refactron_par.analyze(files[0].parent)
    parallel_time = time.time() - start
    
    results["parallel_time"] = parallel_time
    results["speedup"] = sequential_time / parallel_time if parallel_time > 0 else 0
    
    return results


def benchmark_memory_usage(refactron: Refactron, file_path: Path) -> Dict[str, Any]:
    """Benchmark memory usage during analysis."""
    # Enable memory profiling
    refactron.memory_profiler.enabled = True
    
    refactron.memory_profiler.snapshot("before")
    refactron.analyze(file_path)
    refactron.memory_profiler.snapshot("after")
    
    diff = refactron.memory_profiler.compare("before", "after")
    
    return {
        "name": "memory_usage",
        "rss_mb_diff": diff.get("rss_mb_diff", 0),
        "vms_mb_diff": diff.get("vms_mb_diff", 0),
    }


def save_benchmark_results(results: List[Dict[str, Any]]) -> None:
    """Save benchmark results to history file."""
    history = []
    
    if BENCHMARK_HISTORY_FILE.exists():
        try:
            with open(BENCHMARK_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    
    # Add current results
    history.append({
        "timestamp": datetime.now().isoformat(),
        "results": results,
    })
    
    # Keep only last 100 runs
    history = history[-100:]
    
    # Save back
    try:
        with open(BENCHMARK_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n📊 Results saved to {BENCHMARK_HISTORY_FILE}")
    except Exception as e:
        print(f"\n⚠️  Failed to save results: {e}")


def print_results(results: List[Dict[str, Any]]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("REFACTRON PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80 + "\n")

    for result in results:
        print(f"Operation: {result['name']}")
        
        if "mean" in result:
            print(f"  Mean time:   {result['mean']:.4f}s")
            print(f"  Median time: {result['median']:.4f}s")
            print(f"  Min time:    {result['min']:.4f}s")
            print(f"  Max time:    {result['max']:.4f}s")
            print(f"  Std dev:     {result['stdev']:.4f}s")
            print(f"  Iterations:  {result['iterations']}")
        elif "sequential_time" in result:
            print(f"  Sequential time: {result['sequential_time']:.4f}s")
            print(f"  Parallel time:   {result['parallel_time']:.4f}s")
            print(f"  Speedup:         {result['speedup']:.2f}x")
        elif "rss_mb_diff" in result:
            print(f"  Memory delta (RSS): {result['rss_mb_diff']:.2f} MB")
            print(f"  Memory delta (VMS): {result['vms_mb_diff']:.2f} MB")
        
        print()


def main():
    """Run the benchmark suite."""
    print("🚀 Starting Refactron Performance Benchmarks...\n")

    # Initialize Refactron with default config
    config = RefactronConfig.default()
    refactron = Refactron(config)

    results = []
    test_files = []

    # Test different file sizes
    for size in ["small", "medium", "large"]:
        print(f"Benchmarking with {size} file...")
        test_file = create_test_file(size)
        test_files.append(test_file)

        try:
            # Benchmark analysis
            result = benchmark_analysis(refactron, test_file)
            result["name"] = f"analyze_{size}"
            results.append(result)

            # Benchmark refactoring
            result = benchmark_refactoring(refactron, test_file)
            result["name"] = f"refactor_{size}"
            results.append(result)
            
            # Benchmark caching (only for medium file)
            if size == "medium":
                result = benchmark_with_caching(refactron, test_file)
                result["name"] = "analyze_with_cache_medium"
                results.append(result)
            
            # Benchmark memory usage
            result = benchmark_memory_usage(refactron, test_file)
            result["name"] = f"memory_usage_{size}"
            results.append(result)

        finally:
            pass  # Keep files for now

    # Benchmark parallel processing
    if len(test_files) > 1:
        print("Benchmarking parallel processing...")
        result = benchmark_parallel_processing(config, test_files)
        result["name"] = "parallel_processing"
        results.append(result)

    # Clean up test files
    for test_file in test_files:
        try:
            test_file.unlink()
        except Exception:
            pass

    # Print results
    print_results(results)
    
    # Print performance statistics
    print("=" * 80)
    print("PERFORMANCE OPTIMIZATION STATISTICS")
    print("=" * 80 + "\n")
    
    perf_stats = refactron.get_performance_stats()
    
    print("AST Cache:")
    for key, value in perf_stats["ast_cache"].items():
        print(f"  {key}: {value}")
    
    print("\nIncremental Analysis:")
    for key, value in perf_stats["incremental_analysis"].items():
        print(f"  {key}: {value}")
    
    print("\nParallel Processing:")
    for key, value in perf_stats["parallel_processing"].items():
        print(f"  {key}: {value}")
    
    print("\nMemory Profiler:")
    for key, value in perf_stats["memory_profiler"].items():
        print(f"  {key}: {value}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal operations benchmarked: {len(results)}")

    analyze_times = [r["mean"] for r in results if "analyze" in r["name"] and "mean" in r]
    if analyze_times:
        print(f"Average analysis time: {statistics.mean(analyze_times):.4f}s")

    refactor_times = [r["mean"] for r in results if "refactor" in r["name"] and "mean" in r]
    if refactor_times:
        print(f"Average refactoring time: {statistics.mean(refactor_times):.4f}s")
    
    # Save results to history
    save_benchmark_results(results)

    print("\n✅ Benchmarking complete!")


if __name__ == "__main__":
    main()
