"""
Example: Enhanced Error Handling in Refactron

This example demonstrates the new error handling features including:
- Custom exception types
- Graceful degradation
- Error recovery suggestions
- Detailed logging
"""

import logging
import tempfile
from pathlib import Path

from refactron import ConfigError, Refactron
from refactron.core.config import RefactronConfig

# Enable logging to see error details
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_1_graceful_degradation() -> None:
    """Example 1: Graceful degradation when files fail."""
    print("\n" + "=" * 80)
    print("Example 1: Graceful Degradation")
    print("=" * 80)

    # Create a temporary directory with mixed files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a valid Python file
        valid_file = tmpdir_path / "valid.py"
        valid_file.write_text(
            """
def calculate_sum(a, b):
    '''Add two numbers.'''
    return a + b

def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""
        )

        # Create another valid file
        valid_file2 = tmpdir_path / "valid2.py"
        valid_file2.write_text(
            """
class DataProcessor:
    '''Process data efficiently.'''

    def __init__(self, config):
        self.config = config

    def process(self, items):
        return [self.transform(item) for item in items]

    def transform(self, item):
        return item.upper()
"""
        )

        # Create a file with encoding issues (simulated)
        invalid_file = tmpdir_path / "invalid.py"
        # Write binary data that will cause encoding error
        invalid_file.write_bytes(b"\xff\xfe\x00# Invalid encoding")

        # Analyze the directory - Refactron will gracefully skip the invalid file
        refactron = Refactron()
        result = refactron.analyze(tmpdir_path)

        # Show results
        print(f"\nTotal files found: {result.total_files}")
        print(f"Files analyzed successfully: {result.files_analyzed_successfully}")
        print(f"Files failed: {result.files_failed}")
        print(f"Total issues found: {result.total_issues}")

        # Show failed files with recovery suggestions
        if result.failed_files:
            print("\n--- Failed Files ---")
            for error in result.failed_files:
                print(f"\n❌ File: {error.file_path.name}")
                print(f"   Error: {error.error_message}")
                print(f"   Type: {error.error_type}")
                if error.recovery_suggestion:
                    print(f"   💡 Suggestion: {error.recovery_suggestion}")


def example_2_config_error_handling() -> None:
    """Example 2: Configuration error handling."""
    print("\n" + "=" * 80)
    print("Example 2: Configuration Error Handling")
    print("=" * 80)

    # Try to load a non-existent config file
    print("\nTrying to load non-existent config file...")
    try:
        RefactronConfig.from_file(Path("/nonexistent/config.yaml"))
    except ConfigError as e:
        print(f"❌ ConfigError caught: {e}")
        print(f"   File path: {e.file_path}")
        print(f"   💡 Suggestion: {e.recovery_suggestion}")

    # Try to load config with invalid YAML
    print("\nTrying to load config with invalid YAML...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: syntax:\n  - bad\n    - indentation")
        invalid_config_path = Path(f.name)

    try:
        RefactronConfig.from_file(invalid_config_path)
    except ConfigError as e:
        print(f"❌ ConfigError caught: {e}")
        print(f"   💡 Suggestion: {e.recovery_suggestion}")
    finally:
        invalid_config_path.unlink()

    # Successfully load a valid config
    print("\nLoading valid configuration...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
max_function_complexity: 15
enabled_analyzers:
  - complexity
  - code_smells
"""
        )
        valid_config_path = Path(f.name)

    try:
        config = RefactronConfig.from_file(valid_config_path)
        print("✅ Config loaded successfully!")
        print(f"   Max complexity: {config.max_function_complexity}")
        print(f"   Enabled analyzers: {', '.join(config.enabled_analyzers)}")
    finally:
        valid_config_path.unlink()


def example_3_custom_exception_handling() -> None:
    """Example 3: Catching specific exception types."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Exception Handling")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a file with syntax error
        syntax_error_file = tmpdir_path / "syntax_error.py"
        syntax_error_file.write_text(
            """
def broken_function(
    # Missing closing parenthesis and body
"""
        )

        refactron = Refactron()

        # Example of catching specific exceptions
        try:
            result = refactron.analyze(syntax_error_file)

            if result.failed_files:
                print("\nHandling analysis errors:")
                for error in result.failed_files:
                    if error.error_type == "AnalysisError":
                        print(f"   Analysis failed for {error.file_path.name}")
                        print(f"   Reason: {error.error_message}")
                        print(f"   💡 {error.recovery_suggestion}")

        except Exception as e:
            print(f"Unexpected error: {e}")


def example_4_detailed_report() -> None:
    """Example 4: Detailed analysis report with error information."""
    print("\n" + "=" * 80)
    print("Example 4: Detailed Analysis Report")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create files with various issues
        valid_file = tmpdir_path / "good_code.py"
        valid_file.write_text(
            """
def clean_function():
    '''A well-written function.'''
    return 42
"""
        )

        bad_file = tmpdir_path / "bad_code.py"
        bad_file.write_text(
            """
def bad_function(a, b, c, d, e, f, g):
    # Too many parameters
    x = 42  # Magic number
    if a:
        if b:
            if c:
                if d:  # Deep nesting
                    return e + f + g + x
"""
        )

        # Add an invalid file
        invalid_file = tmpdir_path / "invalid.py"
        invalid_file.write_bytes(b"\xff\xfe\x00Invalid")

        # Analyze and generate report
        refactron = Refactron()
        result = refactron.analyze(tmpdir_path)

        # Print the report
        print(result.report(detailed=True))


def example_5_best_practices() -> None:
    """Example 5: Best practices for error handling."""
    print("\n" + "=" * 80)
    print("Example 5: Best Practices")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        file1 = tmpdir_path / "module1.py"
        file1.write_text("def func1():\n    return 'Hello'\n")

        file2 = tmpdir_path / "module2.py"
        file2.write_text("def func2():\n    return 'World'\n")

        # Best Practice 1: Start with small scope
        print("\n1. Start with small scope:")
        refactron = Refactron()
        result = refactron.analyze(file1)
        print(f"   Single file analysis: {result.files_analyzed_successfully} file(s)")

        # Best Practice 2: Check results before expanding
        print("\n2. Check results before expanding:")
        if result.files_failed == 0:
            result = refactron.analyze(tmpdir_path)
            print(f"   Directory analysis: {result.files_analyzed_successfully} file(s)")

        # Best Practice 3: Review summary
        print("\n3. Review summary:")
        summary = result.summary()
        print(f"   Total files: {summary['total_files']}")
        print(f"   Analyzed: {summary['files_analyzed']}")
        print(f"   Failed: {summary['files_failed']}")
        print(f"   Issues: {summary['total_issues']}")

        # Best Practice 4: Handle errors gracefully
        print("\n4. Handle errors gracefully:")
        success_rate = (
            summary["files_analyzed"] / summary["total_files"] * 100
            if summary["total_files"] > 0
            else 0
        )
        print(f"   Success rate: {success_rate:.1f}%")

        if summary["files_failed"] > 0:
            print("   Review failed files and address issues")


def main() -> None:
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("REFACTRON ENHANCED ERROR HANDLING EXAMPLES")
    print("=" * 80)

    try:
        example_1_graceful_degradation()
        example_2_config_error_handling()
        example_3_custom_exception_handling()
        example_4_detailed_report()
        example_5_best_practices()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
