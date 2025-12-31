# Error Handling and Recovery Guide

This guide provides comprehensive information about error handling in Refactron, common error scenarios, and recovery strategies.

## Table of Contents

- [Overview](#overview)
- [Exception Types](#exception-types)
- [Graceful Degradation](#graceful-degradation)
- [Common Error Scenarios](#common-error-scenarios)
- [Recovery Strategies](#recovery-strategies)
- [Logging and Debugging](#logging-and-debugging)
- [Best Practices](#best-practices)

## Overview

Refactron implements robust error handling with the following principles:

1. **Granular Exception Types**: Specific exception classes for different failure scenarios
2. **Graceful Degradation**: Continue processing when individual files fail
3. **Informative Error Messages**: Clear descriptions with recovery suggestions
4. **Comprehensive Logging**: Detailed error logs for debugging

## Exception Types

Refactron defines four main exception types, all inheriting from `RefactronError`:

### RefactronError (Base Exception)

The base exception for all Refactron-specific errors.

```python
from refactron.core.exceptions import RefactronError

try:
    # Refactron operations
    pass
except RefactronError as e:
    print(f"Error: {e}")
    if e.recovery_suggestion:
        print(f"Suggestion: {e.recovery_suggestion}")
```

**Attributes:**
- `message`: Error description
- `file_path`: Optional path to the file that caused the error
- `recovery_suggestion`: Optional suggestion for recovery

### AnalysisError

Raised when code analysis fails.

**Common Causes:**
- Syntax errors in source code
- File encoding issues (non-UTF-8)
- AST parsing failures
- Unsupported Python features

**Example:**
```python
from refactron.core.exceptions import AnalysisError

try:
    result = refactron.analyze("mycode.py")
except AnalysisError as e:
    print(f"Analysis failed: {e}")
    print(f"Analyzer: {e.analyzer_name}")
```

**Attributes:**
- `analyzer_name`: Name of the analyzer that failed

### RefactoringError

Raised when code refactoring fails.

**Common Causes:**
- Unable to parse source code
- File write permission issues
- Backup creation failures
- Refactoring would break semantics

**Example:**
```python
from refactron.core.exceptions import RefactoringError

try:
    result = refactron.refactor("mycode.py", preview=False)
except RefactoringError as e:
    print(f"Refactoring failed: {e}")
    print(f"Operation: {e.operation_type}")
```

**Attributes:**
- `operation_type`: Type of refactoring that failed

### ConfigError

Raised when configuration is invalid or cannot be loaded.

**Common Causes:**
- Configuration file not found
- Invalid YAML syntax
- Missing required configuration options
- Invalid configuration values

**Example:**
```python
from refactron.core.exceptions import ConfigError
from refactron.core.config import RefactronConfig

try:
    config = RefactronConfig.from_file("config.yaml")
except ConfigError as e:
    print(f"Config error: {e}")
    print(f"Config key: {e.config_key}")
```

**Attributes:**
- `config_key`: Specific configuration key that caused the error

## Graceful Degradation

Refactron implements graceful degradation to ensure analysis and refactoring continue even when individual files fail.

### How It Works

1. **Directory Analysis**: When analyzing a directory, if a file fails:
   - The error is logged
   - The file is added to `failed_files` list
   - Analysis continues with remaining files

2. **Results Tracking**: `AnalysisResult` tracks:
   - Successfully analyzed files
   - Failed files with error details
   - Recovery suggestions for each failure

3. **Reporting**: Failed files are shown in the analysis report with:
   - File path
   - Error message
   - Error type
   - Recovery suggestion

### Example

```python
from refactron import Refactron

refactron = Refactron()
result = refactron.analyze("/path/to/project")

# Check results
print(f"Files analyzed: {result.files_analyzed_successfully}")
print(f"Files failed: {result.files_failed}")

# Review failed files
for error in result.failed_files:
    print(f"Failed: {error.file_path}")
    print(f"Reason: {error.error_message}")
    print(f"Suggestion: {error.recovery_suggestion}")
```

## Common Error Scenarios

### Scenario 1: File Encoding Issues

**Problem**: File contains non-UTF-8 characters

**Error Message**:
```
AnalysisError: Failed to read file due to encoding error
```

**Recovery Steps:**
1. Convert the file to UTF-8 encoding:
   ```bash
   iconv -f ISO-8859-1 -t UTF-8 file.py > file_utf8.py
   ```
2. Or specify encoding in Python:
   ```python
   # In your source file
   # -*- coding: utf-8 -*-
   ```

### Scenario 2: Syntax Errors

**Problem**: Source file contains Python syntax errors

**Error Message**:
```
AnalysisError: Failed to parse source code: invalid syntax
```

**Recovery Steps:**
1. Run Python linter to identify syntax errors:
   ```bash
   python -m py_compile file.py
   flake8 file.py
   ```
2. Fix the syntax errors
3. Re-run analysis

### Scenario 3: Permission Denied

**Problem**: Insufficient permissions to read/write files

**Error Message**:
```
RefactoringError: Failed to write file: Permission denied
```

**Recovery Steps:**
1. Check file permissions:
   ```bash
   ls -la file.py
   ```
2. Grant appropriate permissions:
   ```bash
   chmod u+w file.py
   ```
3. Or run with appropriate user privileges

### Scenario 4: Invalid Configuration

**Problem**: Configuration file has invalid YAML syntax

**Error Message**:
```
ConfigError: Invalid YAML syntax in configuration file
```

**Recovery Steps:**
1. Validate YAML syntax:
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```
2. Check for common YAML issues:
   - Incorrect indentation
   - Missing colons
   - Unquoted special characters
3. Use default configuration if needed:
   ```python
   refactron = Refactron()  # Uses default config
   ```

### Scenario 5: Large Files

**Problem**: File is too large and causes memory issues

**Error Message**:
```
AnalysisError: Failed to analyze file: Memory error
```

**Recovery Steps:**
1. Exclude large files from analysis:
   ```yaml
   # config.yaml
   exclude_patterns:
     - "**/large_file.py"
     - "**/data/*.py"
   ```
2. Split large files into smaller modules
3. Increase available memory

### Scenario 6: Missing Dependencies

**Problem**: Analyzer requires packages that aren't installed

**Error Message**:
```
AnalysisError: Failed to run analyzer: Module not found
```

**Recovery Steps:**
1. Install required dependencies:
   ```bash
   pip install refactron[dev]
   ```
2. Check your Python environment
3. Verify all required packages are installed

## Recovery Strategies

### Strategy 1: Incremental Analysis

Start with a small subset of files and gradually expand:

```python
from pathlib import Path
from refactron import Refactron

refactron = Refactron()

# Start with one file
result = refactron.analyze("src/module.py")
if result.files_failed == 0:
    # Expand to directory
    result = refactron.analyze("src/")
```

### Strategy 2: Exclude Problematic Patterns

Use configuration to exclude known problematic files:

```yaml
# config.yaml
exclude_patterns:
  - "**/test_*.py"
  - "**/migrations/**"
  - "**/generated/**"
  - "**/__pycache__/**"
```

### Strategy 3: Enable Verbose Logging

Get detailed information for debugging:

```python
import logging
from refactron import Refactron

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

refactron = Refactron()
result = refactron.analyze("mycode.py")
```

### Strategy 4: Use Preview Mode

Always preview refactoring changes before applying:

```python
from refactron import Refactron

refactron = Refactron()

# Preview changes first
result = refactron.refactor("mycode.py", preview=True)
print(result.report())

# Review carefully, then apply
if input("Apply changes? (y/n): ").lower() == 'y':
    result = refactron.refactor("mycode.py", preview=False)
```

### Strategy 5: Backup and Rollback

Use the backup system for safety:

```python
from refactron.core.backup import BackupRollbackSystem
from pathlib import Path

backup_system = BackupRollbackSystem()

# Create backup before refactoring
file_path = Path("mycode.py")
backup_path = backup_system.backup_file(file_path)

try:
    # Perform refactoring
    refactron.refactor(file_path, preview=False)
except Exception as e:
    # Restore from backup if something goes wrong
    backup_system.rollback_file(file_path)
    print(f"Restored from backup: {backup_path}")
```

## Logging and Debugging

### Enable Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("refactron.log"),
        logging.StreamHandler()
    ]
)

# Run Refactron with logging enabled
from refactron import Refactron
refactron = Refactron()
```

### Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Something unexpected happened (e.g., file skipped)
- **ERROR**: A serious problem occurred

### Viewing Logs

Check logs to understand failures:

```bash
# View recent errors
tail -f refactron.log | grep ERROR

# Search for specific file
grep "mycode.py" refactron.log

# View all warnings
grep WARNING refactron.log
```

## Best Practices

### 1. Always Use Version Control

Before running refactoring operations:
```bash
git commit -am "Before refactoring"
```

### 2. Start with Analysis

Always analyze before refactoring:
```python
# First, analyze to understand issues
result = refactron.analyze("src/")
print(result.report())

# Then refactor based on findings
if result.critical_issues:
    refactron.refactor("src/", preview=True)
```

### 3. Review Failed Files

Check and address failed files:
```python
result = refactron.analyze("src/")

if result.files_failed > 0:
    print("\nFailed files:")
    for error in result.failed_files:
        print(f"  {error.file_path}: {error.error_message}")
        print(f"  Suggestion: {error.recovery_suggestion}")
```

### 4. Use Configuration Files

Create a `.refactron.yaml` in your project:
```yaml
# .refactron.yaml
enabled_analyzers:
  - complexity
  - code_smells
  - security

max_function_complexity: 10
max_parameters: 5

exclude_patterns:
  - "**/test_*.py"
  - "**/migrations/**"
```

### 5. Handle Exceptions Properly

```python
from refactron import Refactron
from refactron.core.exceptions import RefactronError

try:
    refactron = Refactron()
    result = refactron.analyze("src/")
except RefactronError as e:
    print(f"Refactron error: {e}")
    if e.recovery_suggestion:
        print(f"Try this: {e.recovery_suggestion}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 6. Monitor Analysis Results

Track success rates:
```python
result = refactron.analyze("src/")
summary = result.summary()

success_rate = (
    summary["files_analyzed"] / summary["total_files"] * 100
    if summary["total_files"] > 0 else 0
)

print(f"Success rate: {success_rate:.1f}%")

if success_rate < 90:
    print("Warning: High failure rate. Review failed files.")
```

## Getting Help

If you encounter issues not covered in this guide:

1. **Check Logs**: Review detailed error logs for more context
2. **Search Issues**: Look for similar issues on [GitHub Issues](https://github.com/Refactron-ai/Refactron_lib/issues)
3. **Report Bug**: Create a new issue with:
   - Error message and stack trace
   - Minimal reproduction example
   - Refactron version (`refactron --version`)
   - Python version
4. **Community Support**: Ask questions in discussions

## Summary

Refactron's error handling system provides:

- ✅ **Granular exception types** for different failure scenarios
- ✅ **Graceful degradation** to continue processing despite failures
- ✅ **Detailed error messages** with recovery suggestions
- ✅ **Comprehensive logging** for debugging
- ✅ **Practical recovery strategies** for common issues

By following this guide, you can effectively handle errors and recover from failures when using Refactron.
