# Refactron

A Python library for code analysis and refactoring.

[![CI](https://github.com/Refactron-ai/Refactron_lib/workflows/CI/badge.svg)](https://github.com/Refactron-ai/Refactron_lib/actions)
[![PyPI version](https://badge.fury.io/py/refactron.svg)](https://pypi.org/project/refactron/)
[![Python Version](https://img.shields.io/pypi/pyversions/refactron.svg)](https://pypi.org/project/refactron/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Refactron analyzes Python code for security vulnerabilities, performance issues, code smells, and complexity problems. It provides refactoring suggestions with safety previews and supports automated code fixes.

## Features

### Code Analysis

- Security scanning: SQL injection, code injection, hardcoded secrets, SSRF vulnerabilities
- Code quality: magic numbers, long functions, excessive parameters, deep nesting
- Complexity metrics: cyclomatic complexity, maintainability index, nested loops
- Type hints: missing or incomplete annotations
- Dead code detection: unused functions and unreachable code
- Dependency analysis: circular imports, wildcard imports
- Performance issues: N+1 queries, inefficient iterations

### Refactoring

- Extract constants, simplify conditionals, reduce parameters
- Add docstrings, extract methods
- Preview changes before applying
- Risk scoring for each refactoring
- **Pattern Learning** - Learns from your feedback to improve suggestions over time

### Auto-Fix

- 14 automated fixers for common issues
- Configurable safety levels
- Automatic backups and rollback support

## Installation

```bash
pip install refactron
```

## Usage

### Python API

```python
from refactron import Refactron

refactron = Refactron()
analysis = refactron.analyze("path/to/code.py")
print(analysis.report())

result = refactron.refactor("path/to/code.py", preview=True)
result.show_diff()
```

### Command Line

```bash
# Analyze code
refactron analyze myproject/

# Generate report
refactron report myproject/ --format json -o report.json

# Preview refactoring suggestions
refactron refactor myfile.py --preview

# Auto-fix issues
refactron autofix myfile.py --preview
refactron autofix myfile.py --apply
```

## Documentation

- [Quick Reference](docs/QUICK_REFERENCE.md)
- [Tutorial](docs/TUTORIAL.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing Guide](CONTRIBUTING.md)

## Development Status

Stable release (v1.0.1). Tested on Python 3.8-3.12.

- 135 tests, 84% code coverage
- 96.8% analyzer module coverage
- Validated on 5,800+ lines of production code

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/Refactron-ai/Refactron_lib.git
cd Refactron_lib
bash setup_dev.sh  # or setup_dev.bat on Windows
```

## Security

Report security issues to security@refactron.dev See [SECURITY.md](SECURITY.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.
