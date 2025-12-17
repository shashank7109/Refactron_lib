# 🤖 Refactron

**The Intelligent Code Refactoring Transformer**

Refactron helps you write better Python code by finding issues, suggesting improvements, and safely refactoring your codebase.

[![CI](https://github.com/Refactron-ai/Refactron_lib/workflows/CI/badge.svg)](https://github.com/Refactron-ai/Refactron_lib/actions)
[![PyPI version](https://badge.fury.io/py/refactron.svg)](https://pypi.org/project/refactron/)
[![Python Version](https://img.shields.io/pypi/pyversions/refactron.svg)](https://pypi.org/project/refactron/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-135%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen)]()

## ✨ What It Does

Refactron analyzes your Python code to find:
- **Security issues** - SQL injection, hardcoded secrets, unsafe code patterns
- **Performance problems** - N+1 queries, inefficient loops, slow patterns
- **Code smells** - Magic numbers, long functions, too many parameters
- **Complexity** - Deep nesting, high cyclomatic complexity
- **Type hints** - Missing or incomplete annotations
- **Dead code** - Unused functions and unreachable code

Then it helps you fix them with safe, previewable refactorings.

## 🚀 Quick Start

```bash
pip install refactron
```

Analyze your code:
```bash
refactron analyze myproject/
```

Get refactoring suggestions:
```bash
refactron refactor myfile.py --preview
```

Or use it in Python:
```python
from refactron import Refactron

refactron = Refactron()
analysis = refactron.analyze("mycode.py")
print(analysis.report())
```

## 📊 Features

### Security Analysis
Finds vulnerabilities like SQL injection, hardcoded secrets, and unsafe patterns. Includes context-aware analysis that understands when code is in test files vs production.

### Performance Analysis
Detects N+1 query problems, inefficient string concatenation in loops, redundant operations, and other performance bottlenecks that slow down your code.

### Code Quality
Identifies code smells, complexity issues, missing type hints, dead code, and dependency problems. Helps you maintain clean, maintainable code.

### Safe Refactoring
Preview changes before applying them. Each refactoring comes with a risk score so you know what's safe to apply automatically.

## 💡 Why Refactron?

- **Finds real issues** - Not just style problems, but actual bugs and performance issues
- **Safe by default** - Preview everything before applying changes
- **Context-aware** - Understands your code structure, not just syntax
- **Fast and reliable** - Analyzes thousands of lines in seconds

## 📚 Documentation

- [Quick Reference](docs/QUICK_REFERENCE.md) - Command cheatsheet
- [Tutorial](docs/TUTORIAL.md) - Step-by-step guide
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Architecture](ARCHITECTURE.md) - Technical details

## 🤝 Contributing

We'd love your help! Whether it's fixing bugs, adding features, or improving docs - every contribution matters.

Quick start:
```bash
git clone https://github.com/Refactron-ai/Refactron_lib.git
cd Refactron_lib
./setup_dev.sh  # macOS/Linux - automatically sets everything up
# OR setup_dev.bat  # Windows
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. We follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## 🔒 Security

Found a security issue? Please report it privately to security@refactron.us.kg. See [SECURITY.md](SECURITY.md) for details.

## 📄 License

MIT License - feel free to use it in your projects!

---

**Made with ❤️ for the Python community**
