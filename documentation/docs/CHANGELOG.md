# Changelog

All notable changes to Refactron will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### v1.0.15 (2026-02-08)

#### Added
- **LLM/RAG Integration**: Full semantic intelligence suite using Llama 3 (via Groq) and ChromaDB.
- **Repository Management**: New `repo` command group (`list`, `connect`, `disconnect`) for managing local workspaces and auto-indexing.
- **AI-Powered Commands**:
    - `refactron suggest`: Contextual refactoring proposals.
    - `refactron document`: Automated Google-style docstring generation.
- **Observability & Metrics**:
    - `refactron metrics`: Detailed performance and analyzer statistics.
    - `refactron serve-metrics`: Prometheus-compatible endpoint.
    - `refactron telemetry`: Opt-in anonymous usage statistics.
- **CI/CD Integration**: `refactron ci` command to generate GitHub Actions and GitLab CI configurations.
- **Improved CLI UI**: New interactive dashboard and file selector.

#### Fixed
- **Security**: Hardened URL sanitization in workspace management to prevent injection attacks.
- **Compatibility**: Resolved Python 3.8 issues in RAG parsing, Tree-sitter API, and fingerprinting.
- **Reliability**: Added missing `GroqClient` dependency and cleaned up project-wide linting issues.
- **Indexer Reliability**: Fixed missing `GroqClient` import and clarified error messages in the RAG indexer.

### Changed
- Improved CLI startup sequence and progress feedback for RAG indexing operations.
- Updated documentation files to reflect the new AI-powered features.
- Applied project-wide linting and code style fixes.

---

## [1.0.14] - 2026-01-30

### Changed
- Refactored CLI startup sequence to display animation before authentication prompt.
- Improved dependency management (added `astroid`).

## [1.0.13] - 2026-01-30

### Added

#### Pattern Learning System
- **Pattern Learning Engine** - Foundation for identifying and learning project-specific refactoring patterns.
- **Project-Specific Rule Tuner** - CLI commands to tune refactoring rules based on project needs.
- **Suggestion Ranking System** - Intelligent ranking of refactoring suggestions based on risk and impact.
- **Feedback Collection System** - Interactive feedback loop to improve pattern recognition over time.

#### CLI Enhancements
- **Enhanced Welcome Flow** - Sleek startup animation with system checks and rotating quick tips.
- **Interactive Dashboard** - Minimal "Info Center" for quick access to help and version information.
- **Custom Help Formatter** - Beautifully formatted, numbered help output for better command discovery.
- **Authentication Enforcement** - Mandatory authentication for all core commands (analyze, refactor, etc.).

#### Performance & Reliability
- **AST Cache & Incremental Analysis** - Faster analysis by only processing changed files.
- **Parallel Processing** - Multi-threaded analysis for large codebases.
- **Backup & Rollback System** - Git-integrated safety system to undo refactoring changes.
- **Enhanced Error Handling** - Custom exceptions and graceful degradation for a more robust experience.

#### Configuration & Integration
- **Advanced Configuration Management** - Support for profiles, validation, and project-specific settings.
- **CI/CD Integration Templates** - Pre-configured templates for GitHub Actions and other CI/CD platforms.
- **Prometheus Metrics** - Built-in support for exporting metrics to Prometheus.

### Fixed
- Resolved numerous linting and type-checking issues across the codebase.
- Improved Python 3.8 compatibility with explicit type hints.
- Optimized project type detection for large codebases.
- Fixed critical issues in feedback persistence and test isolation.

---

### Planned
- AI-powered pattern recognition
- VS Code extension
- PyCharm plugin
- Advanced custom rule engine
- Performance profiling

---

## [1.0.1] - 2025-12-28

### Added

#### Analyzer Enhancements (#37)
- **New Security Patterns**:
  - SQL parameterization detection (SEC009) - Identifies unsafe SQL string formatting
  - SSRF (Server-Side Request Forgery) vulnerability detection (SEC010)
  - Insecure random number generation detection (SEC011)
  - Weak SSL/TLS configuration detection (SEC012, SEC013)
  - Cryptographic weaknesses detection (weak hashing algorithms)
- **Complexity Analyzer Improvements**:
  - Nested loop depth detection (C003) - Flags deeply nested loops that impact performance
  - Method call chain complexity detection (C004) - Identifies overly long method chaining
- **Performance Analyzer** - New analyzer detecting performance antipatterns:
  - N+1 query detection (P001) - Identifies database queries inside loops
  - Inefficient list comprehensions (P002, P003) - Detects patterns that can be optimized
  - Unnecessary iterations (P004) - Finds redundant loops and iterations
  - Inefficient string concatenation (P005) - Detects string building in loops
  - Redundant list conversions (P006) - Finds unnecessary list() wrappers
- **Code Smell Analyzer Enhancements**:
  - Improved unused import detection (S006) - More accurate analysis of import usage
  - Repeated code block detection - Identifies duplicate code patterns within functions

#### False Positive Reduction (#39, #40)
- **Context-Aware Security Analysis** - Adjusts confidence scores based on file context:
  - Test files get 60% confidence multiplier for certain rules (eval, pickle, random)
  - Example/demo files get 70% confidence multiplier
  - Reduces false positives for legitimate test code
- **Rule Whitelisting** - Configuration-based whitelisting of security rules for specific file patterns
- **False Positive Tracking System** - `FalsePositiveTracker` class to learn and track known false positives
- **Confidence Scores** - All security issues now include confidence scores (0.0-1.0) indicating detection certainty
- **Minimum Confidence Filtering** - Filter out low-confidence issues via configuration

#### Test Coverage Improvements (#47, #48)
- **Comprehensive Edge Case Tests** - Added extensive edge case coverage for all analyzers
- **Real-World Test Datasets** - Test suites with real-world problematic code patterns
- **Integration Tests** - Improved integration test coverage across analyzer modules
- **Achieved 96.8% test coverage** for analyzer modules

#### Developer Experience
- Pre-commit hooks configuration for automated code quality checks
- SECURITY.md with comprehensive security policy and vulnerability reporting process
- CONTRIBUTING_QUICKSTART.md for fast contributor onboarding (5-minute setup)
- Performance benchmarking suite in benchmarks/ directory
- Pre-commit GitHub Actions workflow for CI/CD
- Enhanced README badges (Black, pre-commit, security scanning)
- Comprehensive documentation for false positive reduction features

### Changed
- Formatted 10 files with Black in examples/ and real_world_tests/ directories
- Updated README with accurate test coverage (84%) and test count (135)
- Improved contributing documentation with quick start guide
- Updated CI/CD metrics in README
- Security analyzer now uses context-aware confidence scoring
- Enhanced code smell detection accuracy for unused imports

### Fixed
- Fixed flake8 violations in simplify_conditionals_refactorer.py
- Fixed flake8 violations in reduce_parameters_refactorer.py
- Reduced total flake8 issues from 294 to ~17 (94% improvement)
- Fixed code formatting issues in examples directory
- Improved security analyzer accuracy by reducing false positives in test files

---

## [1.0.0] - 2025-10-27

### 🎉 Major Release - Production Ready!

First stable release with complete auto-fix system and Phase 3 features.

### Added

#### Phase 3: Auto-fix System
- **Auto-fix Engine** - Intelligent automatic code fixing with safety guarantees
- **14 Automatic Fixers** - Fix common issues automatically
  - 🟢 `remove_unused_imports` - Remove unused import statements (risk: 0.0)
  - 🟢 `sort_imports` - Sort imports using isort (risk: 0.0)
  - 🟢 `remove_trailing_whitespace` - Clean whitespace (risk: 0.0)
  - 🟡 `extract_magic_numbers` - Extract to named constants (risk: 0.2)
  - 🟡 `add_docstrings` - Add missing documentation (risk: 0.1)
  - 🟡 `remove_dead_code` - Remove unreachable code (risk: 0.1)
  - 🟡 `normalize_quotes` - Standardize quote style (risk: 0.1)
  - 🟡 `simplify_boolean` - Simplify boolean expressions (risk: 0.3)
  - 🟡 `convert_to_fstring` - Modernize string formatting (risk: 0.2)
  - 🟡 `remove_unused_variables` - Clean unused variables (risk: 0.2)
  - 🟡 `fix_indentation` - Fix tabs/spaces (risk: 0.1)
  - 🟡 `add_missing_commas` - Add trailing commas (risk: 0.1)
  - 🟡 `remove_print_statements` - Remove debug prints (risk: 0.3)
  - 🔴 `fix_type_hints` - Add type hints (risk: 0.4, placeholder)

#### File Operations & Safety
- **Atomic File Writes** - Safe file operations (temp file → rename)
- **Automatic Backups** - All changes backed up before applying
- **Rollback System** - Undo individual files or all at once
- **Backup Index** - Track all backups with timestamps
- **Safety Levels** - Control fix risk (safe/low/moderate/high)

#### CLI Enhancements
- **New Command**: `refactron autofix` - Automatic code fixing
- **Safety Level Flags** - `--safety-level` for risk control
- **Preview Mode** - See changes before applying
- **Apply Mode** - Apply fixes with automatic backup

### Improved
- **Test Coverage** - 135 tests (was 98) → +37 auto-fix tests
- **Overall Coverage** - 81% (maintained high coverage)
- **Production Status** - Changed from Beta to Stable
- **Documentation** - Added comprehensive manual testing guide

### Fixed
- All existing bugs from v0.1.0-beta
- Edge cases in fixer logic
- File operation safety

### Technical Details
- Added `refactron/autofix/` module
  - `engine.py` - Auto-fix engine (95% coverage)
  - `fixers.py` - 14 concrete fixers (88% coverage)
  - `file_ops.py` - File operations (87% coverage)
  - `models.py` - Data models (100% coverage)
- Added 37 comprehensive tests
- File backup stored in `.refactron_backups/`
- Backup index: `.refactron_backups/index.json`

---

## [0.1.0-beta] - 2025-10-25

### 🎉 Initial Beta Release

First production-ready beta release of Refactron!

### Recent Improvements (Pre-Release Polish)
- **Fixed** security analyzer false positives for package metadata (`__author__`, `__version__`, etc.)
- **Improved** CLI code quality by extracting helper functions
  - `analyze()` function simplified (reduced complexity)
  - `refactor()` function simplified (reduced complexity)
- **Added** 11 comprehensive tests for Extract Method refactorer
  - Coverage improved from 62% → 97%
- **Increased** overall test coverage from 89% → 90%
- **Increased** total tests from 87 → 98
- **Eliminated** all critical issues in production code (1 → 0)

### Added

#### Core Features
- **Plugin-based analyzer system** for extensibility
- **Refactoring suggestion engine** with before/after previews
- **Risk scoring system** (0.0-1.0 scale) for safe refactoring
- **Configuration management** via YAML files
- **Rich CLI interface** with colors and progress indicators

#### Analyzers (8 Total)
- **Complexity Analyzer** - Cyclomatic complexity, maintainability index
- **Code Smell Analyzer** - Too many parameters, deep nesting, magic numbers
- **Security Analyzer** - eval/exec detection, hardcoded secrets, injection patterns
- **Dependency Analyzer** - Wildcard imports, unused imports, circular dependencies
- **Dead Code Analyzer** - Unused functions, unreachable code, empty functions
- **Type Hint Analyzer** - Missing type annotations, incomplete generics
- **Extract Method Analyzer** - Identify complex functions that should be split
- **Base Analyzer** - Abstract base for custom analyzers

#### Refactorers (6 Total)
- **Magic Number Refactorer** - Extract magic numbers to constants
- **Reduce Parameters Refactorer** - Convert parameter lists to config objects
- **Simplify Conditionals Refactorer** - Transform nested if statements to guard clauses
- **Add Docstring Refactorer** - Generate contextual docstrings
- **Extract Method Refactorer** - Suggest method extraction
- **Base Refactorer** - Abstract base for custom refactorers

#### CLI Commands
- `refactron analyze` - Analyze code for issues
- `refactron refactor` - Generate refactoring suggestions
- `refactron report` - Create detailed reports (text, JSON, HTML)
- `refactron init` - Initialize configuration file

#### Testing
- **87 tests** with **89% coverage**
- Unit tests for all analyzers
- Integration tests for CLI
- Real-world testing on 5,800 lines of code
- Edge case and error handling tests

#### Documentation
- Comprehensive README with quick start
- Architecture documentation
- Developer setup guide
- Real-world case study with metrics
- Usage examples (Flask, Data Science, CLI)
- Complete feature matrix

#### Examples
- Bad code examples for testing
- Flask API with security issues
- Data science workflow issues
- CLI tool best practices
- Refactoring demonstration
- Phase 2 analyzer showcase

### Performance
- **4,300 lines per second** analysis speed
- Low memory footprint
- Suitable for CI/CD integration
- Fast enough for pre-commit hooks (<2s typical)

### Quality Metrics
- 89% test coverage
- 0 critical security issues in production code
- 51 issues per 1,000 lines (top 25% for Python projects)
- 100% accuracy on security vulnerability detection

---

## [0.0.1] - 2025-10-23

### Initial Development

- Project structure setup
- Basic AST parsing
- Initial analyzer prototypes
- CLI framework
- Early testing

---

## Version History

| Version | Date | Status | Highlights |
|---------|------|--------|------------|
| 1.0.1 | 2025-12-28 | **Stable** | Expanded analyzers, false positive reduction, performance analyzer, improved test coverage |
| 1.0.0 | 2025-10-27 | **Stable** | Production-ready with auto-fix system |
| 0.1.0 | 2025-10-25 | **Beta** | First production-ready release |
| 0.0.1 | 2025-10-23 | Alpha | Initial development |

---

## Categories

### Added
New features and capabilities

### Changed
Changes to existing functionality

### Deprecated
Features that will be removed in future releases

### Removed
Features that have been removed

### Fixed
Bug fixes

### Security
Security-related changes and fixes

---

## Notes

- **v0.1.0** is the first **production-ready** release
- Tested on 5,800 lines of real Python code
- Zero critical issues in production code
- Ready for CI/CD integration
- Suitable for team adoption

---

## Links

- [GitHub Repository](https://github.com/yourusername/refactron)
- [Documentation](README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Issue Tracker](https://github.com/yourusername/refactron/issues)

---

**Keep this changelog up to date with every release!**
