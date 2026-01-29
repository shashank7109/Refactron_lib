# 🚀 Refactron - Complete Feature List

**The Intelligent Code Refactoring Transformer**

This document provides a comprehensive overview of all features and capabilities in the Refactron library.

---

## 📊 Core Capabilities

### 🔍 **Code Analysis** (7 Analyzers)

Refactron analyzes Python code using AST (Abstract Syntax Tree) to detect issues across multiple categories:

---

## 1. 🔒 Security Analyzer (13 Rules)

Detects security vulnerabilities and unsafe code patterns with context-aware confidence scoring.

### Rule IDs & What They Detect:

- **SEC001** - Dangerous functions (`eval()`, `exec()`, `compile()`, `__import__`, `input()`)
- **SEC002** - Dangerous modules (`pickle`, `marshal`, `shelve`) - unsafe deserialization
- **SEC003** - Hardcoded secrets (passwords, API keys, tokens, credentials)
- **SEC004** - SQL injection vulnerabilities (f-strings, % formatting in SQL queries)
- **SEC005** - Command injection risks (`os.system`, `subprocess` with `shell=True`)
- **SEC006** - Weak cryptographic algorithms (MD5, SHA1)
- **SEC007** - Unsafe YAML loading (`yaml.load()` instead of `yaml.safe_load()`)
- **SEC008** - Assert statements used for security checks (can be disabled with -O flag)
- **SEC009** - SQL parameterization issues (string concatenation, `.format()` in SQL queries)
- **SEC010** - SSRF vulnerabilities (dynamic URLs in HTTP requests)
- **SEC011** - Insecure random module usage (warns about `random` module)
- **SEC012** - Weak SSL/TLS configuration (CERT_NONE in SSL contexts)
- **SEC013** - SSL verification disabled (`verify=False` in requests)

### Features:
- Context-aware confidence scoring (0.0-1.0)
- Lower confidence for test files and examples
- File-based ignore patterns
- Rule-specific whitelist support
- False positive filtering

---

## 2. ⚡ Performance Analyzer (6 Rules)

Detects performance bottlenecks and inefficient code patterns.

### Rule IDs & What They Detect:

- **P001** - N+1 query antipattern (database queries inside loops)
- **P002** - Inefficient list comprehension patterns (`list(filter(...))`, `list(map(...))`)
- **P003** - Deeply nested list comprehensions (depth > 2 levels)
- **P004** - Unnecessary iterations (same variable iterated multiple times in a function)
- **P005** - Inefficient string concatenation (`+=` in loops)
- **P006** - Redundant list() calls (wrapping list comprehensions)

---

## 3. 🧠 Complexity Analyzer (4 Rules)

Measures and detects code complexity issues.

### Rule IDs & What They Detect:

- **C001** - High cyclomatic complexity (functions exceeding threshold)
- **C002** - Long functions (exceeding max_function_length threshold)
- **C003** - Deeply nested loops (nesting depth > 3)
- **C004** - Complex method call chains (chains longer than 4 calls)

### Additional Metrics:
- **M001** - Low maintainability index (MI score < 20)

---

## 4. 👃 Code Smell Analyzer (7 Rules)

Detects code smells and anti-patterns that reduce code quality.

### Rule IDs & What They Detect:

- **S001** - Too many parameters (functions with excessive parameters)
- **S002** - Deep nesting (excessive nesting depth)
- **S003** - Duplicate code (functions with numbered suffixes suggesting duplication)
- **S004** - Magic numbers (unexplained numeric constants)
- **S005** - Missing docstrings (functions/classes without documentation)
- **S006** - Unused imports (imported modules that are never used)
- **S007** - Repeated code blocks (3+ consecutive statements repeated within functions)

---

## 5. 💀 Dead Code Analyzer (6 Rules)

Identifies unused and unreachable code.

### Rule IDs & What They Detect:

- **DEAD001** - Unused functions (defined but never called)
- **DEAD002** - Unused variables (assigned but never used)
- **DEAD003** - Unreachable code (code after return/break statements)
- **DEAD004** - Empty functions (functions with only `pass`)
- **DEAD005** - Always-true/false conditions (`if True:`, `if False:`)
- **DEAD006** - Redundant comparisons (`if x == True:`, `if x == False:`)

---

## 6. 🔗 Dependency Analyzer (7 Rules)

Analyzes import statements and module dependencies.

### Rule IDs & What They Detect:

- **DEP001** - Unused imports (imported modules never used)
- **DEP002** - Wildcard imports (`from module import *`)
- **DEP003** - Circular imports (imports inside functions, suggesting circular dependencies)
- **DEP004** - Import order violations (not following PEP 8: stdlib, third-party, local)
- **DEP005** - Relative imports (using `from . import` instead of absolute imports)
- **DEP006** - Duplicate imports (same module imported multiple times)
- **DEP007** - Deprecated modules (`imp`, `optparse`, `xml.etree.cElementTree`)

---

## 7. 📝 Type Hint Analyzer (5 Rules)

Checks for missing or incomplete type annotations.

### Rule IDs & What They Detect:

- **TYPE001** - Missing return type annotations
- **TYPE002** - Missing parameter type annotations
- **TYPE003** - Missing class attribute type annotations
- **TYPE004** - Usage of `Any` type (defeats type checking)
- **TYPE005** - Incomplete generic types (`List`, `Dict`, `Set`, `Tuple` without element types)

---

## 🔧 Refactoring Operations (5 Types)

Suggests and applies safe code transformations with preview and risk scoring.

### Refactoring Types:

1. **Extract Method** (`extract_method`)
   - Suggests extracting complex code blocks into separate methods
   - Identifies functions with multiple logical blocks

2. **Extract Constant** (`extract_constant`)
   - Replaces magic numbers with named constants
   - Groups related magic numbers in functions

3. **Simplify Conditionals** (`simplify_conditionals`)
   - Transforms nested `if` statements into guard clauses
   - Improves code readability

4. **Reduce Parameters** (`reduce_parameters`)
   - Suggests converting parameter lists into configuration objects
   - Reduces function parameter count

5. **Add Docstrings** (`add_docstring`)
   - Automatically generates docstrings for functions and classes
   - Uses context-aware documentation generation

### Refactoring Features:
- Preview mode (see changes before applying)
- Risk scoring (0.0 = perfectly safe, 1.0 = high risk)
- Before/after code previews
- Selective application (apply specific refactoring types)
- **Unique operation IDs** - Each refactoring gets a unique identifier for tracking
- **Pattern fingerprinting** - Code patterns are automatically fingerprinted for learning
- **Feedback collection** - Track which refactorings are accepted, rejected, or ignored

### Suggestion Ranking:

Refactron ranks refactoring suggestions based on learned patterns:

- **RefactoringRanker** (`refactron/patterns/ranker.py`) ranks operations by predicted value
- **Scoring Factors**:
  - Pattern acceptance rate
  - Project-specific weights from project profiles
  - Pattern recency and frequency
  - Metrics improvements (complexity, maintainability)
  - Risk penalty (higher risk = lower score)
- **Integration**:
  - `Refactron.refactor()` ranks operations before display
  - Ranking scores are stored in `operation.metadata["ranking_score"]`
  - CLI preview shows `Ranking Score: X.XXX` per operation
  - Summary shows `📊 N operations ranked by learned patterns`

---

## 🤖 Auto-Fix Engine (14 Fixers)

Automatically fixes common issues using AST-based transformations (no AI APIs required!).

### Available Fixers:

1. **RemoveUnusedImportsFixer** - Removes unused import statements
2. **ExtractMagicNumbersFixer** - Extracts magic numbers into constants
3. **AddDocstringsFixer** - Adds docstrings to functions/classes
4. **RemoveDeadCodeFixer** - Removes unreachable code
5. **FixTypeHintsFixer** - Adds missing type hints
6. **SortImportsFixer** - Sorts and organizes imports (PEP 8 compliant)
7. **RemoveTrailingWhitespaceFixer** - Removes trailing whitespace
8. **NormalizeQuotesFixer** - Normalizes quote style (single/double)
9. **SimplifyBooleanFixer** - Simplifies boolean expressions
10. **ConvertToFStringFixer** - Converts string formatting to f-strings
11. **RemoveUnusedVariablesFixer** - Removes unused variables
12. **FixIndentationFixer** - Fixes indentation issues
13. **AddMissingCommasFixer** - Adds missing commas in sequences
14. **RemovePrintStatementsFixer** - Removes debug print statements

### Auto-Fix Features:
- Risk-based safety levels (safe, low, moderate, high)
- Preview mode before applying
- Backup and rollback support
- AST-based transformations (reliable, deterministic)

---

## 📊 Reporting & Output

### Report Formats:
- **Text** - Human-readable console output
- **JSON** - Machine-readable format for CI/CD integration
- **HTML** - Visual report with formatting

### Report Contents:
- Issue categorization by type and severity
- Detailed issue descriptions with suggestions
- Code snippets for each issue
- File-level and project-level summaries
- Technical debt quantification

---

## 🎛️ CLI Commands

### Core Commands:

1. **`refactron analyze <target>`**
   - Analyze code for issues
   - Options: `--detailed`, `--config`, `--log-level`, `--metrics`, `--show-metrics`

2. **`refactron refactor <target>`**
   - Suggest refactoring operations
   - Options: `--preview`, `--apply`, `-t` (filter by type), `--feedback` (collect feedback)

3. **`refactron report <target>`**
   - Generate detailed reports
   - Options: `--format` (text/json/html), `--output`

4. **`refactron autofix <target>`**
   - Automatically fix issues
   - Options: `--preview`, `--apply`, `--safety-level`

5. **`refactron init`**
   - Initialize configuration file

6. **`refactron rollback`**
   - Rollback changes using backups or Git
   - Options: `--session`, `--use-git`, `--list`, `--clear`

7. **`refactron telemetry <action>`**
   - Manage telemetry (enable/disable/show)

8. **`refactron metrics <format>`**
   - Show metrics (json/text)

9. **`refactron serve-metrics <host> <port>`**
   - Start Prometheus metrics server

10. **`refactron feedback <operation-id>`**
   - Provide feedback on refactoring operations
   - Options: `--action` (accepted/rejected/ignored), `--reason`, `--config`

11. **`refactron patterns <subcommand>`**
   - Project-specific pattern analysis and tuning
   - Subcommands:
     - `analyze` - Show project pattern statistics
     - `recommend` - Show tuning recommendations
     - `tune` - Apply recommendations
     - `profile` - Show current project profile

---

## ⚙️ Configuration & Customization

### Configurable Settings:

- **Enabled Analyzers** - Enable/disable specific analyzers
- **Enabled Refactorers** - Enable/disable specific refactoring types
- **Complexity Thresholds** - Customize max complexity, function length, parameters
- **File Patterns** - Include/exclude specific files or directories
- **Security Settings** - Ignore patterns, rule whitelists, confidence thresholds
- **Performance Optimizations** - AST caching, incremental analysis, parallel processing
- **Logging** - Log levels, formats (JSON/text), file logging
- **Metrics** - Enable metrics collection, Prometheus integration
- **Telemetry** - Opt-in telemetry collection
- **Pattern Learning** - Enable/disable pattern learning, ranking, and custom storage paths

### 🎯 Advanced Configuration Management

Refactron now supports enterprise-grade configuration management with profiles, validation, templates, and versioning.

#### **Environment-Specific Profiles**

Switch between development, staging, and production configurations:

```bash
# Use development profile
refactron analyze . --profile dev

# Use production environment (overrides profile)
refactron refactor . --environment prod
```

**Available Profiles:**
- **dev** - Development settings (DEBUG logging, detailed metrics)
- **staging** - Staging settings (INFO logging, standard metrics)
- **prod** - Production settings (WARNING logging, Prometheus enabled)

**Profile Features:**
- Base configuration + profile overrides
- Environment overrides profile
- Nested configuration merging
- CLI options: `--profile` (`-p`) and `--environment` (`-e`)

#### **Schema Validation**

Strict configuration validation with actionable error messages:

- **Type Checking** - Validates all configuration field types
- **Value Validation** - Ensures values are within acceptable ranges
- **Required Fields** - Validates required configuration fields
- **Clear Error Messages** - Actionable errors with recovery suggestions
- **Version Compatibility** - Validates configuration version compatibility

#### **Config Inheritance & Composition**

Powerful configuration composition with base + overrides:

```yaml
version: "1.0"
base:
  enabled_analyzers: [complexity, security, performance]
  max_function_complexity: 10

profiles:
  dev:
    log_level: DEBUG
    show_details: true
  prod:
    log_level: WARNING
    enable_prometheus: true
```

**Features:**
- Base configuration with shared settings
- Profile-specific overrides
- Nested dictionary merging
- Deep merging for complex configurations

#### **Framework-Specific Templates**

Ready-made configuration templates for popular Python frameworks:

```bash
# Generate Django-specific config
refactron init --template django

# Generate FastAPI-specific config
refactron init --template fastapi

# Generate Flask-specific config
refactron init --template flask
```

**Available Templates:**
- **base** - Default template (all frameworks)
- **django** - Django-specific exclusions (migrations, settings.py, etc.)
- **fastapi** - FastAPI-specific settings (higher complexity thresholds for routes)
- **flask** - Flask-specific settings (blueprint support)

**Template Features:**
- Pre-configured exclude patterns
- Framework-specific complexity thresholds
- Custom rule configurations
- Profile definitions included

#### **Configuration Versioning**

Version-aware configuration system with backward compatibility:

- **Version Tracking** - Config files include version field
- **Migration Support** - Automatic version compatibility checking
- **Backward Compatibility** - Legacy configs continue to work
- **Future-Proof** - Version support for future enhancements

#### **Enhanced CLI Options**

All CLI commands now support profile and environment options:

```bash
# Analyze with dev profile
refactron analyze . --profile dev

# Refactor with production environment
refactron refactor . --environment prod --apply

# Generate report with staging config
refactron report . --profile staging --format html

# Auto-fix with dev profile
refactron autofix . --profile dev --safety-level safe
```

**New Options:**
- `--profile` / `-p` - Select configuration profile (dev, staging, prod)
- `--environment` / `-e` - Override with environment (overrides profile)
- `--template` / `-t` - Select template when running `refactron init`

#### **Configuration Files**

Enhanced YAML configuration structure:

```yaml
version: "1.0"
base:
  # All base settings here
profiles:
  dev:
    # Dev overrides
  staging:
    # Staging overrides
  prod:
    # Prod overrides
```

**Benefits:**
- Single config file for all environments
- Easy switching between environments
- Reduced configuration duplication
- Better organization and maintainability

---

## 🧠 Pattern Learning System

Refactron learns from developer feedback to improve refactoring suggestions over time.

### **Feedback Collection**

Tracks developer actions on refactoring suggestions:

- **Operation Tracking** - Each refactoring operation has a unique UUID for tracking
- **Feedback Recording** - Record acceptance, rejection, or ignoring of suggestions
- **Interactive Collection** - Prompt for feedback during preview with `--feedback` flag
- **Auto-Recording** - Automatically records feedback when using `--apply` flag
- **Manual Entry** - Use `refactron feedback <operation-id>` to provide feedback later

### **Pattern Fingerprinting**

AST-based code pattern identification:

- **Normalized Fingerprinting** - Removes whitespace, comments for consistent matching
- **AST Pattern Extraction** - Captures structural patterns, not just text
- **Hash-Based Identification** - SHA256 hashing for fast pattern lookups
- **Automatic Integration** - Pattern hashes stored in operation metadata

### **Pattern Storage**

Thread-safe, persistent storage system:

- **Project-Specific Storage** - Stores patterns in `.refactron/patterns/` directory
- **Fallback to User Home** - Uses `~/.refactron/patterns/` if no project root found
- **JSON Persistence** - Human-readable JSON format for easy inspection
- **Thread-Safe Operations** - Uses RLock for concurrent access
- **In-Memory Caching** - Reduces file I/O for better performance

### **Pattern Matching**

Intelligent pattern matching with scoring:

- **Hash-Based Lookups** - O(1) pattern matching using hash indexing
- **Similarity Scoring** - Combines acceptance rate, project weights, recency, and frequency
- **Project Context** - Project-specific pattern weights for personalized suggestions
- **Cache Management** - TTL-based caching for pattern data

### **Pattern Learning Engine**

Automatic learning from feedback to improve pattern database:

- **Automatic Learning** - Learns from feedback automatically when recorded
- **Single Feedback Learning** - `learn_from_feedback()` processes individual feedback records
- **Batch Learning** - `batch_learn()` efficiently processes multiple feedback records at once
- **Pattern Metrics Updates** - Tracks before/after code metrics (complexity, maintainability, LOC)
- **Pattern Score Recalculation** - Automatically updates pattern benefit scores based on metrics
- **Pattern Statistics** - Updates acceptance rates, occurrence counts, and benefit scores
- **Background Processing** - `LearningService` processes pending feedback in background
- **Pattern Cleanup** - Automatically removes old patterns (90+ days inactive by default)
- **Operation Reconstruction** - Reconstructs operations from feedback metadata for learning

### Suggestion Ranking:

Refactron ranks refactoring suggestions based on learned patterns:

- **RefactoringRanker** (`refactron/patterns/ranker.py`) ranks operations by predicted value
- **Scoring Factors**:
  - Pattern acceptance rate
  - Project-specific weights from project profiles
  - Pattern recency and frequency
  - Metrics improvements (complexity, maintainability)
  - Risk penalty (higher risk = lower score)
- **Integration**:
  - `Refactron.refactor()` ranks operations before display
  - Ranking scores are stored in `operation.metadata["ranking_score"]`
  - CLI preview shows `Ranking Score: X.XXX` per operation
  - Summary shows `📊 N operations ranked by learned patterns`

### **Project-Specific Rule Tuning**

Customize pattern behavior per project based on historical feedback:

- **RuleTuner** (`refactron/patterns/tuner.py`) analyzes project-specific pattern history
- **Enable/Disable Patterns** per project based on acceptance rates and feedback volume
- **Pattern Weights** per project to influence ranking scores
- **Safe Defaults** - Requires sufficient feedback before making tuning decisions
- **CLI Integration** - `refactron patterns analyze/recommend/tune/profile`

### **Learning Service Features**

Background service for pattern maintenance:

- **Pending Feedback Processing** - Processes all unprocessed feedback records
- **Score Updates** - Recalculates all pattern scores based on latest feedback
- **Pattern Cleanup** - Removes patterns that haven't been seen recently
- **Configurable Retention** - Customizable retention period for old patterns
- **Efficient Batch Operations** - Optimized for processing large amounts of feedback

### **Usage Examples**

```bash
# Collect feedback interactively during preview
refactron refactor file.py --preview --feedback

# Auto-record feedback when applying changes
refactron refactor file.py --apply

# Provide feedback manually later
refactron feedback abc-123-def --action accepted --reason "Improved readability"
```

### **Configuration**

Pattern learning can be configured via `.refactron.yaml` or programmatically:

**YAML Configuration:**
```yaml
# Pattern learning settings
enable_pattern_learning: true          # Master switch (default: true)
pattern_storage_dir: null              # Custom storage path (null = auto-detect)
pattern_learning_enabled: true         # Enable learning from feedback (default: true)
pattern_ranking_enabled: true          # Enable ranking based on patterns (default: true)
```

**Python API:**
```python
from refactron import Refactron
from refactron.core.config import RefactronConfig

# Disable pattern learning entirely
config = RefactronConfig(enable_pattern_learning=False)
refactron = Refactron(config)

# Enable pattern learning but disable ranking
config = RefactronConfig(
    enable_pattern_learning=True,
    pattern_ranking_enabled=False
)
refactron = Refactron(config)

# Use custom storage directory
config = RefactronConfig(
    enable_pattern_learning=True,
    pattern_storage_dir=Path("/custom/path/patterns")
)
refactron = Refactron(config)
```

**Configuration Options:**

- **`enable_pattern_learning`** (bool, default: `true`)
  - Master switch for all pattern learning features
  - When `false`: No pattern components initialized, no storage, no learning, no ranking
  - When `true`: Pattern system initializes (subject to other flags)

- **`pattern_storage_dir`** (Path | null, default: `null`)
  - Custom directory for pattern data storage
  - `null`: Auto-detects project root (`.refactron/patterns/`) or falls back to `~/.refactron/patterns/`
  - Set to a Path to use a specific directory

- **`pattern_learning_enabled`** (bool, default: `true`)
  - Controls whether Refactron learns from feedback
  - When `false`: Feedback is recorded but not used for learning
  - When `true`: Feedback automatically triggers pattern learning

- **`pattern_ranking_enabled`** (bool, default: `true`)
  - Controls whether refactoring suggestions are ranked by learned patterns
  - When `false`: Operations returned in original order, no ranking scores
  - When `true`: Operations ranked by pattern acceptance rates and project-specific weights

**Disabling Pattern Learning:**

To completely disable pattern learning:
```yaml
enable_pattern_learning: false
```

To disable only learning (keep ranking):
```yaml
enable_pattern_learning: true
pattern_learning_enabled: false
pattern_ranking_enabled: true
```

To disable only ranking (keep learning):
```yaml
enable_pattern_learning: true
pattern_learning_enabled: true
pattern_ranking_enabled: false
```

### **Integration**

Seamlessly integrated into Refactron:

- **Automatic Fingerprinting** - Code patterns fingerprinted automatically during refactoring
- **Automatic Learning** - Pattern learning triggered automatically when feedback is recorded
- **Non-Blocking** - Learning failures don't interrupt feedback recording
- **Graceful Degradation** - Works even if pattern storage unavailable
- **Non-Invasive** - Doesn't slow down or interfere with normal refactoring operations
- **Backward Compatible** - Existing workflows continue to work without changes
- **Config-Driven** - All features can be enabled/disabled via configuration

### **Learning Workflow**

How pattern learning works:

1. **Feedback Collection** - Developer provides feedback (accepted/rejected/ignored)
2. **Pattern Identification** - System identifies or creates pattern from code hash
3. **Statistics Update** - Pattern acceptance rate and counts updated
4. **Metrics Calculation** - If accepted, before/after metrics calculated and stored
5. **Score Recalculation** - Pattern benefit score updated based on metrics
6. **Pattern Storage** - Updated pattern saved to persistent storage
7. **Background Maintenance** - LearningService periodically updates scores and cleans old patterns

---

## 🚀 Performance Features

### Optimizations:

1. **AST Caching** - Caches parsed ASTs for faster repeated analysis
2. **Incremental Analysis** - Only analyzes changed files
3. **Parallel Processing** - Multi-threaded/multi-process file analysis
4. **Memory Profiling** - Tracks memory usage during analysis
5. **Metrics Collection** - Detailed performance metrics

---

## 🔐 Safety Features

### Backup & Recovery:
- **Automatic Backups** - Creates backups before applying changes
- **Git Integration** - Uses Git for rollback when available
- **Session Management** - Tracks backup sessions
- **Rollback Support** - Easy restoration of previous versions

### Preview Mode:
- **Safe Default** - Preview mode enabled by default
- **Before/After Diffs** - See exact changes before applying
- **Risk Scoring** - Know the safety level of each operation

---

## 📈 Monitoring & Metrics

### Metrics Collection:
- Analysis time per file
- Issues found by category
- Analyzer performance
- Memory usage tracking
- File analysis success/failure rates

### Prometheus Integration:
- Metrics server for production monitoring
- Customizable host and port
- Real-time metrics export

### Telemetry (Opt-in):
- Usage statistics
- Feature adoption
- Error tracking
- Performance metrics

---

## 🔌 Integration Features

### Python API:
```python
from refactron import Refactron

refactron = Refactron()
analysis = refactron.analyze("mycode.py")
result = refactron.refactor("mycode.py", preview=True)
```

### CLI Integration:
- Can be integrated into CI/CD pipelines
- Exit codes for automation
- JSON output for programmatic processing

---

## 📚 Advanced Features

### False Positive Tracking:
- Learn from false positives
- Filter known false positives
- Improve accuracy over time

### Context-Aware Analysis:
- Different confidence scores for test files
- Lower confidence for example/demo files
- File-type specific rules

### Structured Logging:
- JSON format for log aggregation
- Text format for human reading
- Configurable log levels
- File rotation support

---

## 📊 Summary Statistics

### Total Detection Rules: **48 Rules**
- Security: 13 rules
- Performance: 6 rules
- Complexity: 4 rules
- Code Smells: 7 rules
- Dead Code: 6 rules
- Dependencies: 7 rules
- Type Hints: 5 rules

### Refactoring Operations: **5 Types**
### Auto-Fix Capabilities: **14 Fixers**
### CLI Commands: **11 Commands**
### Report Formats: **3 Formats** (Text, JSON, HTML)

---

## 🎯 Use Cases

Refactron is perfect for:

- **Code Reviews** - Automated issue detection before review
- **Legacy Code Modernization** - Identify and fix technical debt
- **Security Audits** - Find security vulnerabilities automatically
- **Performance Optimization** - Detect performance bottlenecks
- **Code Quality Improvement** - Maintain consistent code quality
- **Technical Debt Tracking** - Quantify and monitor technical debt
- **CI/CD Integration** - Automated quality checks in pipelines
- **Developer Onboarding** - Help new developers understand code quality standards
- **Pattern Learning** - Improve suggestions based on historical feedback
- **Smart Recommendations** - Get better refactoring suggestions over time

---

**Refactron - Making Python code better, one refactoring at a time!** 🚀

**Now with Pattern Learning** - The more you use Refactron, the smarter it gets! 🧠
