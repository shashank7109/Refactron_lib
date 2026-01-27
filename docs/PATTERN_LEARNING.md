# Pattern Learning System Guide

**Learn from your refactoring decisions to improve suggestions over time**

---

## Overview

Refactron's Pattern Learning System learns from your feedback on refactoring suggestions, building a knowledge base that improves the quality and relevance of future recommendations. The more you use Refactron, the smarter it gets!

### Key Features

- **Automatic Learning** - Learns from every refactoring decision you make
- **Smart Ranking** - Ranks suggestions by historical acceptance rates
- **Project-Specific** - Adapts to your project's coding style and preferences
- **Persistent Storage** - Patterns persist across sessions
- **Configurable** - Enable/disable features as needed

---

## How It Works

### 1. **Pattern Fingerprinting**

When Refactron suggests a refactoring, it creates a unique "fingerprint" (hash) of the code pattern:

```python
# Original code
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * 100  # Magic number
    return total

# Pattern fingerprint captures the structural pattern
# (not the exact code, so similar patterns match)
```

### 2. **Feedback Collection**

You provide feedback on each suggestion:

- **Accepted** - You applied the refactoring
- **Rejected** - You decided not to apply it
- **Ignored** - You skipped it

### 3. **Pattern Learning**

Refactron learns from your feedback:

- Tracks acceptance rates for each pattern
- Updates pattern statistics (frequency, recency)
- Calculates benefit scores based on code metrics

### 4. **Smart Ranking**

Future suggestions are ranked based on:

- Pattern acceptance rate (higher = better)
- Project-specific weights
- Pattern recency and frequency
- Code metrics improvements
- Risk scores

---

## Configuration

### Enable/Disable Pattern Learning

Pattern learning is **enabled by default**. You can control it via configuration:

**YAML Configuration (`.refactron.yaml`):**

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

# Enable learning but disable ranking
config = RefactronConfig(
    enable_pattern_learning=True,
    pattern_ranking_enabled=False
)
refactron = Refactron(config)

# Use custom storage directory
from pathlib import Path
config = RefactronConfig(
    enable_pattern_learning=True,
    pattern_storage_dir=Path("/custom/path/patterns")
)
refactron = Refactron(config)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_pattern_learning` | bool | `true` | Master switch for all pattern learning features |
| `pattern_storage_dir` | Path \| null | `null` | Custom storage directory (null = auto-detect) |
| `pattern_learning_enabled` | bool | `true` | Enable learning from feedback |
| `pattern_ranking_enabled` | bool | `true` | Enable ranking based on learned patterns |

---

## Usage Examples

### Basic Usage

**1. Refactor with Preview:**

```bash
refactron refactor myfile.py --preview
```

Refactron shows suggestions with ranking scores:

```
📊 Refactoring Suggestions (ranked by learned patterns)

Operation 1: Extract Constant
  File: myfile.py:42
  Ranking Score: 0.85 ⭐
  Risk: Low (0.2)
  ...
```

**2. Provide Feedback:**

```bash
# Interactive feedback during preview
refactron refactor myfile.py --preview --feedback

# Auto-record when applying
refactron refactor myfile.py --apply

# Manual feedback later
refactron feedback <operation-id> --action accepted --reason "Improved readability"
```

**3. View Pattern Statistics:**

```bash
# Analyze project patterns
refactron patterns analyze

# Get tuning recommendations
refactron patterns recommend

# Apply recommendations
refactron patterns tune --auto

# View current profile
refactron patterns profile
```

### Python API Usage

```python
from refactron import Refactron
from refactron.core.config import RefactronConfig

# Initialize with pattern learning enabled
config = RefactronConfig(
    enable_pattern_learning=True,
    pattern_learning_enabled=True,
    pattern_ranking_enabled=True
)
refactron = Refactron(config)

# Refactor (automatically fingerprints and ranks)
result = refactron.refactor("myfile.py", preview=True)

for operation in result.operations:
    print(f"Operation: {operation.operation_type}")
    print(f"Ranking Score: {operation.metadata.get('ranking_score', 'N/A')}")
    print(f"Risk: {operation.risk_score}")

# Record feedback
if result.operations:
    op = result.operations[0]
    refactron.record_feedback(
        operation_id=op.operation_id,
        action="accepted",
        reason="Good refactoring",
        operation=op
    )
```

---

## API Reference

### Core Classes

#### `PatternStorage`

Manages persistent storage for pattern learning data.

```python
from refactron.patterns.storage import PatternStorage
from pathlib import Path

# Initialize with custom directory
storage = PatternStorage(storage_dir=Path("/custom/path"))

# Load patterns
patterns = storage.load_patterns()

# Load feedback
feedback = storage.load_feedback()

# Get project profile
profile = storage.get_project_profile(project_path=Path("/project"))
```

#### `PatternFingerprinter`

Generates fingerprints for code patterns.

```python
from refactron.patterns.fingerprint import PatternFingerprinter

fingerprinter = PatternFingerprinter()

# Fingerprint code
code_hash = fingerprinter.fingerprint_code("def foo(): pass")

# Fingerprint refactoring operation
from refactron.core.models import RefactoringOperation
op = RefactoringOperation(...)
pattern_hash = fingerprinter.fingerprint_refactoring(op)
```

#### `PatternLearner`

Learns patterns from feedback.

```python
from refactron.patterns.learner import PatternLearner
from refactron.patterns.storage import PatternStorage
from refactron.patterns.fingerprint import PatternFingerprinter

storage = PatternStorage()
fingerprinter = PatternFingerprinter()
learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

# Learn from feedback
from refactron.patterns.models import RefactoringFeedback
feedback = RefactoringFeedback.create(...)
pattern_id = learner.learn_from_feedback(operation, feedback)
```

#### `RefactoringRanker`

Ranks refactoring suggestions based on learned patterns.

```python
from refactron.patterns.ranker import RefactoringRanker
from refactron.patterns.storage import PatternStorage
from refactron.patterns.matcher import PatternMatcher
from refactron.patterns.fingerprint import PatternFingerprinter

storage = PatternStorage()
matcher = PatternMatcher(storage=storage)
fingerprinter = PatternFingerprinter()

ranker = RefactoringRanker(
    storage=storage,
    matcher=matcher,
    fingerprinter=fingerprinter
)

# Rank operations
from refactron.core.models import RefactoringOperation
operations = [RefactoringOperation(...), ...]
ranked = ranker.rank_operations(operations, project_path=Path("/project"))

# Get top suggestions
top_5 = ranker.get_top_suggestions(operations, project_path=Path("/project"), top_n=5)
```

#### `RuleTuner`

Tunes rules based on project-specific pattern history.

```python
from refactron.patterns.tuner import RuleTuner
from refactron.patterns.storage import PatternStorage

storage = PatternStorage()
tuner = RuleTuner(storage=storage)

# Analyze project patterns
analysis = tuner.analyze_project_patterns(project_path=Path("/project"))

# Generate recommendations
recommendations = tuner.generate_recommendations(project_path=Path("/project"))

# Apply tuning
profile = tuner.apply_tuning(project_path=Path("/project"), recommendations=recommendations)
```

#### `LearningService`

Background service for pattern maintenance.

```python
from refactron.patterns.learning_service import LearningService
from refactron.patterns.storage import PatternStorage

storage = PatternStorage()
service = LearningService(storage=storage)

# Process pending feedback
stats = service.process_pending_feedback()
print(f"Processed: {stats['processed']}, Created: {stats['created']}")

# Update pattern scores
service.update_pattern_scores()

# Cleanup old patterns (90+ days inactive)
service.cleanup_old_patterns(days=90)
```

---

## Storage Location

Pattern data is stored in JSON files:

**Default Locations:**
- Project root: `.refactron/patterns/` (if project root detected)
- User home: `~/.refactron/patterns/` (fallback)

**Custom Location:**
```python
config = RefactronConfig(
    pattern_storage_dir=Path("/custom/path/patterns")
)
```

**Storage Files:**
- `patterns.json` - Learned patterns
- `feedback.json` - Feedback records
- `project_profiles.json` - Project-specific configurations
- `pattern_metrics.json` - Pattern metrics

---

## Best Practices

### 1. **Provide Consistent Feedback**

The more feedback you provide, the better the system learns:

- Accept good refactorings
- Reject inappropriate suggestions
- Ignore suggestions you're not sure about

### 2. **Use Project-Specific Tuning**

For large projects, use project-specific tuning:

```bash
# Analyze patterns for your project
refactron patterns analyze

# Apply recommendations
refactron patterns tune --auto
```

### 3. **Review Pattern Statistics**

Periodically review pattern statistics:

```bash
refactron patterns analyze
```

This shows:
- Most accepted patterns
- Patterns with low acceptance rates
- Recommendations for tuning

### 4. **Custom Storage for CI/CD**

In CI/CD environments, use custom storage:

```python
config = RefactronConfig(
    enable_pattern_learning=True,
    pattern_storage_dir=Path("/ci/patterns")
)
```

### 5. **Disable When Not Needed**

If you don't want pattern learning:

```yaml
enable_pattern_learning: false
```

This disables all pattern learning features and saves resources.

---

## Troubleshooting

### Patterns Not Learning

**Problem:** Patterns aren't being learned from feedback.

**Solutions:**
1. Check that `pattern_learning_enabled` is `true`
2. Verify storage directory is writable
3. Check logs for errors: `refactron refactor --log-level DEBUG`

### Ranking Not Working

**Problem:** Suggestions aren't being ranked.

**Solutions:**
1. Check that `pattern_ranking_enabled` is `true`
2. Ensure patterns have been learned (provide feedback first)
3. Verify `PatternMatcher` is initialized (only when ranking enabled)

### Storage Issues

**Problem:** Storage directory errors.

**Solutions:**
1. Check directory permissions
2. Use custom `pattern_storage_dir` if needed
3. Verify disk space available

---

## Advanced Usage

### Custom Pattern Weights

Adjust pattern weights per project:

```python
from refactron.patterns.storage import PatternStorage
from refactron.patterns.models import ProjectPatternProfile

storage = PatternStorage()
profile = storage.get_project_profile(project_path=Path("/project"))

# Set custom weight for a pattern
profile.set_pattern_weight("pattern-id-123", 0.9)

# Save profile
storage.save_project_profile(profile)
```

### Batch Learning

Process multiple feedback records efficiently:

```python
from refactron.patterns.learner import PatternLearner

learner = PatternLearner(storage=storage, fingerprinter=fingerprinter)

# Load pending feedback
feedback_list = storage.load_feedback()

# Batch learn
learner.batch_learn(feedback_list)
```

### Pattern Cleanup

Clean up old patterns:

```python
from refactron.patterns.learning_service import LearningService

service = LearningService(storage=storage)

# Remove patterns older than 90 days
service.cleanup_old_patterns(days=90)
```

---

## Performance Considerations

### Memory Usage

- Pattern storage uses in-memory caching
- Cache is invalidated when patterns are updated
- Large pattern databases may use more memory

### Storage Size

- Each pattern: ~1-2 KB
- Each feedback record: ~500 bytes
- Typical project: 100-1000 patterns (~100-200 KB)

### Learning Speed

- Learning from feedback: < 100ms
- Pattern score updates: < 50ms per pattern
- Batch learning: ~10ms per feedback record

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Refactor with Pattern Learning

on: [push, pull_request]

jobs:
  refactor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install Refactron
        run: pip install refactron

      - name: Run Refactron
        run: |
          # Pattern storage is configured via RefactronConfig.pattern_storage_dir
          # or your project's .refactron.yaml configuration file
          refactron refactor . --preview
        # Note: Pattern storage directory is configured via config file or Python API,
        # not via environment variables. See Configuration section for details.
```

---

## See Also

- [Quick Reference Guide](QUICK_REFERENCE.md) - Quick command reference
- [Features Documentation](../FEATURES.md) - Complete feature list
- [Configuration Guide](../README.md#configuration) - Configuration options

---

**Pattern Learning System** - Making Refactron smarter with every refactoring! 🧠✨
