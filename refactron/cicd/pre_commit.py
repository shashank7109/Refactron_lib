"""Pre-commit hook template generation."""

from pathlib import Path
from typing import List, Optional


class PreCommitGenerator:
    """Generate pre-commit hook templates for Refactron."""

    @staticmethod
    def generate_pre_commit_config(
        stages: Optional[List[str]] = None,
        fail_on_critical: bool = True,
        fail_on_errors: bool = False,
        max_critical: int = 0,
        max_errors: int = 10,
    ) -> str:
        """Generate pre-commit hook configuration.

        Args:
            stages: Pre-commit stages to run on (default: ['commit', 'push'])
            fail_on_critical: Fail commit if critical issues found
            fail_on_errors: Fail commit if error-level issues found
            max_critical: Maximum allowed critical issues
            max_errors: Maximum allowed error-level issues

        Returns:
            YAML configuration content
        """
        if stages is None:
            stages = ["commit", "push"]

        stages_str = ", ".join([f'"{s}"' for s in stages])

        config = f"""repos:
  - repo: local
    hooks:
      - id: refactron-analyze
        name: Refactron Code Analysis
        entry: bash -c 'refactron analyze --format json --output .refactron-report.json || true'
        language: system
        pass_filenames: false
        always_run: true
        stages: [{stages_str}]
      - id: refactron-quality-gate
        name: Refactron Quality Gate
        entry: |
          bash -c '
          python << "EOF"
import json
import sys

try:
    with open(".refactron-report.json", "r") as f:
        data = json.load(f)

    summary = data.get("summary", {{}})
    critical = summary.get("critical", 0)
    errors = summary.get("errors", 0)
    warnings = summary.get("warnings", 0)

    print(f"📊 Refactron Pre-Commit Analysis:")
    print(f"  Critical: {{critical}}")
    print(f"  Errors: {{errors}}")
    print(f"  Warnings: {{warnings}}")

    fail = False

    if {str(fail_on_critical)} and critical > {max_critical}:
        print(f"❌ Quality gate failed: Critical issues ({{critical}}) > {max_critical}")
        fail = True

    if {str(fail_on_errors)} and errors > {max_errors}:
        print(f"❌ Quality gate failed: Error issues ({{errors}}) > {max_errors}")
        fail = True

    if fail:
        sys.exit(1)
EOF
          '
        language: system
        pass_filenames: false
        always_run: true
        stages: [{stages_str}]
"""

        return config

    @staticmethod
    def generate_simple_hook() -> str:
        """Generate simple pre-commit hook script.

        Returns:
            Bash script content
        """
        hook = """#!/bin/bash
# Refactron Pre-Commit Hook

# Install refactron if not available
if ! command -v refactron &> /dev/null; then
    echo "Installing Refactron..."
    pip install refactron
fi

# Run analysis on staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\\.py$' || true)

if [ -z "$STAGED_FILES" ]; then
    echo "No Python files staged"
    exit 0
fi

echo "Running Refactron analysis on staged files..."
echo "$STAGED_FILES" | xargs refactron analyze --format json --log-format json

# Check exit code (0 = success, 1 = critical issues found)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Refactron found critical issues. Commit blocked."
    echo "Run 'refactron analyze .' to see details."
    exit 1
fi

echo "✅ Refactron pre-commit check passed"
exit 0
"""

        return hook

    @staticmethod
    def save_config(config_content: str, output_path: Path) -> None:
        """Save pre-commit configuration to file.

        Args:
            config_content: YAML configuration content
            output_path: Path to save configuration file

        Raises:
            IOError: If file cannot be written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(config_content)

    @staticmethod
    def save_hook(hook_content: str, output_path: Path) -> None:
        """Save pre-commit hook script to file.

        Args:
            hook_content: Hook script content
            output_path: Path to save hook file

        Raises:
            IOError: If file cannot be written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(hook_content)

        # Make executable
        output_path.chmod(0o755)
