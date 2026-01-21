"""GitHub Actions workflow template generation."""

from pathlib import Path
from typing import Any, Dict, List, Optional


class GitHubActionsGenerator:
    """Generate GitHub Actions workflow templates for Refactron."""

    @staticmethod
    def generate_analysis_workflow(
        python_versions: Optional[List[str]] = None,
        trigger_on: Optional[List[str]] = None,
        quality_gate: Optional[Dict[str, Any]] = None,
        cache_enabled: bool = True,
        upload_artifacts: bool = True,
    ) -> str:
        """Generate GitHub Actions workflow for code analysis.

        Args:
            python_versions: Python versions to test
                (default: ['3.8', '3.9', '3.10', '3.11', '3.12'])
            trigger_on: Events to trigger on
                (default: ['push', 'pull_request'])
            quality_gate: Quality gate configuration
            cache_enabled: Enable dependency caching
            upload_artifacts: Upload analysis reports as artifacts

        Returns:
            YAML workflow content
        """
        if python_versions is None:
            python_versions = ["3.8", "3.9", "3.10", "3.11", "3.12"]

        if trigger_on is None:
            trigger_on = ["push", "pull_request"]

        if quality_gate is None:
            quality_gate = {"fail_on_critical": True, "max_critical": 0}

        python_matrix = ", ".join([f'"{v}"' for v in python_versions])

        trigger_yaml = "\n".join(
            [
                f"  {t}:\n    paths:\n      - '**.py'\n      - '.github/workflows/refactron.yml'"
                for t in trigger_on
            ]
        )

        workflow = f"""name: Refactron Code Analysis

on:
{trigger_yaml}
  workflow_dispatch:

jobs:
  analyze:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [{python_matrix}]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{{{ matrix.python-version }}}}
        uses: actions/setup-python@v5
        with:
          python-version: ${{{{ matrix.python-version }}}}
          cache: 'pip'

      - name: Install Refactron
        run: |
          python -m pip install --upgrade pip
          pip install refactron

      - name: Run Refactron Analysis
        id: analyze
        run: |
          mkdir -p .refactron-reports
          refactron analyze . --format json \\
            --output .refactron-reports/analysis.json \\
            --log-format json || ANALYSIS_EXIT=$?
          ANALYSIS_EXIT=${{ANALYSIS_EXIT:-$?}}
          echo "exit_code=$ANALYSIS_EXIT" >> $GITHUB_OUTPUT
        continue-on-error: true

      - name: Parse Quality Gate
        if: always()
        run: |
          if [ -f .refactron-reports/analysis.json ]; then
            python << 'EOF'
import json
import sys

with open('.refactron-reports/analysis.json', 'r') as f:
    data = json.load(f)

summary = data.get('summary', {{}})
critical = summary.get('critical', 0)
errors = summary.get('errors', 0)
warnings = summary.get('warnings', 0)

print(f"📊 Analysis Summary:")
print(f"  Critical: {{critical}}")
print(f"  Errors: {{errors}}")
print(f"  Warnings: {{warnings}}")
print(f"  Total: {{summary.get('total_issues', 0)}}")

# Quality gate checks
fail = False
fail_on_critical = {str(quality_gate.get('fail_on_critical', True))}
max_critical = {quality_gate.get('max_critical', 0)}
if fail_on_critical and critical > max_critical:
    threshold = max_critical
    msg = (
        f"❌ Quality gate failed: Critical issues ({{critical}}) "
        f"exceed threshold ({{threshold}})"
    )
    print(msg)
    fail = True

fail_on_errors = {str(quality_gate.get('fail_on_errors', False))}
max_errors = {quality_gate.get('max_errors', 10)}
if fail_on_errors and errors > max_errors:
    error_threshold = max_errors
    msg = (
        f"❌ Quality gate failed: Error issues ({{errors}}) "
        f"exceed threshold ({{error_threshold}})"
    )
    print(msg)
    fail = True

if fail:
    sys.exit(1)
EOF
          else
            echo "⚠️ No analysis report found"
            exit 1
          fi

      - name: Upload Analysis Report
        if: always() && {str(upload_artifacts).lower()}
        uses: actions/upload-artifact@v4
        with:
          name: refactron-analysis-${{{{ matrix.python-version }}}}
          path: .refactron-reports/analysis.json
          retention-days: 30

      - name: Comment PR with Results
        if: github.event_name == 'pull_request' && always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            let summary = '## 📊 Refactron Analysis Results\\n\\n';

            try {{
              const data = JSON.parse(fs.readFileSync('.refactron-reports/analysis.json', 'utf8'));
              const s = data.summary || {{}};

              summary += `| Metric | Value |\\n`;
              summary += `|--------|-------|\\n`;
              files_str = `${{s.files_analyzed || 0}}/${{s.total_files || 0}}`;
              summary += `| Files Analyzed | ${{files_str}} |\\n`;
              summary += `| Critical Issues | ${{s.critical || 0}} |\\n`;
              summary += `| Error Issues | ${{s.errors || 0}} |\\n`;
              summary += `| Warning Issues | ${{s.warnings || 0}} |\\n`;
              summary += `| Total Issues | ${{s.total_issues || 0}} |\\n`;

              if (s.critical > 0) {{
                summary += '\\n⚠️ **Critical issues found - please review before merging**\\n';
              }}
            }} catch (e) {{
              summary += '⚠️ Could not parse analysis results\\n';
            }}

            github.rest.issues.createComment({{
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: summary
            }});
"""

        return workflow

    @staticmethod
    def generate_pre_commit_workflow(
        python_version: str = "3.11",
        trigger_on: Optional[List[str]] = None,
    ) -> str:
        """Generate GitHub Actions workflow for pre-commit analysis.

        Args:
            python_version: Python version to use
            trigger_on: Events to trigger on

        Returns:
            YAML workflow content
        """
        if trigger_on is None:
            trigger_on = ["pull_request"]

        trigger_yaml = "\n".join([f"  {t}:" for t in trigger_on])

        workflow = f"""name: Refactron Pre-Commit

on:
{trigger_yaml}

jobs:
  pre-commit:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python {python_version}
        uses: actions/setup-python@v5
        with:
          python-version: '{python_version}'
          cache: 'pip'

      - name: Install Refactron
        run: |
          python -m pip install --upgrade pip
          pip install refactron

      - name: Run Refactron on Changed Files
        run: |
          if [ "${{{{ github.event_name }}}}" == "pull_request" ]; then
            git fetch origin ${{{{ github.base_ref }}}}
            CHANGED_FILES=$(git diff --name-only origin/${{{{ github.base_ref }}}} \\
              | grep '\\.py$' || true)

            if [ -n "$CHANGED_FILES" ]; then
              echo "$CHANGED_FILES" | xargs refactron analyze --format json --log-format json
            else
              echo "No Python files changed"
            fi
          else
            refactron analyze . --format json --log-format json
          fi
"""

        return workflow

    @staticmethod
    def save_workflow(workflow_content: str, output_path: Path) -> None:
        """Save workflow to file.

        Args:
            workflow_content: Workflow YAML content
            output_path: Path to save workflow file

        Raises:
            IOError: If file cannot be written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(workflow_content)
