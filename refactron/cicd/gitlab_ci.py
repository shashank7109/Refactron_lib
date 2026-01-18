"""GitLab CI pipeline configuration generation."""

from pathlib import Path
from typing import Dict, Optional


class GitLabCIGenerator:
    """Generate GitLab CI pipeline configurations for Refactron."""

    @staticmethod
    def generate_analysis_pipeline(
        python_versions: Optional[list[str]] = None,
        quality_gate: Optional[Dict] = None,
        cache_enabled: bool = True,
        artifacts_enabled: bool = True,
    ) -> str:
        """Generate GitLab CI pipeline for code analysis.

        Args:
            python_versions: Python versions to test
            quality_gate: Quality gate configuration
            cache_enabled: Enable dependency caching
            artifacts_enabled: Save analysis reports as artifacts

        Returns:
            YAML pipeline content
        """
        if python_versions is None:
            python_versions = ["3.8", "3.9", "3.10", "3.11", "3.12"]

        if quality_gate is None:
            quality_gate = {"fail_on_critical": True, "max_critical": 0}

        python_matrix = ", ".join([f'"{v}"' for v in python_versions])

        pipeline = f"""stages:
  - analyze

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/
    - venv/

.refactron_template: &refactron_template
  before_script:
    - python -m pip install --upgrade pip
    - pip install refactron
  script:
    - mkdir -p .refactron-reports
    - refactron analyze . --format json --output .refactron-reports/analysis.json --log-format json || true
    - |
      python << 'EOF'
import json
import sys

try:
    with open('.refactron-reports/analysis.json', 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary', {{}})
    critical = summary.get('critical', 0)
    errors = summary.get('errors', 0)
    warnings = summary.get('warnings', 0)
    total = summary.get('total_issues', 0)
    
    print(f"📊 Analysis Summary:")
    print(f"  Critical: {{critical}}")
    print(f"  Errors: {{errors}}")
    print(f"  Warnings: {{warnings}}")
    print(f"  Total: {{total}}")
    
    # Quality gate enforcement
    fail = False
    
    if {str(quality_gate.get('fail_on_critical', True))} and critical > {quality_gate.get('max_critical', 0)}:
        print(f"❌ Quality gate failed: Critical issues ({{critical}}) > threshold ({quality_gate.get('max_critical', 0)})")
        fail = True
    
    if {str(quality_gate.get('fail_on_errors', False))} and errors > {quality_gate.get('max_errors', 10)}:
        print(f"❌ Quality gate failed: Error issues ({{errors}}) > threshold ({quality_gate.get('max_errors', 10)})")
        fail = True
    
    if fail:
        sys.exit(1)
except FileNotFoundError:
    print("⚠️ Analysis report not found")
    sys.exit(1)
except Exception as e:
    print(f"⚠️ Error parsing report: {{e}}")
    sys.exit(1)
EOF
  artifacts:
    when: always
    paths:
      - .refactron-reports/
    expire_in: 30 days
  only:
    changes:
      - "**/*.py"
      - ".gitlab-ci.yml"

analyze:python3.8:
  <<: *refactron_template
  image: python:3.8
  stage: analyze

analyze:python3.9:
  <<: *refactron_template
  image: python:3.9
  stage: analyze

analyze:python3.10:
  <<: *refactron_template
  image: python:3.10
  stage: analyze

analyze:python3.11:
  <<: *refactron_template
  image: python:3.11
  stage: analyze

analyze:python3.12:
  <<: *refactron_template
  image: python:3.12
  stage: analyze
"""

        return pipeline

    @staticmethod
    def generate_pre_commit_pipeline(python_version: str = "3.11") -> str:
        """Generate GitLab CI pipeline for pre-commit analysis.

        Args:
            python_version: Python version to use

        Returns:
            YAML pipeline content
        """
        pipeline = f"""stages:
  - pre-commit

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

pre-commit:refactron:
  image: python:{python_version}
  stage: pre-commit
  before_script:
    - python -m pip install --upgrade pip
    - pip install refactron
  script:
    - |
      if [ "$CI_MERGE_REQUEST_ID" != "" ]; then
        git fetch origin $CI_MERGE_REQUEST_TARGET_BRANCH_NAME
        CHANGED_FILES=$(git diff --name-only origin/$CI_MERGE_REQUEST_TARGET_BRANCH_NAME | grep '\\.py$' || true)
        
        if [ -n "$CHANGED_FILES" ]; then
          echo "$CHANGED_FILES" | xargs refactron analyze --format json --log-format json
        else
          echo "No Python files changed"
        fi
      else
        refactron analyze . --format json --log-format json
      fi
  only:
    - merge_requests
    - branches
"""

        return pipeline

    @staticmethod
    def save_pipeline(pipeline_content: str, output_path: Path) -> None:
        """Save pipeline configuration to file.

        Args:
            pipeline_content: Pipeline YAML content
            output_path: Path to save pipeline file

        Raises:
            IOError: If file cannot be written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pipeline_content)
