# refactron.cicd

CI/CD integration templates and utilities for Refactron.

## Classes

## Functions


---

# refactron.cicd.github_actions

GitHub Actions workflow template generation.

## Classes

### GitHubActionsGenerator

```python
GitHubActionsGenerator()
```

Generate GitHub Actions workflow templates for Refactron.

#### GitHubActionsGenerator.generate_analysis_workflow

```python
GitHubActionsGenerator.generate_analysis_workflow(python_versions: Optional[List[str]] = None, trigger_on: Optional[List[str]] = None, quality_gate: Optional[Dict[str, Any]] = None, cache_enabled: bool = True, upload_artifacts: bool = True) -> str
```

Generate GitHub Actions workflow for code analysis.

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

#### GitHubActionsGenerator.generate_pre_commit_workflow

```python
GitHubActionsGenerator.generate_pre_commit_workflow(python_version: str = '3.11', trigger_on: Optional[List[str]] = None) -> str
```

Generate GitHub Actions workflow for pre-commit analysis.

Args:
    python_version: Python version to use
    trigger_on: Events to trigger on

Returns:
    YAML workflow content

#### GitHubActionsGenerator.save_workflow

```python
GitHubActionsGenerator.save_workflow(workflow_content: str, output_path: pathlib._local.Path) -> None
```

Save workflow to file.

Args:
    workflow_content: Workflow YAML content
    output_path: Path to save workflow file

Raises:
    IOError: If file cannot be written

## Functions


---

# refactron.cicd.gitlab_ci

GitLab CI pipeline configuration generation.

## Classes

### GitLabCIGenerator

```python
GitLabCIGenerator()
```

Generate GitLab CI pipeline configurations for Refactron.

#### GitLabCIGenerator.generate_analysis_pipeline

```python
GitLabCIGenerator.generate_analysis_pipeline(python_versions: Optional[List[str]] = None, quality_gate: Optional[Dict[str, Any]] = None, cache_enabled: bool = True, artifacts_enabled: bool = True) -> str
```

Generate GitLab CI pipeline for code analysis.

Args:
    python_versions: Python versions to test
    quality_gate: Quality gate configuration
    cache_enabled: Enable dependency caching
    artifacts_enabled: Save analysis reports as artifacts

Returns:
    YAML pipeline content

#### GitLabCIGenerator.generate_pre_commit_pipeline

```python
GitLabCIGenerator.generate_pre_commit_pipeline(python_version: str = '3.11') -> str
```

Generate GitLab CI pipeline for pre-commit analysis.

Args:
    python_version: Python version to use

Returns:
    YAML pipeline content

#### GitLabCIGenerator.save_pipeline

```python
GitLabCIGenerator.save_pipeline(pipeline_content: str, output_path: pathlib._local.Path) -> None
```

Save pipeline configuration to file.

Args:
    pipeline_content: Pipeline YAML content
    output_path: Path to save pipeline file

Raises:
    IOError: If file cannot be written

## Functions


---

# refactron.cicd.pr_integration

Pull Request integration utilities for posting comments and suggestions.

## Classes

### PRComment

```python
PRComment(file_path: str, line: int, message: str, level: str, rule_id: Optional[str] = None, suggestion: Optional[str] = None) -> None
```

Represents a PR comment.

#### PRComment.__init__

```python
PRComment.__init__(self, file_path: str, line: int, message: str, level: str, rule_id: Optional[str] = None, suggestion: Optional[str] = None) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### PRComment.to_markdown

```python
PRComment.to_markdown(self) -> str
```

Convert comment to markdown format.

Returns:
    Markdown formatted comment

### PRIntegration

```python
PRIntegration()
```

Utilities for PR integration and inline comments.

#### PRIntegration.format_comment_for_github_api

```python
PRIntegration.format_comment_for_github_api(comment: refactron.cicd.pr_integration.PRComment) -> Dict
```

Format comment for GitHub API.

Args:
    comment: PR comment

Returns:
    GitHub API comment format

#### PRIntegration.generate_github_comment_body

```python
PRIntegration.generate_github_comment_body(result: refactron.core.analysis_result.AnalysisResult) -> str
```

Generate GitHub PR comment body.

Args:
    result: Analysis result

Returns:
    Markdown comment body

#### PRIntegration.generate_inline_comments

```python
PRIntegration.generate_inline_comments(result: refactron.core.analysis_result.AnalysisResult, file_path: pathlib._local.Path) -> List[refactron.cicd.pr_integration.PRComment]
```

Generate inline comments for a specific file.

Args:
    result: Analysis result
    file_path: File to generate comments for

Returns:
    List of PR comments

#### PRIntegration.generate_pr_summary

```python
PRIntegration.generate_pr_summary(result: refactron.core.analysis_result.AnalysisResult) -> str
```

Generate PR summary from analysis result.

Args:
    result: Analysis result

Returns:
    Markdown formatted summary

#### PRIntegration.save_comments_json

```python
PRIntegration.save_comments_json(comments: List[refactron.cicd.pr_integration.PRComment], output_path: pathlib._local.Path) -> None
```

Save comments to JSON file for CI/CD integration.

Args:
    comments: List of PR comments
    output_path: Path to save JSON file

Raises:
    IOError: If file cannot be written

## Functions


---

# refactron.cicd.pre_commit

Pre-commit hook template generation.

## Classes

### PreCommitGenerator

```python
PreCommitGenerator()
```

Generate pre-commit hook templates for Refactron.

#### PreCommitGenerator.generate_pre_commit_config

```python
PreCommitGenerator.generate_pre_commit_config(stages: Optional[List[str]] = None, fail_on_critical: bool = True, fail_on_errors: bool = False, max_critical: int = 0, max_errors: int = 10) -> str
```

Generate pre-commit hook configuration.

Args:
    stages: Pre-commit stages to run on (default: ['commit', 'push'])
    fail_on_critical: Fail commit if critical issues found
    fail_on_errors: Fail commit if error-level issues found
    max_critical: Maximum allowed critical issues
    max_errors: Maximum allowed error-level issues

Returns:
    YAML configuration content

#### PreCommitGenerator.generate_simple_hook

```python
PreCommitGenerator.generate_simple_hook() -> str
```

Generate simple pre-commit hook script.

Returns:
    Bash script content

#### PreCommitGenerator.save_config

```python
PreCommitGenerator.save_config(config_content: str, output_path: pathlib._local.Path) -> None
```

Save pre-commit configuration to file.

Args:
    config_content: YAML configuration content
    output_path: Path to save configuration file

Raises:
    IOError: If file cannot be written

#### PreCommitGenerator.save_hook

```python
PreCommitGenerator.save_hook(hook_content: str, output_path: pathlib._local.Path) -> None
```

Save pre-commit hook script to file.

Args:
    hook_content: Hook script content
    output_path: Path to save hook file

Raises:
    IOError: If file cannot be written

## Functions


---

# refactron.cicd.quality_gates

Quality gate parsing and enforcement for CI/CD pipelines.

## Classes

### QualityGate

```python
QualityGate(max_critical: int = 0, max_errors: int = 10, max_warnings: int = 50, max_total: Optional[int] = None, fail_on_critical: bool = True, fail_on_errors: bool = False, fail_on_warnings: bool = False, min_success_rate: float = 0.95) -> None
```

Configuration for quality gate thresholds.

#### QualityGate.__init__

```python
QualityGate.__init__(self, max_critical: int = 0, max_errors: int = 10, max_warnings: int = 50, max_total: Optional[int] = None, fail_on_critical: bool = True, fail_on_errors: bool = False, fail_on_warnings: bool = False, min_success_rate: float = 0.95) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### QualityGate.check

```python
QualityGate.check(self, result: refactron.core.analysis_result.AnalysisResult) -> Tuple[bool, str]
```

Check if quality gate passes.

Args:
    result: Analysis result to check

Returns:
    Tuple of (passed, message)

#### QualityGate.to_dict

```python
QualityGate.to_dict(self) -> Dict
```

Convert quality gate to dictionary.

### QualityGateParser

```python
QualityGateParser()
```

Parse CLI output and enforce quality gates.

#### QualityGateParser.enforce_gate

```python
QualityGateParser.enforce_gate(result: refactron.core.analysis_result.AnalysisResult, gate: refactron.cicd.quality_gates.QualityGate) -> Tuple[bool, str, int]
```

Enforce quality gate on analysis result.

Args:
    result: Analysis result
    gate: Quality gate configuration

Returns:
    Tuple of (passed, message, exit_code)

#### QualityGateParser.generate_summary

```python
QualityGateParser.generate_summary(result: refactron.core.analysis_result.AnalysisResult) -> str
```

Generate quality gate summary for CI/CD.

Args:
    result: Analysis result

Returns:
    Formatted summary string

#### QualityGateParser.parse_exit_code

```python
QualityGateParser.parse_exit_code(exit_code: int) -> Dict[str, int]
```

Parse exit code from refactron analyze.

Args:
    exit_code: Process exit code

Returns:
    Dictionary indicating if build should fail

#### QualityGateParser.parse_json_output

```python
QualityGateParser.parse_json_output(json_path: pathlib._local.Path) -> Dict
```

Parse JSON output from refactron analyze --format json.

Args:
    json_path: Path to JSON output file

Returns:
    Parsed JSON dictionary

Raises:
    ValueError: If JSON is invalid
    FileNotFoundError: If file doesn't exist

#### QualityGateParser.parse_text_output

```python
QualityGateParser.parse_text_output(text: str) -> Dict[str, int]
```

Parse text output from refactron analyze command.

Args:
    text: Text output from CLI

Returns:
    Dictionary with issue counts

## Functions

