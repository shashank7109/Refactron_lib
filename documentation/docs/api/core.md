# refactron.core

Core functionality for Refactron.

## Classes

## Functions


---

# refactron.core.analysis_result

Analysis result representation.

## Classes

### AnalysisResult

```python
AnalysisResult(file_metrics: List[refactron.core.models.FileMetrics] = <factory>, total_files: int = 0, total_issues: int = 0, failed_files: List[refactron.core.analysis_result.FileAnalysisError] = <factory>) -> None
```

Result of code analysis.

#### AnalysisResult.__init__

```python
AnalysisResult.__init__(self, file_metrics: List[refactron.core.models.FileMetrics] = <factory>, total_files: int = 0, total_issues: int = 0, failed_files: List[refactron.core.analysis_result.FileAnalysisError] = <factory>) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### AnalysisResult.issues_by_file

```python
AnalysisResult.issues_by_file(self, file_path: pathlib._local.Path) -> List[refactron.core.models.CodeIssue]
```

Get issues for a specific file.

#### AnalysisResult.issues_by_level

```python
AnalysisResult.issues_by_level(self, level: refactron.core.models.IssueLevel) -> List[refactron.core.models.CodeIssue]
```

Get issues filtered by severity level.

#### AnalysisResult.report

```python
AnalysisResult.report(self, detailed: bool = True) -> str
```

Generate a text report of the analysis.

#### AnalysisResult.summary

```python
AnalysisResult.summary(self) -> Dict[str, int]
```

Get a summary of the analysis.

### FileAnalysisError

```python
FileAnalysisError(file_path: pathlib._local.Path, error_message: str, error_type: str, recovery_suggestion: Optional[str] = None) -> None
```

Represents an error that occurred while analyzing a file.

#### FileAnalysisError.__init__

```python
FileAnalysisError.__init__(self, file_path: pathlib._local.Path, error_message: str, error_type: str, recovery_suggestion: Optional[str] = None) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

## Functions


---

# refactron.core.backup

Backup and Rollback System for Refactron.

Provides functionality to:
- Auto-create .refactron-backup/ directory with original files before changes
- Git integration for automatic commits before major refactoring
- Rollback capability to restore original files

## Classes

### BackupManager

```python
BackupManager(root_dir: Optional[pathlib._local.Path] = None)
```

Manage backups and rollbacks for refactoring operations.

#### BackupManager.__init__

```python
BackupManager.__init__(self, root_dir: Optional[pathlib._local.Path] = None)
```

Initialize the backup manager.

Args:
    root_dir: Root directory for backups. Defaults to current working directory.

#### BackupManager.backup_file

```python
BackupManager.backup_file(self, file_path: pathlib._local.Path, session_id: str) -> pathlib._local.Path
```

Backup a single file to the backup directory.

Args:
    file_path: Path to the file to backup.
    session_id: Session ID for this backup operation.

Returns:
    Path to the backup file.

#### BackupManager.backup_files

```python
BackupManager.backup_files(self, file_paths: List[pathlib._local.Path], session_id: str) -> Tuple[List[pathlib._local.Path], List[pathlib._local.Path]]
```

Backup multiple files.

Args:
    file_paths: List of file paths to backup.
    session_id: Session ID for this backup operation.

Returns:
    Tuple of (successful backup paths, failed file paths).

#### BackupManager.clear_all_sessions

```python
BackupManager.clear_all_sessions(self) -> int
```

Clear all backup sessions.

Returns:
    Number of sessions cleared.

#### BackupManager.clear_session

```python
BackupManager.clear_session(self, session_id: str) -> bool
```

Clear a specific backup session.

Args:
    session_id: Session ID to clear.

Returns:
    True if successful, False otherwise.

#### BackupManager.create_backup_session

```python
BackupManager.create_backup_session(self, description: str = '') -> str
```

Create a new backup session.

Args:
    description: Description of the operation being performed.

Returns:
    Session ID for the backup session.

#### BackupManager.get_latest_session

```python
BackupManager.get_latest_session(self) -> Optional[Dict[str, Any]]
```

Get the latest backup session.

Returns:
    Latest session information or None if no sessions exist.

#### BackupManager.get_session

```python
BackupManager.get_session(self, session_id: str) -> Optional[Dict[str, Any]]
```

Get information about a specific session.

Args:
    session_id: Session ID to look up.

Returns:
    Session information or None if not found.

#### BackupManager.list_sessions

```python
BackupManager.list_sessions(self) -> List[Dict[str, Any]]
```

List all backup sessions.

Returns:
    List of session information dictionaries.

#### BackupManager.rollback_session

```python
BackupManager.rollback_session(self, session_id: Optional[str] = None) -> Tuple[int, List[str]]
```

Rollback files from a backup session.

Args:
    session_id: Session ID to rollback. If None, uses the latest session.

Returns:
    Tuple of (number of files restored, list of failed file paths).

#### BackupManager.update_session_git_commit

```python
BackupManager.update_session_git_commit(self, session_id: str, commit_hash: Optional[str]) -> bool
```

Update the Git commit hash for a session.

Args:
    session_id: Session ID to update.
    commit_hash: Git commit hash to associate with the session.

Returns:
    True if successful, False if session not found.

### BackupRollbackSystem

```python
BackupRollbackSystem(root_dir: Optional[pathlib._local.Path] = None)
```

Combined backup and rollback system that integrates file backups with Git.

#### BackupRollbackSystem.__init__

```python
BackupRollbackSystem.__init__(self, root_dir: Optional[pathlib._local.Path] = None)
```

Initialize the backup and rollback system.

Args:
    root_dir: Root directory for operations.

#### BackupRollbackSystem.clear_all

```python
BackupRollbackSystem.clear_all(self) -> int
```

Clear all backup sessions.

#### BackupRollbackSystem.list_sessions

```python
BackupRollbackSystem.list_sessions(self) -> List[Dict[str, Any]]
```

List all backup sessions.

#### BackupRollbackSystem.prepare_for_refactoring

```python
BackupRollbackSystem.prepare_for_refactoring(self, files: List[pathlib._local.Path], description: str = 'refactoring operation', create_git_commit: bool = True) -> Tuple[str, List[pathlib._local.Path]]
```

Prepare for a refactoring operation by creating backups and optionally a Git commit.

Args:
    files: List of files to be refactored.
    description: Description of the refactoring operation.
    create_git_commit: Whether to create a Git commit before refactoring.

Returns:
    Tuple of (session ID, list of files that failed to backup).

#### BackupRollbackSystem.rollback

```python
BackupRollbackSystem.rollback(self, session_id: Optional[str] = None, use_git: bool = False) -> Dict[str, Any]
```

Rollback changes from a refactoring session.

Args:
    session_id: Session ID to rollback. If None, uses the latest session.
    use_git: Whether to use Git rollback instead of file backup.

Returns:
    Dictionary with rollback results.

### GitIntegration

```python
GitIntegration(repo_path: Optional[pathlib._local.Path] = None)
```

Git integration for automatic commits before refactoring.

#### GitIntegration.__init__

```python
GitIntegration.__init__(self, repo_path: Optional[pathlib._local.Path] = None)
```

Initialize Git integration.

Args:
    repo_path: Path to the Git repository. Defaults to current directory.

#### GitIntegration.create_pre_refactor_commit

```python
GitIntegration.create_pre_refactor_commit(self, message: Optional[str] = None, files: Optional[List[pathlib._local.Path]] = None) -> Optional[str]
```

Create a commit before refactoring.

Args:
    message: Commit message. Defaults to auto-generated message.
    files: Specific files to commit. If None, stages and commits all
           uncommitted changes (git add -A). Note: This may include
           unintended files like temporary files or build artifacts.

Returns:
    Commit hash if successful, None otherwise.

#### GitIntegration.get_current_branch

```python
GitIntegration.get_current_branch(self) -> Optional[str]
```

Get the current Git branch name.

#### GitIntegration.get_current_commit

```python
GitIntegration.get_current_commit(self) -> Optional[str]
```

Get the current commit hash.

#### GitIntegration.git_rollback_to_commit

```python
GitIntegration.git_rollback_to_commit(self, commit_hash: str) -> bool
```

Rollback to a specific commit (soft reset).

Args:
    commit_hash: Commit hash to rollback to.

Returns:
    True if successful, False otherwise.

#### GitIntegration.has_uncommitted_changes

```python
GitIntegration.has_uncommitted_changes(self) -> bool
```

Check if there are uncommitted changes.

#### GitIntegration.is_git_repo

```python
GitIntegration.is_git_repo(self) -> bool
```

Check if the current directory is a Git repository.

## Functions


---

# refactron.core.cache

AST caching layer for performance optimization.

## Classes

### ASTCache

```python
ASTCache(cache_dir: Optional[pathlib._local.Path] = None, enabled: bool = True, max_cache_size_mb: int = 100, cleanup_threshold_percent: float = 0.8)
```

Cache for parsed AST trees to avoid re-parsing identical files.

Uses file content hashing to determine cache validity.

#### ASTCache.__init__

```python
ASTCache.__init__(self, cache_dir: Optional[pathlib._local.Path] = None, enabled: bool = True, max_cache_size_mb: int = 100, cleanup_threshold_percent: float = 0.8)
```

Initialize the AST cache.

Args:
    cache_dir: Directory to store cache files. If None, uses temporary directory.
    enabled: Whether caching is enabled.
    max_cache_size_mb: Maximum cache size in megabytes.
    cleanup_threshold_percent: Cleanup to this percentage of max when limit exceeded.

#### ASTCache.clear

```python
ASTCache.clear(self) -> None
```

Clear all cached data.

#### ASTCache.get

```python
ASTCache.get(self, file_path: pathlib._local.Path, content: str) -> Optional[Tuple[libcst._nodes.module.Module, Dict[str, Any]]]
```

Get cached AST and metadata for a file.

Args:
    file_path: Path to the file.
    content: Current content of the file.

Returns:
    Tuple of (AST module, metadata) if cached, None otherwise.

#### ASTCache.get_stats

```python
ASTCache.get_stats(self) -> Dict[str, Any]
```

Get cache statistics.

Returns:
    Dictionary containing cache statistics.

#### ASTCache.put

```python
ASTCache.put(self, file_path: pathlib._local.Path, content: str, ast_module: libcst._nodes.module.Module, metadata: Optional[Dict[str, Any]] = None) -> None
```

Store AST and metadata in cache.

Args:
    file_path: Path to the file.
    content: Content of the file.
    ast_module: Parsed AST module.
    metadata: Optional metadata to cache alongside the AST.

## Functions


---

# refactron.core.config

Configuration management for Refactron.

## Classes

### RefactronConfig

```python
RefactronConfig(version: str = <factory>, environment: Optional[str] = None, enabled_analyzers: List[str] = <factory>, enabled_refactorers: List[str] = <factory>, max_function_complexity: int = 10, max_function_length: int = 50, max_file_length: int = 500, max_parameters: int = 5, report_format: str = 'text', show_details: bool = True, require_preview: bool = True, backup_enabled: bool = True, include_patterns: List[str] = <factory>, exclude_patterns: List[str] = <factory>, custom_rules: Dict[str, Any] = <factory>, security_ignore_patterns: List[str] = <factory>, security_rule_whitelist: Dict[str, List[str]] = <factory>, security_min_confidence: float = 0.5, enable_ast_cache: bool = True, ast_cache_dir: Optional[pathlib._local.Path] = None, max_ast_cache_size_mb: int = 100, enable_incremental_analysis: bool = True, incremental_state_file: Optional[pathlib._local.Path] = None, enable_parallel_processing: bool = True, max_parallel_workers: Optional[int] = None, use_multiprocessing: bool = False, enable_memory_profiling: bool = False, memory_optimization_threshold_mb: float = 5.0, memory_pressure_threshold_percent: float = 80.0, memory_pressure_threshold_available_mb: float = 500.0, cache_cleanup_threshold_percent: float = 0.8, log_level: str = 'INFO', log_format: str = 'text', log_file: Optional[pathlib._local.Path] = None, log_max_bytes: int = 10485760, log_backup_count: int = 5, enable_console_logging: bool = True, enable_file_logging: bool = True, enable_metrics: bool = True, metrics_detailed: bool = True, enable_prometheus: bool = False, prometheus_host: str = '127.0.0.1', prometheus_port: int = 9090, enable_telemetry: bool = False, telemetry_endpoint: Optional[str] = None, enable_pattern_learning: bool = True, pattern_storage_dir: Optional[pathlib._local.Path] = None, pattern_learning_enabled: bool = True, pattern_ranking_enabled: bool = True) -> None
```

Configuration for Refactron analysis and refactoring.

#### RefactronConfig.__init__

```python
RefactronConfig.__init__(self, version: str = <factory>, environment: Optional[str] = None, enabled_analyzers: List[str] = <factory>, enabled_refactorers: List[str] = <factory>, max_function_complexity: int = 10, max_function_length: int = 50, max_file_length: int = 500, max_parameters: int = 5, report_format: str = 'text', show_details: bool = True, require_preview: bool = True, backup_enabled: bool = True, include_patterns: List[str] = <factory>, exclude_patterns: List[str] = <factory>, custom_rules: Dict[str, Any] = <factory>, security_ignore_patterns: List[str] = <factory>, security_rule_whitelist: Dict[str, List[str]] = <factory>, security_min_confidence: float = 0.5, enable_ast_cache: bool = True, ast_cache_dir: Optional[pathlib._local.Path] = None, max_ast_cache_size_mb: int = 100, enable_incremental_analysis: bool = True, incremental_state_file: Optional[pathlib._local.Path] = None, enable_parallel_processing: bool = True, max_parallel_workers: Optional[int] = None, use_multiprocessing: bool = False, enable_memory_profiling: bool = False, memory_optimization_threshold_mb: float = 5.0, memory_pressure_threshold_percent: float = 80.0, memory_pressure_threshold_available_mb: float = 500.0, cache_cleanup_threshold_percent: float = 0.8, log_level: str = 'INFO', log_format: str = 'text', log_file: Optional[pathlib._local.Path] = None, log_max_bytes: int = 10485760, log_backup_count: int = 5, enable_console_logging: bool = True, enable_file_logging: bool = True, enable_metrics: bool = True, metrics_detailed: bool = True, enable_prometheus: bool = False, prometheus_host: str = '127.0.0.1', prometheus_port: int = 9090, enable_telemetry: bool = False, telemetry_endpoint: Optional[str] = None, enable_pattern_learning: bool = True, pattern_storage_dir: Optional[pathlib._local.Path] = None, pattern_learning_enabled: bool = True, pattern_ranking_enabled: bool = True) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### RefactronConfig.to_file

```python
RefactronConfig.to_file(self, config_path: pathlib._local.Path) -> None
```

Save configuration to a YAML file.

Args:
    config_path: Path where configuration should be saved

Raises:
    ConfigError: If config file cannot be written

## Functions


---

# refactron.core.config_loader

Enhanced configuration loader with profiles, inheritance, and versioning.

## Classes

### ConfigLoader

```python
ConfigLoader()
```

Loads and merges configuration with support for profiles and inheritance.

## Functions


---

# refactron.core.config_templates

Configuration templates for common Python frameworks.

## Classes

### ConfigTemplates

```python
ConfigTemplates()
```

Pre-configured templates for common Python frameworks.

#### ConfigTemplates.get_base_template

```python
ConfigTemplates.get_base_template() -> Dict
```

Get base configuration template.

#### ConfigTemplates.get_django_template

```python
ConfigTemplates.get_django_template() -> Dict
```

Get Django-specific configuration template.

#### ConfigTemplates.get_fastapi_template

```python
ConfigTemplates.get_fastapi_template() -> Dict
```

Get FastAPI-specific configuration template.

#### ConfigTemplates.get_flask_template

```python
ConfigTemplates.get_flask_template() -> Dict
```

Get Flask-specific configuration template.

#### ConfigTemplates.get_template

```python
ConfigTemplates.get_template(framework: str) -> Dict
```

Get configuration template for a specific framework.

Args:
    framework: Framework name (django, fastapi, flask, base)

Returns:
    Configuration template dictionary

Raises:
    ValueError: If framework is not supported

## Functions


---

# refactron.core.config_validator

Configuration schema validation for Refactron.

## Classes

### ConfigValidator

```python
ConfigValidator()
```

Validates Refactron configuration against schema.

## Functions


---

# refactron.core.credentials

Local credential storage for Refactron CLI.

This is intentionally minimal: credentials are stored in a user-only readable file
under ~/.refactron/. For production hardening, an OS keychain integration can be
added later.

## Classes

### RefactronCredentials

```python
RefactronCredentials(api_base_url: 'str', access_token: 'str', token_type: 'str', expires_at: 'Optional[datetime]' = None, email: 'Optional[str]' = None, plan: 'Optional[str]' = None, api_key: 'Optional[str]' = None) -> None
```

Stored CLI credentials.

#### RefactronCredentials.__init__

```python
RefactronCredentials.__init__(self, api_base_url: 'str', access_token: 'str', token_type: 'str', expires_at: 'Optional[datetime]' = None, email: 'Optional[str]' = None, plan: 'Optional[str]' = None, api_key: 'Optional[str]' = None) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### RefactronCredentials.to_dict

```python
RefactronCredentials.to_dict(self) -> 'Dict[str, Any]'
```

No documentation available.

## Functions

### credentials_path

```python
credentials_path() -> 'Path'
```

Default credentials file path.

### delete_credentials

```python
delete_credentials(path: 'Optional[Path]' = None) -> 'bool'
```

Delete stored credentials. Returns True if deleted, False if not present.

### load_credentials

```python
load_credentials(path: 'Optional[Path]' = None) -> 'Optional[RefactronCredentials]'
```

Load credentials from disk. Returns None if missing or invalid.

### save_credentials

```python
save_credentials(creds: 'RefactronCredentials', path: 'Optional[Path]' = None) -> 'None'
```

Save credentials to disk (0600 permissions where supported).


---

# refactron.core.device_auth

Device-code authentication helpers for Refactron CLI.

Implements a minimal Device Authorization Grant-like flow against the Refactron API:
- POST /oauth/device to get (device_code, user_code, verification_uri)
- POST /oauth/token to poll until authorized and receive tokens

## Classes

### DeviceAuthorization

```python
DeviceAuthorization(device_code: 'str', user_code: 'str', verification_uri: 'str', expires_in: 'int', interval: 'int') -> None
```

DeviceAuthorization(device_code: 'str', user_code: 'str', verification_uri: 'str', expires_in: 'int', interval: 'int')

#### DeviceAuthorization.__init__

```python
DeviceAuthorization.__init__(self, device_code: 'str', user_code: 'str', verification_uri: 'str', expires_in: 'int', interval: 'int') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### TokenResponse

```python
TokenResponse(access_token: 'str', token_type: 'str', expires_in: 'int', email: 'Optional[str]' = None, plan: 'Optional[str]' = None, api_key: 'Optional[str]' = None) -> None
```

TokenResponse(access_token: 'str', token_type: 'str', expires_in: 'int', email: 'Optional[str]' = None, plan: 'Optional[str]' = None, api_key: 'Optional[str]' = None)

#### TokenResponse.__init__

```python
TokenResponse.__init__(self, access_token: 'str', token_type: 'str', expires_in: 'int', email: 'Optional[str]' = None, plan: 'Optional[str]' = None, api_key: 'Optional[str]' = None) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### TokenResponse.expires_at

```python
TokenResponse.expires_at(self) -> 'datetime'
```

No documentation available.

## Functions

### _normalize_base_url

```python
_normalize_base_url(api_base_url: 'str') -> 'str'
```

No documentation available.

### _post_json

```python
_post_json(url: 'str', payload: 'Dict[str, Any]', timeout_seconds: 'int' = 10) -> 'Dict[str, Any]'
```

No documentation available.

### poll_for_token

```python
poll_for_token(device_code: 'str', api_base_url: 'str' = 'https://api.refactron.dev', client_id: 'str' = 'refactron-cli', interval_seconds: 'int' = 5, expires_in_seconds: 'int' = 900, timeout_seconds: 'int' = 10, sleep_fn: 'Callable[[float], None]' = <built-in function sleep>) -> 'TokenResponse'
```

No documentation available.

### start_device_authorization

```python
start_device_authorization(api_base_url: 'str' = 'https://api.refactron.dev', client_id: 'str' = 'refactron-cli', timeout_seconds: 'int' = 10) -> 'DeviceAuthorization'
```

No documentation available.


---

# refactron.core.exceptions

Custom exception types for Refactron.

This module defines granular exception types for different failure scenarios,
enabling better error handling and recovery strategies.

## Classes

### AnalysisError

```python
AnalysisError(message: str, file_path: Optional[pathlib._local.Path] = None, analyzer_name: Optional[str] = None, recovery_suggestion: Optional[str] = None)
```

Raised when code analysis fails.

This exception is raised when an analyzer encounters an error
while processing source code. Common causes include:
- Syntax errors in the source code
- Unsupported Python language features
- AST parsing failures
- File encoding issues

#### AnalysisError.__init__

```python
AnalysisError.__init__(self, message: str, file_path: Optional[pathlib._local.Path] = None, analyzer_name: Optional[str] = None, recovery_suggestion: Optional[str] = None)
```

Initialize the exception.

Args:
    message: Error message describing what went wrong
    file_path: Optional path to the file being analyzed
    analyzer_name: Name of the analyzer that failed
    recovery_suggestion: Optional suggestion for how to recover

### ConfigError

```python
ConfigError(message: str, config_path: Optional[pathlib._local.Path] = None, config_key: Optional[str] = None, recovery_suggestion: Optional[str] = None)
```

Raised when configuration is invalid or cannot be loaded.

This exception is raised when there are problems with the
configuration. Common causes include:
- Invalid YAML syntax in config file
- Missing required configuration options
- Invalid configuration values (e.g., negative thresholds)
- Configuration file not found or not readable

#### ConfigError.__init__

```python
ConfigError.__init__(self, message: str, config_path: Optional[pathlib._local.Path] = None, config_key: Optional[str] = None, recovery_suggestion: Optional[str] = None)
```

Initialize the exception.

Args:
    message: Error message describing what went wrong
    config_path: Optional path to the config file
    config_key: Optional specific config key that caused the error
    recovery_suggestion: Optional suggestion for how to recover

### RefactoringError

```python
RefactoringError(message: str, file_path: Optional[pathlib._local.Path] = None, operation_type: Optional[str] = None, recovery_suggestion: Optional[str] = None)
```

Raised when code refactoring fails.

This exception is raised when a refactoring operation cannot be
completed successfully. Common causes include:
- Unable to parse the source code
- Refactoring would break code semantics
- File write permission issues
- Backup creation failures

#### RefactoringError.__init__

```python
RefactoringError.__init__(self, message: str, file_path: Optional[pathlib._local.Path] = None, operation_type: Optional[str] = None, recovery_suggestion: Optional[str] = None)
```

Initialize the exception.

Args:
    message: Error message describing what went wrong
    file_path: Optional path to the file being refactored
    operation_type: Type of refactoring operation that failed
    recovery_suggestion: Optional suggestion for how to recover

### RefactronError

```python
RefactronError(message: str, file_path: Optional[pathlib._local.Path] = None, recovery_suggestion: Optional[str] = None)
```

Base exception for all Refactron errors.

All custom exceptions in Refactron inherit from this class,
allowing for easy catching of all Refactron-specific errors.

#### RefactronError.__init__

```python
RefactronError.__init__(self, message: str, file_path: Optional[pathlib._local.Path] = None, recovery_suggestion: Optional[str] = None)
```

Initialize the exception.

Args:
    message: Error message describing what went wrong
    file_path: Optional path to the file that caused the error
    recovery_suggestion: Optional suggestion for how to recover from the error

## Functions


---

# refactron.core.false_positive_tracker

False positive tracking system for security rules.

## Classes

### FalsePositiveTracker

```python
FalsePositiveTracker(storage_path: pathlib._local.Path = None)
```

Tracks and learns from false positive patterns.

#### FalsePositiveTracker.__init__

```python
FalsePositiveTracker.__init__(self, storage_path: pathlib._local.Path = None)
```

Initialize the false positive tracker.

Args:
    storage_path: Path to store false positive data

#### FalsePositiveTracker.clear_all

```python
FalsePositiveTracker.clear_all(self) -> None
```

Clear all false positive data.

#### FalsePositiveTracker.clear_rule

```python
FalsePositiveTracker.clear_rule(self, rule_id: str) -> None
```

Clear all false positives for a specific rule.

Args:
    rule_id: The rule to clear

#### FalsePositiveTracker.get_false_positive_patterns

```python
FalsePositiveTracker.get_false_positive_patterns(self, rule_id: str) -> List[str]
```

Get all false positive patterns for a rule.

Args:
    rule_id: The rule ID

Returns:
    List of false positive patterns

#### FalsePositiveTracker.is_false_positive

```python
FalsePositiveTracker.is_false_positive(self, rule_id: str, pattern: str) -> bool
```

Check if a pattern is marked as a false positive.

Args:
    rule_id: The rule to check
    pattern: The pattern to check

Returns:
    True if the pattern is a known false positive

#### FalsePositiveTracker.load

```python
FalsePositiveTracker.load(self) -> None
```

Load false positive data from storage.

#### FalsePositiveTracker.mark_false_positive

```python
FalsePositiveTracker.mark_false_positive(self, rule_id: str, pattern: str) -> None
```

Mark a pattern as a false positive for a specific rule.

Args:
    rule_id: The rule that produced the false positive
    pattern: The pattern that was incorrectly flagged

#### FalsePositiveTracker.save

```python
FalsePositiveTracker.save(self) -> None
```

Save false positive data to storage.

## Functions


---

# refactron.core.incremental

Incremental analysis tracking for performance optimization.

## Classes

### IncrementalAnalysisTracker

```python
IncrementalAnalysisTracker(state_file: Optional[pathlib._local.Path] = None, enabled: bool = True)
```

Track file changes to enable incremental analysis.

Only analyzes files that have changed since the last run.
Thread-safe for concurrent updates.

#### IncrementalAnalysisTracker.__init__

```python
IncrementalAnalysisTracker.__init__(self, state_file: Optional[pathlib._local.Path] = None, enabled: bool = True)
```

Initialize the incremental analysis tracker.

Args:
    state_file: Path to the state file. If None, uses default location.
    enabled: Whether incremental analysis is enabled.

#### IncrementalAnalysisTracker.cleanup_missing_files

```python
IncrementalAnalysisTracker.cleanup_missing_files(self, valid_file_paths: Set[pathlib._local.Path]) -> None
```

Remove files from state that no longer exist or are not in the valid set.

Args:
    valid_file_paths: Set of file paths that are still valid.

#### IncrementalAnalysisTracker.clear

```python
IncrementalAnalysisTracker.clear(self) -> None
```

Clear all state data.

#### IncrementalAnalysisTracker.get_changed_files

```python
IncrementalAnalysisTracker.get_changed_files(self, file_paths: List[pathlib._local.Path]) -> List[pathlib._local.Path]
```

Filter list of files to only those that have changed.

Args:
    file_paths: List of file paths to check.

Returns:
    List of files that have changed or are new.

#### IncrementalAnalysisTracker.get_stats

```python
IncrementalAnalysisTracker.get_stats(self) -> Dict[str, int]
```

Get statistics about the tracked state.

Returns:
    Dictionary containing statistics.

#### IncrementalAnalysisTracker.has_file_changed

```python
IncrementalAnalysisTracker.has_file_changed(self, file_path: pathlib._local.Path) -> bool
```

Check if a file has changed since the last analysis.

Args:
    file_path: Path to the file to check.

Returns:
    True if the file has changed or is new, False otherwise.

#### IncrementalAnalysisTracker.remove_file_state

```python
IncrementalAnalysisTracker.remove_file_state(self, file_path: pathlib._local.Path) -> None
```

Remove a file from the state tracking.

Args:
    file_path: Path to the file to remove.

#### IncrementalAnalysisTracker.save

```python
IncrementalAnalysisTracker.save(self) -> None
```

Save the current state to disk.

#### IncrementalAnalysisTracker.update_file_state

```python
IncrementalAnalysisTracker.update_file_state(self, file_path: pathlib._local.Path) -> None
```

Update the state for a file after analysis.

Args:
    file_path: Path to the file that was analyzed.

## Functions


---

# refactron.core.inference

Inference engine wrapping astroid for semantic analysis.
Provides capabilities to infer types, values, and resolve symbols.

## Classes

### InferenceEngine

```python
InferenceEngine()
```

Wrapper around astroid to provide high-level semantic analysis capabilities.

#### InferenceEngine.get_node_type_name

```python
InferenceEngine.get_node_type_name(node: astroid.nodes.node_ng.NodeNG) -> str
```

Get the string representation of the inferred type.

#### InferenceEngine.infer_node

```python
InferenceEngine.infer_node(node: astroid.nodes.node_ng.NodeNG, context: Optional[astroid.context.InferenceContext] = None) -> List[Any]
```

Attempt to infer the value/type of a given node.
Returns a list of potential values (astroid nodes).

#### InferenceEngine.is_subtype_of

```python
InferenceEngine.is_subtype_of(node: astroid.nodes.node_ng.NodeNG, type_name: str) -> bool
```

Check if node infers to a specific type name (e.g. 'str', 'int', 'MyClass').

#### InferenceEngine.parse_file

```python
InferenceEngine.parse_file(file_path: str) -> astroid.nodes.scoped_nodes.scoped_nodes.Module
```

Parse a file into an astroid node tree.

#### InferenceEngine.parse_string

```python
InferenceEngine.parse_string(code: str, module_name: str = '') -> astroid.nodes.scoped_nodes.scoped_nodes.Module
```

Parse source code string into an astroid node tree.

## Functions


---

# refactron.core.logging_config

Structured logging configuration for Refactron.

This module provides JSON-formatted logging for CI/CD and log aggregation systems,
with configurable log levels and rotation support.

## Classes

### JSONFormatter

```python
JSONFormatter(fmt=None, datefmt=None, style='%', validate=True, *, defaults=None)
```

Custom JSON formatter for structured logging.

#### JSONFormatter.__init__

```python
JSONFormatter.__init__(self, fmt=None, datefmt=None, style='%', validate=True, *, defaults=None)
```

Initialize the formatter with specified format strings.

Initialize the formatter either with the specified format string, or a
default as described above. Allow for specialized date formatting with
the optional datefmt argument. If datefmt is omitted, you get an
ISO8601-like (or RFC 3339-like) format.

Use a style parameter of '%', '\{' or '$' to specify that you want to
use one of %-formatting, :meth:`str.format` (``{}``) formatting or
:class:`string.Template` formatting in your format string.

.. versionchanged:: 3.2
   Added the ``style`` parameter.

#### JSONFormatter.format

```python
JSONFormatter.format(self, record: logging.LogRecord) -> str
```

Format log record as JSON.

Args:
    record: The log record to format

Returns:
    JSON-formatted log string

#### JSONFormatter.formatException

```python
JSONFormatter.formatException(self, ei)
```

Format and return the specified exception information as a string.

This default implementation just uses
traceback.print_exception()

#### JSONFormatter.formatMessage

```python
JSONFormatter.formatMessage(self, record)
```

No documentation available.

#### JSONFormatter.formatStack

```python
JSONFormatter.formatStack(self, stack_info)
```

This method is provided as an extension point for specialized
formatting of stack information.

The input data is a string as returned from a call to
:func:`traceback.print_stack`, but with the last trailing newline
removed.

The base implementation just returns the value passed in.

#### JSONFormatter.formatTime

```python
JSONFormatter.formatTime(self, record, datefmt=None)
```

Return the creation time of the specified LogRecord as formatted text.

This method should be called from format() by a formatter which
wants to make use of a formatted time. This method can be overridden
in formatters to provide for any specific requirement, but the
basic behaviour is as follows: if datefmt (a string) is specified,
it is used with time.strftime() to format the creation time of the
record. Otherwise, an ISO8601-like (or RFC 3339-like) format is used.
The resulting string is returned. This function uses a user-configurable
function to convert the creation time to a tuple. By default,
time.localtime() is used; to change this for a particular formatter
instance, set the 'converter' attribute to a function with the same
signature as time.localtime() or time.gmtime(). To change it for all
formatters, for example if you want all logging times to be shown in GMT,
set the 'converter' attribute in the Formatter class.

#### JSONFormatter.usesTime

```python
JSONFormatter.usesTime(self)
```

Check if the format uses the creation time of the record.

### StructuredLogger

```python
StructuredLogger(name: str = 'refactron', level: str = 'INFO', log_file: Optional[pathlib._local.Path] = None, log_format: str = 'json', max_bytes: int = 10485760, backup_count: int = 5, enable_console: bool = True, enable_file: bool = True)
```

Structured logger with JSON formatting and rotation support.

#### StructuredLogger.__init__

```python
StructuredLogger.__init__(self, name: str = 'refactron', level: str = 'INFO', log_file: Optional[pathlib._local.Path] = None, log_format: str = 'json', max_bytes: int = 10485760, backup_count: int = 5, enable_console: bool = True, enable_file: bool = True)
```

Initialize structured logger.

Args:
    name: Logger name
    level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file: Path to log file (if None, uses default location)
    log_format: Log format ('json' or 'text')
    max_bytes: Maximum log file size before rotation
    backup_count: Number of backup files to keep
    enable_console: Enable console logging
    enable_file: Enable file logging

#### StructuredLogger.get_logger

```python
StructuredLogger.get_logger(self) -> logging.Logger
```

Get the configured logger instance.

Returns:
    Configured logger instance

#### StructuredLogger.log_with_context

```python
StructuredLogger.log_with_context(self, level: str, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None
```

Log message with additional context data.

Args:
    level: Log level (debug, info, warning, error, critical)
    message: Log message
    extra_data: Additional context data to include in log

## Functions

### setup_logging

```python
setup_logging(level: str = 'INFO', log_file: Optional[pathlib._local.Path] = None, log_format: str = 'json', max_bytes: int = 10485760, backup_count: int = 5, enable_console: bool = True, enable_file: bool = True) -> refactron.core.logging_config.StructuredLogger
```

Setup structured logging for Refactron.

Args:
    level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file: Path to log file
    log_format: Log format ('json' or 'text')
    max_bytes: Maximum log file size before rotation
    backup_count: Number of backup files to keep
    enable_console: Enable console logging
    enable_file: Enable file logging

Returns:
    Configured StructuredLogger instance


---

# refactron.core.memory_profiler

Memory profiling and optimization utilities.

## Classes

### MemoryProfiler

```python
MemoryProfiler(enabled: bool = True, pressure_threshold_percent: float = 80.0, pressure_threshold_available_mb: float = 500.0)
```

Memory profiling and optimization utilities.

Helps track and optimize memory usage for large codebases.

#### MemoryProfiler.__init__

```python
MemoryProfiler.__init__(self, enabled: bool = True, pressure_threshold_percent: float = 80.0, pressure_threshold_available_mb: float = 500.0)
```

Initialize the memory profiler.

Args:
    enabled: Whether memory profiling is enabled.
    pressure_threshold_percent: Percent threshold for high memory pressure.
    pressure_threshold_available_mb: Available memory threshold in MB.

#### MemoryProfiler.check_memory_pressure

```python
MemoryProfiler.check_memory_pressure(self) -> bool
```

Check if the system is under memory pressure.

Returns:
    True if memory pressure is high (>80% usage).

#### MemoryProfiler.clear_snapshots

```python
MemoryProfiler.clear_snapshots(self) -> None
```

Clear all stored snapshots.

#### MemoryProfiler.compare

```python
MemoryProfiler.compare(self, start_label: str, end_label: str) -> Dict[str, float]
```

Compare two memory snapshots.

Args:
    start_label: Label of the starting snapshot.
    end_label: Label of the ending snapshot.

Returns:
    Dictionary with memory differences.

#### MemoryProfiler.get_current_memory

```python
MemoryProfiler.get_current_memory(self) -> refactron.core.memory_profiler.MemorySnapshot
```

Get current memory usage snapshot.

Returns:
    MemorySnapshot with current memory usage.

#### MemoryProfiler.get_stats

```python
MemoryProfiler.get_stats(self) -> Dict[str, Any]
```

Get memory profiling statistics.

Returns:
    Dictionary containing statistics.

#### MemoryProfiler.optimize_for_large_files

```python
MemoryProfiler.optimize_for_large_files(self, file_size_mb: float, threshold_mb: Optional[float] = None) -> bool
```

Determine if special optimization is needed for a large file.

Args:
    file_size_mb: File size in megabytes.
    threshold_mb: Optional threshold override. If None, uses default of 5.0 MB.

Returns:
    True if optimization is recommended.

#### MemoryProfiler.profile_function

```python
MemoryProfiler.profile_function(self, func: Callable[..., ~T], *args: Any, label: Optional[str] = None, **kwargs: Any) -> Tuple[~T, Dict[str, Any]]
```

Profile memory usage of a function call.

Args:
    func: Function to profile.
    *args: Positional arguments for the function.
    label: Optional label for logging.
    **kwargs: Keyword arguments for the function.

Returns:
    Tuple of (function result, profiling info).

#### MemoryProfiler.snapshot

```python
MemoryProfiler.snapshot(self, label: str) -> refactron.core.memory_profiler.MemorySnapshot
```

Take a memory snapshot with a label.

Args:
    label: Label for this snapshot.

Returns:
    MemorySnapshot with current memory usage.

#### MemoryProfiler.suggest_gc

```python
MemoryProfiler.suggest_gc(self) -> None
```

Suggest garbage collection if memory pressure is high.

### MemorySnapshot

```python
MemorySnapshot(rss_mb: float, vms_mb: float, percent: float, available_mb: float) -> None
```

Snapshot of memory usage at a point in time.

#### MemorySnapshot.__init__

```python
MemorySnapshot.__init__(self, rss_mb: float, vms_mb: float, percent: float, available_mb: float) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

## Functions

### estimate_file_size_mb

```python
estimate_file_size_mb(file_path: str) -> float
```

Estimate file size in megabytes.

Args:
    file_path: Path to the file.

Returns:
    File size in MB.

### stream_large_file

```python
stream_large_file(file_path: str, chunk_size: int = 8192) -> Any
```

Stream a large file in chunks instead of reading all at once.

Args:
    file_path: Path to the file.
    chunk_size: Size of each chunk in bytes.

Yields:
    Chunks of file content.


---

# refactron.core.metrics

Metrics collection and tracking for Refactron.

This module provides execution metrics tracking including:
- Analysis time per file and total run time
- Refactoring success/failure rates
- Rule hit counts per analyzer/refactorer

## Classes

### FileMetric

```python
FileMetric(file_path: str, analysis_time_ms: float, lines_of_code: int, issues_found: int, analyzers_run: List[str] = <factory>, timestamp: str = <factory>, success: bool = True, error_message: Optional[str] = None) -> None
```

Metrics for a single file analysis.

#### FileMetric.__init__

```python
FileMetric.__init__(self, file_path: str, analysis_time_ms: float, lines_of_code: int, issues_found: int, analyzers_run: List[str] = <factory>, timestamp: str = <factory>, success: bool = True, error_message: Optional[str] = None) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### MetricsCollector

```python
MetricsCollector() -> None
```

Centralized metrics collection for Refactron operations.

#### MetricsCollector.__init__

```python
MetricsCollector.__init__(self) -> None
```

Initialize metrics collector.

#### MetricsCollector.end_analysis

```python
MetricsCollector.end_analysis(self) -> None
```

Mark the end of an analysis session.

#### MetricsCollector.end_refactoring

```python
MetricsCollector.end_refactoring(self) -> None
```

Mark the end of a refactoring session.

#### MetricsCollector.get_analysis_summary

```python
MetricsCollector.get_analysis_summary(self) -> Dict[str, Any]
```

Get summary of analysis metrics.

Returns:
    Dictionary containing analysis summary metrics

#### MetricsCollector.get_combined_summary

```python
MetricsCollector.get_combined_summary(self) -> Dict[str, Any]
```

Get combined summary of all metrics.

Returns:
    Dictionary containing all metrics summaries

#### MetricsCollector.get_refactoring_summary

```python
MetricsCollector.get_refactoring_summary(self) -> Dict[str, Any]
```

Get summary of refactoring metrics.

Returns:
    Dictionary containing refactoring summary metrics

#### MetricsCollector.record_analyzer_hit

```python
MetricsCollector.record_analyzer_hit(self, analyzer_name: str, issue_type: str) -> None
```

Record that an analyzer found an issue.

Args:
    analyzer_name: Name of the analyzer
    issue_type: Type of issue found

#### MetricsCollector.record_file_analysis

```python
MetricsCollector.record_file_analysis(self, file_path: str, analysis_time_ms: float, lines_of_code: int, issues_found: int, analyzers_run: List[str], success: bool = True, error_message: Optional[str] = None) -> None
```

Record metrics for a single file analysis.

Args:
    file_path: Path to the analyzed file
    analysis_time_ms: Time taken to analyze the file in milliseconds
    lines_of_code: Number of lines of code in the file
    issues_found: Number of issues found in the file
    analyzers_run: List of analyzer names that were run
    success: Whether the analysis succeeded
    error_message: Error message if analysis failed

#### MetricsCollector.record_refactoring

```python
MetricsCollector.record_refactoring(self, operation_type: str, file_path: str, execution_time_ms: float, success: bool, risk_level: str = 'safe', error_message: Optional[str] = None) -> None
```

Record metrics for a single refactoring operation.

Args:
    operation_type: Type of refactoring operation
    file_path: Path to the refactored file
    execution_time_ms: Time taken to perform refactoring in milliseconds
    success: Whether the refactoring succeeded
    risk_level: Risk level of the refactoring
    error_message: Error message if refactoring failed

#### MetricsCollector.reset

```python
MetricsCollector.reset(self) -> None
```

Reset all metrics to initial state.

#### MetricsCollector.start_analysis

```python
MetricsCollector.start_analysis(self) -> None
```

Mark the start of an analysis session.

#### MetricsCollector.start_refactoring

```python
MetricsCollector.start_refactoring(self) -> None
```

Mark the start of a refactoring session.

### RefactoringMetric

```python
RefactoringMetric(operation_type: str, file_path: str, execution_time_ms: float, success: bool, risk_level: str, timestamp: str = <factory>, error_message: Optional[str] = None) -> None
```

Metrics for a single refactoring operation.

#### RefactoringMetric.__init__

```python
RefactoringMetric.__init__(self, operation_type: str, file_path: str, execution_time_ms: float, success: bool, risk_level: str, timestamp: str = <factory>, error_message: Optional[str] = None) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

## Functions

### get_metrics_collector

```python
get_metrics_collector() -> refactron.core.metrics.MetricsCollector
```

Get the global metrics collector instance.

Returns:
    Global MetricsCollector instance

### reset_metrics_collector

```python
reset_metrics_collector() -> None
```

Reset the global metrics collector.


---

# refactron.core.models

Data models for Refactron.

## Classes

### CodeIssue

```python
CodeIssue(category: refactron.core.models.IssueCategory, level: refactron.core.models.IssueLevel, message: str, file_path: pathlib._local.Path, line_number: int, column: int = 0, end_line: Optional[int] = None, code_snippet: Optional[str] = None, suggestion: Optional[str] = None, rule_id: Optional[str] = None, confidence: float = 1.0, metadata: Dict[str, Any] = <factory>) -> None
```

Represents a detected code issue.

#### CodeIssue.__init__

```python
CodeIssue.__init__(self, category: refactron.core.models.IssueCategory, level: refactron.core.models.IssueLevel, message: str, file_path: pathlib._local.Path, line_number: int, column: int = 0, end_line: Optional[int] = None, code_snippet: Optional[str] = None, suggestion: Optional[str] = None, rule_id: Optional[str] = None, confidence: float = 1.0, metadata: Dict[str, Any] = <factory>) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### FileMetrics

```python
FileMetrics(file_path: pathlib._local.Path, lines_of_code: int, comment_lines: int, blank_lines: int, complexity: float, maintainability_index: float, functions: int, classes: int, issues: List[refactron.core.models.CodeIssue] = <factory>) -> None
```

Metrics for a single file.

#### FileMetrics.__init__

```python
FileMetrics.__init__(self, file_path: pathlib._local.Path, lines_of_code: int, comment_lines: int, blank_lines: int, complexity: float, maintainability_index: float, functions: int, classes: int, issues: List[refactron.core.models.CodeIssue] = <factory>) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### IssueCategory

```python
IssueCategory(*values)
```

Categories of code issues.

### IssueLevel

```python
IssueLevel(*values)
```

Severity level of code issues.

### RefactoringOperation

```python
RefactoringOperation(operation_type: str, file_path: pathlib._local.Path, line_number: int, description: str, old_code: str, new_code: str, risk_score: float, operation_id: str = <factory>, reasoning: Optional[str] = None, metadata: Dict[str, Any] = <factory>) -> None
```

Represents a refactoring operation to be applied.

#### RefactoringOperation.__init__

```python
RefactoringOperation.__init__(self, operation_type: str, file_path: pathlib._local.Path, line_number: int, description: str, old_code: str, new_code: str, risk_score: float, operation_id: str = <factory>, reasoning: Optional[str] = None, metadata: Dict[str, Any] = <factory>) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

## Functions


---

# refactron.core.parallel

Parallel processing utilities for performance optimization.

## Classes

### ParallelProcessor

```python
ParallelProcessor(max_workers: Optional[int] = None, use_processes: bool = True, enabled: bool = True)
```

Parallel processing manager for analyzing multiple files concurrently.

Supports both multiprocessing and threading based on the task type.

#### ParallelProcessor.__init__

```python
ParallelProcessor.__init__(self, max_workers: Optional[int] = None, use_processes: bool = True, enabled: bool = True)
```

Initialize the parallel processor.

Args:
    max_workers: Maximum number of worker processes/threads.
                If None, uses CPU count capped at 8 workers to avoid resource exhaustion.
    use_processes: If True, uses multiprocessing; if False, uses threading.
    enabled: Whether parallel processing is enabled.

#### ParallelProcessor.get_config

```python
ParallelProcessor.get_config(self) -> Dict[str, Any]
```

Get the current configuration.

Returns:
    Dictionary containing configuration details.

#### ParallelProcessor.process_files

```python
ParallelProcessor.process_files(self, files: List[pathlib._local.Path], process_func: Callable[[pathlib._local.Path], Tuple[Optional[refactron.core.models.FileMetrics], Optional[refactron.core.analysis_result.FileAnalysisError]]], progress_callback: Optional[Callable[[int, int], NoneType]] = None) -> Tuple[List[refactron.core.models.FileMetrics], List[refactron.core.analysis_result.FileAnalysisError]]
```

Process multiple files in parallel.

Args:
    files: List of file paths to process.
    process_func: Function to process a single file. Should return
                 (FileMetrics, None) on success or (None, FileAnalysisError) on error.
    progress_callback: Optional callback for progress updates (completed, total).

Returns:
    Tuple of (successful results, failed files).

## Functions


---

# refactron.core.prometheus_metrics

Prometheus metrics exporter for Refactron.

This module provides Prometheus-compatible metrics endpoint for monitoring
Refactron's performance and usage in production environments.

## Classes

### MetricsHTTPHandler

```python
MetricsHTTPHandler(request, client_address, server)
```

HTTP handler for Prometheus metrics endpoint.

#### MetricsHTTPHandler.__init__

```python
MetricsHTTPHandler.__init__(self, request, client_address, server)
```

Initialize self.  See help(type(self)) for accurate signature.

#### MetricsHTTPHandler.address_string

```python
MetricsHTTPHandler.address_string(self)
```

Return the client address.

#### MetricsHTTPHandler.date_time_string

```python
MetricsHTTPHandler.date_time_string(self, timestamp=None)
```

Return the current date and time formatted for a message header.

#### MetricsHTTPHandler.do_GET

```python
MetricsHTTPHandler.do_GET(self) -> None
```

Handle GET requests to /metrics endpoint.

#### MetricsHTTPHandler.end_headers

```python
MetricsHTTPHandler.end_headers(self)
```

Send the blank line ending the MIME headers.

#### MetricsHTTPHandler.finish

```python
MetricsHTTPHandler.finish(self)
```

No documentation available.

#### MetricsHTTPHandler.flush_headers

```python
MetricsHTTPHandler.flush_headers(self)
```

No documentation available.

#### MetricsHTTPHandler.handle

```python
MetricsHTTPHandler.handle(self)
```

Handle multiple requests if necessary.

#### MetricsHTTPHandler.handle_expect_100

```python
MetricsHTTPHandler.handle_expect_100(self)
```

Decide what to do with an "Expect: 100-continue" header.

If the client is expecting a 100 Continue response, we must
respond with either a 100 Continue or a final response before
waiting for the request body. The default is to always respond
with a 100 Continue. You can behave differently (for example,
reject unauthorized requests) by overriding this method.

This method should either return True (possibly after sending
a 100 Continue response) or send an error response and return
False.

#### MetricsHTTPHandler.handle_one_request

```python
MetricsHTTPHandler.handle_one_request(self)
```

Handle a single HTTP request.

You normally don't need to override this method; see the class
__doc__ string for information on how to handle specific HTTP
commands such as GET and POST.

#### MetricsHTTPHandler.log_date_time_string

```python
MetricsHTTPHandler.log_date_time_string(self)
```

Return the current time formatted for logging.

#### MetricsHTTPHandler.log_error

```python
MetricsHTTPHandler.log_error(self, format, *args)
```

Log an error.

This is called when a request cannot be fulfilled.  By
default it passes the message on to log_message().

Arguments are the same as for log_message().

XXX This should go to the separate error log.

#### MetricsHTTPHandler.log_message

```python
MetricsHTTPHandler.log_message(self, format: str, *args: Any) -> None
```

Suppress default logging.

#### MetricsHTTPHandler.log_request

```python
MetricsHTTPHandler.log_request(self, code='-', size='-')
```

Log an accepted request.

This is called by send_response().

#### MetricsHTTPHandler.parse_request

```python
MetricsHTTPHandler.parse_request(self)
```

Parse a request (internal).

The request should be stored in self.raw_requestline; the results
are in self.command, self.path, self.request_version and
self.headers.

Return True for success, False for failure; on failure, any relevant
error response has already been sent back.

#### MetricsHTTPHandler.send_error

```python
MetricsHTTPHandler.send_error(self, code, message=None, explain=None)
```

Send and log an error reply.

Arguments are
* code:    an HTTP error code
           3 digits
* message: a simple optional 1 line reason phrase.
           *( HTAB / SP / VCHAR / %x80-FF )
           defaults to short entry matching the response code
* explain: a detailed message defaults to the long entry
           matching the response code.

This sends an error response (so it must be called before any
output has been generated), logs the error, and finally sends
a piece of HTML explaining the error to the user.

#### MetricsHTTPHandler.send_header

```python
MetricsHTTPHandler.send_header(self, keyword, value)
```

Send a MIME header to the headers buffer.

#### MetricsHTTPHandler.send_response

```python
MetricsHTTPHandler.send_response(self, code, message=None)
```

Add the response header to the headers buffer and log the
response code.

Also send two standard headers with the server software
version and the current date.

#### MetricsHTTPHandler.send_response_only

```python
MetricsHTTPHandler.send_response_only(self, code, message=None)
```

Send the response header only.

#### MetricsHTTPHandler.setup

```python
MetricsHTTPHandler.setup(self)
```

No documentation available.

#### MetricsHTTPHandler.version_string

```python
MetricsHTTPHandler.version_string(self)
```

Return the server software version string.

### PrometheusMetrics

```python
PrometheusMetrics() -> None
```

Prometheus metrics formatter and exporter.

#### PrometheusMetrics.__init__

```python
PrometheusMetrics.__init__(self) -> None
```

Initialize Prometheus metrics.

#### PrometheusMetrics.format_metrics

```python
PrometheusMetrics.format_metrics(self) -> str
```

Format metrics in Prometheus exposition format.

Returns:
    String containing Prometheus-formatted metrics

### PrometheusMetricsServer

```python
PrometheusMetricsServer(host: str = '127.0.0.1', port: int = 9090) -> None
```

HTTP server for exposing Prometheus metrics.

#### PrometheusMetricsServer.__init__

```python
PrometheusMetricsServer.__init__(self, host: str = '127.0.0.1', port: int = 9090) -> None
```

Initialize Prometheus metrics server.

Args:
    host: Host to bind to (default: 127.0.0.1 for localhost-only access)
    port: Port to listen on

#### PrometheusMetricsServer.is_running

```python
PrometheusMetricsServer.is_running(self) -> bool
```

Check if the metrics server is running.

Returns:
    True if server is running, False otherwise

#### PrometheusMetricsServer.start

```python
PrometheusMetricsServer.start(self) -> None
```

Start the metrics server in a background thread.

#### PrometheusMetricsServer.stop

```python
PrometheusMetricsServer.stop(self) -> None
```

Stop the metrics server.

## Functions

### get_metrics_server

```python
get_metrics_server() -> Optional[refactron.core.prometheus_metrics.PrometheusMetricsServer]
```

Get the global metrics server instance.

Returns:
    PrometheusMetricsServer instance or None if not started

### start_metrics_server

```python
start_metrics_server(host: str = '127.0.0.1', port: int = 9090) -> refactron.core.prometheus_metrics.PrometheusMetricsServer
```

Start the global Prometheus metrics server.

Args:
    host: Host to bind to (default: 127.0.0.1 for localhost-only access)
    port: Port to listen on

Returns:
    PrometheusMetricsServer instance

### stop_metrics_server

```python
stop_metrics_server() -> None
```

Stop the global Prometheus metrics server.


---

# refactron.core.refactor_result

Refactoring result representation.

## Classes

### RefactorResult

```python
RefactorResult(operations: List[refactron.core.models.RefactoringOperation] = <factory>, applied: bool = False, preview_mode: bool = True) -> None
```

Result of refactoring operations.

#### RefactorResult.__init__

```python
RefactorResult.__init__(self, operations: List[refactron.core.models.RefactoringOperation] = <factory>, applied: bool = False, preview_mode: bool = True) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### RefactorResult.apply

```python
RefactorResult.apply(self) -> bool
```

Apply the refactoring operations to the files.

#### RefactorResult.get_ranking_score

```python
RefactorResult.get_ranking_score(self, operation: refactron.core.models.RefactoringOperation) -> float
```

Get ranking score for an operation (0.0 if not ranked).

#### RefactorResult.operations_by_file

```python
RefactorResult.operations_by_file(self, file_path: pathlib._local.Path) -> List[refactron.core.models.RefactoringOperation]
```

Get operations for a specific file.

#### RefactorResult.operations_by_type

```python
RefactorResult.operations_by_type(self, operation_type: str) -> List[refactron.core.models.RefactoringOperation]
```

Get operations of a specific type.

#### RefactorResult.show_diff

```python
RefactorResult.show_diff(self) -> str
```

Show a diff of all operations.

#### RefactorResult.summary

```python
RefactorResult.summary(self) -> Dict[str, int]
```

Get a summary of refactoring operations.

#### RefactorResult.top_ranked_operations

```python
RefactorResult.top_ranked_operations(self, top_n: int = 10) -> List[refactron.core.models.RefactoringOperation]
```

Get top N ranked operations by ranking score.

## Functions


---

# refactron.core.refactron

Main Refactron class - the entry point for all operations.

## Classes

### Refactron

```python
Refactron(config: Optional[refactron.core.config.RefactronConfig] = None)
```

Main Refactron class for code analysis and refactoring.

Example:
    >>> refactron = Refactron()
    >>> result = refactron.analyze("mycode.py")
    >>> print(result.report())

#### Refactron.__init__

```python
Refactron.__init__(self, config: Optional[refactron.core.config.RefactronConfig] = None)
```

Initialize Refactron.

Args:
    config: Configuration object. If None, uses default config.

#### Refactron.analyze

```python
Refactron.analyze(self, target: Union[str, pathlib._local.Path]) -> refactron.core.analysis_result.AnalysisResult
```

Analyze a file or directory.

Args:
    target: Path to file or directory to analyze

Returns:
    AnalysisResult containing all detected issues and any errors encountered

Note:
    This method implements graceful degradation - if individual files fail
    to analyze, they are logged and skipped, allowing analysis to continue
    on remaining files.

#### Refactron.clear_caches

```python
Refactron.clear_caches(self) -> None
```

Clear all performance-related caches.

#### Refactron.detect_project_root

```python
Refactron.detect_project_root(self, file_path: pathlib._local.Path) -> pathlib._local.Path
```

Detect project root by looking for common markers in parent directories.

The search walks up the directory tree from the file's parent directory,
checking for common project markers up to a fixed maximum depth.

Args:
    file_path: Path to a file in the project.

Returns:
    The path to the project root if any of the known markers are found
    within the search depth limit, or the file's parent directory if no
    markers are detected.

#### Refactron.get_performance_stats

```python
Refactron.get_performance_stats(self) -> dict
```

Get performance statistics from all optimization components.

Returns:
    Dictionary containing performance statistics.

#### Refactron.get_python_files

```python
Refactron.get_python_files(self, directory: pathlib._local.Path) -> List[pathlib._local.Path]
```

Get all Python files in a directory, respecting exclude patterns.

#### Refactron.record_feedback

```python
Refactron.record_feedback(self, operation_id: str, action: str, reason: Optional[str] = None, operation: Optional[refactron.core.models.RefactoringOperation] = None) -> None
```

Record developer feedback on a refactoring suggestion.

Args:
    operation_id: Unique identifier for the refactoring operation
    action: Feedback action - "accepted", "rejected", or "ignored"
    reason: Optional reason for the feedback
    operation: Optional RefactoringOperation object (used to extract metadata)

Note:
    If pattern storage is not initialized, this method will silently fail.

#### Refactron.refactor

```python
Refactron.refactor(self, target: Union[str, pathlib._local.Path], preview: bool = True, operation_types: Optional[List[str]] = None) -> refactron.core.refactor_result.RefactorResult
```

Refactor a file or directory.

Args:
    target: Path to file or directory to refactor
    preview: If True, show changes without applying them
    operation_types: Specific refactoring operations to apply (None = all)

Returns:
    RefactorResult containing all proposed operations

Note:
    This method implements graceful degradation - if individual files fail
    to refactor, they are logged and skipped, allowing refactoring to continue
    on remaining files.

## Functions


---

# refactron.core.repositories

GitHub repository integration for Refactron CLI.

This module provides functionality to interact with the Refactron backend API
to fetch GitHub repositories connected to the user's account.

## Classes

### Repository

```python
Repository(id: 'int', name: 'str', full_name: 'str', description: 'Optional[str]', private: 'bool', html_url: 'str', clone_url: 'str', ssh_url: 'str', default_branch: 'str', language: 'Optional[str]', updated_at: 'str') -> None
```

Represents a GitHub repository.

#### Repository.__init__

```python
Repository.__init__(self, id: 'int', name: 'str', full_name: 'str', description: 'Optional[str]', private: 'bool', html_url: 'str', clone_url: 'str', ssh_url: 'str', default_branch: 'str', language: 'Optional[str]', updated_at: 'str') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

## Functions

### list_repositories

```python
list_repositories(api_base_url: 'str', timeout_seconds: 'int' = 10) -> 'List[Repository]'
```

Fetch all GitHub repositories connected to the user's account.

Args:
    api_base_url: The Refactron API base URL
    timeout_seconds: Request timeout in seconds

Returns:
    List of Repository objects

Raises:
    RuntimeError: If the request fails or user is not authenticated


---

# refactron.core.telemetry

Opt-in telemetry system for Refactron.

This module provides anonymous usage data collection to understand real-world
usage patterns and performance characteristics. All telemetry is opt-in and
respects user privacy.

## Classes

### TelemetryCollector

```python
TelemetryCollector(enabled: bool = False, anonymous_id: Optional[str] = None, telemetry_file: Optional[pathlib._local.Path] = None)
```

Collects and manages telemetry data with privacy considerations.

#### TelemetryCollector.__init__

```python
TelemetryCollector.__init__(self, enabled: bool = False, anonymous_id: Optional[str] = None, telemetry_file: Optional[pathlib._local.Path] = None)
```

Initialize telemetry collector.

Args:
    enabled: Whether telemetry collection is enabled
    anonymous_id: Anonymous identifier for this installation
    telemetry_file: Path to file where telemetry data is stored

#### TelemetryCollector.flush

```python
TelemetryCollector.flush(self) -> None
```

Write collected events to disk.

#### TelemetryCollector.get_summary

```python
TelemetryCollector.get_summary(self) -> Dict[str, Any]
```

Get a summary of collected telemetry events.

Returns:
    Dictionary containing telemetry summary

#### TelemetryCollector.record_analysis_completed

```python
TelemetryCollector.record_analysis_completed(self, files_analyzed: int, total_time_ms: float, issues_found: int, analyzers_used: List[str]) -> None
```

Record an analysis completion event.

Args:
    files_analyzed: Number of files analyzed
    total_time_ms: Total analysis time in milliseconds
    issues_found: Number of issues found
    analyzers_used: List of analyzers that were used

#### TelemetryCollector.record_error

```python
TelemetryCollector.record_error(self, error_type: str, error_category: str, context: Optional[str] = None) -> None
```

Record an error event.

Args:
    error_type: Type of error (generic, no specific error messages)
    error_category: Category of error (e.g., 'analysis', 'refactoring')
    context: Optional context information (should not contain PII)

#### TelemetryCollector.record_event

```python
TelemetryCollector.record_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None
```

Record a telemetry event.

Args:
    event_type: Type of event (e.g., 'analysis_completed', 'refactoring_applied')
    data: Additional event data (should not contain PII)

#### TelemetryCollector.record_feature_usage

```python
TelemetryCollector.record_feature_usage(self, feature_name: str, metadata: Optional[Dict[str, Any]] = None) -> None
```

Record a feature usage event.

Args:
    feature_name: Name of the feature used
    metadata: Optional metadata about feature usage

#### TelemetryCollector.record_refactoring_applied

```python
TelemetryCollector.record_refactoring_applied(self, operation_type: str, files_affected: int, total_time_ms: float, success: bool) -> None
```

Record a refactoring operation event.

Args:
    operation_type: Type of refactoring operation
    files_affected: Number of files affected
    total_time_ms: Total refactoring time in milliseconds
    success: Whether the refactoring succeeded

### TelemetryConfig

```python
TelemetryConfig(config_file: Optional[pathlib._local.Path] = None)
```

Configuration for telemetry system.

#### TelemetryConfig.__init__

```python
TelemetryConfig.__init__(self, config_file: Optional[pathlib._local.Path] = None)
```

Initialize telemetry configuration.

Args:
    config_file: Path to telemetry configuration file

#### TelemetryConfig.disable

```python
TelemetryConfig.disable(self) -> None
```

Disable telemetry collection.

#### TelemetryConfig.enable

```python
TelemetryConfig.enable(self, anonymous_id: Optional[str] = None) -> None
```

Enable telemetry collection.

Args:
    anonymous_id: Optional anonymous identifier (generated if not provided)

#### TelemetryConfig.save_config

```python
TelemetryConfig.save_config(self) -> None
```

Save telemetry configuration to file.

### TelemetryEvent

```python
TelemetryEvent(event_type: str, timestamp: str = <factory>, session_id: str = <factory>, data: Dict[str, Any] = <factory>) -> None
```

Represents a single telemetry event.

#### TelemetryEvent.__init__

```python
TelemetryEvent.__init__(self, event_type: str, timestamp: str = <factory>, session_id: str = <factory>, data: Dict[str, Any] = <factory>) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

## Functions

### disable_telemetry

```python
disable_telemetry() -> None
```

Disable telemetry collection globally.

### enable_telemetry

```python
enable_telemetry() -> None
```

Enable telemetry collection globally.

### get_telemetry_collector

```python
get_telemetry_collector(enabled: Optional[bool] = None) -> refactron.core.telemetry.TelemetryCollector
```

Get the global telemetry collector instance.

Args:
    enabled: Override enabled status (uses config if None)

Returns:
    Global TelemetryCollector instance


---

# refactron.core.workspace

Workspace management for Refactron CLI.

This module handles the mapping between remote GitHub repositories and local
directory paths, enabling seamless navigation and context switching.

## Classes

### WorkspaceManager

```python
WorkspaceManager(config_path: 'Optional[Path]' = None) -> 'None'
```

Manages workspace mappings between repositories and local paths.

#### WorkspaceManager.__init__

```python
WorkspaceManager.__init__(self, config_path: 'Optional[Path]' = None) -> 'None'
```

Initialize the workspace manager.

Args:
    config_path: Path to the workspaces.json file (default: ~/.refactron/workspaces.json)

#### WorkspaceManager.add_workspace

```python
WorkspaceManager.add_workspace(self, mapping: 'WorkspaceMapping') -> 'None'
```

Add or update a workspace mapping.

Args:
    mapping: The workspace mapping to add

#### WorkspaceManager.detect_repository

```python
WorkspaceManager.detect_repository(self, directory: 'Optional[Path]' = None) -> 'Optional[str]'
```

Attempt to detect the GitHub repository from the .git config.

Args:
    directory: Directory to search (default: current directory)

Returns:
    The repository full name (e.g., "user/repo"), or None if not detected

#### WorkspaceManager.get_workspace

```python
WorkspaceManager.get_workspace(self, repo_name: 'str') -> 'Optional[WorkspaceMapping]'
```

Get a workspace mapping by repository name.

Args:
    repo_name: The repository name (e.g., "repo" or "user/repo")

Returns:
    The workspace mapping, or None if not found

#### WorkspaceManager.get_workspace_by_path

```python
WorkspaceManager.get_workspace_by_path(self, local_path: 'str') -> 'Optional[WorkspaceMapping]'
```

Get a workspace mapping by local path.

Args:
    local_path: The local directory path

Returns:
    The workspace mapping, or None if not found

#### WorkspaceManager.list_workspaces

```python
WorkspaceManager.list_workspaces(self) -> 'list[WorkspaceMapping]'
```

List all workspace mappings.

Returns:
    List of all workspace mappings

#### WorkspaceManager.remove_workspace

```python
WorkspaceManager.remove_workspace(self, repo_full_name: 'str') -> 'bool'
```

Remove a workspace mapping.

Args:
    repo_full_name: The full name of the repository

Returns:
    True if removed, False if not found

### WorkspaceMapping

```python
WorkspaceMapping(repo_id: 'int', repo_name: 'str', repo_full_name: 'str', local_path: 'str', connected_at: 'str') -> None
```

Represents a mapping between a remote repository and a local path.

#### WorkspaceMapping.__init__

```python
WorkspaceMapping.__init__(self, repo_id: 'int', repo_name: 'str', repo_full_name: 'str', local_path: 'str', connected_at: 'str') -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### WorkspaceMapping.to_dict

```python
WorkspaceMapping.to_dict(self) -> 'Dict[str, Any]'
```

Convert to dictionary for JSON serialization.

## Functions

