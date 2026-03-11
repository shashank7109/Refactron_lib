# refactron.patterns

Pattern Learning System for Refactron.

## Classes

## Functions


---

# refactron.patterns.fingerprint

Pattern fingerprinting for code pattern identification.

## Classes

### PatternFingerprinter

```python
PatternFingerprinter() -> None
```

Generates fingerprints for code patterns using AST-based hashing.

#### PatternFingerprinter.__init__

```python
PatternFingerprinter.__init__(self) -> None
```

Initialize the pattern fingerprinter.

#### PatternFingerprinter.fingerprint_code

```python
PatternFingerprinter.fingerprint_code(self, code_snippet: str) -> str
```

Generate hash fingerprint for a code snippet.

Args:
    code_snippet: Source code to fingerprint

Returns:
    SHA256 hash of the normalized code pattern

#### PatternFingerprinter.fingerprint_issue_context

```python
PatternFingerprinter.fingerprint_issue_context(self, issue: refactron.core.models.CodeIssue, source_code: str, context_lines: int = 3) -> str
```

Generate fingerprint for issue context.

Args:
    issue: CodeIssue to fingerprint
    source_code: Full source code of the file
    context_lines: Number of lines before/after to include (default: 3)

Returns:
    SHA256 hash of the normalized issue context pattern

#### PatternFingerprinter.fingerprint_refactoring

```python
PatternFingerprinter.fingerprint_refactoring(self, operation: refactron.core.models.RefactoringOperation) -> str
```

Generate fingerprint for refactoring operation.

Args:
    operation: RefactoringOperation to fingerprint

Returns:
    SHA256 hash of the normalized refactoring pattern

## Functions


---

# refactron.patterns.learner

Pattern learning engine that learns from feedback and refactoring history.

## Classes

### PatternLearner

```python
PatternLearner(storage: refactron.patterns.storage.PatternStorage, fingerprinter: refactron.patterns.fingerprint.PatternFingerprinter) -> None
```

Learns patterns from feedback and refactoring history.

#### PatternLearner.__init__

```python
PatternLearner.__init__(self, storage: refactron.patterns.storage.PatternStorage, fingerprinter: refactron.patterns.fingerprint.PatternFingerprinter) -> None
```

Initialize pattern learner.

Args:
    storage: PatternStorage instance for loading/saving patterns
    fingerprinter: PatternFingerprinter for generating code fingerprints

Raises:
    ValueError: If storage or fingerprinter is None

#### PatternLearner.batch_learn

```python
PatternLearner.batch_learn(self, feedback_list: List[Tuple[refactron.core.models.RefactoringOperation, refactron.patterns.models.RefactoringFeedback]]) -> Dict[str, int]
```

Process multiple feedback records efficiently.

Args:
    feedback_list: List of (operation, feedback) tuples to process

Returns:
    Dictionary with statistics: \{'processed': int, 'created': int,
    'updated': int, 'failed': int\}

Raises:
    ValueError: If feedback_list is None or contains None values

#### PatternLearner.learn_from_feedback

```python
PatternLearner.learn_from_feedback(self, operation: refactron.core.models.RefactoringOperation, feedback: refactron.patterns.models.RefactoringFeedback) -> Optional[str]
```

Learn from a single feedback record.

Args:
    operation: RefactoringOperation that was evaluated
    feedback: Feedback record containing developer decision

Returns:
    Pattern ID if pattern was created/updated, None if skipped

Raises:
    ValueError: If operation or feedback is None
    RuntimeError: If pattern storage operations fail

#### PatternLearner.update_pattern_metrics

```python
PatternLearner.update_pattern_metrics(self, pattern_id: str, before_metrics: refactron.core.models.FileMetrics, after_metrics: refactron.core.models.FileMetrics) -> None
```

Update metrics for a pattern based on before/after comparison.

Args:
    pattern_id: ID of the pattern to update
    before_metrics: FileMetrics before refactoring
    after_metrics: FileMetrics after refactoring

Raises:
    ValueError: If pattern_id is empty or metrics are None
    RuntimeError: If pattern not found or update fails

## Functions


---

# refactron.patterns.learning_service

Background service for pattern learning and maintenance.

## Classes

### LearningService

```python
LearningService(storage: refactron.patterns.storage.PatternStorage, learner: Optional[refactron.patterns.learner.PatternLearner] = None) -> None
```

Background service for pattern learning and maintenance.

#### LearningService.__init__

```python
LearningService.__init__(self, storage: refactron.patterns.storage.PatternStorage, learner: Optional[refactron.patterns.learner.PatternLearner] = None) -> None
```

Initialize learning service.

Args:
    storage: PatternStorage instance for data access
    learner: PatternLearner instance (created if None)

Raises:
    ValueError: If storage is None

#### LearningService.cleanup_old_patterns

```python
LearningService.cleanup_old_patterns(self, days: int = 90) -> Dict[str, int]
```

Remove patterns that haven't been seen recently.

Args:
    days: Number of days of inactivity before removal (default: 90)

Returns:
    Dictionary with cleanup statistics: \{'removed': int, 'total': int\}

Raises:
    ValueError: If days is negative
    RuntimeError: If cleanup fails

#### LearningService.process_pending_feedback

```python
LearningService.process_pending_feedback(self, limit: Optional[int] = None) -> Dict[str, int]
```

Process any pending feedback records that haven't been learned from yet.

Args:
    limit: Maximum number of feedback records to process (None = all)

Returns:
    Dictionary with processing statistics

Raises:
    RuntimeError: If processing fails critically

#### LearningService.update_pattern_scores

```python
LearningService.update_pattern_scores(self) -> Dict[str, int]
```

Recalculate scores for all patterns.

This updates acceptance rates and benefit scores based on current feedback.

Returns:
    Dictionary with update statistics: \{'updated': int, 'total': int\}

Raises:
    RuntimeError: If update fails

## Functions


---

# refactron.patterns.matcher

Pattern matching for finding similar code patterns.

## Classes

### PatternMatcher

```python
PatternMatcher(storage: refactron.patterns.storage.PatternStorage, cache_ttl_seconds: int = 300)
```

Matches code patterns against learned patterns with scoring.

#### PatternMatcher.__init__

```python
PatternMatcher.__init__(self, storage: refactron.patterns.storage.PatternStorage, cache_ttl_seconds: int = 300)
```

Initialize pattern matcher.

Args:
    storage: PatternStorage instance for loading patterns
    cache_ttl_seconds: Cache time-to-live in seconds (default: 300 seconds / 5 minutes)

#### PatternMatcher.calculate_pattern_score

```python
PatternMatcher.calculate_pattern_score(self, pattern: refactron.patterns.models.RefactoringPattern, project_profile: Optional[refactron.patterns.models.ProjectPatternProfile] = None) -> float
```

Calculate score for pattern suggestion.

The scoring algorithm applies multiple bonuses multiplicatively:
- Project weight: 0.0-1.0 (disabled patterns get 0.0)
- Enabled pattern bonus: 1.2x (20% bonus)
- Recency bonus: up to 1.2x (20% bonus for patterns seen in last 30 days)
- Frequency bonus: up to 1.3x (30% bonus based on log scale of occurrences)
- Benefit bonus: up to 1.15x (15% bonus based on average_benefit_score)

These bonuses can compound to exceed 1.0, but the final score is normalized
to the range [0.0, 1.0] using min/max clipping.

Args:
    pattern: Pattern to score
    project_profile: Optional project-specific profile for weighting

Returns:
    Score between 0.0 and 1.0 (higher = better suggestion)

#### PatternMatcher.clear_cache

```python
PatternMatcher.clear_cache(self) -> None
```

Clear the pattern cache (force reload on next access).

#### PatternMatcher.find_best_matches

```python
PatternMatcher.find_best_matches(self, code_hash: str, operation_type: Optional[str] = None, project_profile: Optional[refactron.patterns.models.ProjectPatternProfile] = None, limit: int = 10) -> List[Tuple[refactron.patterns.models.RefactoringPattern, float]]
```

Find best matching patterns with scores.

Args:
    code_hash: Hash of the code pattern to match
    operation_type: Optional operation type to filter by
    project_profile: Optional project-specific profile for weighting
    limit: Maximum number of results to return

Returns:
    List of tuples (pattern, score) sorted by score (highest first)

#### PatternMatcher.find_similar_patterns

```python
PatternMatcher.find_similar_patterns(self, code_hash: str, operation_type: Optional[str] = None, limit: Optional[int] = None) -> List[refactron.patterns.models.RefactoringPattern]
```

Find patterns similar to given code hash.

Optimized with O(1) hash-based lookup instead of O(n) linear search.

Args:
    code_hash: Hash of the code pattern to match
    operation_type: Optional operation type to filter by
    limit: Optional maximum number of results to return

Returns:
    List of similar patterns, sorted by acceptance rate

## Functions


---

# refactron.patterns.models

Data models for Pattern Learning System.

## Classes

### PatternMetric

```python
PatternMetric(pattern_id: str, complexity_reduction: float = 0.0, maintainability_improvement: float = 0.0, lines_of_code_change: int = 0, issue_resolution_count: int = 0, before_metrics: Dict[str, float] = <factory>, after_metrics: Dict[str, float] = <factory>, total_evaluations: int = 0) -> None
```

Metrics for evaluating pattern effectiveness.

#### PatternMetric.__init__

```python
PatternMetric.__init__(self, pattern_id: str, complexity_reduction: float = 0.0, maintainability_improvement: float = 0.0, lines_of_code_change: int = 0, issue_resolution_count: int = 0, before_metrics: Dict[str, float] = <factory>, after_metrics: Dict[str, float] = <factory>, total_evaluations: int = 0) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### PatternMetric.to_dict

```python
PatternMetric.to_dict(self) -> Dict[str, Any]
```

Convert metrics to dictionary for serialization.

#### PatternMetric.update

```python
PatternMetric.update(self, complexity_reduction: float, maintainability_improvement: float, lines_of_code_change: int, issue_resolution_count: int, before_metrics: Dict[str, float], after_metrics: Dict[str, float]) -> 'PatternMetric'
```

Update metrics with new evaluation data (in-place mutation).

Returns:
    self to enable method chaining

Note:
    This method modifies the object in-place. The return value
    is provided to enable method chaining.

### ProjectPatternProfile

```python
ProjectPatternProfile(project_id: str, project_path: pathlib._local.Path, enabled_patterns: Set[str] = <factory>, disabled_patterns: Set[str] = <factory>, pattern_weights: Dict[str, float] = <factory>, rule_thresholds: Dict[str, float] = <factory>, last_updated: datetime.datetime = <factory>, metadata: Dict[str, Any] = <factory>) -> None
```

Project-specific pattern tuning and rules.

#### ProjectPatternProfile.__init__

```python
ProjectPatternProfile.__init__(self, project_id: str, project_path: pathlib._local.Path, enabled_patterns: Set[str] = <factory>, disabled_patterns: Set[str] = <factory>, pattern_weights: Dict[str, float] = <factory>, rule_thresholds: Dict[str, float] = <factory>, last_updated: datetime.datetime = <factory>, metadata: Dict[str, Any] = <factory>) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### ProjectPatternProfile.disable_pattern

```python
ProjectPatternProfile.disable_pattern(self, pattern_id: str) -> None
```

Disable a pattern for this project.

#### ProjectPatternProfile.enable_pattern

```python
ProjectPatternProfile.enable_pattern(self, pattern_id: str) -> None
```

Enable a pattern for this project.

#### ProjectPatternProfile.get_pattern_weight

```python
ProjectPatternProfile.get_pattern_weight(self, pattern_id: str, default: float = 1.0) -> float
```

Get weight for a pattern, returning default if not set.

#### ProjectPatternProfile.is_pattern_enabled

```python
ProjectPatternProfile.is_pattern_enabled(self, pattern_id: str) -> bool
```

Check if a pattern is enabled for this project.

#### ProjectPatternProfile.set_pattern_weight

```python
ProjectPatternProfile.set_pattern_weight(self, pattern_id: str, weight: float) -> None
```

Set custom weight for a pattern.

#### ProjectPatternProfile.set_rule_threshold

```python
ProjectPatternProfile.set_rule_threshold(self, rule_id: str, threshold: float) -> None
```

Set custom threshold for a rule.

#### ProjectPatternProfile.to_dict

```python
ProjectPatternProfile.to_dict(self) -> Dict[str, Any]
```

Convert profile to dictionary for serialization.

### RefactoringFeedback

```python
RefactoringFeedback(operation_id: str, operation_type: str, file_path: pathlib._local.Path, timestamp: datetime.datetime, action: str, reason: Optional[str] = None, code_pattern_hash: Optional[str] = None, project_path: Optional[pathlib._local.Path] = None, metadata: Dict[str, Any] = <factory>) -> None
```

Tracks developer acceptance/rejection of refactoring suggestions.

#### RefactoringFeedback.__init__

```python
RefactoringFeedback.__init__(self, operation_id: str, operation_type: str, file_path: pathlib._local.Path, timestamp: datetime.datetime, action: str, reason: Optional[str] = None, code_pattern_hash: Optional[str] = None, project_path: Optional[pathlib._local.Path] = None, metadata: Dict[str, Any] = <factory>) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### RefactoringFeedback.to_dict

```python
RefactoringFeedback.to_dict(self) -> Dict[str, Any]
```

Convert feedback to dictionary for serialization.

### RefactoringPattern

```python
RefactoringPattern(pattern_id: str, pattern_hash: str, operation_type: str, code_snippet_before: str, code_snippet_after: str, acceptance_rate: float = 0.0, total_occurrences: int = 0, accepted_count: int = 0, rejected_count: int = 0, ignored_count: int = 0, average_benefit_score: float = 0.0, first_seen: datetime.datetime = <factory>, last_seen: datetime.datetime = <factory>, project_context: Dict[str, Any] = <factory>, metadata: Dict[str, Any] = <factory>) -> None
```

Represents a learned pattern from successful refactorings.

#### RefactoringPattern.__init__

```python
RefactoringPattern.__init__(self, pattern_id: str, pattern_hash: str, operation_type: str, code_snippet_before: str, code_snippet_after: str, acceptance_rate: float = 0.0, total_occurrences: int = 0, accepted_count: int = 0, rejected_count: int = 0, ignored_count: int = 0, average_benefit_score: float = 0.0, first_seen: datetime.datetime = <factory>, last_seen: datetime.datetime = <factory>, project_context: Dict[str, Any] = <factory>, metadata: Dict[str, Any] = <factory>) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

#### RefactoringPattern.calculate_benefit_score

```python
RefactoringPattern.calculate_benefit_score(self, metric: Optional[refactron.patterns.models.PatternMetric] = None) -> float
```

Calculate overall benefit score for this pattern.

#### RefactoringPattern.to_dict

```python
RefactoringPattern.to_dict(self) -> Dict[str, Any]
```

Convert pattern to dictionary for serialization.

#### RefactoringPattern.update_from_feedback

```python
RefactoringPattern.update_from_feedback(self, action: str) -> None
```

Update pattern statistics from feedback.

## Functions


---

# refactron.patterns.ranker

Ranking engine for refactoring suggestions based on learned patterns.

## Classes

### RefactoringRanker

```python
RefactoringRanker(storage: refactron.patterns.storage.PatternStorage, matcher: refactron.patterns.matcher.PatternMatcher, fingerprinter: refactron.patterns.fingerprint.PatternFingerprinter)
```

Ranks refactoring suggestions based on learned patterns and project context.

#### RefactoringRanker.__init__

```python
RefactoringRanker.__init__(self, storage: refactron.patterns.storage.PatternStorage, matcher: refactron.patterns.matcher.PatternMatcher, fingerprinter: refactron.patterns.fingerprint.PatternFingerprinter)
```

Initialize refactoring ranker.

Args:
    storage: PatternStorage instance for accessing patterns and profiles
    matcher: PatternMatcher instance for finding similar patterns
    fingerprinter: PatternFingerprinter instance for generating code hashes

#### RefactoringRanker.get_ranked_with_scores

```python
RefactoringRanker.get_ranked_with_scores(self, operations: List[refactron.core.models.RefactoringOperation], project_path: Optional[pathlib._local.Path] = None) -> List[Tuple[refactron.core.models.RefactoringOperation, float, Optional[refactron.patterns.models.RefactoringPattern]]]
```

Get ranked operations with scores and matching patterns.

Args:
    operations: List of refactoring operations to rank
    project_path: Optional project path for project-specific scoring

Returns:
    List of tuples (operation, score, best_matching_pattern)
    sorted by score descending

#### RefactoringRanker.get_top_suggestions

```python
RefactoringRanker.get_top_suggestions(self, operations: List[refactron.core.models.RefactoringOperation], project_path: Optional[pathlib._local.Path] = None, top_n: int = 10) -> List[refactron.core.models.RefactoringOperation]
```

Get top N ranked suggestions.

Args:
    operations: List of refactoring operations to rank
    project_path: Optional project path for project-specific scoring
    top_n: Number of top suggestions to return (default: 10)

Returns:
    List of top N RefactoringOperation instances, sorted by score descending

#### RefactoringRanker.rank_operations

```python
RefactoringRanker.rank_operations(self, operations: List[refactron.core.models.RefactoringOperation], project_path: Optional[pathlib._local.Path] = None) -> List[Tuple[refactron.core.models.RefactoringOperation, float]]
```

Rank refactoring operations by predicted value based on learned patterns.

Scoring factors:
1. Pattern acceptance rate (base score)
2. Project-specific weights (from ProjectPatternProfile)
3. Pattern recency (recent patterns weighted higher)
4. Pattern frequency (more occurrences = more reliable)
5. Metrics improvement (complexity reduction, maintainability)
6. Risk penalty (higher risk = lower score)

Args:
    operations: List of refactoring operations to rank
    project_path: Optional project path for project-specific scoring

Returns:
    List of tuples (operation, score) sorted by score descending
    Score range: 0.0 (lowest priority) to 1.0 (highest priority)

## Functions


---

# refactron.patterns.storage

Storage management for Pattern Learning System.

## Classes

### PatternStorage

```python
PatternStorage(storage_dir: Optional[pathlib._local.Path] = None)
```

Manages persistent storage for pattern learning data.

#### PatternStorage.__init__

```python
PatternStorage.__init__(self, storage_dir: Optional[pathlib._local.Path] = None)
```

Initialize pattern storage.

Args:
    storage_dir: Directory to store pattern data. If None, uses default:
                - First checks project root (.refactron/patterns/)
                - Falls back to ~/.refactron/patterns/

#### PatternStorage.clear_cache

```python
PatternStorage.clear_cache(self) -> None
```

Clear in-memory caches (force reload from disk).

#### PatternStorage.get_pattern

```python
PatternStorage.get_pattern(self, pattern_id: str) -> Optional[refactron.patterns.models.RefactoringPattern]
```

Get a specific pattern by ID.

Args:
    pattern_id: Pattern ID to retrieve

Returns:
    Pattern if found, None otherwise

#### PatternStorage.get_pattern_metric

```python
PatternStorage.get_pattern_metric(self, pattern_id: str) -> Optional[refactron.patterns.models.PatternMetric]
```

Get metric for a specific pattern.

Args:
    pattern_id: Pattern ID to retrieve metric for

Returns:
    Metric if found, None otherwise

#### PatternStorage.get_project_profile

```python
PatternStorage.get_project_profile(self, project_path: pathlib._local.Path) -> refactron.patterns.models.ProjectPatternProfile
```

Get or create project profile for a project.

Args:
    project_path: Path to project root

Returns:
    Project profile (created if doesn't exist)

#### PatternStorage.load_feedback

```python
PatternStorage.load_feedback(self, pattern_id: Optional[str] = None, project_path: Optional[pathlib._local.Path] = None) -> List[refactron.patterns.models.RefactoringFeedback]
```

Load feedback records from storage.

Note: For large feedback datasets, this filters in Python after loading
all records. Consider implementing pagination or separate indices if
performance becomes an issue with very large datasets.

Args:
    pattern_id: Optional pattern ID to filter by
    project_path: Optional project path to filter by

Returns:
    List of feedback records matching filters

#### PatternStorage.load_pattern_metrics

```python
PatternStorage.load_pattern_metrics(self) -> Dict[str, refactron.patterns.models.PatternMetric]
```

Load all pattern metrics from storage.

Returns:
    Dictionary mapping pattern_id to PatternMetric
    Note: Returns a copy to prevent external modifications to cache.

#### PatternStorage.load_patterns

```python
PatternStorage.load_patterns(self) -> Dict[str, refactron.patterns.models.RefactoringPattern]
```

Load all patterns from storage.

Returns:
    Dictionary mapping pattern_id to RefactoringPattern
    Note: Returns a copy to prevent external modifications to cache.

#### PatternStorage.load_project_profiles

```python
PatternStorage.load_project_profiles(self) -> Dict[str, refactron.patterns.models.ProjectPatternProfile]
```

Load all project profiles from storage.

Returns:
    Dictionary mapping project_id to ProjectPatternProfile
    Note: Returns a copy to prevent external modifications to cache.

#### PatternStorage.replace_patterns

```python
PatternStorage.replace_patterns(self, patterns: Dict[str, refactron.patterns.models.RefactoringPattern]) -> None
```

Replace all patterns in storage with the provided dictionary.

This method completely replaces the pattern storage, useful for cleanup
operations where patterns need to be removed.

Args:
    patterns: Dictionary mapping pattern_id to RefactoringPattern

#### PatternStorage.save_feedback

```python
PatternStorage.save_feedback(self, feedback: refactron.patterns.models.RefactoringFeedback) -> None
```

Save feedback record to storage.

Args:
    feedback: Feedback record to save

#### PatternStorage.save_pattern

```python
PatternStorage.save_pattern(self, pattern: refactron.patterns.models.RefactoringPattern) -> None
```

Save pattern to storage.

Args:
    pattern: Pattern to save

#### PatternStorage.save_pattern_metric

```python
PatternStorage.save_pattern_metric(self, metric: refactron.patterns.models.PatternMetric) -> None
```

Save pattern metric to storage.

Args:
    metric: Metric to save

#### PatternStorage.save_project_profile

```python
PatternStorage.save_project_profile(self, profile: refactron.patterns.models.ProjectPatternProfile) -> None
```

Save project profile to storage.

Args:
    profile: Profile to save

#### PatternStorage.update_pattern_stats

```python
PatternStorage.update_pattern_stats(self, pattern_id: str, action: str) -> None
```

Update pattern statistics from feedback.

Args:
    pattern_id: Pattern ID to update
    action: Action taken ("accepted", "rejected", "ignored")

## Functions


---

# refactron.patterns.tuner

Project-specific rule tuning based on pattern learning history.

## Classes

### PatternStats

```python
PatternStats(pattern_id: 'str', pattern_hash: 'str', operation_type: 'str', accepted_count: 'int' = 0, rejected_count: 'int' = 0, ignored_count: 'int' = 0) -> None
```

Aggregated statistics for a pattern within a specific project.

#### PatternStats.__init__

```python
PatternStats.__init__(self, pattern_id: 'str', pattern_hash: 'str', operation_type: 'str', accepted_count: 'int' = 0, rejected_count: 'int' = 0, ignored_count: 'int' = 0) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### RuleTuner

```python
RuleTuner(storage: 'PatternStorage') -> 'None'
```

Tunes rules based on project-specific pattern history.

#### RuleTuner.__init__

```python
RuleTuner.__init__(self, storage: 'PatternStorage') -> 'None'
```

Initialize self.  See help(type(self)) for accurate signature.

#### RuleTuner.analyze_project_patterns

```python
RuleTuner.analyze_project_patterns(self, project_path: 'Path') -> 'Dict[str, Any]'
```

Analyze patterns for a specific project.

Returns a dictionary with:
- project_id
- project_path
- patterns: list of per-pattern statistics combining project and global data

#### RuleTuner.apply_tuning

```python
RuleTuner.apply_tuning(self, project_path: 'Path', recommendations: 'Dict[str, Any]') -> 'ProjectPatternProfile'
```

Apply tuning recommendations to project profile.

recommendations is expected to have keys:
- "to_disable": List[str] of pattern_ids to disable
- "to_enable": List[str] of pattern_ids to enable
- "weights": Dict[str, float] of pattern_id -> weight

#### RuleTuner.generate_recommendations

```python
RuleTuner.generate_recommendations(self, project_path: 'Path') -> 'Dict[str, Any]'
```

Generate rule tuning recommendations for a project.

Heuristics:
- Disable patterns with sufficient feedback and low acceptance.
- Enable patterns with high acceptance.
- Adjust pattern weights based on project acceptance.

## Functions
