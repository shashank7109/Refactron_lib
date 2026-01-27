"""Storage management for Pattern Learning System."""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from refactron.patterns.models import (
    PatternMetric,
    ProjectPatternProfile,
    RefactoringFeedback,
    RefactoringPattern,
)

logger = logging.getLogger(__name__)


class PatternStorage:
    """Manages persistent storage for pattern learning data."""

    PATTERNS_FILE = "patterns.json"
    FEEDBACK_FILE = "feedback.json"
    PROJECT_PROFILES_FILE = "project_profiles.json"
    PATTERN_METRICS_FILE = "pattern_metrics.json"
    MAX_PROJECT_ROOT_SEARCH_DEPTH = 5  # Maximum directory levels to search for project root

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize pattern storage.

        Args:
            storage_dir: Directory to store pattern data. If None, uses default:
                        - First checks project root (.refactron/patterns/)
                        - Falls back to ~/.refactron/patterns/
        """
        if storage_dir is None:
            # Try project root first
            project_root = self._find_project_root()
            if project_root:
                self.storage_dir = project_root / ".refactron" / "patterns"
            else:
                # Fall back to user home directory
                self.storage_dir = Path.home() / ".refactron" / "patterns"
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Thread lock for safe concurrent access (RLock for reentrant access)
        self._lock = threading.RLock()

        # File paths
        self.patterns_file = self.storage_dir / self.PATTERNS_FILE
        self.feedback_file = self.storage_dir / self.FEEDBACK_FILE
        self.profiles_file = self.storage_dir / self.PROJECT_PROFILES_FILE
        self.metrics_file = self.storage_dir / self.PATTERN_METRICS_FILE

        # In-memory caches (loaded on demand)
        self._patterns_cache: Optional[Dict[str, RefactoringPattern]] = None
        self._feedback_cache: Optional[List[RefactoringFeedback]] = None
        self._profiles_cache: Optional[Dict[str, ProjectPatternProfile]] = None
        self._metrics_cache: Optional[Dict[str, PatternMetric]] = None

    def _find_project_root(self) -> Optional[Path]:
        """
        Find project root by looking for .git, .refactron.yaml, or setup.py/pyproject.toml.

        Returns:
            Project root path if found, None otherwise
        """
        current = Path.cwd().resolve()

        # Check up to MAX_PROJECT_ROOT_SEARCH_DEPTH levels up
        for _ in range(self.MAX_PROJECT_ROOT_SEARCH_DEPTH):
            if any(
                (current / marker).exists()
                for marker in [".git", ".refactron.yaml", "setup.py", "pyproject.toml"]
            ):
                return current
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent

        return None

    def save_feedback(self, feedback: RefactoringFeedback) -> None:
        """
        Save feedback record to storage.

        Args:
            feedback: Feedback record to save
        """
        with self._lock:
            feedbacks = self.load_feedback()
            feedbacks.append(feedback)
            self._save_feedback_list(feedbacks)

    def load_feedback(
        self, pattern_id: Optional[str] = None, project_path: Optional[Path] = None
    ) -> List[RefactoringFeedback]:
        """
        Load feedback records from storage.

        Note: For large feedback datasets, this filters in Python after loading
        all records. Consider implementing pagination or separate indices if
        performance becomes an issue with very large datasets.

        Args:
            pattern_id: Optional pattern ID to filter by
            project_path: Optional project path to filter by

        Returns:
            List of feedback records matching filters
        """
        with self._lock:
            if self._feedback_cache is None:
                self._feedback_cache = self._load_feedback_list()

            feedbacks = self._feedback_cache

            # Optimize: Apply all filters in a single pass
            if pattern_id or project_path:
                # Pre-resolve project_path once to avoid repeated calls
                resolved_project_str = str(project_path.resolve()) if project_path else None

                filtered = []
                for f in feedbacks:
                    # Apply pattern_id filter
                    if pattern_id and f.code_pattern_hash != pattern_id:
                        continue

                    # Apply project_path filter (only resolve f.project_path when needed)
                    if resolved_project_str:
                        if not f.project_path:
                            continue
                        if str(f.project_path.resolve()) != resolved_project_str:
                            continue

                    filtered.append(f)

                return filtered

            return feedbacks

    def save_pattern(self, pattern: RefactoringPattern) -> None:
        """
        Save pattern to storage.

        Args:
            pattern: Pattern to save
        """
        with self._lock:
            patterns = self.load_patterns()
            patterns[pattern.pattern_id] = pattern
            self._save_patterns_dict(patterns)

    def replace_patterns(self, patterns: Dict[str, RefactoringPattern]) -> None:
        """
        Replace all patterns in storage with the provided dictionary.

        This method completely replaces the pattern storage, useful for cleanup
        operations where patterns need to be removed.

        Args:
            patterns: Dictionary mapping pattern_id to RefactoringPattern
        """
        with self._lock:
            self._save_patterns_dict(patterns)

    def load_patterns(self) -> Dict[str, RefactoringPattern]:
        """
        Load all patterns from storage.

        Returns:
            Dictionary mapping pattern_id to RefactoringPattern
            Note: Returns a copy to prevent external modifications to cache.
        """
        with self._lock:
            if self._patterns_cache is None:
                self._patterns_cache = self._load_patterns_dict()
            # Return copy to prevent cache mutation (required for thread safety)
            return dict(self._patterns_cache)  # dict() is faster than .copy() for large dicts

    def get_pattern(self, pattern_id: str) -> Optional[RefactoringPattern]:
        """
        Get a specific pattern by ID.

        Args:
            pattern_id: Pattern ID to retrieve

        Returns:
            Pattern if found, None otherwise
        """
        patterns = self.load_patterns()
        return patterns.get(pattern_id)

    def update_pattern_stats(self, pattern_id: str, action: str) -> None:
        """
        Update pattern statistics from feedback.

        Args:
            pattern_id: Pattern ID to update
            action: Action taken ("accepted", "rejected", "ignored")
        """
        with self._lock:
            pattern = self.get_pattern(pattern_id)
            if pattern:
                pattern.update_from_feedback(action)
                self.save_pattern(pattern)

    def save_pattern_metric(self, metric: PatternMetric) -> None:
        """
        Save pattern metric to storage.

        Args:
            metric: Metric to save
        """
        with self._lock:
            metrics = self.load_pattern_metrics()
            metrics[metric.pattern_id] = metric
            self._save_metrics_dict(metrics)

    def load_pattern_metrics(self) -> Dict[str, PatternMetric]:
        """
        Load all pattern metrics from storage.

        Returns:
            Dictionary mapping pattern_id to PatternMetric
            Note: Returns a copy to prevent external modifications to cache.
        """
        with self._lock:
            if self._metrics_cache is None:
                self._metrics_cache = self._load_metrics_dict()
            # Return copy to prevent cache mutation (required for thread safety)
            return dict(self._metrics_cache)  # dict() is faster than .copy() for large dicts

    def get_pattern_metric(self, pattern_id: str) -> Optional[PatternMetric]:
        """
        Get metric for a specific pattern.

        Args:
            pattern_id: Pattern ID to retrieve metric for

        Returns:
            Metric if found, None otherwise
        """
        metrics = self.load_pattern_metrics()
        return metrics.get(pattern_id)

    def get_project_profile(self, project_path: Path) -> ProjectPatternProfile:
        """
        Get or create project profile for a project.

        Args:
            project_path: Path to project root

        Returns:
            Project profile (created if doesn't exist)
        """
        with self._lock:
            project_id = ProjectPatternProfile.generate_project_id(project_path)
            profiles = self.load_project_profiles()

            if project_id in profiles:
                profile = profiles[project_id]
                # Update path if it changed (same project, different location)
                # Use filesystem-aware comparison when possible
                paths_differ = True
                try:
                    if profile.project_path.exists() and project_path.exists():
                        # Use filesystem-aware comparison when both paths exist
                        paths_differ = not profile.project_path.samefile(project_path)
                    else:
                        # Fall back to resolved path string comparison
                        paths_differ = str(profile.project_path.resolve()) != str(
                            project_path.resolve()
                        )
                except OSError:
                    # If samefile/exists checks fail, fall back to resolved string comparison
                    paths_differ = str(profile.project_path.resolve()) != str(
                        project_path.resolve()
                    )

                if paths_differ:
                    profile.project_path = project_path
                    self.save_project_profile(profile)
                return profile

            # Create new profile
            profile = ProjectPatternProfile.create(project_path)
            self.save_project_profile(profile)
            return profile

    def save_project_profile(self, profile: ProjectPatternProfile) -> None:
        """
        Save project profile to storage.

        Args:
            profile: Profile to save
        """
        with self._lock:
            profiles = self.load_project_profiles()
            profiles[profile.project_id] = profile
            self._save_profiles_dict(profiles)

    def load_project_profiles(self) -> Dict[str, ProjectPatternProfile]:
        """
        Load all project profiles from storage.

        Returns:
            Dictionary mapping project_id to ProjectPatternProfile
            Note: Returns a copy to prevent external modifications to cache.
        """
        with self._lock:
            if self._profiles_cache is None:
                self._profiles_cache = self._load_profiles_dict()
            # Return copy to prevent cache mutation (required for thread safety)
            return dict(self._profiles_cache)  # dict() is faster than .copy() for large dicts

    def clear_cache(self) -> None:
        """Clear in-memory caches (force reload from disk)."""
        with self._lock:
            self._patterns_cache = None
            self._feedback_cache = None
            self._profiles_cache = None
            self._metrics_cache = None

    # Private methods for file I/O

    def _load_patterns_dict(self) -> Dict[str, RefactoringPattern]:
        """
        Load patterns from file.

        Note: This method performs basic JSON validation but does not
        validate against maliciously crafted data. Ensure storage directory
        has appropriate access controls in production environments.
        """
        if not self.patterns_file.exists():
            return {}

        try:
            with open(self.patterns_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Validate basic structure
                if not isinstance(data, dict):
                    logger.warning(
                        f"Invalid patterns file structure: expected dict, got {type(data)}"
                    )
                    return {}
                return {
                    pattern_id: RefactoringPattern.from_dict(pattern_data)
                    for pattern_id, pattern_data in data.items()
                    if isinstance(pattern_data, dict)  # Skip invalid entries
                }
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load patterns from {self.patterns_file}: {e}")
            return {}

    def _save_patterns_dict(self, patterns: Dict[str, RefactoringPattern]) -> None:
        """Save patterns to file."""
        try:
            data = {pattern_id: pattern.to_dict() for pattern_id, pattern in patterns.items()}
            with open(self.patterns_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._patterns_cache = patterns
        except IOError as e:
            logger.error(f"Failed to save patterns to {self.patterns_file}: {e}")
            raise

    def _load_feedback_list(self) -> List[RefactoringFeedback]:
        """Load feedback from file."""
        if not self.feedback_file.exists():
            return []

        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [RefactoringFeedback.from_dict(item) for item in data]
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load feedback from {self.feedback_file}: {e}")
            return []

    def _save_feedback_list(self, feedbacks: List[RefactoringFeedback]) -> None:
        """Save feedback to file."""
        try:
            data = [feedback.to_dict() for feedback in feedbacks]
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._feedback_cache = feedbacks
        except IOError as e:
            logger.error(f"Failed to save feedback to {self.feedback_file}: {e}")
            raise

    def _load_profiles_dict(self) -> Dict[str, ProjectPatternProfile]:
        """Load project profiles from file."""
        if not self.profiles_file.exists():
            return {}

        try:
            with open(self.profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    project_id: ProjectPatternProfile.from_dict(profile_data)
                    for project_id, profile_data in data.items()
                }
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load project profiles from {self.profiles_file}: {e}")
            return {}

    def _save_profiles_dict(self, profiles: Dict[str, ProjectPatternProfile]) -> None:
        """Save project profiles to file."""
        try:
            data = {project_id: profile.to_dict() for project_id, profile in profiles.items()}
            with open(self.profiles_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._profiles_cache = profiles
        except IOError as e:
            logger.error(f"Failed to save project profiles to {self.profiles_file}: {e}")
            raise

    def _load_metrics_dict(self) -> Dict[str, PatternMetric]:
        """Load pattern metrics from file."""
        if not self.metrics_file.exists():
            return {}

        try:
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    pattern_id: PatternMetric.from_dict(metric_data)
                    for pattern_id, metric_data in data.items()
                }
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load metrics from {self.metrics_file}: {e}")
            return {}

    def _save_metrics_dict(self, metrics: Dict[str, PatternMetric]) -> None:
        """Save pattern metrics to file."""
        try:
            data = {pattern_id: metric.to_dict() for pattern_id, metric in metrics.items()}
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._metrics_cache = metrics
        except IOError as e:
            logger.error(f"Failed to save metrics to {self.metrics_file}: {e}")
            raise
