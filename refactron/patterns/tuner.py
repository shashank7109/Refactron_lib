"""Project-specific rule tuning based on pattern learning history."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from refactron.patterns.models import ProjectPatternProfile, RefactoringPattern
from refactron.patterns.storage import PatternStorage

logger = logging.getLogger(__name__)


@dataclass
class PatternStats:
    """Aggregated statistics for a pattern within a specific project."""

    pattern_id: str
    pattern_hash: str
    operation_type: str
    accepted_count: int = 0
    rejected_count: int = 0
    ignored_count: int = 0

    @property
    def total_decisions(self) -> int:
        return self.accepted_count + self.rejected_count

    @property
    def total_feedback(self) -> int:
        return self.accepted_count + self.rejected_count + self.ignored_count

    @property
    def acceptance_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.accepted_count / self.total_decisions


class RuleTuner:
    """Tunes rules based on project-specific pattern history."""

    def __init__(self, storage: PatternStorage) -> None:
        if storage is None:
            raise ValueError("PatternStorage cannot be None")
        self.storage = storage

    def analyze_project_patterns(self, project_path: Path) -> Dict[str, Any]:
        """
        Analyze patterns for a specific project.

        Returns a dictionary with:
        - project_id
        - project_path
        - patterns: list of per-pattern statistics combining project and global data
        """
        if not isinstance(project_path, Path):
            raise TypeError("project_path must be a pathlib.Path instance")

        # Load project profile (creates if missing)
        profile = self.storage.get_project_profile(project_path)

        # Load all feedback for this project
        feedbacks = self.storage.load_feedback(project_path=project_path)
        if not feedbacks:
            return {
                "project_id": profile.project_id,
                "project_path": str(profile.project_path),
                "patterns": [],
            }

        # Build index of patterns by (hash, operation_type) for efficient lookups
        patterns_by_key: Dict[Tuple[str, str], RefactoringPattern] = {}
        for pattern in self.storage.load_patterns().values():
            key = (pattern.pattern_hash, pattern.operation_type)
            patterns_by_key[key] = pattern

        # Aggregate project-level stats per pattern_id
        stats_by_pattern_id: Dict[str, PatternStats] = {}

        for fb in feedbacks:
            if not fb.code_pattern_hash:
                # Feedback without a pattern hash cannot be mapped; skip safely
                continue

            key = (fb.code_pattern_hash, fb.operation_type)
            pattern = patterns_by_key.get(key)
            if not pattern:
                # Unknown pattern for this feedback; skip but log at debug level
                logger.debug(
                    "No pattern found for feedback operation_id=%s hash=%s type=%s",
                    fb.operation_id,
                    fb.code_pattern_hash,
                    fb.operation_type,
                )
                continue

            if pattern.pattern_id not in stats_by_pattern_id:
                stats_by_pattern_id[pattern.pattern_id] = PatternStats(
                    pattern_id=pattern.pattern_id,
                    pattern_hash=pattern.pattern_hash,
                    operation_type=pattern.operation_type,
                )

            stats = stats_by_pattern_id[pattern.pattern_id]
            if fb.action == "accepted":
                stats.accepted_count += 1
            elif fb.action == "rejected":
                stats.rejected_count += 1
            elif fb.action == "ignored":
                stats.ignored_count += 1

        # Build analysis list with global pattern data
        patterns_analysis: List[Dict[str, Any]] = []
        all_patterns = self.storage.load_patterns()

        for pattern_id, stats in stats_by_pattern_id.items():
            pattern = all_patterns.get(pattern_id)
            if not pattern:
                # Pattern may have been cleaned up; skip but do not fail
                logger.debug("Pattern %s not found when building analysis", pattern_id)
                continue

            patterns_analysis.append(
                {
                    "pattern_id": pattern.pattern_id,
                    "pattern_hash": pattern.pattern_hash,
                    "operation_type": pattern.operation_type,
                    "project_accepted": stats.accepted_count,
                    "project_rejected": stats.rejected_count,
                    "project_ignored": stats.ignored_count,
                    "project_total_feedback": stats.total_feedback,
                    "project_total_decisions": stats.total_decisions,
                    "project_acceptance_rate": stats.acceptance_rate,
                    "global_acceptance_rate": pattern.acceptance_rate,
                    "global_total_occurrences": pattern.total_occurrences,
                    "enabled": profile.is_pattern_enabled(pattern.pattern_id),
                    "weight": profile.get_pattern_weight(pattern.pattern_id, default=1.0),
                }
            )

        # Sort patterns by project acceptance rate (descending) for convenience
        patterns_analysis.sort(
            key=lambda p: (p["project_acceptance_rate"], p["project_total_feedback"]),
            reverse=True,
        )

        return {
            "project_id": profile.project_id,
            "project_path": str(profile.project_path),
            "patterns": patterns_analysis,
        }

    def generate_recommendations(self, project_path: Path) -> Dict[str, Any]:
        """
        Generate rule tuning recommendations for a project.

        Heuristics:
        - Disable patterns with sufficient feedback and low acceptance.
        - Enable patterns with high acceptance.
        - Adjust pattern weights based on project acceptance.
        """
        analysis = self.analyze_project_patterns(project_path)
        patterns = analysis["patterns"]

        to_disable: List[str] = []
        to_enable: List[str] = []
        weights: Dict[str, float] = {}

        # Thresholds for decisions
        min_feedback_for_decision = 5
        disable_threshold = 0.2  # Very low acceptance
        enable_threshold = 0.7  # High acceptance

        for p in patterns:
            total_feedback = p["project_total_feedback"]
            acc = p["project_acceptance_rate"]
            pattern_id = p["pattern_id"]

            if total_feedback < min_feedback_for_decision:
                # Not enough data to make a reliable decision
                continue

            if acc <= disable_threshold:
                to_disable.append(pattern_id)
                # Strongly down-weight disabled patterns
                weights[pattern_id] = 0.1
            elif acc >= enable_threshold:
                to_enable.append(pattern_id)
                # Slightly up-weight well-accepted patterns
                weights[pattern_id] = 0.9
            else:
                # For medium-acceptance patterns, gently reduce weight
                weights[pattern_id] = 0.7

        return {
            "project_path": str(project_path),
            "analysis": analysis,
            "to_disable": sorted(set(to_disable)),
            "to_enable": sorted(set(to_enable)),
            "weights": weights,
        }

    def apply_tuning(
        self,
        project_path: Path,
        recommendations: Dict[str, Any],
    ) -> ProjectPatternProfile:
        """
        Apply tuning recommendations to project profile.

        recommendations is expected to have keys:
        - "to_disable": List[str] of pattern_ids to disable
        - "to_enable": List[str] of pattern_ids to enable
        - "weights": Dict[str, float] of pattern_id -> weight
        """
        if not isinstance(recommendations, dict):
            raise TypeError("recommendations must be a dict")

        to_disable = recommendations.get("to_disable", [])
        to_enable = recommendations.get("to_enable", [])
        weights = recommendations.get("weights", {})

        if not isinstance(to_disable, list) or not all(isinstance(p, str) for p in to_disable):
            raise ValueError("to_disable must be a list of pattern_id strings")

        if not isinstance(to_enable, list) or not all(isinstance(p, str) for p in to_enable):
            raise ValueError("to_enable must be a list of pattern_id strings")

        if not isinstance(weights, dict) or not all(
            isinstance(k, str) and isinstance(v, (int, float)) for k, v in weights.items()
        ):
            raise ValueError("weights must be a dict[str, float] mapping pattern_id to weight")

        profile = self.storage.get_project_profile(project_path)

        # Apply enable/disable decisions
        for pattern_id in to_disable:
            profile.disable_pattern(pattern_id)

        for pattern_id in to_enable:
            profile.enable_pattern(pattern_id)

        # Apply weight adjustments
        for pattern_id, weight in weights.items():
            try:
                profile.set_pattern_weight(pattern_id, float(weight))
            except ValueError as e:
                # Invalid weight; log and continue with other patterns
                logger.warning("Skipping invalid weight for pattern %s: %s", pattern_id, e)

        # Persist updated profile
        self.storage.save_project_profile(profile)
        return profile
