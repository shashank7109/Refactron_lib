"""Ranking engine for refactoring suggestions based on learned patterns."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from refactron.core.models import RefactoringOperation
from refactron.patterns.fingerprint import PatternFingerprinter
from refactron.patterns.matcher import PatternMatcher
from refactron.patterns.models import ProjectPatternProfile, RefactoringPattern
from refactron.patterns.storage import PatternStorage

logger = logging.getLogger(__name__)


class RefactoringRanker:
    """Ranks refactoring suggestions based on learned patterns and project context."""

    def __init__(
        self,
        storage: PatternStorage,
        matcher: PatternMatcher,
        fingerprinter: PatternFingerprinter,
    ):
        """
        Initialize refactoring ranker.

        Args:
            storage: PatternStorage instance for accessing patterns and profiles
            matcher: PatternMatcher instance for finding similar patterns
            fingerprinter: PatternFingerprinter instance for generating code hashes
        """
        if storage is None:
            raise ValueError("PatternStorage cannot be None")
        if matcher is None:
            raise ValueError("PatternMatcher cannot be None")
        if fingerprinter is None:
            raise ValueError("PatternFingerprinter cannot be None")

        self.storage = storage
        self.matcher = matcher
        self.fingerprinter = fingerprinter

    def rank_operations(
        self,
        operations: List[RefactoringOperation],
        project_path: Optional[Path] = None,
    ) -> List[Tuple[RefactoringOperation, float]]:
        """
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
        """
        if not operations:
            return []

        # Load project profile if project path provided
        project_profile: Optional[ProjectPatternProfile] = None
        if project_path:
            try:
                project_profile = self.storage.get_project_profile(project_path)
            except Exception as e:
                logger.debug(f"Failed to load project profile for {project_path}: {e}")

        # Rank each operation
        ranked_operations: List[Tuple[RefactoringOperation, float]] = []

        for operation in operations:
            try:
                score = self._calculate_operation_score(operation, project_profile)
                ranked_operations.append((operation, score))
            except Exception as e:
                # If scoring fails, assign default score and log warning
                logger.warning(
                    f"Failed to calculate score for operation {operation.operation_id}: {e}"
                )
                # Default score based on risk (lower risk = higher default score)
                # Clamp to [0.0, 1.0] to satisfy documented score range.
                default_score = 1.0 - operation.risk_score
                default_score = max(0.0, min(1.0, default_score))
                ranked_operations.append((operation, default_score))

        # Sort by score descending (highest score first)
        ranked_operations.sort(key=lambda x: x[1], reverse=True)

        return ranked_operations

    def _calculate_operation_score(
        self,
        operation: RefactoringOperation,
        project_profile: Optional[ProjectPatternProfile] = None,
    ) -> float:
        """
        Calculate ranking score for a single operation.

        Args:
            operation: RefactoringOperation to score
            project_profile: Optional project-specific profile

        Returns:
            Score between 0.0 and 1.0 (higher = better suggestion)
        """
        # Get or generate pattern hash
        pattern_hash = operation.metadata.get("code_pattern_hash")
        if not pattern_hash:
            try:
                pattern_hash = self.fingerprinter.fingerprint_code(operation.old_code)
                # Store hash in metadata for future use
                operation.metadata["code_pattern_hash"] = pattern_hash
            except Exception as e:
                logger.debug(
                    f"Failed to fingerprint code for operation {operation.operation_id}: {e}"
                )
                # Fallback: score based only on risk, clamped to [0.0, 1.0]
                fallback_score = 1.0 - operation.risk_score
                return max(0.0, min(1.0, fallback_score))

        # Find matching patterns
        matching_patterns = self.matcher.find_similar_patterns(
            code_hash=pattern_hash,
            operation_type=operation.operation_type,
            limit=5,  # Consider top 5 matches
        )

        if not matching_patterns:
            # No matching patterns found - use risk-based scoring
            base_score = 1.0 - operation.risk_score
            base_score = max(0.0, min(1.0, base_score))
            # Apply slight penalty for unknown patterns and clamp result
            unknown_pattern_score = base_score * 0.8
            return max(0.0, min(1.0, unknown_pattern_score))

        # Use best matching pattern for scoring
        best_pattern = matching_patterns[0]

        # Calculate base score from pattern
        base_score = self.matcher.calculate_pattern_score(best_pattern, project_profile)

        # Apply risk penalty (higher risk = lower score)
        # Risk penalty: score *= (1.0 - risk_score * 0.5)
        # This means:
        # - risk_score 0.0: no penalty (multiply by 1.0)
        # - risk_score 0.5: moderate penalty (multiply by 0.75)
        # - risk_score 1.0: maximum penalty (multiply by 0.5)
        risk_penalty_factor = 1.0 - (operation.risk_score * 0.5)
        base_score *= risk_penalty_factor

        # Apply pattern confidence bonus if multiple patterns match
        # (indicates this is a well-known pattern)
        if len(matching_patterns) > 1:
            confidence_bonus = 1.0 + (0.05 * min(len(matching_patterns) - 1, 3))
            base_score *= min(confidence_bonus, 1.15)  # Cap at 15% bonus

        # Normalize to 0.0-1.0 range
        final_score = min(1.0, max(0.0, base_score))

        return final_score

    def get_top_suggestions(
        self,
        operations: List[RefactoringOperation],
        project_path: Optional[Path] = None,
        top_n: int = 10,
    ) -> List[RefactoringOperation]:
        """
        Get top N ranked suggestions.

        Args:
            operations: List of refactoring operations to rank
            project_path: Optional project path for project-specific scoring
            top_n: Number of top suggestions to return (default: 10)

        Returns:
            List of top N RefactoringOperation instances, sorted by score descending
        """
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")

        ranked = self.rank_operations(operations, project_path)
        return [op for op, _ in ranked[:top_n]]

    def get_ranked_with_scores(
        self,
        operations: List[RefactoringOperation],
        project_path: Optional[Path] = None,
    ) -> List[Tuple[RefactoringOperation, float, Optional[RefactoringPattern]]]:
        """
        Get ranked operations with scores and matching patterns.

        Args:
            operations: List of refactoring operations to rank
            project_path: Optional project path for project-specific scoring

        Returns:
            List of tuples (operation, score, best_matching_pattern)
            sorted by score descending
        """
        if not operations:
            return []

        # Load project profile if project path provided
        project_profile: Optional[ProjectPatternProfile] = None
        if project_path:
            try:
                project_profile = self.storage.get_project_profile(project_path)
            except Exception as e:
                logger.debug(f"Failed to load project profile for {project_path}: {e}")

        ranked_with_patterns: List[
            Tuple[RefactoringOperation, float, Optional[RefactoringPattern]]
        ] = []

        for operation in operations:
            try:
                # Get pattern hash
                pattern_hash = operation.metadata.get("code_pattern_hash")
                if not pattern_hash:
                    try:
                        pattern_hash = self.fingerprinter.fingerprint_code(operation.old_code)
                        operation.metadata["code_pattern_hash"] = pattern_hash
                    except Exception as e:
                        logger.debug(
                            f"Failed to fingerprint code for operation "
                            f"{operation.operation_id}: {e}"
                        )
                        pattern_hash = None

                # Find matching pattern
                best_pattern: Optional[RefactoringPattern] = None
                if pattern_hash:
                    matching_patterns = self.matcher.find_similar_patterns(
                        code_hash=pattern_hash,
                        operation_type=operation.operation_type,
                        limit=1,
                    )
                    if matching_patterns:
                        best_pattern = matching_patterns[0]

                # Calculate score
                score = self._calculate_operation_score(operation, project_profile)
                ranked_with_patterns.append((operation, score, best_pattern))

            except Exception as e:
                logger.warning(f"Failed to rank operation {operation.operation_id}: {e}")
                fallback_score = 1.0 - operation.risk_score
                fallback_score = max(0.0, min(1.0, fallback_score))
                ranked_with_patterns.append((operation, fallback_score, None))

        # Sort by score descending
        ranked_with_patterns.sort(key=lambda x: x[1], reverse=True)

        return ranked_with_patterns
