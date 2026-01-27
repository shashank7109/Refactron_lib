"""Pattern learning engine that learns from feedback and refactoring history."""

import copy
import logging
from typing import Dict, List, Optional, Tuple

from refactron.core.models import FileMetrics, RefactoringOperation
from refactron.patterns.fingerprint import PatternFingerprinter
from refactron.patterns.models import PatternMetric, RefactoringFeedback, RefactoringPattern
from refactron.patterns.storage import PatternStorage

logger = logging.getLogger(__name__)


class PatternLearner:
    """Learns patterns from feedback and refactoring history."""

    def __init__(self, storage: PatternStorage, fingerprinter: PatternFingerprinter) -> None:
        """
        Initialize pattern learner.

        Args:
            storage: PatternStorage instance for loading/saving patterns
            fingerprinter: PatternFingerprinter for generating code fingerprints

        Raises:
            ValueError: If storage or fingerprinter is None
        """
        if storage is None:
            raise ValueError("PatternStorage cannot be None")
        if fingerprinter is None:
            raise ValueError("PatternFingerprinter cannot be None")

        self.storage = storage
        self.fingerprinter = fingerprinter

    def learn_from_feedback(
        self, operation: RefactoringOperation, feedback: RefactoringFeedback
    ) -> Optional[str]:
        """
        Learn from a single feedback record.

        Args:
            operation: RefactoringOperation that was evaluated
            feedback: Feedback record containing developer decision

        Returns:
            Pattern ID if pattern was created/updated, None if skipped

        Raises:
            ValueError: If operation or feedback is None
            RuntimeError: If pattern storage operations fail
        """
        if operation is None:
            raise ValueError("RefactoringOperation cannot be None")
        if feedback is None:
            raise ValueError("RefactoringFeedback cannot be None")

        try:
            # Get pattern hash from operation metadata or generate it
            pattern_hash = operation.metadata.get("code_pattern_hash")
            if not pattern_hash:
                # Fallback: generate hash from old_code
                try:
                    pattern_hash = self.fingerprinter.fingerprint_code(operation.old_code)
                except Exception as e:
                    logger.warning(
                        f"Failed to fingerprint code pattern for operation "
                        f"{operation.operation_id}: {e}"
                    )
                    return None

            # Find existing pattern by hash and operation type
            pattern = self._find_pattern_by_hash(pattern_hash, operation.operation_type)

            if pattern is None:
                # Create new pattern
                pattern = self._create_pattern_from_operation(operation, pattern_hash, feedback)

            # Update pattern statistics from feedback
            pattern.update_from_feedback(feedback.action)

            # Update benefit score if accepted and we have metrics
            if feedback.action == "accepted":
                # Only update benefit score if we have metrics to calculate from
                # Otherwise calculate_benefit_score() would just return the existing value
                metric = self.storage.get_pattern_metric(pattern.pattern_id)
                if metric is not None:
                    pattern.average_benefit_score = pattern.calculate_benefit_score(metric)

            # Save updated pattern
            try:
                self.storage.save_pattern(pattern)
                logger.debug(
                    f"Learned from feedback: pattern {pattern.pattern_id}, "
                    f"action: {feedback.action}"
                )
                return pattern.pattern_id
            except Exception as e:
                logger.error(f"Failed to save pattern {pattern.pattern_id}: {e}", exc_info=True)
                raise RuntimeError(f"Failed to save pattern: {e}") from e

        except Exception as e:
            logger.error(
                f"Error learning from feedback for operation {operation.operation_id}: {e}",
                exc_info=True,
            )
            # Don't raise - allow other feedback to be processed
            return None

    def batch_learn(
        self, feedback_list: List[Tuple[RefactoringOperation, RefactoringFeedback]]
    ) -> Dict[str, int]:
        """
        Process multiple feedback records efficiently.

        Args:
            feedback_list: List of (operation, feedback) tuples to process

        Returns:
            Dictionary with statistics: {'processed': int, 'created': int,
            'updated': int, 'failed': int}

        Raises:
            ValueError: If feedback_list is None or contains None values
        """
        if feedback_list is None:
            raise ValueError("feedback_list cannot be None")

        stats = {"processed": 0, "created": 0, "updated": 0, "failed": 0}
        existing_patterns: Dict[str, RefactoringPattern] = {}

        try:
            # Pre-load all patterns for batch processing
            all_patterns = self.storage.load_patterns()
            for pattern in all_patterns.values():
                key = self._pattern_key(pattern.pattern_hash, pattern.operation_type)
                existing_patterns[key] = pattern

            # Process feedback in batch
            patterns_to_save: Dict[str, RefactoringPattern] = {}

            for operation, feedback in feedback_list:
                if operation is None or feedback is None:
                    logger.warning("Skipping None operation or feedback in batch")
                    stats["failed"] += 1
                    continue

                try:
                    stats["processed"] += 1

                    # Get pattern hash
                    pattern_hash = operation.metadata.get("code_pattern_hash")
                    if not pattern_hash:
                        try:
                            pattern_hash = self.fingerprinter.fingerprint_code(operation.old_code)
                        except Exception as e:
                            logger.debug(
                                f"Failed to fingerprint for operation {operation.operation_id}: {e}"
                            )
                            stats["failed"] += 1
                            continue

                    key = self._pattern_key(pattern_hash, operation.operation_type)
                    pattern = existing_patterns.get(key)

                    if pattern is None:
                        # Create new pattern
                        pattern = self._create_pattern_from_operation(
                            operation, pattern_hash, feedback
                        )
                        existing_patterns[key] = pattern
                        patterns_to_save[pattern.pattern_id] = pattern
                        stats["created"] += 1
                    else:
                        # Deep-copy pattern before mutation to avoid thread-safety issues
                        # (load_patterns returns cached instances that could be modified
                        # concurrently)
                        if pattern.pattern_id not in patterns_to_save:
                            pattern = copy.deepcopy(pattern)
                            patterns_to_save[pattern.pattern_id] = pattern
                        else:
                            # Already have a copy, use it
                            pattern = patterns_to_save[pattern.pattern_id]
                        stats["updated"] += 1

                    # Update pattern statistics on copy
                    pattern.update_from_feedback(feedback.action)
                    if feedback.action == "accepted":
                        # Only update benefit score if we have metrics
                        metric = self.storage.get_pattern_metric(pattern.pattern_id)
                        if metric is not None:
                            pattern.average_benefit_score = pattern.calculate_benefit_score(metric)

                except Exception as e:
                    logger.warning(
                        f"Failed to process feedback for operation {operation.operation_id}: {e}"
                    )
                    stats["failed"] += 1

            # Batch save all updated patterns
            if patterns_to_save:
                try:
                    # Save patterns individually (storage handles merging)
                    for pattern in patterns_to_save.values():
                        self.storage.save_pattern(pattern)

                except Exception as e:
                    logger.error(f"Failed to batch save patterns: {e}", exc_info=True)
                    # Fallback: save individually
                    for pattern in patterns_to_save.values():
                        try:
                            self.storage.save_pattern(pattern)
                        except Exception as save_error:
                            logger.error(
                                f"Failed to save pattern {pattern.pattern_id}: {save_error}"
                            )
                            stats["failed"] += 1

            logger.info(
                f"Batch learning complete: {stats['processed']} processed, "
                f"{stats['created']} created, {stats['updated']} updated, "
                f"{stats['failed']} failed"
            )

        except Exception as e:
            logger.error(f"Error in batch learning: {e}", exc_info=True)
            raise RuntimeError(f"Batch learning failed: {e}") from e

        return stats

    def update_pattern_metrics(
        self,
        pattern_id: str,
        before_metrics: FileMetrics,
        after_metrics: FileMetrics,
    ) -> None:
        """
        Update metrics for a pattern based on before/after comparison.

        Args:
            pattern_id: ID of the pattern to update
            before_metrics: FileMetrics before refactoring
            after_metrics: FileMetrics after refactoring

        Raises:
            ValueError: If pattern_id is empty or metrics are None
            RuntimeError: If pattern not found or update fails
        """
        if not pattern_id:
            raise ValueError("pattern_id cannot be empty")
        if before_metrics is None:
            raise ValueError("before_metrics cannot be None")
        if after_metrics is None:
            raise ValueError("after_metrics cannot be None")

        try:
            pattern = self.storage.get_pattern(pattern_id)
            if pattern is None:
                logger.warning(f"Pattern {pattern_id} not found for metrics update")
                return

            # Calculate improvements
            complexity_reduction = before_metrics.complexity - after_metrics.complexity
            maintainability_improvement = (
                after_metrics.maintainability_index - before_metrics.maintainability_index
            )
            issue_count_reduction = before_metrics.issue_count - after_metrics.issue_count

            # Get or create pattern metric
            metric = self.storage.get_pattern_metric(pattern_id)
            lines_change = after_metrics.lines_of_code - before_metrics.lines_of_code
            before_metrics_dict = {
                "complexity": before_metrics.complexity,
                "maintainability_index": before_metrics.maintainability_index,
                "lines_of_code": before_metrics.lines_of_code,
            }
            after_metrics_dict = {
                "complexity": after_metrics.complexity,
                "maintainability_index": after_metrics.maintainability_index,
                "lines_of_code": after_metrics.lines_of_code,
            }

            if metric is None:
                metric = PatternMetric(
                    pattern_id=pattern_id,
                    complexity_reduction=complexity_reduction,
                    maintainability_improvement=maintainability_improvement,
                    lines_of_code_change=lines_change,
                    issue_resolution_count=issue_count_reduction,
                    before_metrics=before_metrics_dict,
                    after_metrics=after_metrics_dict,
                    total_evaluations=1,
                )
            else:
                # Update existing metric with weighted average
                metric.update(
                    complexity_reduction=complexity_reduction,
                    maintainability_improvement=maintainability_improvement,
                    lines_of_code_change=lines_change,
                    issue_resolution_count=issue_count_reduction,
                    before_metrics=before_metrics_dict,
                    after_metrics=after_metrics_dict,
                )

            # Save metric
            self.storage.save_pattern_metric(metric)

            # Update pattern's average benefit score
            pattern.average_benefit_score = pattern.calculate_benefit_score(metric)
            self.storage.save_pattern(pattern)

            logger.debug(
                f"Updated metrics for pattern {pattern_id}: "
                f"complexity_reduction={complexity_reduction:.2f}, "
                f"maintainability_improvement={maintainability_improvement:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to update metrics for pattern {pattern_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to update pattern metrics: {e}") from e

    def _find_pattern_by_hash(
        self, pattern_hash: str, operation_type: str
    ) -> Optional[RefactoringPattern]:
        """
        Find existing pattern by hash and operation type.

        Args:
            pattern_hash: Hash of the code pattern
            operation_type: Type of refactoring operation

        Returns:
            RefactoringPattern if found, None otherwise
        """
        try:
            patterns = self.storage.load_patterns()
            for pattern in patterns.values():
                if (
                    pattern.pattern_hash == pattern_hash
                    and pattern.operation_type == operation_type
                ):
                    return pattern
            return None
        except Exception as e:
            logger.warning(f"Error finding pattern by hash: {e}")
            return None

    def _create_pattern_from_operation(
        self,
        operation: RefactoringOperation,
        pattern_hash: str,
        feedback: RefactoringFeedback,
    ) -> RefactoringPattern:
        """
        Create a new pattern from a refactoring operation.

        Args:
            operation: RefactoringOperation to create pattern from
            pattern_hash: Hash of the code pattern
            feedback: Feedback record for context

        Returns:
            New RefactoringPattern instance
        """
        project_context = {}
        if feedback.project_path:
            project_context["project_path"] = str(feedback.project_path)

        return RefactoringPattern.create(
            pattern_hash=pattern_hash,
            operation_type=operation.operation_type,
            code_snippet_before=operation.old_code,
            code_snippet_after=operation.new_code,
            project_context=project_context,
            metadata={"risk_score": operation.risk_score},
        )

    @staticmethod
    def _pattern_key(pattern_hash: str, operation_type: str) -> str:
        """Generate a unique key for pattern lookup."""
        return f"{pattern_hash}:{operation_type}"
