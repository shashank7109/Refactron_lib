"""Background service for pattern learning and maintenance."""

import copy
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from refactron.core.models import RefactoringOperation
from refactron.patterns.fingerprint import PatternFingerprinter
from refactron.patterns.learner import PatternLearner
from refactron.patterns.models import RefactoringFeedback, RefactoringPattern
from refactron.patterns.storage import PatternStorage

logger = logging.getLogger(__name__)


class LearningService:
    """Background service for pattern learning and maintenance."""

    def __init__(
        self,
        storage: PatternStorage,
        learner: Optional[PatternLearner] = None,
    ) -> None:
        """
        Initialize learning service.

        Args:
            storage: PatternStorage instance for data access
            learner: PatternLearner instance (created if None)

        Raises:
            ValueError: If storage is None
        """
        if storage is None:
            raise ValueError("PatternStorage cannot be None")

        self.storage = storage
        self.learner = learner or PatternLearner(
            storage=storage, fingerprinter=PatternFingerprinter()
        )

    def process_pending_feedback(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Process any pending feedback records that haven't been learned from yet.

        Args:
            limit: Maximum number of feedback records to process (None = all)

        Returns:
            Dictionary with processing statistics

        Raises:
            RuntimeError: If processing fails critically
        """
        try:
            # Load all feedback
            all_feedback = self.storage.load_feedback()

            if not all_feedback:
                logger.debug("No feedback records to process")
                return {"processed": 0, "created": 0, "updated": 0, "failed": 0}

            # Filter feedback that needs processing
            # For now, we'll process all feedback (in future, could track processed status)
            if limit is None:
                feedback_to_process = all_feedback
            else:
                if limit < 0:
                    raise ValueError(f"limit must be non-negative or None, got {limit}")
                feedback_to_process = all_feedback[:limit]

            logger.info(f"Processing {len(feedback_to_process)} feedback records")

            # Group feedback by operation_id to get operations
            # Note: In a real scenario, we'd need to store operations alongside feedback
            # For now, we'll reconstruct operations from feedback metadata
            operations_with_feedback: List[Tuple[RefactoringOperation, RefactoringFeedback]] = []

            for feedback in feedback_to_process:
                try:
                    # Reconstruct operation from feedback
                    operation = self._reconstruct_operation_from_feedback(feedback)
                    if operation:
                        operations_with_feedback.append((operation, feedback))
                except Exception as e:
                    logger.warning(
                        f"Failed to reconstruct operation for feedback {feedback.operation_id}: {e}"
                    )

            if not operations_with_feedback:
                logger.warning("No valid operations found for feedback records")
                return {"processed": 0, "created": 0, "updated": 0, "failed": 0}

            # Batch learn from feedback
            stats = self.learner.batch_learn(operations_with_feedback)

            logger.info(
                f"Processed {stats['processed']} feedback records: "
                f"{stats['created']} patterns created, {stats['updated']} updated"
            )

            return stats

        except Exception as e:
            logger.error(f"Error processing pending feedback: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process pending feedback: {e}") from e

    def update_pattern_scores(self) -> Dict[str, int]:
        """
        Recalculate scores for all patterns.

        This updates acceptance rates and benefit scores based on current feedback.

        Returns:
            Dictionary with update statistics: {'updated': int, 'total': int}

        Raises:
            RuntimeError: If update fails
        """
        try:
            patterns = self.storage.load_patterns()
            total = len(patterns)
            updated = 0

            logger.info(f"Updating scores for {total} patterns")

            for pattern_id, pattern in patterns.items():
                try:
                    # Deep-copy pattern before mutation to avoid thread-safety issues
                    # (load_patterns returns cached instances that could be modified concurrently)
                    pattern_copy = copy.deepcopy(pattern)

                    # Get latest metrics
                    metric = self.storage.get_pattern_metric(pattern_id)

                    # Recalculate benefit score on copy
                    old_score = pattern_copy.average_benefit_score
                    pattern_copy.average_benefit_score = pattern_copy.calculate_benefit_score(
                        metric
                    )

                    # Save if changed
                    if abs(pattern_copy.average_benefit_score - old_score) > 0.01:
                        self.storage.save_pattern(pattern_copy)
                        updated += 1

                except Exception as e:
                    logger.warning(f"Failed to update score for pattern {pattern_id}: {e}")

            logger.info(f"Updated scores for {updated}/{total} patterns")

            return {"updated": updated, "total": total}

        except Exception as e:
            logger.error(f"Error updating pattern scores: {e}", exc_info=True)
            raise RuntimeError(f"Failed to update pattern scores: {e}") from e

    def cleanup_old_patterns(self, days: int = 90) -> Dict[str, int]:
        """
        Remove patterns that haven't been seen recently.

        Args:
            days: Number of days of inactivity before removal (default: 90)

        Returns:
            Dictionary with cleanup statistics: {'removed': int, 'total': int}

        Raises:
            ValueError: If days is negative
            RuntimeError: If cleanup fails
        """
        if days < 0:
            raise ValueError("days must be non-negative")

        try:
            patterns = self.storage.load_patterns()
            total = len(patterns)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            removed_count = 0

            logger.info(f"Cleaning up patterns older than {days} days (before {cutoff_date})")

            patterns_to_keep: Dict[str, RefactoringPattern] = {}

            for pattern_id, pattern in patterns.items():
                if pattern.last_seen >= cutoff_date:
                    patterns_to_keep[pattern_id] = pattern
                else:
                    removed_count += 1
                    logger.debug(
                        f"Removing old pattern {pattern_id} " f"(last seen: {pattern.last_seen})"
                    )

            # Replace all patterns with only those to keep (truly removes old ones)
            if removed_count > 0:
                # Use replace_patterns to completely replace storage, removing old patterns
                self.storage.replace_patterns(patterns_to_keep)
                logger.info(f"Removed {removed_count} old patterns")

            return {"removed": removed_count, "total": total}

        except Exception as e:
            logger.error(f"Error cleaning up old patterns: {e}", exc_info=True)
            raise RuntimeError(f"Failed to cleanup old patterns: {e}") from e

    def _reconstruct_operation_from_feedback(
        self, feedback: RefactoringFeedback
    ) -> Optional[RefactoringOperation]:
        """
        Reconstruct RefactoringOperation from feedback metadata.

        This is a best-effort reconstruction since we don't store full operations.
        In production, consider storing operation data alongside feedback.

        Args:
            feedback: Feedback record to reconstruct from

        Returns:
            RefactoringOperation if reconstruction succeeds, None otherwise
        """
        try:
            from refactron.core.models import RefactoringOperation

            # Extract basic info from feedback
            file_path = feedback.file_path
            operation_type = feedback.operation_type

            # Try to extract code snippets from metadata if available
            metadata = feedback.metadata or {}
            old_code = metadata.get("old_code", "")
            new_code = metadata.get("new_code", "")

            # Create minimal operation (some fields may be missing)
            operation = RefactoringOperation(
                operation_type=operation_type,
                file_path=file_path,
                line_number=metadata.get("line_number", 0),
                description=metadata.get("description", ""),
                old_code=old_code,
                new_code=new_code,
                risk_score=metadata.get("risk_score", 0.5),
                operation_id=feedback.operation_id,
                reasoning=metadata.get("reasoning"),
                metadata={"code_pattern_hash": feedback.code_pattern_hash},
            )

            return operation

        except Exception as e:
            logger.debug(
                f"Failed to reconstruct operation from feedback {feedback.operation_id}: {e}"
            )
            return None
