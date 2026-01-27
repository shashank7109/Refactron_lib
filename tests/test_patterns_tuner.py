"""Tests for RuleTuner (Phase 6: Project-Specific Rule Tuning)."""

import tempfile
from pathlib import Path

import pytest

from refactron.patterns.models import RefactoringFeedback, RefactoringPattern
from refactron.patterns.storage import PatternStorage
from refactron.patterns.tuner import RuleTuner


class IsolatedStorage:
    """Context manager for isolated PatternStorage in tests."""

    def __init__(self) -> None:
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir_obj.name)
        self.storage = PatternStorage(storage_dir=self.tmpdir)

    def __enter__(self) -> PatternStorage:
        return self.storage

    def __exit__(self, *args: object) -> None:
        self._tmpdir_obj.cleanup()


class TestRuleTuner:
    """Test RuleTuner behavior."""

    def test_init_requires_storage(self) -> None:
        """RuleTuner must be initialized with a PatternStorage."""
        with IsolatedStorage() as storage:
            tuner = RuleTuner(storage=storage)
            assert tuner.storage is storage

        with pytest.raises(ValueError, match="PatternStorage cannot be None"):
            RuleTuner(storage=None)  # type: ignore[arg-type]

    def test_analyze_project_patterns_no_feedback(self) -> None:
        """Analyze should return empty patterns when no feedback exists."""
        with IsolatedStorage() as storage:
            tuner = RuleTuner(storage=storage)
            project_path = Path("/tmp/project")

            analysis = tuner.analyze_project_patterns(project_path)

            assert analysis["project_id"]
            assert analysis["project_path"] == str(project_path)
            assert analysis["patterns"] == []

    def test_analyze_project_patterns_with_feedback(self) -> None:
        """Analyze should aggregate project-specific pattern stats."""
        with IsolatedStorage() as storage:
            tuner = RuleTuner(storage=storage)
            project_path = Path("/tmp/project")

            # Create patterns
            pattern1 = RefactoringPattern.create(
                pattern_hash="hash1",
                operation_type="extract_method",
                code_snippet_before="def foo(): pass",
                code_snippet_after="def bar(): pass",
            )
            pattern1.acceptance_rate = 0.8
            storage.save_pattern(pattern1)

            pattern2 = RefactoringPattern.create(
                pattern_hash="hash2",
                operation_type="extract_constant",
                code_snippet_before="x = 42",
                code_snippet_after="ANSWER = 42",
            )
            pattern2.acceptance_rate = 0.3
            storage.save_pattern(pattern2)

            # Create feedback for this project
            fb1 = RefactoringFeedback.create(
                operation_id="op1",
                operation_type="extract_method",
                file_path=project_path / "test1.py",
                action="accepted",
                code_pattern_hash="hash1",
                project_path=project_path,
            )
            fb2 = RefactoringFeedback.create(
                operation_id="op2",
                operation_type="extract_method",
                file_path=project_path / "test1.py",
                action="rejected",
                code_pattern_hash="hash1",
                project_path=project_path,
            )
            fb3 = RefactoringFeedback.create(
                operation_id="op3",
                operation_type="extract_constant",
                file_path=project_path / "test2.py",
                action="rejected",
                code_pattern_hash="hash2",
                project_path=project_path,
            )

            storage.save_feedback(fb1)
            storage.save_feedback(fb2)
            storage.save_feedback(fb3)

            analysis = tuner.analyze_project_patterns(project_path)
            patterns = analysis["patterns"]

            # There should be entries for both patterns
            ids = {p["pattern_id"] for p in patterns}
            assert pattern1.pattern_id in ids
            assert pattern2.pattern_id in ids

            # Find stats for pattern1
            p1_stats = next(p for p in patterns if p["pattern_id"] == pattern1.pattern_id)
            assert p1_stats["project_accepted"] == 1
            assert p1_stats["project_rejected"] == 1
            assert p1_stats["project_total_feedback"] == 2
            assert p1_stats["project_total_decisions"] == 2
            assert abs(p1_stats["project_acceptance_rate"] - 0.5) < 1e-6
            assert p1_stats["global_acceptance_rate"] == pytest.approx(0.8)

    def test_generate_recommendations(self) -> None:
        """Generate basic recommendations based on project stats."""
        with IsolatedStorage() as storage:
            tuner = RuleTuner(storage=storage)
            project_path = Path("/tmp/project")

            # Create patterns
            good_pattern = RefactoringPattern.create(
                pattern_hash="hash_good",
                operation_type="extract_method",
                code_snippet_before="def good(): pass",
                code_snippet_after="def better(): pass",
            )
            bad_pattern = RefactoringPattern.create(
                pattern_hash="hash_bad",
                operation_type="extract_constant",
                code_snippet_before="x = 42",
                code_snippet_after="ANSWER = 42",
            )
            storage.save_pattern(good_pattern)
            storage.save_pattern(bad_pattern)

            # Create feedback: good_pattern mostly accepted, bad_pattern mostly rejected
            for i in range(5):
                fb_good = RefactoringFeedback.create(
                    operation_id=f"good-{i}",
                    operation_type="extract_method",
                    file_path=project_path / "file_good.py",
                    action="accepted",
                    code_pattern_hash="hash_good",
                    project_path=project_path,
                )
                storage.save_feedback(fb_good)

            for i in range(5):
                fb_bad = RefactoringFeedback.create(
                    operation_id=f"bad-{i}",
                    operation_type="extract_constant",
                    file_path=project_path / "file_bad.py",
                    action="rejected",
                    code_pattern_hash="hash_bad",
                    project_path=project_path,
                )
                storage.save_feedback(fb_bad)

            recs = tuner.generate_recommendations(project_path)

            assert recs["project_path"] == str(project_path)
            to_disable = set(recs["to_disable"])
            to_enable = set(recs["to_enable"])
            weights = recs["weights"]

            # Good pattern should be enabled and get higher weight
            assert good_pattern.pattern_id in to_enable
            assert weights[good_pattern.pattern_id] >= 0.8

            # Bad pattern should be disabled and strongly down-weighted
            assert bad_pattern.pattern_id in to_disable
            assert weights[bad_pattern.pattern_id] <= 0.2

    def test_apply_tuning_updates_profile(self) -> None:
        """Apply tuning should update and persist project profile."""
        with IsolatedStorage() as storage:
            tuner = RuleTuner(storage=storage)
            project_path = Path("/tmp/project")

            profile_before = storage.get_project_profile(project_path)

            recs = {
                "to_disable": ["pattern-disable"],
                "to_enable": ["pattern-enable"],
                "weights": {"pattern-weighted": 0.75},
            }

            profile_after = tuner.apply_tuning(project_path, recs)

            # Profile instance should be the same logical project
            assert profile_before.project_id == profile_after.project_id

            # Disabled pattern
            assert "pattern-disable" in profile_after.disabled_patterns

            # Enabled pattern
            assert "pattern-enable" in profile_after.enabled_patterns

            # Weight applied and clamped via ProjectPatternProfile validation
            assert profile_after.get_pattern_weight("pattern-weighted") == pytest.approx(0.75)
