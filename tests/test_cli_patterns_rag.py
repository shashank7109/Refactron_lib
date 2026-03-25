"""Coverage tests for CLI patterns and rag commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from refactron.cli.patterns import patterns
from refactron.cli.rag import rag


@dataclass
class _FakeStats:
    total_files: int = 2
    total_chunks: int = 4
    index_path: str = ".rag/chroma"
    embedding_model: str = "mini"
    chunk_types: dict = None

    def __post_init__(self):
        if self.chunk_types is None:
            self.chunk_types = {"function": 1, "class": 1, "method": 1, "module": 1}


def test_patterns_analyze_recommend_tune_and_profile(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setattr("refactron.cli.patterns._load_config", lambda *a, **k: {})
    monkeypatch.setattr("refactron.cli.patterns._setup_logging", lambda *a, **k: None)
    monkeypatch.setattr("refactron.cli.patterns._auth_banner", lambda *a, **k: None)
    monkeypatch.setattr(
        "refactron.cli.patterns._get_pattern_storage_from_config", lambda cfg: object()
    )

    class _FakeTuner:
        def __init__(self, storage):  # noqa: ARG002
            pass

        def analyze_project_patterns(self, project_root):
            return {
                "project_path": str(project_root),
                "patterns": [
                    {
                        "pattern_id": "p1",
                        "operation_type": "refactor",
                        "project_acceptance_rate": 0.8,
                        "project_total_decisions": 10,
                        "global_acceptance_rate": 0.6,
                        "enabled": True,
                        "weight": 1.2,
                    }
                ],
            }

        def generate_recommendations(self, project_root):
            return {
                "project_path": str(project_root),
                "to_disable": ["p2"],
                "to_enable": ["p3"],
                "weights": {"p1": 0.75},
            }

        def apply_tuning(self, project_root, recs):  # noqa: ARG002
            return SimpleNamespace(project_path=str(project_root), project_id="proj-1")

    monkeypatch.setattr("refactron.cli.patterns.RuleTuner", _FakeTuner)

    class _FakeProfile:
        project_id = "proj-1"
        project_path = "x/y"
        last_updated = datetime.now(timezone.utc)
        enabled_patterns = {"p1"}
        disabled_patterns = {"p2"}
        pattern_weights = {"p1": 1.0, "p2": 0.6}

        def is_pattern_enabled(self, pid):
            return pid in self.enabled_patterns

        def get_pattern_weight(self, pid, default=1.0):
            return self.pattern_weights.get(pid, default)

    monkeypatch.setattr(
        "refactron.cli.patterns._get_pattern_storage_from_config",
        lambda cfg: SimpleNamespace(get_project_profile=lambda *_: _FakeProfile()),
    )

    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(patterns, ["analyze"]).exit_code == 0
        assert runner.invoke(patterns, ["recommend"]).exit_code == 0
        assert runner.invoke(patterns, ["tune", "--auto"]).exit_code == 0
        assert runner.invoke(patterns, ["profile"]).exit_code == 0


def test_patterns_empty_and_error_paths(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.setattr("refactron.cli.patterns._load_config", lambda *a, **k: {})
    monkeypatch.setattr("refactron.cli.patterns._setup_logging", lambda *a, **k: None)
    monkeypatch.setattr("refactron.cli.patterns._auth_banner", lambda *a, **k: None)

    class _NoDataTuner:
        def __init__(self, storage):  # noqa: ARG002
            pass

        def analyze_project_patterns(self, project_root):  # noqa: ARG002
            return {"project_path": "x", "patterns": []}

        def generate_recommendations(self, project_root):  # noqa: ARG002
            return {"project_path": "x", "to_disable": [], "to_enable": [], "weights": {}}

    monkeypatch.setattr(
        "refactron.cli.patterns._get_pattern_storage_from_config", lambda cfg: object()
    )
    monkeypatch.setattr("refactron.cli.patterns.RuleTuner", _NoDataTuner)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(patterns, ["analyze"]).exit_code == 0
        assert runner.invoke(patterns, ["recommend"]).exit_code == 0
        assert runner.invoke(patterns, ["tune", "--auto"]).exit_code == 0


def test_rag_index_search_status_paths(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setattr("refactron.cli.rag._setup_logging", lambda *a, **k: None)
    monkeypatch.setattr("refactron.cli.rag._auth_banner", lambda *a, **k: None)

    fake_workspace = SimpleNamespace(local_path=str(tmp_path), repo_full_name="u/repo")
    monkeypatch.setattr(
        "refactron.cli.rag.WorkspaceManager",
        lambda: SimpleNamespace(get_workspace_by_path=lambda *_: fake_workspace),
    )

    class _FakeIndexer:
        def __init__(self, local_path):  # noqa: ARG002
            pass

        def index_repository(self, local_path, summarize=False):  # noqa: ARG002
            return _FakeStats()

        def get_stats(self):
            return _FakeStats()

    class _FakeRetriever:
        def __init__(self, local_path):  # noqa: ARG002
            pass

        def retrieve_similar(self, query, top_k=5, chunk_type=None):  # noqa: ARG002
            return [
                SimpleNamespace(
                    name="f",
                    chunk_type="function",
                    file_path="a.py",
                    line_range=(1, 2),
                    content="def f(): pass",
                    distance=0.1,
                )
            ]

    monkeypatch.setattr("refactron.cli.rag.RAGIndexer", _FakeIndexer)
    monkeypatch.setattr("refactron.cli.rag.ContextRetriever", _FakeRetriever)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(rag, ["index"]).exit_code == 0
        assert runner.invoke(rag, ["search", "find function"]).exit_code == 0
        assert runner.invoke(rag, ["status"]).exit_code == 0


def test_rag_not_connected_and_index_error_paths(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.setattr("refactron.cli.rag._setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        "refactron.cli.rag.WorkspaceManager",
        lambda: SimpleNamespace(get_workspace_by_path=lambda *_: None),
    )

    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(rag, ["index"]).exit_code == 1
        assert runner.invoke(rag, ["search", "q"]).exit_code == 1
        assert runner.invoke(rag, ["status"]).exit_code == 1
