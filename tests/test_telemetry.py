"""Tests for telemetry system."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from refactron.core.telemetry import (
    TelemetryCollector,
    TelemetryConfig,
    TelemetryEvent,
    disable_telemetry,
    enable_telemetry,
    get_telemetry_collector,
)


class TestTelemetryEvent:
    """Test TelemetryEvent dataclass."""

    def test_creation(self):
        """Test creating a telemetry event."""
        event = TelemetryEvent(
            event_type="analysis_completed",
            data={"files_analyzed": 10},
        )

        assert event.event_type == "analysis_completed"
        assert event.data["files_analyzed"] == 10
        assert "timestamp" in event.__dict__
        assert "session_id" in event.__dict__


class TestTelemetryCollector:
    """Test TelemetryCollector class."""

    def test_initialization_disabled(self):
        """Test collector initialization with telemetry disabled."""
        collector = TelemetryCollector(enabled=False)

        assert collector.enabled is False
        assert len(collector.events) == 0

    def test_initialization_enabled(self):
        """Test collector initialization with telemetry enabled."""
        with TemporaryDirectory() as tmpdir:
            telemetry_file = Path(tmpdir) / "events.jsonl"
            collector = TelemetryCollector(
                enabled=True,
                telemetry_file=telemetry_file,
            )

            assert collector.enabled is True
            assert collector.telemetry_file == telemetry_file
            assert collector.anonymous_id is not None

    def test_anonymous_id_generation(self):
        """Test anonymous ID generation is consistent."""
        collector1 = TelemetryCollector(enabled=True)
        collector2 = TelemetryCollector(enabled=True)

        # Same machine should generate same anonymous ID
        assert collector1.anonymous_id == collector2.anonymous_id
        # Should be a hex string
        assert len(collector1.anonymous_id) == 16
        assert all(c in "0123456789abcdef" for c in collector1.anonymous_id)

    def test_record_event_disabled(self):
        """Test that events are not recorded when disabled."""
        collector = TelemetryCollector(enabled=False)

        collector.record_event("test_event", {"key": "value"})

        assert len(collector.events) == 0

    def test_record_event_enabled(self):
        """Test recording events when enabled."""
        collector = TelemetryCollector(enabled=True)

        collector.record_event("test_event", {"key": "value"})

        assert len(collector.events) == 1
        event = collector.events[0]
        assert event.event_type == "test_event"
        assert event.data["key"] == "value"
        # Should include system info
        assert "anonymous_id" in event.data
        assert "python_version" in event.data
        assert "platform" in event.data

    def test_record_analysis_completed(self):
        """Test recording analysis completion event."""
        collector = TelemetryCollector(enabled=True)

        collector.record_analysis_completed(
            files_analyzed=10,
            total_time_ms=5000.0,
            issues_found=25,
            analyzers_used=["complexity", "security"],
        )

        assert len(collector.events) == 1
        event = collector.events[0]
        assert event.event_type == "analysis_completed"
        assert event.data["files_analyzed"] == 10
        assert event.data["total_time_ms"] == 5000.0
        assert event.data["issues_found"] == 25
        assert event.data["analyzers_used"] == ["complexity", "security"]

    def test_record_refactoring_applied(self):
        """Test recording refactoring event."""
        collector = TelemetryCollector(enabled=True)

        collector.record_refactoring_applied(
            operation_type="extract_method",
            files_affected=3,
            total_time_ms=1500.0,
            success=True,
        )

        assert len(collector.events) == 1
        event = collector.events[0]
        assert event.event_type == "refactoring_applied"
        assert event.data["operation_type"] == "extract_method"
        assert event.data["files_affected"] == 3
        assert event.data["success"] is True

    def test_record_error(self):
        """Test recording error event."""
        collector = TelemetryCollector(enabled=True)

        collector.record_error(
            error_type="AnalysisError",
            error_category="analysis",
            context="File parsing",
        )

        assert len(collector.events) == 1
        event = collector.events[0]
        assert event.event_type == "error_occurred"
        assert event.data["error_type"] == "AnalysisError"
        assert event.data["error_category"] == "analysis"
        assert event.data["context"] == "File parsing"

    def test_record_feature_usage(self):
        """Test recording feature usage event."""
        collector = TelemetryCollector(enabled=True)

        collector.record_feature_usage(
            feature_name="parallel_processing",
            metadata={"workers": 4},
        )

        assert len(collector.events) == 1
        event = collector.events[0]
        assert event.event_type == "feature_used"
        assert event.data["feature_name"] == "parallel_processing"
        assert event.data["workers"] == 4

    def test_flush_disabled(self):
        """Test that flush does nothing when disabled."""
        with TemporaryDirectory() as tmpdir:
            telemetry_file = Path(tmpdir) / "events.jsonl"
            collector = TelemetryCollector(
                enabled=False,
                telemetry_file=telemetry_file,
            )

            collector.events.append(
                TelemetryEvent(event_type="test", data={})
            )
            collector.flush()

            # File should not be created
            assert not telemetry_file.exists()

    def test_flush_enabled(self):
        """Test flushing events to file."""
        with TemporaryDirectory() as tmpdir:
            telemetry_file = Path(tmpdir) / "events.jsonl"
            collector = TelemetryCollector(
                enabled=True,
                telemetry_file=telemetry_file,
            )

            collector.record_event("event1", {"key1": "value1"})
            collector.record_event("event2", {"key2": "value2"})
            collector.flush()

            # Verify file was created and events were written
            assert telemetry_file.exists()

            with open(telemetry_file, "r") as f:
                lines = f.readlines()

            assert len(lines) == 2

            event1 = json.loads(lines[0])
            event2 = json.loads(lines[1])

            assert event1["event_type"] == "event1"
            assert event2["event_type"] == "event2"

            # Events should be cleared after flush
            assert len(collector.events) == 0

    def test_get_summary(self):
        """Test getting telemetry summary."""
        collector = TelemetryCollector(enabled=True)

        collector.record_event("event1", {})
        collector.record_event("event1", {})
        collector.record_event("event2", {})

        summary = collector.get_summary()

        assert summary["enabled"] is True
        assert summary["total_events"] == 3
        assert summary["event_counts"]["event1"] == 2
        assert summary["event_counts"]["event2"] == 1
        assert "anonymous_id" in summary
        assert "session_id" in summary


class TestTelemetryConfig:
    """Test TelemetryConfig class."""

    def test_initialization_new_config(self):
        """Test initializing with non-existent config file."""
        with TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "telemetry.json"
            config = TelemetryConfig(config_file=config_file)

            # Should default to disabled
            assert config.enabled is False
            assert config.anonymous_id is None

    def test_enable(self):
        """Test enabling telemetry."""
        with TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "telemetry.json"
            config = TelemetryConfig(config_file=config_file)

            config.enable()

            assert config.enabled is True
            assert config.anonymous_id is not None
            assert config_file.exists()

            # Verify saved config
            with open(config_file, "r") as f:
                saved_config = json.load(f)

            assert saved_config["enabled"] is True
            assert saved_config["anonymous_id"] == config.anonymous_id

    def test_disable(self):
        """Test disabling telemetry."""
        with TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "telemetry.json"
            config = TelemetryConfig(config_file=config_file)

            config.enable()
            config.disable()

            assert config.enabled is False
            assert config_file.exists()

            # Verify saved config
            with open(config_file, "r") as f:
                saved_config = json.load(f)

            assert saved_config["enabled"] is False

    def test_load_existing_config(self):
        """Test loading existing config file."""
        with TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "telemetry.json"

            # Create existing config
            with open(config_file, "w") as f:
                json.dump({
                    "enabled": True,
                    "anonymous_id": "test123456789abc",
                }, f)

            config = TelemetryConfig(config_file=config_file)

            assert config.enabled is True
            assert config.anonymous_id == "test123456789abc"

    def test_corrupted_config_handling(self):
        """Test handling corrupted config file."""
        with TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "telemetry.json"

            # Create corrupted config
            with open(config_file, "w") as f:
                f.write("invalid json{")

            config = TelemetryConfig(config_file=config_file)

            # Should fall back to defaults
            assert config.enabled is False
            assert config.anonymous_id is None


def test_get_telemetry_collector():
    """Test global telemetry collector."""
    collector1 = get_telemetry_collector(enabled=False)
    collector2 = get_telemetry_collector()

    # Should return same instance
    assert collector1 is collector2


def test_enable_disable_telemetry():
    """Test global enable/disable functions."""
    with TemporaryDirectory() as tmpdir:
        # Mock home directory
        import os
        original_home = os.environ.get("HOME")
        os.environ["HOME"] = tmpdir

        try:
            # Enable telemetry
            enable_telemetry()
            config = TelemetryConfig()
            assert config.enabled is True

            # Disable telemetry
            disable_telemetry()
            config = TelemetryConfig()
            assert config.enabled is False
        finally:
            # Restore original HOME (or remove it if it was unset)
            if original_home is not None:
                os.environ["HOME"] = original_home
            else:
                os.environ.pop("HOME", None)
