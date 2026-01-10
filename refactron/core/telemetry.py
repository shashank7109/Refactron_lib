"""Opt-in telemetry system for Refactron.

This module provides anonymous usage data collection to understand real-world
usage patterns and performance characteristics. All telemetry is opt-in and
respects user privacy.
"""

import hashlib
import json
import platform
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TelemetryEvent:
    """Represents a single telemetry event."""

    event_type: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    """Collects and manages telemetry data with privacy considerations."""

    def __init__(
        self,
        enabled: bool = False,
        anonymous_id: Optional[str] = None,
        telemetry_file: Optional[Path] = None,
    ):
        """Initialize telemetry collector.

        Args:
            enabled: Whether telemetry collection is enabled
            anonymous_id: Anonymous identifier for this installation
            telemetry_file: Path to file where telemetry data is stored
        """
        self.enabled = enabled
        self.anonymous_id = anonymous_id or self._generate_anonymous_id()

        # Set default telemetry file location
        if telemetry_file is None:
            telemetry_dir = Path.home() / ".refactron" / "telemetry"
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            self.telemetry_file = telemetry_dir / "events.jsonl"
        else:
            self.telemetry_file = telemetry_file

        self.session_id = str(uuid.uuid4())
        self.events: List[TelemetryEvent] = []

    def _generate_anonymous_id(self) -> str:
        """Generate an anonymous identifier for this installation.

        Returns:
            Anonymous identifier string
        """
        # Create a hash based on hardware characteristics only (no hostname)
        # This is anonymous but consistent for the same machine
        machine_info = f"{platform.machine()}{platform.processor()}{sys.platform}"
        return hashlib.sha256(machine_info.encode()).hexdigest()[:16]

    def record_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a telemetry event.

        Args:
            event_type: Type of event (e.g., 'analysis_completed', 'refactoring_applied')
            data: Additional event data (should not contain PII)
        """
        if not self.enabled:
            return

        event_data = data or {}

        # Add system information (anonymous)
        event_data.update(
            {
                "anonymous_id": self.anonymous_id,
                "python_version": (
                    f"{sys.version_info.major}."
                    f"{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "platform": platform.system(),
                "platform_version": platform.release(),
            }
        )

        event = TelemetryEvent(
            event_type=event_type,
            session_id=self.session_id,
            data=event_data,
        )
        self.events.append(event)

    def record_analysis_completed(
        self,
        files_analyzed: int,
        total_time_ms: float,
        issues_found: int,
        analyzers_used: List[str],
    ) -> None:
        """Record an analysis completion event.

        Args:
            files_analyzed: Number of files analyzed
            total_time_ms: Total analysis time in milliseconds
            issues_found: Number of issues found
            analyzers_used: List of analyzers that were used
        """
        self.record_event(
            "analysis_completed",
            {
                "files_analyzed": files_analyzed,
                "total_time_ms": total_time_ms,
                "issues_found": issues_found,
                "analyzers_used": analyzers_used,
            },
        )

    def record_refactoring_applied(
        self,
        operation_type: str,
        files_affected: int,
        total_time_ms: float,
        success: bool,
    ) -> None:
        """Record a refactoring operation event.

        Args:
            operation_type: Type of refactoring operation
            files_affected: Number of files affected
            total_time_ms: Total refactoring time in milliseconds
            success: Whether the refactoring succeeded
        """
        self.record_event(
            "refactoring_applied",
            {
                "operation_type": operation_type,
                "files_affected": files_affected,
                "total_time_ms": total_time_ms,
                "success": success,
            },
        )

    def record_error(
        self,
        error_type: str,
        error_category: str,
        context: Optional[str] = None,
    ) -> None:
        """Record an error event.

        Args:
            error_type: Type of error (generic, no specific error messages)
            error_category: Category of error (e.g., 'analysis', 'refactoring')
            context: Optional context information (should not contain PII)
        """
        self.record_event(
            "error_occurred",
            {
                "error_type": error_type,
                "error_category": error_category,
                "context": context,
            },
        )

    def record_feature_usage(
        self, feature_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a feature usage event.

        Args:
            feature_name: Name of the feature used
            metadata: Optional metadata about feature usage
        """
        self.record_event(
            "feature_used",
            {
                "feature_name": feature_name,
                **(metadata or {}),
            },
        )

    def flush(self) -> None:
        """Write collected events to disk."""
        if not self.enabled or not self.events:
            return

        try:
            # Ensure directory exists
            self.telemetry_file.parent.mkdir(parents=True, exist_ok=True)

            # Append events to JSONL file
            with open(self.telemetry_file, "a", encoding="utf-8") as f:
                for event in self.events:
                    event_dict = {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp,
                        "session_id": event.session_id,
                        "data": event.data,
                    }
                    f.write(json.dumps(event_dict) + "\n")

            # Clear events after writing
            self.events.clear()
        except (IOError, OSError):
            # Silently fail if we can't write telemetry
            pass

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected telemetry events.

        Returns:
            Dictionary containing telemetry summary
        """
        event_counts: Dict[str, int] = {}
        for event in self.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        return {
            "enabled": self.enabled,
            "anonymous_id": self.anonymous_id,
            "session_id": self.session_id,
            "total_events": len(self.events),
            "event_counts": event_counts,
        }


class TelemetryConfig:
    """Configuration for telemetry system."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize telemetry configuration.

        Args:
            config_file: Path to telemetry configuration file
        """
        if config_file is None:
            config_dir = Path.home() / ".refactron"
            config_dir.mkdir(parents=True, exist_ok=True)
            self.config_file = config_dir / "telemetry.json"
        else:
            self.config_file = config_file

        self._load_config()

    def _load_config(self) -> None:
        """Load telemetry configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.enabled = config.get("enabled", False)
                    self.anonymous_id = config.get("anonymous_id")
            except (IOError, json.JSONDecodeError):
                # Use defaults if config is corrupted
                self.enabled = False
                self.anonymous_id = None
        else:
            # Defaults for new installations
            self.enabled = False
            self.anonymous_id = None

    def save_config(self) -> None:
        """Save telemetry configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "enabled": self.enabled,
                        "anonymous_id": self.anonymous_id,
                    },
                    f,
                    indent=2,
                )
        except (IOError, OSError):
            # Silently fail if we can't write config
            pass

    def enable(self, anonymous_id: Optional[str] = None) -> None:
        """Enable telemetry collection.

        Args:
            anonymous_id: Optional anonymous identifier (generated if not provided)
        """
        self.enabled = True
        if anonymous_id:
            self.anonymous_id = anonymous_id
        elif not self.anonymous_id:
            # Generate new anonymous ID without using host-identifying data
            machine_info = f"{platform.machine()}{platform.processor()}{sys.platform}"
            self.anonymous_id = hashlib.sha256(machine_info.encode()).hexdigest()[:16]
        self.save_config()

    def disable(self) -> None:
        """Disable telemetry collection."""
        self.enabled = False
        self.save_config()


# Global telemetry collector instance
_global_telemetry_collector: Optional[TelemetryCollector] = None


def get_telemetry_collector(enabled: Optional[bool] = None) -> TelemetryCollector:
    """Get the global telemetry collector instance.

    Args:
        enabled: Override enabled status (uses config if None)

    Returns:
        Global TelemetryCollector instance
    """
    global _global_telemetry_collector

    if _global_telemetry_collector is None:
        # Load configuration
        config = TelemetryConfig()
        is_enabled = enabled if enabled is not None else config.enabled

        _global_telemetry_collector = TelemetryCollector(
            enabled=is_enabled,
            anonymous_id=config.anonymous_id,
        )

    return _global_telemetry_collector


def enable_telemetry() -> None:
    """Enable telemetry collection globally."""
    config = TelemetryConfig()
    config.enable()

    global _global_telemetry_collector
    _global_telemetry_collector = None  # Reset to pick up new config


def disable_telemetry() -> None:
    """Disable telemetry collection globally."""
    config = TelemetryConfig()
    config.disable()

    global _global_telemetry_collector
    _global_telemetry_collector = None  # Reset to pick up new config
