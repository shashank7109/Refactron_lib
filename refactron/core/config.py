"""Configuration management for Refactron."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from refactron.core.exceptions import ConfigError


@dataclass
class RefactronConfig:
    """Configuration for Refactron analysis and refactoring."""

    # Analysis settings
    enabled_analyzers: List[str] = field(
        default_factory=lambda: [
            "complexity",
            "code_smells",
            "security",
            "dependency",
            "dead_code",
            "type_hints",
            "performance",
        ]
    )

    # Refactoring settings
    enabled_refactorers: List[str] = field(
        default_factory=lambda: [
            "extract_method",
            "extract_constant",
            "simplify_conditionals",
            "reduce_parameters",
            "add_docstring",
        ]
    )

    # Complexity thresholds
    max_function_complexity: int = 10
    max_function_length: int = 50
    max_file_length: int = 500
    max_parameters: int = 5

    # Reporting settings
    report_format: str = "text"  # text, json, html
    show_details: bool = True

    # Safety settings
    require_preview: bool = True
    backup_enabled: bool = True

    # File patterns
    include_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            "**/test_*.py",
            "**/__pycache__/**",
            "**/venv/**",
            "**/env/**",
            "**/.git/**",
        ]
    )

    # Custom rules
    custom_rules: Dict[str, Any] = field(default_factory=dict)

    # Security analyzer settings
    security_ignore_patterns: List[str] = field(
        default_factory=lambda: [
            "**/test_*.py",
            "**/tests/**/*.py",
            "**/*_test.py",
        ]
    )
    security_rule_whitelist: Dict[str, List[str]] = field(default_factory=dict)
    security_min_confidence: float = 0.5  # Minimum confidence to report issues

    # Performance optimization settings
    enable_ast_cache: bool = True
    ast_cache_dir: Optional[Path] = None
    max_ast_cache_size_mb: int = 100

    enable_incremental_analysis: bool = True
    incremental_state_file: Optional[Path] = None

    enable_parallel_processing: bool = True
    max_parallel_workers: Optional[int] = None
    use_multiprocessing: bool = False  # Threading by default (more compatible)

    enable_memory_profiling: bool = False
    memory_optimization_threshold_mb: float = 5.0
    memory_pressure_threshold_percent: float = 80.0
    memory_pressure_threshold_available_mb: float = 500.0
    cache_cleanup_threshold_percent: float = 0.8

    # Logging and monitoring settings
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "json"  # json or text
    log_file: Optional[Path] = None  # If None, uses default location
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB default
    log_backup_count: int = 5
    enable_console_logging: bool = True
    enable_file_logging: bool = True

    # Metrics collection settings
    enable_metrics: bool = True
    metrics_detailed: bool = True  # Track detailed per-file metrics

    # Prometheus settings
    enable_prometheus: bool = False
    prometheus_host: str = "127.0.0.1"
    prometheus_port: int = 9090

    # Telemetry settings
    enable_telemetry: bool = False  # Opt-in only
    telemetry_endpoint: Optional[str] = None  # For future remote submission

    @classmethod
    def from_file(cls, config_path: Path) -> "RefactronConfig":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            RefactronConfig instance with loaded settings

        Raises:
            ConfigError: If config file cannot be loaded or parsed
        """
        if not config_path.exists():
            raise ConfigError(
                f"Configuration file not found: {config_path}",
                config_path=config_path,
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Invalid YAML syntax in configuration file: {e}",
                config_path=config_path,
            ) from e
        except (IOError, OSError) as e:
            raise ConfigError(
                f"Failed to read configuration file: {e}",
                config_path=config_path,
            ) from e

        try:
            return cls(**config_dict)
        except TypeError as e:
            raise ConfigError(
                f"Invalid configuration options: {e}",
                config_path=config_path,
            ) from e

    def to_file(self, config_path: Path) -> None:
        """Save configuration to a YAML file.

        Args:
            config_path: Path where configuration should be saved

        Raises:
            ConfigError: If config file cannot be written
        """
        config_dict = {
            "enabled_analyzers": self.enabled_analyzers,
            "enabled_refactorers": self.enabled_refactorers,
            "max_function_complexity": self.max_function_complexity,
            "max_function_length": self.max_function_length,
            "max_file_length": self.max_file_length,
            "max_parameters": self.max_parameters,
            "report_format": self.report_format,
            "show_details": self.show_details,
            "require_preview": self.require_preview,
            "backup_enabled": self.backup_enabled,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "custom_rules": self.custom_rules,
            "security_ignore_patterns": self.security_ignore_patterns,
            "security_rule_whitelist": self.security_rule_whitelist,
            "security_min_confidence": self.security_min_confidence,
            "enable_ast_cache": self.enable_ast_cache,
            "ast_cache_dir": str(self.ast_cache_dir) if self.ast_cache_dir else None,
            "max_ast_cache_size_mb": self.max_ast_cache_size_mb,
            "enable_incremental_analysis": self.enable_incremental_analysis,
            "incremental_state_file": (
                str(self.incremental_state_file) if self.incremental_state_file else None
            ),
            "enable_parallel_processing": self.enable_parallel_processing,
            "max_parallel_workers": self.max_parallel_workers,
            "use_multiprocessing": self.use_multiprocessing,
            "enable_memory_profiling": self.enable_memory_profiling,
            "memory_optimization_threshold_mb": self.memory_optimization_threshold_mb,
            "memory_pressure_threshold_percent": self.memory_pressure_threshold_percent,
            "memory_pressure_threshold_available_mb": self.memory_pressure_threshold_available_mb,
            "cache_cleanup_threshold_percent": self.cache_cleanup_threshold_percent,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_file": str(self.log_file) if self.log_file else None,
            "log_max_bytes": self.log_max_bytes,
            "log_backup_count": self.log_backup_count,
            "enable_console_logging": self.enable_console_logging,
            "enable_file_logging": self.enable_file_logging,
            "enable_metrics": self.enable_metrics,
            "metrics_detailed": self.metrics_detailed,
            "enable_prometheus": self.enable_prometheus,
            "prometheus_host": self.prometheus_host,
            "prometheus_port": self.prometheus_port,
            "enable_telemetry": self.enable_telemetry,
            "telemetry_endpoint": self.telemetry_endpoint,
        }

        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        except (IOError, OSError) as e:
            raise ConfigError(
                f"Failed to write configuration file: {e}",
                config_path=config_path,
            ) from e

    @classmethod
    def default(cls) -> "RefactronConfig":
        """Return default configuration."""
        return cls()
