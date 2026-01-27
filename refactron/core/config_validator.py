"""Configuration schema validation for Refactron."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from refactron.core.exceptions import ConfigError


class ConfigValidator:
    """Validates Refactron configuration against schema."""

    # Current config version
    CURRENT_VERSION = "1.0"

    # Supported config versions (for backward compatibility)
    SUPPORTED_VERSIONS = {"1.0"}

    # Valid analyzer names
    VALID_ANALYZERS = {
        "complexity",
        "code_smells",
        "security",
        "dependency",
        "dead_code",
        "type_hints",
        "performance",
    }

    # Valid refactorer names
    VALID_REFACTORERS = {
        "extract_method",
        "extract_constant",
        "simplify_conditionals",
        "reduce_parameters",
        "add_docstring",
    }

    # Valid log levels
    VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    # Valid report formats
    VALID_REPORT_FORMATS = {"text", "json", "html"}

    # Valid environments
    VALID_ENVIRONMENTS = {"dev", "staging", "prod", "development", "production"}

    @classmethod
    def validate(cls, config_dict: Dict[str, Any], config_path: Optional[Path] = None) -> List[str]:
        """
        Validate configuration dictionary against schema.

        Args:
            config_dict: Configuration dictionary to validate
            config_path: Optional path to config file for error messages

        Returns:
            List of validation error messages (empty if valid)

        Raises:
            ConfigError: If validation fails with actionable error messages
        """
        errors: List[str] = []

        # Validate version
        version = config_dict.get("version", cls.CURRENT_VERSION)
        if version not in cls.SUPPORTED_VERSIONS:
            errors.append(
                f"Unsupported config version '{version}'. "
                f"Supported versions: {', '.join(cls.SUPPORTED_VERSIONS)}. "
                f"Current version: {cls.CURRENT_VERSION}"
            )

        # Validate environment if present (None is allowed)
        if "environment" in config_dict:
            env = config_dict["environment"]
            if env is not None and env not in cls.VALID_ENVIRONMENTS:
                errors.append(
                    f"Invalid environment '{env}'. "
                    f"Valid environments: {', '.join(sorted(cls.VALID_ENVIRONMENTS))}"
                )

        # Validate enabled_analyzers
        if "enabled_analyzers" in config_dict:
            analyzers = config_dict["enabled_analyzers"]
            if not isinstance(analyzers, list):
                errors.append("'enabled_analyzers' must be a list")
            else:
                invalid = set(analyzers) - cls.VALID_ANALYZERS
                if invalid:
                    errors.append(
                        f"Invalid analyzer(s): {', '.join(sorted(invalid))}. "
                        f"Valid analyzers: {', '.join(sorted(cls.VALID_ANALYZERS))}"
                    )

        # Validate enabled_refactorers
        if "enabled_refactorers" in config_dict:
            refactorers = config_dict["enabled_refactorers"]
            if not isinstance(refactorers, list):
                errors.append("'enabled_refactorers' must be a list")
            else:
                invalid = set(refactorers) - cls.VALID_REFACTORERS
                if invalid:
                    errors.append(
                        f"Invalid refactorer(s): {', '.join(sorted(invalid))}. "
                        f"Valid refactorers: {', '.join(sorted(cls.VALID_REFACTORERS))}"
                    )

        # Validate numeric thresholds
        numeric_fields = {
            "max_function_complexity": (int, 1, 100, False),  # allow_none
            "max_function_length": (int, 1, 1000, False),
            "max_file_length": (int, 1, 10000, False),
            "max_parameters": (int, 1, 50, False),
            "max_ast_cache_size_mb": (int, 1, 10000, False),
            "max_parallel_workers": (int, 1, 128, True),  # None means use system default
            "log_max_bytes": (int, 1024, None, False),  # At least 1KB
            "log_backup_count": (int, 1, 100, False),
            "prometheus_port": (int, 1, 65535, False),
        }

        for field, (field_type, min_val, max_val, allow_none) in numeric_fields.items():
            if field in config_dict:
                value = config_dict[field]
                if value is None:
                    if not allow_none:
                        errors.append(f"'{field}' cannot be None")
                elif not isinstance(value, field_type):
                    errors.append(
                        f"'{field}' must be {field_type.__name__}, got {type(value).__name__}"
                    )
                elif min_val is not None and value < min_val:
                    errors.append(f"'{field}' must be >= {min_val}, got {value}")
                elif max_val is not None and value > max_val:
                    errors.append(f"'{field}' must be <= {max_val}, got {value}")

        # Validate float fields
        float_fields = {
            "security_min_confidence": (0.0, 1.0),
            "memory_optimization_threshold_mb": (0.0, None),
            "memory_pressure_threshold_percent": (0.0, 100.0),
            "memory_pressure_threshold_available_mb": (0.0, None),
            "cache_cleanup_threshold_percent": (0.0, 1.0),
        }

        for float_field, value_range in float_fields.items():
            if float_field in config_dict:
                float_min: Optional[float] = value_range[0]
                float_max: Optional[float] = value_range[1]
                float_value_raw = config_dict[float_field]
                if not isinstance(float_value_raw, (int, float)):
                    errors.append(
                        f"'{float_field}' must be a number, "
                        f"got {type(float_value_raw).__name__}"
                    )
                else:
                    float_value: float = float(float_value_raw)
                    if float_min is not None and float_value < float_min:
                        errors.append(f"'{float_field}' must be >= {float_min}, got {float_value}")
                    elif float_max is not None and float_value > float_max:
                        errors.append(f"'{float_field}' must be <= {float_max}, got {float_value}")

        # Validate boolean fields
        boolean_fields = {
            "show_details",
            "require_preview",
            "backup_enabled",
            "enable_ast_cache",
            "enable_incremental_analysis",
            "enable_parallel_processing",
            "use_multiprocessing",
            "enable_memory_profiling",
            "enable_console_logging",
            "enable_file_logging",
            "enable_metrics",
            "metrics_detailed",
            "enable_prometheus",
            "enable_telemetry",
            "enable_pattern_learning",
            "pattern_learning_enabled",
            "pattern_ranking_enabled",
        }

        for field in boolean_fields:
            if field in config_dict:
                value = config_dict[field]
                if not isinstance(value, bool):
                    errors.append(f"'{field}' must be a boolean, got {type(value).__name__}")

        # Validate string fields with specific values
        if "log_level" in config_dict:
            log_level = config_dict["log_level"]
            if log_level not in cls.VALID_LOG_LEVELS:
                errors.append(
                    f"Invalid log_level '{log_level}'. "
                    f"Valid levels: {', '.join(sorted(cls.VALID_LOG_LEVELS))}"
                )

        if "log_format" in config_dict:
            log_format = config_dict["log_format"]
            if log_format not in {"json", "text"}:
                errors.append(f"Invalid log_format '{log_format}'. Valid formats: json, text")

        if "report_format" in config_dict:
            report_format = config_dict["report_format"]
            if report_format not in cls.VALID_REPORT_FORMATS:
                errors.append(
                    f"Invalid report_format '{report_format}'. "
                    f"Valid formats: {', '.join(sorted(cls.VALID_REPORT_FORMATS))}"
                )

        # Validate list fields
        list_fields = {
            "include_patterns",
            "exclude_patterns",
            "security_ignore_patterns",
        }

        for field in list_fields:
            if field in config_dict:
                value = config_dict[field]
                if not isinstance(value, list):
                    errors.append(f"'{field}' must be a list")
                elif value and not all(isinstance(item, str) for item in value):
                    errors.append(f"All items in '{field}' must be strings")

        # Validate dict fields
        dict_fields = {"custom_rules", "security_rule_whitelist"}

        for field in dict_fields:
            if field in config_dict:
                value = config_dict[field]
                if not isinstance(value, dict):
                    errors.append(f"'{field}' must be a dictionary")

        # Validate path fields
        path_fields = {
            "ast_cache_dir",
            "incremental_state_file",
            "log_file",
            "pattern_storage_dir",
        }

        for field in path_fields:
            if field in config_dict:
                value = config_dict[field]
                if value is not None and not isinstance(value, (str, Path)):
                    errors.append(
                        f"'{field}' must be a string path or null, got {type(value).__name__}"
                    )

        # Validate host/port combinations
        if "prometheus_host" in config_dict:
            host = config_dict["prometheus_host"]
            if not isinstance(host, str) or not host.strip():
                errors.append("'prometheus_host' must be a non-empty string")

        # If errors found, raise ConfigError with actionable messages
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            recovery = (
                "Review the error messages above and correct the configuration. "
                "Run 'refactron init' to generate a valid template, or see "
                "documentation for the correct configuration format."
            )
            raise ConfigError(
                error_msg,
                config_path=config_path,
                recovery_suggestion=recovery,
            )

        return []

    @classmethod
    def validate_version_compatibility(
        cls, version: str, config_path: Optional[Path] = None
    ) -> None:
        """
        Validate config version and provide migration guidance if needed.

        Args:
            version: Config version to validate
            config_path: Optional path for error messages

        Raises:
            ConfigError: If version is incompatible
        """
        if version not in cls.SUPPORTED_VERSIONS:
            raise ConfigError(
                f"Unsupported configuration version '{version}'. "
                f"Supported versions: {', '.join(cls.SUPPORTED_VERSIONS)}. "
                f"Current version: {cls.CURRENT_VERSION}",
                config_path=config_path,
                recovery_suggestion=(
                    f"Update your configuration to version {cls.CURRENT_VERSION}. "
                    "Run 'refactron init' to generate a new configuration file, "
                    "or manually update the 'version' field in your config."
                ),
            )
