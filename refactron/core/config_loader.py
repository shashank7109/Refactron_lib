"""Enhanced configuration loader with profiles, inheritance, and versioning."""

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from refactron.core.config_validator import ConfigValidator
from refactron.core.exceptions import ConfigError


class ConfigLoader:
    """Loads and merges configuration with support for profiles and inheritance."""

    @staticmethod
    def _normalize_environment(env: Optional[str]) -> Optional[str]:
        """Normalize environment name (development -> dev, production -> prod)."""
        if not env:
            return None
        env_lower = env.lower()
        if env_lower in ("development", "dev"):
            return "dev"
        if env_lower in ("production", "prod"):
            return "prod"
        if env_lower == "staging":
            return "staging"
        return env_lower

    @classmethod
    def load_with_profile(
        cls,
        config_path: Path,
        profile: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load configuration with profile and environment support.

        Args:
            config_path: Path to configuration file
            profile: Profile name to use (dev, staging, prod)
            environment: Environment name (overrides profile if set)

        Returns:
            Merged configuration dictionary

        Raises:
            ConfigError: If config cannot be loaded or is invalid
        """
        if not config_path.exists():
            raise ConfigError(
                f"Configuration file not found: {config_path}",
                config_path=config_path,
                recovery_suggestion="Run 'refactron init' to create a configuration file",
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Invalid YAML syntax in configuration file: {e}",
                config_path=config_path,
                recovery_suggestion=(
                    "Check the YAML syntax in your configuration file, "
                    "including indentation, missing colons, and unclosed brackets"
                ),
            ) from e
        except (IOError, OSError) as e:
            raise ConfigError(
                f"Failed to read configuration file: {e}",
                config_path=config_path,
            ) from e

        # Determine which profile/environment to use
        effective_env = cls._normalize_environment(environment or profile)

        # Start with base configuration
        base_config = config_data.get("base")
        if "base" not in config_data:
            # No "base" key - use entire config (excluding profiles) as base
            base_config = {k: v for k, v in config_data.items() if k != "profiles"}
        else:
            # "base" key exists
            if base_config is None:
                # Explicit null - use empty dict
                base_config = {}
            elif not isinstance(base_config, dict):
                raise ConfigError(
                    "'base' must be a dictionary when provided; use null or omit "
                    "'base' to use top-level keys as base configuration",
                    config_path=config_path,
                    config_key="base",
                )

        # Merge profile-specific configuration
        final_config = copy.deepcopy(base_config)

        # Apply profile overrides if profiles section exists
        if "profiles" in config_data and effective_env:
            profiles = config_data["profiles"]
            if not isinstance(profiles, dict):
                raise ConfigError(
                    "'profiles' must be a dictionary",
                    config_path=config_path,
                    config_key="profiles",
                )

            if effective_env in profiles:
                profile_config = profiles[effective_env]
                if not isinstance(profile_config, dict):
                    raise ConfigError(
                        f"Profile '{effective_env}' must be a dictionary",
                        config_path=config_path,
                        config_key=f"profiles.{effective_env}",
                    )
                final_config = cls._merge_config(final_config, profile_config)
            else:
                # Warn but don't fail - use base config
                pass

        # Set environment in final config
        if effective_env:
            final_config["environment"] = effective_env

        # Ensure version is set
        if "version" not in final_config:
            final_config["version"] = ConfigValidator.CURRENT_VERSION

        # Validate the merged configuration
        ConfigValidator.validate(final_config, config_path)

        result: Dict[str, Any] = final_config
        return result

    @staticmethod
    def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.

        Args:
            base: Base configuration (dict or None; None is treated as an empty dict)
            override: Override configuration

        Returns:
            Merged configuration

        Raises:
            ValueError: If base is not a dict or None
        """
        # Ensure base is a dict (defensive programming)
        if base is None:
            base = {}
        if not isinstance(base, dict):
            raise ValueError(f"Base configuration must be a dict, got {type(base).__name__}")

        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = ConfigLoader._merge_config(result[key], value)
            else:
                # Override with new value
                result[key] = copy.deepcopy(value)

        return result

    @classmethod
    def load_from_file(
        cls,
        config_path: Path,
        profile: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load configuration from file with profile support.

        This is the main entry point for loading configuration.

        Args:
            config_path: Path to configuration file
            profile: Profile name (dev, staging, prod)
            environment: Environment name (overrides profile)

        Returns:
            Configuration dictionary ready for RefactronConfig

        Raises:
            ConfigError: If configuration is invalid
        """
        return cls.load_with_profile(config_path, profile, environment)

    @classmethod
    def migrate_config(
        cls, config_dict: Dict[str, Any], from_version: str, to_version: str
    ) -> Dict[str, Any]:
        """
        Migrate configuration from one version to another.

        Args:
            config_dict: Configuration dictionary
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated configuration dictionary

        Raises:
            ConfigError: If migration is not supported
        """
        if from_version == to_version:
            return config_dict

        # For now, we only support version 1.0
        # Future versions can add migration logic here
        if from_version not in ConfigValidator.SUPPORTED_VERSIONS:
            raise ConfigError(
                f"Cannot migrate from unsupported version '{from_version}'",
                recovery_suggestion=(
                    f"Please update your configuration manually to version {to_version}. "
                    "Run 'refactron init' to generate a new configuration template."
                ),
            )

        # Ensure version is updated
        migrated = copy.deepcopy(config_dict)
        migrated["version"] = to_version

        return migrated
