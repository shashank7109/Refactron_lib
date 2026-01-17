"""Tests for enhanced configuration management features."""

import tempfile
from pathlib import Path

import pytest
import yaml

from refactron.core.config import RefactronConfig
from refactron.core.config_loader import ConfigLoader
from refactron.core.config_templates import ConfigTemplates
from refactron.core.config_validator import ConfigValidator
from refactron.core.exceptions import ConfigError


class TestConfigValidator:
    """Tests for configuration schema validation."""

    def test_validate_valid_config(self) -> None:
        """Test validation of valid configuration."""
        config = {
            "version": "1.0",
            "enabled_analyzers": ["complexity", "security"],
            "max_function_complexity": 10,
        }
        errors = ConfigValidator.validate(config)
        assert len(errors) == 0

    def test_validate_invalid_analyzer(self) -> None:
        """Test validation catches invalid analyzer names."""
        config = {
            "version": "1.0",
            "enabled_analyzers": ["invalid_analyzer"],
        }
        with pytest.raises(ConfigError) as exc_info:
            ConfigValidator.validate(config)
        assert "Invalid analyzer" in str(exc_info.value)

    def test_validate_invalid_version(self) -> None:
        """Test validation catches unsupported version."""
        config = {
            "version": "2.0",
            "enabled_analyzers": ["complexity"],
        }
        with pytest.raises(ConfigError) as exc_info:
            ConfigValidator.validate(config)
        assert "Unsupported config version" in str(exc_info.value)

    def test_validate_invalid_numeric_value(self) -> None:
        """Test validation catches invalid numeric values."""
        config = {
            "version": "1.0",
            "max_function_complexity": -5,  # Invalid: must be >= 1
        }
        with pytest.raises(ConfigError) as exc_info:
            ConfigValidator.validate(config)
        assert "max_function_complexity" in str(exc_info.value)

    def test_validate_invalid_environment(self) -> None:
        """Test validation catches invalid environment."""
        config = {
            "version": "1.0",
            "environment": "invalid_env",
        }
        with pytest.raises(ConfigError) as exc_info:
            ConfigValidator.validate(config)
        assert "Invalid environment" in str(exc_info.value)

    def test_validate_invalid_log_level(self) -> None:
        """Test validation catches invalid log level."""
        config = {
            "version": "1.0",
            "log_level": "INVALID",
        }
        with pytest.raises(ConfigError) as exc_info:
            ConfigValidator.validate(config)
        assert "Invalid log_level" in str(exc_info.value)


class TestConfigLoader:
    """Tests for configuration loading with profiles."""

    def test_load_base_config(self) -> None:
        """Test loading base configuration without profiles."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": {
                    "enabled_analyzers": ["complexity"],
                    "log_level": "INFO",
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded = ConfigLoader.load_from_file(config_path)
            assert loaded["enabled_analyzers"] == ["complexity"]
            assert loaded["log_level"] == "INFO"
        finally:
            config_path.unlink()

    def test_load_with_profile(self) -> None:
        """Test loading configuration with profile override."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": {
                    "enabled_analyzers": ["complexity"],
                    "log_level": "INFO",
                },
                "profiles": {
                    "dev": {
                        "log_level": "DEBUG",
                    },
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded = ConfigLoader.load_from_file(config_path, profile="dev")
            assert loaded["enabled_analyzers"] == ["complexity"]  # From base
            assert loaded["log_level"] == "DEBUG"  # From profile
            assert loaded["environment"] == "dev"
        finally:
            config_path.unlink()

    def test_load_with_environment_override(self) -> None:
        """Test environment overrides profile."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": {
                    "log_level": "INFO",
                },
                "profiles": {
                    "dev": {"log_level": "DEBUG"},
                    "prod": {"log_level": "WARNING"},
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded = ConfigLoader.load_from_file(config_path, profile="dev", environment="prod")
            assert loaded["log_level"] == "WARNING"  # Environment takes precedence
            assert loaded["environment"] == "prod"
        finally:
            config_path.unlink()

    def test_load_legacy_config(self) -> None:
        """Test loading legacy config without base/profiles structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "enabled_analyzers": ["complexity"],
                "log_level": "INFO",
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded = ConfigLoader.load_from_file(config_path)
            assert loaded["enabled_analyzers"] == ["complexity"]
            assert loaded["log_level"] == "INFO"
        finally:
            config_path.unlink()

    def test_merge_nested_config(self) -> None:
        """Test deep merging of nested configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": {
                    "custom_rules": {
                        "rule1": "value1",
                        "rule2": "value2",
                    },
                },
                "profiles": {
                    "dev": {
                        "custom_rules": {
                            "rule2": "overridden",
                            "rule3": "value3",
                        },
                    },
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded = ConfigLoader.load_from_file(config_path, profile="dev")
            assert loaded["custom_rules"]["rule1"] == "value1"  # From base
            assert loaded["custom_rules"]["rule2"] == "overridden"  # From profile
            assert loaded["custom_rules"]["rule3"] == "value3"  # From profile
        finally:
            config_path.unlink()


class TestConfigTemplates:
    """Tests for configuration templates."""

    def test_get_base_template(self) -> None:
        """Test base template generation."""
        template = ConfigTemplates.get_base_template()
        assert "version" in template
        assert "base" in template
        assert "profiles" in template
        assert "dev" in template["profiles"]
        assert "staging" in template["profiles"]
        assert "prod" in template["profiles"]

    def test_get_django_template(self) -> None:
        """Test Django template generation."""
        template = ConfigTemplates.get_django_template()
        assert "migrations" in str(template["base"]["exclude_patterns"])
        assert "django_specific" in template["base"]["custom_rules"]

    def test_get_fastapi_template(self) -> None:
        """Test FastAPI template generation."""
        template = ConfigTemplates.get_fastapi_template()
        assert template["base"]["max_function_complexity"] == 15
        assert template["base"]["max_parameters"] == 10
        assert "fastapi_specific" in template["base"]["custom_rules"]

    def test_get_flask_template(self) -> None:
        """Test Flask template generation."""
        template = ConfigTemplates.get_flask_template()
        assert "flask_specific" in template["base"]["custom_rules"]

    def test_get_template_invalid_framework(self) -> None:
        """Test error handling for invalid framework."""
        with pytest.raises(ValueError) as exc_info:
            ConfigTemplates.get_template("invalid")
        assert "Unsupported framework" in str(exc_info.value)


class TestRefactronConfigEnhanced:
    """Tests for enhanced RefactronConfig with profiles."""

    def test_config_with_version(self) -> None:
        """Test config includes version field."""
        config = RefactronConfig()
        assert config.version == ConfigValidator.CURRENT_VERSION

    def test_config_with_environment(self) -> None:
        """Test config includes environment field."""
        config = RefactronConfig(environment="dev")
        assert config.environment == "dev"

    def test_from_file_with_profile(self) -> None:
        """Test loading config from file with profile."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": {
                    "enabled_analyzers": ["complexity"],
                    "log_level": "INFO",
                },
                "profiles": {
                    "dev": {
                        "log_level": "DEBUG",
                    },
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded_config = RefactronConfig.from_file(config_path, profile="dev")
            assert loaded_config.log_level == "DEBUG"
            assert loaded_config.environment == "dev"
        finally:
            config_path.unlink()

    def test_from_file_with_environment(self) -> None:
        """Test loading config from file with environment."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": {
                    "log_level": "INFO",
                },
                "profiles": {
                    "prod": {
                        "log_level": "WARNING",
                    },
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded_config = RefactronConfig.from_file(config_path, environment="prod")
            assert loaded_config.log_level == "WARNING"
            assert loaded_config.environment == "prod"
        finally:
            config_path.unlink()

    def test_from_file_legacy_format(self) -> None:
        """Test loading legacy config format (backward compatibility)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "enabled_analyzers": ["complexity"],
                "log_level": "INFO",
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            loaded_config = RefactronConfig.from_file(config_path)
            assert loaded_config.enabled_analyzers == ["complexity"]
            assert loaded_config.log_level == "INFO"
            # Version should be set to current
            assert loaded_config.version == ConfigValidator.CURRENT_VERSION
        finally:
            config_path.unlink()

    def test_to_file_includes_version(self) -> None:
        """Test saving config includes version."""
        config = RefactronConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            config.to_file(config_path)
            with open(config_path) as f:
                saved = yaml.safe_load(f)
            assert saved["version"] == ConfigValidator.CURRENT_VERSION
        finally:
            config_path.unlink()


class TestConfigVersioning:
    """Tests for configuration versioning and backward compatibility."""

    def test_version_validation(self) -> None:
        """Test version validation."""
        ConfigValidator.validate_version_compatibility("1.0")
        with pytest.raises(ConfigError):
            ConfigValidator.validate_version_compatibility("2.0")

    def test_migrate_config_same_version(self) -> None:
        """Test migration when versions are the same."""
        config = {"version": "1.0", "log_level": "INFO"}
        migrated = ConfigLoader.migrate_config(config, "1.0", "1.0")
        assert migrated == config

    def test_migrate_config_unsupported_version(self) -> None:
        """Test migration from unsupported version."""
        config = {"version": "0.9", "log_level": "INFO"}
        with pytest.raises(ConfigError):
            ConfigLoader.migrate_config(config, "0.9", "1.0")


class TestConfigErrorMessages:
    """Tests for clear, actionable error messages."""

    def test_validation_error_with_suggestion(self) -> None:
        """Test validation errors include recovery suggestions."""
        config = {
            "version": "1.0",
            "enabled_analyzers": ["invalid"],
        }
        with pytest.raises(ConfigError) as exc_info:
            ConfigValidator.validate(config)
        error = exc_info.value
        assert error.recovery_suggestion is not None
        assert len(error.recovery_suggestion) > 0

    def test_load_error_with_suggestion(self) -> None:
        """Test load errors include recovery suggestions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax: [")
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError) as exc_info:
                ConfigLoader.load_from_file(config_path)
            error = exc_info.value
            assert error.recovery_suggestion is not None
        finally:
            config_path.unlink()
