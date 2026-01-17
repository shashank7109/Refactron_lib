"""Edge case tests for config loader, including base: null handling."""

import tempfile
from pathlib import Path

import pytest
import yaml

from refactron.core.config_loader import ConfigLoader
from refactron.core.exceptions import ConfigError


class TestConfigLoaderEdgeCases:
    """Tests for edge cases in configuration loading."""

    def test_base_null_with_profile(self) -> None:
        """Test that base: null is handled correctly with profiles."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": None,  # Explicitly null
                "profiles": {
                    "dev": {"log_level": "DEBUG", "show_details": True},
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            result = ConfigLoader.load_from_file(config_path, profile="dev")
            assert result["log_level"] == "DEBUG"  # From profile
            assert result["show_details"] is True  # From profile
            assert result["environment"] == "dev"
            assert "version" in result
        finally:
            config_path.unlink()

    def test_base_null_without_profile(self) -> None:
        """Test that base: null works without profile selection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": None,
                "profiles": {
                    "dev": {"log_level": "DEBUG"},
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            result = ConfigLoader.load_from_file(config_path)
            assert result["version"] == "1.0"
            assert "environment" not in result  # No profile selected
        finally:
            config_path.unlink()

    def test_base_invalid_type(self) -> None:
        """Test that invalid base type (not dict or null) raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": "invalid",  # Should be dict or null
                "profiles": {"dev": {"log_level": "DEBUG"}},
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigError) as exc_info:
                ConfigLoader.load_from_file(config_path, profile="dev")
            assert "base" in str(exc_info.value).lower()
        finally:
            config_path.unlink()

    def test_merge_config_with_none_base(self) -> None:
        """Test _merge_config handles None base gracefully."""
        override = {"log_level": "DEBUG", "show_details": True}
        result = ConfigLoader._merge_config(None, override)
        assert result == override
        assert result["log_level"] == "DEBUG"

    def test_merge_config_nested_merge(self) -> None:
        """Test nested dictionary merging works correctly."""
        base = {
            "custom_rules": {"rule1": "value1", "rule2": "value2"},
            "log_level": "INFO",
        }
        override = {
            "custom_rules": {"rule2": "overridden", "rule3": "value3"},
            "log_level": "DEBUG",
        }
        result = ConfigLoader._merge_config(base, override)
        assert result["custom_rules"]["rule1"] == "value1"  # From base
        assert result["custom_rules"]["rule2"] == "overridden"  # From override
        assert result["custom_rules"]["rule3"] == "value3"  # From override
        assert result["log_level"] == "DEBUG"  # From override

    def test_empty_base_with_profile(self) -> None:
        """Test empty base dict with profile."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "version": "1.0",
                "base": {},
                "profiles": {
                    "dev": {"log_level": "DEBUG"},
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            result = ConfigLoader.load_from_file(config_path, profile="dev")
            assert result["log_level"] == "DEBUG"
            assert result["version"] == "1.0"
        finally:
            config_path.unlink()
