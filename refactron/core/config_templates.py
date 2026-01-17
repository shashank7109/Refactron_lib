"""Configuration templates for common Python frameworks."""

from typing import Dict

from refactron.core.config_validator import ConfigValidator


class ConfigTemplates:
    """Pre-configured templates for common Python frameworks."""

    @staticmethod
    def get_base_template() -> Dict:
        """Get base configuration template."""
        return {
            "version": ConfigValidator.CURRENT_VERSION,
            "base": {
                "enabled_analyzers": [
                    "complexity",
                    "code_smells",
                    "security",
                    "dependency",
                    "dead_code",
                    "type_hints",
                    "performance",
                ],
                "enabled_refactorers": [
                    "extract_method",
                    "extract_constant",
                    "simplify_conditionals",
                    "reduce_parameters",
                    "add_docstring",
                ],
                "max_function_complexity": 10,
                "max_function_length": 50,
                "max_file_length": 500,
                "max_parameters": 5,
                "report_format": "text",
                "show_details": True,
                "require_preview": True,
                "backup_enabled": True,
                "include_patterns": ["*.py"],
                "exclude_patterns": [
                    "**/test_*.py",
                    "**/__pycache__/**",
                    "**/venv/**",
                    "**/env/**",
                    "**/.git/**",
                ],
                "security_ignore_patterns": [
                    "**/test_*.py",
                    "**/tests/**/*.py",
                    "**/*_test.py",
                ],
                "security_min_confidence": 0.5,
                "enable_ast_cache": True,
                "max_ast_cache_size_mb": 100,
                "enable_incremental_analysis": True,
                "enable_parallel_processing": True,
                "max_parallel_workers": None,
                "use_multiprocessing": False,
                "log_level": "INFO",
                "log_format": "json",
                "enable_console_logging": True,
                "enable_file_logging": True,
                "enable_metrics": True,
                "metrics_detailed": True,
                "enable_prometheus": False,
                "prometheus_host": "127.0.0.1",
                "prometheus_port": 9090,
                "enable_telemetry": False,
            },
            "profiles": {
                "dev": {
                    "log_level": "DEBUG",
                    "show_details": True,
                    "enable_metrics": True,
                    "metrics_detailed": True,
                },
                "staging": {
                    "log_level": "INFO",
                    "show_details": True,
                    "enable_metrics": True,
                    "metrics_detailed": False,
                },
                "prod": {
                    "log_level": "WARNING",
                    "show_details": False,
                    "enable_metrics": True,
                    "metrics_detailed": False,
                    "enable_prometheus": True,
                },
            },
        }

    @staticmethod
    def get_django_template() -> Dict:
        """Get Django-specific configuration template."""
        template = ConfigTemplates.get_base_template()
        template["base"].update(
            {
                "exclude_patterns": [
                    "**/test_*.py",
                    "**/tests/**/*.py",
                    "**/*_test.py",
                    "**/__pycache__/**",
                    "**/venv/**",
                    "**/env/**",
                    "**/.git/**",
                    "**/migrations/**",
                    "**/manage.py",
                    "**/settings.py",
                    "**/wsgi.py",
                    "**/asgi.py",
                ],
                "security_ignore_patterns": [
                    "**/test_*.py",
                    "**/tests/**/*.py",
                    "**/*_test.py",
                    "**/migrations/**",
                ],
                "custom_rules": {
                    "django_specific": {
                        "ignore_migrations": True,
                        "ignore_settings": True,
                    },
                },
            }
        )
        return template

    @staticmethod
    def get_fastapi_template() -> Dict:
        """Get FastAPI-specific configuration template."""
        template = ConfigTemplates.get_base_template()
        template["base"].update(
            {
                "exclude_patterns": [
                    "**/test_*.py",
                    "**/tests/**/*.py",
                    "**/*_test.py",
                    "**/__pycache__/**",
                    "**/venv/**",
                    "**/env/**",
                    "**/.git/**",
                ],
                "max_function_complexity": 15,  # FastAPI routes can be more complex
                "max_parameters": 10,  # FastAPI endpoints may have more params
                "custom_rules": {
                    "fastapi_specific": {
                        "allow_async_functions": True,
                        "allow_decorators": True,
                    },
                },
            }
        )
        return template

    @staticmethod
    def get_flask_template() -> Dict:
        """Get Flask-specific configuration template."""
        template = ConfigTemplates.get_base_template()
        template["base"].update(
            {
                "exclude_patterns": [
                    "**/test_*.py",
                    "**/tests/**/*.py",
                    "**/*_test.py",
                    "**/__pycache__/**",
                    "**/venv/**",
                    "**/env/**",
                    "**/.git/**",
                ],
                "custom_rules": {
                    "flask_specific": {
                        "allow_decorators": True,
                        "allow_blueprints": True,
                    },
                },
            }
        )
        return template

    @staticmethod
    def get_template(framework: str) -> Dict:
        """
        Get configuration template for a specific framework.

        Args:
            framework: Framework name (django, fastapi, flask, base)

        Returns:
            Configuration template dictionary

        Raises:
            ValueError: If framework is not supported
        """
        framework_lower = framework.lower()
        if framework_lower == "django":
            return ConfigTemplates.get_django_template()
        elif framework_lower == "fastapi":
            return ConfigTemplates.get_fastapi_template()
        elif framework_lower == "flask":
            return ConfigTemplates.get_flask_template()
        elif framework_lower == "base":
            return ConfigTemplates.get_base_template()
        else:
            raise ValueError(
                f"Unsupported framework '{framework}'. "
                f"Supported frameworks: django, fastapi, flask, base"
            )
