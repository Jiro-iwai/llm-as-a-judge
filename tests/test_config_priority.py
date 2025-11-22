"""
Integration tests for configuration priority and high-level behavior.

This module tests that configuration loading follows the correct priority:
1. Environment variables (overrides)
2. YAML config file (recommended)
3. Default values (code defaults)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config.app_config import (
    get_app_config,
    get_max_workers,
    get_output_file_names,
    get_timeout,
    load_config,
    reset_config,
)


class TestConfigPriority:
    """Tests for configuration priority order"""

    def test_default_values_when_no_config(self):
        """Test that default values are used when no config file or env vars are set"""
        reset_config()
        # Remove any existing env vars
        env_vars_to_remove = [
            "APP_CONFIG_FILE",
            "APP_TIMEOUT",
            "APP_MAX_RETRIES",
            "APP_MAX_WORKERS",
        ]
        original_env = {}
        for var in env_vars_to_remove:
            if var in os.environ:
                original_env[var] = os.environ[var]
                del os.environ[var]

        try:
            config = load_config()
            # Should use defaults
            assert config["timeout"] == 120  # Default timeout
            assert config["max_retries"] == 3  # Default max_retries
        finally:
            # Restore original env vars
            for var, value in original_env.items():
                os.environ[var] = value

    def test_yaml_overrides_defaults(self):
        """Test that YAML config file overrides default values"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "timeout": 180,
                "max_retries": 5,
                "max_workers": 4,
            }
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            # Remove env vars that might override
            env_vars_to_remove = ["APP_TIMEOUT", "APP_MAX_RETRIES", "APP_MAX_WORKERS"]
            original_env = {}
            for var in env_vars_to_remove:
                if var in os.environ:
                    original_env[var] = os.environ[var]
                    del os.environ[var]

            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                config = load_config()
                assert config["timeout"] == 180  # From YAML
                assert config["max_retries"] == 5  # From YAML
                assert config["max_workers"] == 4  # From YAML

            # Restore original env vars
            for var, value in original_env.items():
                os.environ[var] = value
        finally:
            os.unlink(temp_file)

    def test_env_vars_override_yaml(self):
        """Test that environment variables override YAML config file values"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "timeout": 180,
                "max_retries": 5,
            }
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            with patch.dict(
                os.environ,
                {
                    "APP_CONFIG_FILE": temp_file,
                    "APP_TIMEOUT": "200",  # Override YAML value
                },
            ):
                config = load_config()
                assert config["timeout"] == 200  # From env var (overrides YAML)
                assert config["max_retries"] == 5  # From YAML (not overridden)
        finally:
            os.unlink(temp_file)

    def test_get_timeout_respects_priority(self):
        """Test that get_timeout respects configuration priority"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"timeout": 150}
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            # Remove APP_TIMEOUT if it exists
            original_timeout = os.environ.get("APP_TIMEOUT")
            if "APP_TIMEOUT" in os.environ:
                del os.environ["APP_TIMEOUT"]

            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                reset_config()  # Reset to reload config
                timeout = get_timeout()
                assert timeout == 150  # From YAML

            # Test env var override
            with patch.dict(
                os.environ,
                {
                    "APP_CONFIG_FILE": temp_file,
                    "APP_TIMEOUT": "200",
                },
            ):
                reset_config()  # Reset to reload config with new env var
                timeout = get_timeout()
                assert timeout == 200  # From env var

            # Restore original timeout
            if original_timeout:
                os.environ["APP_TIMEOUT"] = original_timeout
        finally:
            os.unlink(temp_file)

    def test_get_max_workers_respects_priority(self):
        """Test that get_max_workers respects configuration priority"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"max_workers": 4}
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            # Remove APP_MAX_WORKERS if it exists
            original_max_workers = os.environ.get("APP_MAX_WORKERS")
            if "APP_MAX_WORKERS" in os.environ:
                del os.environ["APP_MAX_WORKERS"]

            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                reset_config()  # Reset to reload config
                max_workers = get_max_workers()
                assert max_workers == 4  # From YAML

            # Test env var override
            with patch.dict(
                os.environ,
                {
                    "APP_CONFIG_FILE": temp_file,
                    "APP_MAX_WORKERS": "8",
                },
            ):
                reset_config()  # Reset to reload config with new env var
                max_workers = get_max_workers()
                assert max_workers == 8  # From env var

            # Restore original max_workers
            if original_max_workers:
                os.environ["APP_MAX_WORKERS"] = original_max_workers
        finally:
            os.unlink(temp_file)

    def test_output_files_from_yaml(self):
        """Test that output_files can be configured via YAML"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "output_files": {
                    "evaluation_comparison": "custom/comparison.png",
                    "evaluation_summary": "custom/summary.txt",
                }
            }
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            # Remove any APP_OUTPUT_FILE_* env vars
            env_vars_to_remove = [
                var for var in os.environ.keys() if var.startswith("APP_OUTPUT_FILE_")
            ]
            original_env = {}
            for var in env_vars_to_remove:
                original_env[var] = os.environ[var]
                del os.environ[var]

            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                output_files = get_output_file_names()
                assert output_files["evaluation_comparison"] == "custom/comparison.png"
                assert output_files["evaluation_summary"] == "custom/summary.txt"

            # Restore original env vars
            for var, value in original_env.items():
                os.environ[var] = value
        finally:
            os.unlink(temp_file)

    def test_config_validation_on_load(self):
        """Test that configuration is validated when loaded"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "max_workers": 0,  # Invalid value
                "timeout": -10,  # Invalid value
            }
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                # Load should succeed but validation should catch issues
                config = load_config()
                # Invalid values should still be in config, but get_max_workers should fix them
                assert config["max_workers"] == 0
                # get_max_workers should return None for invalid values
                max_workers = get_max_workers()
                assert max_workers is None  # Should be fixed to None
        finally:
            os.unlink(temp_file)

    def test_get_app_config_caches_result(self):
        """Test that get_app_config caches the configuration"""
        reset_config()
        config1 = get_app_config()
        config2 = get_app_config()
        # Should return the same instance (cached)
        assert config1 is config2

