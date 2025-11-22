"""
Unit tests for config.app_config module.

This module tests the application configuration system that externalizes
hardcoded values like timeout, max_retries, retry_delay, delay, identity, etc.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Import the config module (will be created)
# Note: This will fail until we create config/app_config.py
from src.config.app_config import (
    DEFAULT_CONFIG,
    get_app_config,
    get_timeout,
    get_max_retries,
    get_retry_delay,
    get_api_delay,
    get_default_identity,
    get_output_file_names,
    get_regex_patterns,
    load_config,
    reset_config,
)


class TestLoadConfig:
    """Tests for load_config function"""

    def test_load_config_default(self):
        """Test load_config loads default values when no config file exists"""
        reset_config()
        config = load_config()
        assert config is not None
        assert "timeout" in config
        assert "max_retries" in config
        assert "retry_delay" in config

    def test_load_config_from_file(self):
        """Test load_config loads from YAML file"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "timeout": 180,
                "max_retries": 5,
                "retry_delay": 3,
            }
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                config = load_config()
                assert config["timeout"] == 180
                assert config["max_retries"] == 5
                assert config["retry_delay"] == 3
        finally:
            os.unlink(temp_file)

    def test_load_config_from_env_var(self):
        """Test load_config reads from environment variables"""
        reset_config()
        with patch.dict(
            os.environ,
            {
                "APP_TIMEOUT": "150",
                "APP_MAX_RETRIES": "4",
                "APP_RETRY_DELAY": "2",
            },
        ):
            config = load_config()
            assert config["timeout"] == 150
            assert config["max_retries"] == 4
            assert config["retry_delay"] == 2

    def test_load_config_env_var_overrides_file(self):
        """Test environment variables override config file values"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"timeout": 180, "max_retries": 5}
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            with patch.dict(
                os.environ,
                {
                    "APP_CONFIG_FILE": temp_file,
                    "APP_TIMEOUT": "200",
                },
            ):
                config = load_config()
                assert config["timeout"] == 200  # Env var overrides file
                assert config["max_retries"] == 5  # From file
        finally:
            os.unlink(temp_file)


class TestGetAppConfig:
    """Tests for get_app_config function"""

    def test_get_app_config_returns_dict(self):
        """Test get_app_config returns a dictionary"""
        reset_config()
        config = get_app_config()
        assert isinstance(config, dict)

    def test_get_app_config_same_instance(self):
        """Test get_app_config returns the same config instance"""
        reset_config()
        config1 = get_app_config()
        config2 = get_app_config()
        assert config1 is config2


class TestGetTimeout:
    """Tests for get_timeout function"""

    def test_get_timeout_default(self):
        """Test get_timeout returns default value"""
        reset_config()
        timeout = get_timeout()
        assert isinstance(timeout, int)
        assert timeout > 0

    def test_get_timeout_from_config(self):
        """Test get_timeout reads from config"""
        reset_config()
        with patch.dict(os.environ, {"APP_TIMEOUT": "150"}):
            timeout = get_timeout()
            assert timeout == 150

    def test_get_timeout_with_override(self):
        """Test get_timeout can be overridden"""
        reset_config()
        timeout = get_timeout(override=200)
        assert timeout == 200


class TestGetMaxRetries:
    """Tests for get_max_retries function"""

    def test_get_max_retries_default(self):
        """Test get_max_retries returns default value"""
        reset_config()
        max_retries = get_max_retries()
        assert isinstance(max_retries, int)
        assert max_retries > 0

    def test_get_max_retries_from_config(self):
        """Test get_max_retries reads from config"""
        reset_config()
        with patch.dict(os.environ, {"APP_MAX_RETRIES": "5"}):
            max_retries = get_max_retries()
            assert max_retries == 5


class TestGetRetryDelay:
    """Tests for get_retry_delay function"""

    def test_get_retry_delay_default(self):
        """Test get_retry_delay returns default value"""
        reset_config()
        retry_delay = get_retry_delay()
        assert isinstance(retry_delay, int)
        assert retry_delay > 0

    def test_get_retry_delay_from_config(self):
        """Test get_retry_delay reads from config"""
        reset_config()
        with patch.dict(os.environ, {"APP_RETRY_DELAY": "3"}):
            retry_delay = get_retry_delay()
            assert retry_delay == 3


class TestGetApiDelay:
    """Tests for get_api_delay function"""

    def test_get_api_delay_default(self):
        """Test get_api_delay returns default value"""
        reset_config()
        delay = get_api_delay()
        assert isinstance(delay, float)
        assert delay > 0

    def test_get_api_delay_from_config(self):
        """Test get_api_delay reads from config"""
        reset_config()
        with patch.dict(os.environ, {"APP_API_DELAY": "2.5"}):
            delay = get_api_delay()
            assert delay == 2.5


class TestGetDefaultIdentity:
    """Tests for get_default_identity function"""

    def test_get_default_identity_default(self):
        """Test get_default_identity returns default value"""
        reset_config()
        identity = get_default_identity()
        assert isinstance(identity, str)
        assert len(identity) > 0

    def test_get_default_identity_from_config(self):
        """Test get_default_identity reads from config"""
        reset_config()
        with patch.dict(os.environ, {"APP_DEFAULT_IDENTITY": "TEST_USER"}):
            identity = get_default_identity()
            assert identity == "TEST_USER"


class TestGetOutputFileNames:
    """Tests for get_output_file_names function"""

    def test_get_output_file_names_returns_dict(self):
        """Test get_output_file_names returns a dictionary"""
        reset_config()
        file_names = get_output_file_names()
        assert isinstance(file_names, dict)

    def test_get_output_file_names_has_expected_keys(self):
        """Test get_output_file_names has expected keys"""
        reset_config()
        file_names = get_output_file_names()
        assert "evaluation_comparison" in file_names
        assert "evaluation_distribution" in file_names
        assert "evaluation_boxplot" in file_names
        assert "evaluation_summary" in file_names
        assert "ragas_evaluation_comparison" in file_names
        assert "ragas_evaluation_distribution" in file_names
        assert "ragas_evaluation_boxplot" in file_names
        assert "ragas_evaluation_summary" in file_names
        assert "processing_time_comparison" in file_names
        assert "processing_time_statistics" in file_names
        assert "processing_time_summary" in file_names

    def test_get_output_file_names_from_config(self):
        """Test get_output_file_names reads from config"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "output_files": {
                    "evaluation_comparison": "custom_comparison.png",
                }
            }
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                file_names = get_output_file_names()
                assert file_names["evaluation_comparison"] == "custom_comparison.png"
        finally:
            os.unlink(temp_file)


class TestGetRegexPatterns:
    """Tests for get_regex_patterns function"""

    def test_get_regex_patterns_returns_dict(self):
        """Test get_regex_patterns returns a dictionary"""
        reset_config()
        patterns = get_regex_patterns()
        assert isinstance(patterns, dict)

    def test_get_regex_patterns_has_expected_keys(self):
        """Test get_regex_patterns has expected keys"""
        reset_config()
        patterns = get_regex_patterns()
        assert "model_a_pattern" in patterns
        assert "model_b_pattern" in patterns

    def test_get_regex_patterns_from_config(self):
        """Test get_regex_patterns reads from config"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "regex_patterns": {
                    "model_a_pattern": r"📥 \[custom-model-a\].*?経過時間: ([\d.]+)秒",
                }
            }
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                patterns = get_regex_patterns()
                assert "custom-model-a" in patterns["model_a_pattern"]
        finally:
            os.unlink(temp_file)


class TestConfigIntegration:
    """Integration tests for configuration system"""

    def test_all_config_functions_work_together(self):
        """Test all config functions work together"""
        reset_config()
        timeout = get_timeout()
        max_retries = get_max_retries()
        retry_delay = get_retry_delay()
        delay = get_api_delay()
        identity = get_default_identity()

        assert timeout > 0
        assert max_retries > 0
        assert retry_delay > 0
        assert delay > 0
        assert len(identity) > 0

    def test_config_file_priority(self):
        """Test config file values are used when env vars are not set"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "timeout": 180,
                "max_retries": 5,
                "retry_delay": 3,
                "api_delay": 2.0,
                "default_identity": "CONFIG_USER",
            }
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            # Remove env vars if they exist
            env_vars_to_remove = [
                "APP_TIMEOUT",
                "APP_MAX_RETRIES",
                "APP_RETRY_DELAY",
                "APP_API_DELAY",
                "APP_DEFAULT_IDENTITY",
            ]
            original_env = {}
            for var in env_vars_to_remove:
                if var in os.environ:
                    original_env[var] = os.environ[var]
                    del os.environ[var]

            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                assert get_timeout() == 180
                assert get_max_retries() == 5
                assert get_retry_delay() == 3
                assert get_api_delay() == 2.0
                assert get_default_identity() == "CONFIG_USER"

            # Restore original env vars
            for var, value in original_env.items():
                os.environ[var] = value
        finally:
            os.unlink(temp_file)


class TestDefaultConfigImmutability:
    """Tests to ensure DEFAULT_CONFIG is never mutated by load_config."""

    def test_environment_override_does_not_mutate_default_output_files(self):
        """Environment overrides for output files must not change DEFAULT_CONFIG."""
        reset_config()
        original_value = DEFAULT_CONFIG["output_files"]["evaluation_summary"]

        with patch.dict(
            os.environ, {"APP_OUTPUT_FILE_EVALUATION_SUMMARY": "custom_summary.txt"}
        ):
            load_config()

        reset_config()
        assert (
            DEFAULT_CONFIG["output_files"]["evaluation_summary"] == original_value
        ), "DEFAULT_CONFIG output_files was mutated by load_config"

        # Also ensure new configs fall back to default once env var is cleared
        config = load_config()
        assert (
            config["output_files"]["evaluation_summary"] == original_value
        ), "Config should revert to default evaluation_summary after reset"

    def test_environment_override_does_not_mutate_default_regex_patterns(self):
        """Environment overrides for regex patterns must not change DEFAULT_CONFIG."""
        reset_config()
        original_pattern = DEFAULT_CONFIG["regex_patterns"]["model_a_pattern"]

        with patch.dict(
            os.environ,
            {"APP_REGEX_MODEL_A_PATTERN": r"📥 \[custom\].*?経過時間: ([\d.]+)秒"},
        ):
            load_config()

        reset_config()
        assert (
            DEFAULT_CONFIG["regex_patterns"]["model_a_pattern"] == original_pattern
        ), "DEFAULT_CONFIG regex_patterns was mutated by load_config"

        config = load_config()
        assert (
            config["regex_patterns"]["model_a_pattern"] == original_pattern
        ), "Config should revert to default regex pattern after reset"

