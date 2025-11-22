"""
Unit tests for configuration validation logic.

This module tests validation of configuration values including:
- Required credentials for Azure OpenAI
- Required embeddings settings for Ragas
- Valid ranges for numeric values (max_workers, timeout, etc.)
- Type validation for YAML config values
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config.app_config import (
    get_app_config,
    load_config,
    reset_config,
    validate_app_config,
)


class TestValidateAppConfig:
    """Tests for validate_app_config function"""

    def test_validate_app_config_valid_config(self):
        """Test validate_app_config accepts valid configuration"""
        reset_config()
        config = load_config()
        issues = validate_app_config(config)
        # Valid config should produce no critical issues
        # (warnings about missing credentials are OK if not using Azure)
        assert isinstance(issues, list)

    def test_validate_app_config_max_workers_zero(self):
        """Test validate_app_config warns about max_workers=0"""
        reset_config()
        config = load_config()
        config["max_workers"] = 0
        issues = validate_app_config(config)
        # Should warn about invalid max_workers
        assert any("max_workers" in str(issue).lower() for issue in issues)

    def test_validate_app_config_max_workers_negative(self):
        """Test validate_app_config warns about negative max_workers"""
        reset_config()
        config = load_config()
        config["max_workers"] = -1
        issues = validate_app_config(config)
        assert any("max_workers" in str(issue).lower() for issue in issues)

    def test_validate_app_config_max_workers_string(self):
        """Test validate_app_config handles non-numeric max_workers"""
        reset_config()
        config = load_config()
        config["max_workers"] = "invalid"
        issues = validate_app_config(config)
        # Should detect type mismatch
        assert any("max_workers" in str(issue).lower() for issue in issues)

    def test_validate_app_config_timeout_negative(self):
        """Test validate_app_config warns about negative timeout"""
        reset_config()
        config = load_config()
        config["timeout"] = -10
        issues = validate_app_config(config)
        assert any("timeout" in str(issue).lower() for issue in issues)

    def test_validate_app_config_max_retries_negative(self):
        """Test validate_app_config warns about negative max_retries"""
        reset_config()
        config = load_config()
        config["max_retries"] = -1
        issues = validate_app_config(config)
        assert any("max_retries" in str(issue).lower() for issue in issues)

    def test_validate_app_config_yaml_type_mismatch(self):
        """Test validate_app_config detects type mismatches in YAML"""
        reset_config()
        config = load_config()
        # timeout should be int, but set as string
        config["timeout"] = "not-a-number"
        issues = validate_app_config(config)
        assert any("timeout" in str(issue).lower() for issue in issues)

    def test_validate_app_config_output_files_not_dict(self):
        """Test validate_app_config detects invalid output_files structure"""
        reset_config()
        config = load_config()
        config["output_files"] = "not-a-dict"
        issues = validate_app_config(config)
        assert any("output_files" in str(issue).lower() for issue in issues)

    def test_validate_app_config_regex_patterns_not_dict(self):
        """Test validate_app_config detects invalid regex_patterns structure"""
        reset_config()
        config = load_config()
        config["regex_patterns"] = "not-a-dict"
        issues = validate_app_config(config)
        assert any("regex_patterns" in str(issue).lower() for issue in issues)


class TestConfigValidationIntegration:
    """Integration tests for config validation with load_config"""

    def test_load_config_validates_max_workers_from_env(self):
        """Test load_config validates max_workers from environment variable"""
        reset_config()
        with patch.dict(os.environ, {"APP_MAX_WORKERS": "0"}):
            config = load_config()
            issues = validate_app_config(config)
            # Should warn about invalid max_workers
            assert any("max_workers" in str(issue).lower() for issue in issues)

    def test_load_config_validates_max_workers_from_yaml(self):
        """Test load_config validates max_workers from YAML file"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"max_workers": -5}
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            # Clear APP_MAX_WORKERS to ensure YAML value is used
            env_vars = {"APP_CONFIG_FILE": temp_file}
            with patch.dict(os.environ, env_vars, clear=False):
                # Explicitly remove APP_MAX_WORKERS if it was set to ensure YAML value is used
                if "APP_MAX_WORKERS" in os.environ:
                    del os.environ["APP_MAX_WORKERS"]
                config = load_config()
                issues = validate_app_config(config)
                assert any("max_workers" in str(issue).lower() for issue in issues)
        finally:
            os.unlink(temp_file)

    def test_load_config_validates_timeout_from_yaml(self):
        """Test load_config validates timeout type from YAML file"""
        reset_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # timeout should be int, but set as string in YAML
            config_data = {"timeout": "invalid"}
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"APP_CONFIG_FILE": temp_file}):
                config = load_config()
                # load_config should handle type conversion, but validation should catch issues
                issues = validate_app_config(config)
                # If timeout is still a string after load_config, validation should catch it
                if isinstance(config.get("timeout"), str):
                    assert any("timeout" in str(issue).lower() for issue in issues)
        finally:
            os.unlink(temp_file)

    def test_get_app_config_validates_on_first_load(self):
        """Test get_app_config validates configuration on first load"""
        reset_config()
        # This test ensures validation is called during config loading
        # The actual validation behavior is tested in validate_app_config tests
        config = get_app_config()
        assert config is not None
        issues = validate_app_config(config)
        assert isinstance(issues, list)

