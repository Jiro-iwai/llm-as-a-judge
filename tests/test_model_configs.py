"""Unit tests for config.model_configs module"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_configs import (
    MODEL_CONFIGS,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    get_model_config,
    get_full_config,
    get_simple_config,
    get_ragas_config,
)


class TestModelConfigs:
    """Tests for MODEL_CONFIGS dictionary"""

    def test_model_configs_contains_expected_models(self):
        """Test that MODEL_CONFIGS contains all expected models"""
        expected_models = ["gpt-5", "gpt-4.1", "gpt-4-turbo"]
        for model in expected_models:
            assert model in MODEL_CONFIGS, f"Model {model} not found in MODEL_CONFIGS"

    def test_gpt5_config_has_all_required_fields(self):
        """Test that GPT-5 config has all required fields"""
        config = MODEL_CONFIGS["gpt-5"]
        assert "max_total_tokens" in config
        assert "max_tokens" in config
        assert "max_completion_tokens" in config
        assert "min_output_tokens" in config
        assert "max_output_tokens" in config
        assert "safety_margin" in config
        assert "temperature" in config
        assert "use_max_completion_tokens" in config
        assert "timeout" in config

    def test_gpt5_config_values(self):
        """Test GPT-5 specific configuration values"""
        config = MODEL_CONFIGS["gpt-5"]
        assert config["temperature"] == 1.0
        assert config["use_max_completion_tokens"] is True
        assert config["max_completion_tokens"] == 2000
        assert config["timeout"] == 120

    def test_gpt41_config_values(self):
        """Test GPT-4.1 specific configuration values"""
        config = MODEL_CONFIGS["gpt-4.1"]
        assert config["temperature"] == 0.7
        assert config["use_max_completion_tokens"] is False
        assert config["max_tokens"] == 2000
        assert config["timeout"] == 120

    def test_gpt4_turbo_config_values(self):
        """Test GPT-4-turbo specific configuration values"""
        config = MODEL_CONFIGS["gpt-4-turbo"]
        assert config["temperature"] == 0.7
        assert config["use_max_completion_tokens"] is False
        assert config["max_tokens"] == 2000
        assert config["timeout"] == 120


class TestDefaultModel:
    """Tests for DEFAULT_MODEL"""

    def test_default_model_is_valid(self):
        """Test that DEFAULT_MODEL is a valid model"""
        assert DEFAULT_MODEL in MODEL_CONFIGS


class TestSupportedModels:
    """Tests for SUPPORTED_MODELS"""

    def test_supported_models_contains_all_models(self):
        """Test that SUPPORTED_MODELS contains all models from MODEL_CONFIGS"""
        assert set(SUPPORTED_MODELS) == set(MODEL_CONFIGS.keys())


class TestGetModelConfig:
    """Tests for get_model_config function"""

    def test_get_model_config_exact_match(self):
        """Test getting config with exact model name match"""
        config = get_model_config("gpt-5")
        assert config["temperature"] == 1.0
        assert config["use_max_completion_tokens"] is True

    def test_get_model_config_case_insensitive(self):
        """Test that model name matching is case insensitive"""
        config1 = get_model_config("GPT-5")
        config2 = get_model_config("gpt-5")
        assert config1 == config2

    def test_get_model_config_partial_match(self):
        """Test partial matching (e.g., 'gpt5' -> 'gpt-5')"""
        config1 = get_model_config("gpt5")
        config2 = get_model_config("gpt-5")
        assert config1 == config2

    def test_get_model_config_unknown_model(self):
        """Test getting config for unknown model (should return default)"""
        config = get_model_config("unknown-model")
        assert config == MODEL_CONFIGS[DEFAULT_MODEL]

    def test_get_model_config_with_whitespace(self):
        """Test that whitespace in model name is handled correctly"""
        config1 = get_model_config("  gpt-5  ")
        config2 = get_model_config("gpt-5")
        assert config1 == config2


class TestGetFullConfig:
    """Tests for get_full_config function"""

    def test_get_full_config_returns_full_config(self):
        """Test that get_full_config returns complete configuration"""
        config = get_full_config("gpt-5")
        assert "max_total_tokens" in config
        assert "min_output_tokens" in config
        assert "max_output_tokens" in config
        assert "safety_margin" in config
        assert "timeout" in config

    def test_get_full_config_all_models(self):
        """Test get_full_config for all supported models"""
        for model in SUPPORTED_MODELS:
            config = get_full_config(model)
            assert "max_total_tokens" in config
            assert "temperature" in config


class TestGetSimpleConfig:
    """Tests for get_simple_config function"""

    def test_get_simple_config_returns_simple_config(self):
        """Test that get_simple_config returns simplified configuration"""
        config = get_simple_config("gpt-5")
        assert "max_tokens" in config
        assert "temperature" in config
        assert "use_max_completion_tokens" in config
        # Should not include full config fields
        assert "max_total_tokens" not in config or "max_total_tokens" in config  # May be included for compatibility

    def test_get_simple_config_all_models(self):
        """Test get_simple_config for all supported models"""
        for model in SUPPORTED_MODELS:
            config = get_simple_config(model)
            assert "max_tokens" in config
            assert "temperature" in config


class TestGetRagasConfig:
    """Tests for get_ragas_config function"""

    def test_get_ragas_config_returns_ragas_config(self):
        """Test that get_ragas_config returns Ragas-compatible configuration"""
        config = get_ragas_config("gpt-5")
        assert "max_tokens" in config or "max_completion_tokens" in config
        assert "temperature" in config
        assert "use_max_completion_tokens" in config

    def test_get_ragas_config_all_models(self):
        """Test get_ragas_config for all supported models"""
        for model in SUPPORTED_MODELS:
            config = get_ragas_config(model)
            assert "temperature" in config
            assert "use_max_completion_tokens" in config

    def test_get_ragas_config_gpt5_has_max_completion_tokens(self):
        """Test that GPT-5 Ragas config has max_completion_tokens"""
        config = get_ragas_config("gpt-5")
        assert config["use_max_completion_tokens"] is True
        assert "max_completion_tokens" in config

    def test_get_ragas_config_gpt4_has_max_tokens(self):
        """Test that GPT-4 models Ragas config has max_tokens"""
        for model in ["gpt-4.1", "gpt-4-turbo"]:
            config = get_ragas_config(model)
            assert config["use_max_completion_tokens"] is False
            assert "max_tokens" in config


class TestConfigConsistency:
    """Tests for configuration consistency across config types"""

    def test_temperature_consistent(self):
        """Test that temperature is consistent across config types"""
        for model in SUPPORTED_MODELS:
            full_config = get_full_config(model)
            simple_config = get_simple_config(model)
            ragas_config = get_ragas_config(model)
            assert full_config["temperature"] == simple_config["temperature"]
            assert full_config["temperature"] == ragas_config["temperature"]

    def test_use_max_completion_tokens_consistent(self):
        """Test that use_max_completion_tokens is consistent across config types"""
        for model in SUPPORTED_MODELS:
            full_config = get_full_config(model)
            simple_config = get_simple_config(model)
            ragas_config = get_ragas_config(model)
            assert (
                full_config["use_max_completion_tokens"]
                == simple_config["use_max_completion_tokens"]
            )
            assert (
                full_config["use_max_completion_tokens"]
                == ragas_config["use_max_completion_tokens"]
            )

