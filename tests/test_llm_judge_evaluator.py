"""Unit tests for llm_judge_evaluator.py"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_judge_evaluator import (
    get_model_config,
    is_gpt5,
    create_user_prompt,
)


class TestGetModelConfig:
    """Tests for get_model_config function"""

    def test_get_model_config_gpt5(self):
        """Test getting config for GPT-5"""
        config = get_model_config("gpt-5")
        assert config["temperature"] == 1.0
        assert config["use_max_completion_tokens"] is True
        assert config["timeout"] == 120

    def test_get_model_config_gpt41(self):
        """Test getting config for GPT-4.1"""
        config = get_model_config("gpt-4.1")
        assert config["temperature"] == 0.7
        assert config["use_max_completion_tokens"] is False
        assert config["timeout"] == 120

    def test_get_model_config_gpt4_turbo(self):
        """Test getting config for GPT-4-turbo"""
        config = get_model_config("gpt-4-turbo")
        assert config["temperature"] == 0.7
        assert config["use_max_completion_tokens"] is False

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


class TestIsGpt5:
    """Tests for is_gpt5 function"""

    def test_is_gpt5_true(self):
        """Test identifying GPT-5"""
        assert is_gpt5("gpt-5") is True
        assert is_gpt5("GPT-5") is True
        assert is_gpt5("gpt5") is True
        assert is_gpt5("GPT5") is True

    def test_is_gpt5_false(self):
        """Test identifying non-GPT-5 models"""
        assert is_gpt5("gpt-4.1") is False
        assert is_gpt5("gpt-4-turbo") is False
        assert is_gpt5("claude") is False


class TestCreateUserPrompt:
    """Tests for create_user_prompt function"""

    def test_create_user_prompt_basic(self):
        """Test creating a basic user prompt"""
        question = "テスト質問"
        model_a_response = "モデルAの回答"
        model_b_response = "モデルBの回答"

        prompt = create_user_prompt(question, model_a_response, model_b_response)

        assert question in prompt
        assert model_a_response in prompt
        assert model_b_response in prompt
        assert "Model A" in prompt or "モデルA" in prompt
        assert "Model B" in prompt or "モデルB" in prompt

    def test_create_user_prompt_empty_strings(self):
        """Test creating prompt with empty strings"""
        prompt = create_user_prompt("", "", "")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

