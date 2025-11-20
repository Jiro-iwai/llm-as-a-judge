"""Unit tests for format_clarity_evaluator.py"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from format_clarity_evaluator import (
    parse_final_answer,
    get_model_config,
    extract_scores_from_evaluation,
)


class TestParseFinalAnswer:
    """Tests for parse_final_answer function"""

    def test_parse_final_answer_with_emoji(self):
        """Test parsing final answer with emoji marker"""
        log = """## 📝 Task タスク
---
情報検索

## ✅ Final Answer 回答
---
これは最終回答です。

## 🔗 URLs URL
---
https://example.com"""
        result = parse_final_answer(log)
        assert "これは最終回答です。" in result
        assert "## 🔗 URLs" not in result

    def test_parse_final_answer_without_emoji(self):
        """Test parsing final answer without emoji"""
        log = """## 📝 Task タスク
---
情報検索

## Final Answer 回答
---
これは最終回答です。

## 🔗 URLs URL
---
https://example.com"""
        result = parse_final_answer(log)
        assert "これは最終回答です。" in result

    def test_parse_final_answer_no_match(self):
        """Test parsing when no final answer section found"""
        log = """## 📝 Task タスク
---
情報検索

## 📚 Raw Search Results
---
検索結果です"""
        result = parse_final_answer(log)
        assert result == log  # Should return original log

    def test_parse_final_answer_empty_string(self):
        """Test parsing empty string"""
        result = parse_final_answer("")
        assert result == ""

    def test_parse_final_answer_none(self):
        """Test parsing None value"""
        result = parse_final_answer(None)
        assert result == ""


class TestGetModelConfig:
    """Tests for get_model_config function"""

    def test_get_model_config_gpt5(self):
        """Test getting config for GPT-5"""
        config = get_model_config("gpt-5")
        assert config["temperature"] == 1.0
        assert config["use_max_completion_tokens"] is True

    def test_get_model_config_gpt41(self):
        """Test getting config for GPT-4.1"""
        config = get_model_config("gpt-4.1")
        assert config["temperature"] == 0.7
        assert config["use_max_completion_tokens"] is False

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

    def test_get_model_config_unknown_model(self):
        """Test getting config for unknown model (should return default)"""
        config = get_model_config("unknown-model")
        # Should return default (gpt-4-turbo) config
        assert config["temperature"] == 0.7


class TestExtractScoresFromEvaluation:
    """Tests for extract_scores_from_evaluation function"""

    def test_extract_scores_valid(self):
        """Test extracting scores from valid evaluation JSON"""
        evaluation = {
            "format_clarity_evaluation": {
                "score": 4,
                "justification": "Good formatting match",
            }
        }
        score, justification = extract_scores_from_evaluation(evaluation)
        assert score == 4
        assert justification == "Good formatting match"

    def test_extract_scores_missing_score(self):
        """Test extracting scores when score is missing"""
        evaluation = {
            "format_clarity_evaluation": {
                "justification": "No score provided",
            }
        }
        score, justification = extract_scores_from_evaluation(evaluation)
        assert score is None
        assert justification == "No score provided"

    def test_extract_scores_empty_evaluation(self):
        """Test extracting scores from empty evaluation"""
        evaluation = {}
        score, justification = extract_scores_from_evaluation(evaluation)
        assert score is None
        assert justification == ""

