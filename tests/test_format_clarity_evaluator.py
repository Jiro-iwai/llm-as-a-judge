"""Unit tests for format_clarity_evaluator.py"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from format_clarity_evaluator import (
    parse_final_answer,
    get_model_config,
    extract_scores_from_evaluation,
    create_user_prompt,
    call_judge_model,
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


class TestCreateUserPrompt:
    """Tests for create_user_prompt function"""

    def test_create_user_prompt_basic(self):
        """Test creating a basic user prompt"""
        question = "テスト質問"
        claude_35_answer = "Claude 3.5の回答"
        claude_45_answer = "Claude 4.5の回答"

        prompt = create_user_prompt(question, claude_35_answer, claude_45_answer)

        assert question in prompt
        assert claude_35_answer in prompt
        assert claude_45_answer in prompt
        assert "Claude 3.5" in prompt
        assert "Claude 4.5" in prompt

    def test_create_user_prompt_empty_strings(self):
        """Test creating prompt with empty strings"""
        prompt = create_user_prompt("", "", "")
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestCallJudgeModel:
    """Tests for call_judge_model function"""

    def test_call_judge_model_success(self):
        """Test successful API call"""
        # Create a mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(
                    content=json.dumps(
                        {
                            "format_clarity_evaluation": {
                                "score": 4,
                                "justification": "Good formatting match",
                            }
                        }
                    )
                ),
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            is_azure=False,
        )

        assert result is not None
        assert "format_clarity_evaluation" in result
        assert result["format_clarity_evaluation"]["score"] == 4

    def test_call_judge_model_json_decode_error(self):
        """Test handling JSON decode error"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(finish_reason="stop", message=Mock(content="invalid json"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        # Should return None after retries exhausted
        assert result is None

    def test_call_judge_model_empty_response(self):
        """Test handling empty response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(finish_reason="stop", message=Mock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_missing_key(self):
        """Test handling response missing required key"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(content=json.dumps({"wrong_key": "value"})),
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_truncated_response(self):
        """Test handling truncated response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                finish_reason="length",
                message=Mock(
                    content=json.dumps(
                        {
                            "format_clarity_evaluation": {
                                "score": 4,
                                "justification": "Good formatting match",
                            }
                        }
                    )
                ),
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
        )

        assert result is not None
        assert result["format_clarity_evaluation"]["score"] == 4

    def test_call_judge_model_api_exception(self):
        """Test handling API exception"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_timeout_error(self):
        """Test handling timeout error"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timeout")

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_json_decode_with_debug_info(self):
        """Test JSON decode error with debug information"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(finish_reason="stop", message=Mock(content="invalid json"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_response_without_choices(self):
        """Test handling response without choices"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_exception_with_backoff(self):
        """Test exception handling with exponential backoff"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "Claude 3.5の回答",
            "Claude 4.5の回答",
            model_name="gpt-4-turbo",
            max_retries=2,
            retry_delay=0.01,  # Short delay for testing
        )

        assert result is None
        # Should have retried multiple times
        assert mock_client.chat.completions.create.call_count == 2

