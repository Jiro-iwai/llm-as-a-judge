"""Unit tests for llm_judge_evaluator.py"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.llm_judge_evaluator import (
    get_model_config,
    is_gpt5,
    create_user_prompt,
    log_info,
    log_section,
    log_warning,
    log_error,
    log_success,
    call_judge_model,
    extract_scores_from_evaluation,
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


class TestLogFunctions:
    """Tests for log functions"""

    def test_log_info(self):
        """Test log_info function"""
        import io
        import logging
        from src.utils.logging_config import setup_logging, reset_logger

        reset_logger()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_info("Test message")
        output = stream.getvalue()
        assert "Test message" in output

    def test_log_info_with_indent(self):
        """Test log_info with indentation"""
        import io
        import logging
        from src.utils.logging_config import setup_logging, reset_logger

        reset_logger()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_info("Test message", indent=2)
        output = stream.getvalue()
        assert "Test message" in output
        assert output.startswith("    ")  # 2 indents = 4 spaces

    def test_log_section(self):
        """Test log_section function"""
        import io
        import logging
        from src.utils.logging_config import setup_logging, reset_logger

        reset_logger()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_section("Test Section")
        output = stream.getvalue()
        assert "Test Section" in output
        assert "=" * 70 in output

    def test_log_warning(self):
        """Test log_warning function"""
        import io
        import logging
        from src.utils.logging_config import setup_logging, reset_logger

        reset_logger()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        log_warning("Warning message")
        output = stream.getvalue()
        assert "Warning message" in output

    def test_log_error(self):
        """Test log_error function"""
        import io
        import logging
        from src.utils.logging_config import setup_logging, reset_logger

        reset_logger()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        log_error("Error message")
        output = stream.getvalue()
        assert "Error message" in output
        assert "❌" in output

    def test_log_success(self):
        """Test log_success function"""
        import io
        import logging
        from src.utils.logging_config import setup_logging, reset_logger

        reset_logger()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = setup_logging(reset=True)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        log_success("Success message")
        output = stream.getvalue()
        assert "Success message" in output
        assert "✓" in output


class TestCallJudgeModel:
    """Tests for call_judge_model function"""

    def test_call_judge_model_success(self):
        """Test successful API call"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(
                    content=json.dumps(
                        {
                            "model_a_evaluation": {
                                "overall_score": 4,
                                "justification": "Good response",
                            },
                            "model_b_evaluation": {
                                "overall_score": 5,
                                "justification": "Excellent response",
                            },
                        }
                    )
                ),
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "モデルAの回答",
            "モデルBの回答",
            model_name="gpt-4-turbo",
            is_azure=False,
        )

        assert result is not None
        assert "model_a_evaluation" in result
        assert "model_b_evaluation" in result

    def test_call_judge_model_json_decode_error(self):
        """Test handling JSON decode error"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(finish_reason="stop", message=Mock(content="invalid json"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "モデルAの回答",
            "モデルBの回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_api_exception(self):
        """Test handling API exception"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "モデルAの回答",
            "モデルBの回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_timeout_error(self):
        """Test handling TimeoutError"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timeout")

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "モデルAの回答",
            "モデルBの回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_max_tokens_error(self):
        """Test handling max_tokens error"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("max_tokens limit exceeded")

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "モデルAの回答",
            "モデルBの回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None

    def test_call_judge_model_input_too_long_error(self):
        """Test handling input too long error"""
        mock_client = Mock()
        # Simulate ValueError for input too long
        mock_client.chat.completions.create.side_effect = ValueError("入力トークン数が長すぎます")

        result = call_judge_model(
            mock_client,
            "テスト質問",
            "モデルAの回答",
            "モデルBの回答",
            model_name="gpt-4-turbo",
            max_retries=1,
        )

        assert result is None


class TestExtractScoresFromEvaluation:
    """Tests for extract_scores_from_evaluation function"""

    def test_extract_scores_complete(self):
        """Test extracting all scores from complete evaluation"""
        evaluation = {
            "model_a_evaluation": {
                "citation_score": {"score": 4, "justification": "Good citations"},
                "relevance_score": {"score": 5, "justification": "Excellent relevance"},
                "react_performance_thought_score": {"score": 3, "justification": "OK"},
                "rag_retrieval_observation_score": {"score": 4, "justification": "Good"},
                "information_integration_score": {"score": 5, "justification": "Perfect"},
            },
            "model_b_evaluation": {
                "citation_score": {"score": 3, "justification": "OK citations"},
                "relevance_score": {"score": 4, "justification": "Good relevance"},
                "react_performance_thought_score": {"score": 4, "justification": "Good"},
                "rag_retrieval_observation_score": {"score": 3, "justification": "OK"},
                "information_integration_score": {"score": 4, "justification": "Good"},
            },
        }

        result_a = extract_scores_from_evaluation(evaluation, "model_a_evaluation")
        result_b = extract_scores_from_evaluation(evaluation, "model_b_evaluation")

        assert result_a["citation_score"] == 4
        assert result_a["relevance_score"] == 5
        assert result_b["citation_score"] == 3
        assert result_b["relevance_score"] == 4

    def test_extract_scores_missing_keys(self):
        """Test extracting scores when some keys are missing"""
        evaluation = {
            "model_a_evaluation": {
                "citation_score": {"score": 4, "justification": "Good"},
                # Missing other scores
            },
        }

        result = extract_scores_from_evaluation(evaluation, "model_a_evaluation")

        assert result["citation_score"] == 4
        # Missing scores should be None or not present
        assert "relevance_score" not in result or result.get("relevance_score") is None

    def test_extract_scores_missing_score_field(self):
        """Test extracting scores when score field is missing"""
        evaluation = {
            "model_a_evaluation": {
                "citation_score": {"justification": "Good but no score"},
            },
        }

        result = extract_scores_from_evaluation(evaluation, "model_a_evaluation")

        # Score should be None if not present
        assert result.get("citation_score") is None or result["citation_score"] is None

