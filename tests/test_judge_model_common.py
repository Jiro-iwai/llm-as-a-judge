"""
Tests for common call_judge_model function.

This test ensures that the common call_judge_model function works correctly
for both llm_judge_evaluator.py and format_clarity_evaluator.py use cases.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the common function (will be created)
try:
    from src.utils.judge_model_common import call_judge_model_common
except ImportError:
    # If not yet created, we'll skip tests
    call_judge_model_common = None


@pytest.mark.skipif(
    call_judge_model_common is None,
    reason="Common function not yet implemented",
)
class TestCallJudgeModelCommon:
    """Tests for common call_judge_model function"""

    @patch("src.utils.judge_model_common.get_max_retries")
    @patch("src.utils.judge_model_common.get_retry_delay")
    def test_call_judge_model_common_success_llm_judge(
        self, mock_get_retry_delay, mock_get_max_retries
    ):
        """Test successful API call for llm_judge evaluator"""
        mock_get_max_retries.return_value = 3
        mock_get_retry_delay.return_value = 1
        model_config = {
            "max_total_tokens": 8000,
            "min_output_tokens": 100,
            "max_output_tokens": 2000,
            "safety_margin": 100,
            "temperature": 0.7,
            "use_max_completion_tokens": False,
            "timeout": 60,
        }

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(
                    content=json.dumps(
                        {
                            "model_a_evaluation": {
                                "citation_score": {"score": 4, "justification": "Good"},
                            },
                            "model_b_evaluation": {
                                "citation_score": {"score": 5, "justification": "Excellent"},
                            },
                        }
                    )
                ),
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        system_prompt = "You are a judge."
        user_prompt = "Question: Test\nModel A: Answer A\nModel B: Answer B"

        result = call_judge_model_common(
            client=mock_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name="gpt-4-turbo",
            model_config=model_config,
            response_validation_keys=["model_a_evaluation", "model_b_evaluation"],
            enable_token_estimation=True,
        )

        assert result is not None
        assert "model_a_evaluation" in result
        assert "model_b_evaluation" in result
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.utils.judge_model_common.get_max_retries")
    @patch("src.utils.judge_model_common.get_retry_delay")
    def test_call_judge_model_common_success_format_clarity(
        self, mock_get_retry_delay, mock_get_max_retries
    ):
        """Test successful API call for format_clarity evaluator"""
        mock_get_max_retries.return_value = 3
        mock_get_retry_delay.return_value = 1
        model_config = {
            "max_total_tokens": 8000,
            "min_output_tokens": 100,
            "max_output_tokens": 2000,
            "safety_margin": 100,
            "temperature": 0.7,
            "use_max_completion_tokens": False,
            "timeout": 60,
        }

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
                                "justification": "Good format match",
                            }
                        }
                    )
                ),
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        system_prompt = "You are a format evaluator."
        user_prompt = "Question: Test\nModel A: Answer A\nModel B: Answer B"

        result = call_judge_model_common(
            client=mock_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name="gpt-4-turbo",
            model_config=model_config,
            response_validation_keys=["format_clarity_evaluation"],
            enable_token_estimation=False,
        )

        assert result is not None
        assert "format_clarity_evaluation" in result
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.utils.judge_model_common.get_max_retries")
    @patch("src.utils.judge_model_common.get_retry_delay")
    @patch("time.sleep")
    def test_call_judge_model_common_json_decode_error(
        self, mock_sleep, mock_get_retry_delay, mock_get_max_retries
    ):
        """Test handling JSON decode error with retry"""
        mock_get_max_retries.return_value = 2
        mock_get_retry_delay.return_value = 0.1
        model_config = {
            "max_total_tokens": 8000,
            "min_output_tokens": 100,
            "max_output_tokens": 2000,
            "safety_margin": 100,
            "temperature": 0.7,
            "use_max_completion_tokens": False,
            "timeout": 60,
        }

        mock_client = Mock()
        # First call returns invalid JSON, second call succeeds
        mock_response_invalid = Mock()
        mock_response_invalid.choices = [Mock(finish_reason="stop", message=Mock(content="invalid json"))]

        mock_response_valid = Mock()
        mock_response_valid.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(
                    content=json.dumps(
                        {
                            "format_clarity_evaluation": {
                                "score": 4,
                                "justification": "Good",
                            }
                        }
                    )
                ),
            )
        ]

        mock_client.chat.completions.create.side_effect = [
            mock_response_invalid,
            mock_response_valid,
        ]

        result = call_judge_model_common(
            client=mock_client,
            system_prompt="Test prompt",
            user_prompt="Test user prompt",
            model_name="gpt-4-turbo",
            model_config=model_config,
            response_validation_keys=["format_clarity_evaluation"],
            enable_token_estimation=False,
        )

        assert result is not None
        assert mock_client.chat.completions.create.call_count == 2

    @patch("src.utils.judge_model_common.get_max_retries")
    @patch("src.utils.judge_model_common.get_retry_delay")
    @patch("time.sleep")
    def test_call_judge_model_common_timeout_error(
        self, mock_sleep, mock_get_retry_delay, mock_get_max_retries
    ):
        """Test handling TimeoutError"""
        mock_get_max_retries.return_value = 2
        mock_get_retry_delay.return_value = 0.1
        model_config = {
            "max_total_tokens": 8000,
            "min_output_tokens": 100,
            "max_output_tokens": 2000,
            "safety_margin": 100,
            "temperature": 0.7,
            "use_max_completion_tokens": False,
            "timeout": 60,
        }

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timeout")

        result = call_judge_model_common(
            client=mock_client,
            system_prompt="Test prompt",
            user_prompt="Test user prompt",
            model_name="gpt-4-turbo",
            model_config=model_config,
            response_validation_keys=["format_clarity_evaluation"],
            enable_token_estimation=False,
            timeout=30,
        )

        assert result is None

    @patch("src.utils.judge_model_common.get_max_retries")
    @patch("src.utils.judge_model_common.get_retry_delay")
    def test_call_judge_model_common_token_estimation_enabled(
        self, mock_get_retry_delay, mock_get_max_retries
    ):
        """Test that token estimation is used when enabled"""
        mock_get_max_retries.return_value = 3
        mock_get_retry_delay.return_value = 1
        model_config = {
            "max_total_tokens": 8000,
            "min_output_tokens": 100,
            "max_output_tokens": 2000,
            "safety_margin": 100,
            "temperature": 0.7,
            "use_max_completion_tokens": False,
            "timeout": 60,
        }

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                finish_reason="stop",
                message=Mock(
                    content=json.dumps(
                        {
                            "model_a_evaluation": {"citation_score": {"score": 4}},
                            "model_b_evaluation": {"citation_score": {"score": 5}},
                        }
                    )
                ),
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        system_prompt = "You are a judge." * 100  # Long prompt
        user_prompt = "Test prompt"

        result = call_judge_model_common(
            client=mock_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name="gpt-4-turbo",
            model_config=model_config,
            response_validation_keys=["model_a_evaluation", "model_b_evaluation"],
            enable_token_estimation=True,
        )

        assert result is not None
        # Verify that API was called with calculated max_tokens
        call_args = mock_client.chat.completions.create.call_args
        assert "max_tokens" in call_args.kwargs or "max_completion_tokens" in call_args.kwargs

    @patch("src.utils.judge_model_common.get_max_retries")
    @patch("src.utils.judge_model_common.get_retry_delay")
    def test_call_judge_model_common_token_estimation_disabled(
        self, mock_get_retry_delay, mock_get_max_retries
    ):
        """Test that token estimation is not used when disabled"""
        mock_get_max_retries.return_value = 3
        mock_get_retry_delay.return_value = 1
        model_config = {
            "max_total_tokens": 8000,
            "min_output_tokens": 100,
            "max_output_tokens": 2000,
            "max_completion_tokens": 2000,
            "safety_margin": 100,
            "temperature": 0.7,
            "use_max_completion_tokens": False,
            "timeout": 60,
        }

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
                                "justification": "Good",
                            }
                        }
                    )
                ),
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = call_judge_model_common(
            client=mock_client,
            system_prompt="Test prompt",
            user_prompt="Test user prompt",
            model_name="gpt-4-turbo",
            model_config=model_config,
            response_validation_keys=["format_clarity_evaluation"],
            enable_token_estimation=False,
        )

        assert result is not None
        # When token estimation is disabled, should use model config values directly
        call_args = mock_client.chat.completions.create.call_args
        assert "max_tokens" in call_args.kwargs or "max_completion_tokens" in call_args.kwargs

