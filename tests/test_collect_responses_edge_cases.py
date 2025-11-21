"""
Unit tests for collect_responses.py edge cases and error handling.

This module tests edge cases and error handling paths that are not
covered by other tests.
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCleanAndFormatLlmLogEdgeCases:
    """Tests for clean_and_format_llm_log edge cases."""

    def test_clean_and_format_llm_log_value_error(self):
        """Test clean_and_format_llm_log handles ValueError."""
        from collect_responses import clean_and_format_llm_log

        # Mock json.loads to raise ValueError
        with patch("json.loads", side_effect=ValueError("Invalid value")):
            result = clean_and_format_llm_log('{"test": "data"}')
            assert "Error parsing JSON" in result

    def test_clean_and_format_llm_log_type_error(self):
        """Test clean_and_format_llm_log handles TypeError."""
        from collect_responses import clean_and_format_llm_log

        # Mock json.loads to raise TypeError
        with patch("json.loads", side_effect=TypeError("Invalid type")):
            result = clean_and_format_llm_log('{"test": "data"}')
            assert "Error parsing JSON" in result

    def test_clean_and_format_llm_log_generic_exception(self):
        """Test clean_and_format_llm_log handles generic Exception."""
        from collect_responses import clean_and_format_llm_log

        # Mock json.loads to raise a generic exception
        with patch("json.loads", side_effect=Exception("Unexpected error")):
            result = clean_and_format_llm_log('{"test": "data"}')
            assert "unexpected error" in result.lower() or "error" in result.lower()

    def test_clean_and_format_llm_log_empty_section(self):
        """Test clean_and_format_llm_log skips empty sections."""
        from collect_responses import clean_and_format_llm_log

        # Create log with empty section
        log_text = "思考：\n\n回答：Answer here"
        result = clean_and_format_llm_log(log_text)
        # Empty section should be skipped
        assert "思考" not in result or "---" not in result


class TestFormatResponseEdgeCases:
    """Tests for format_response edge cases."""

    def test_format_response_exception_handling(self):
        """Test format_response handles exceptions gracefully."""
        from collect_responses import format_response

        # Mock clean_and_format_llm_log to raise exception
        with patch("collect_responses.clean_and_format_llm_log", side_effect=Exception("Test error")):
            result = format_response("test response")
            # Should return original response_text on exception
            assert result == "test response"

    def test_format_response_generic_exception(self):
        """Test format_response handles generic exceptions."""
        from collect_responses import format_response

        # Mock clean_and_format_llm_log to raise generic exception
        with patch("collect_responses.clean_and_format_llm_log", side_effect=RuntimeError("Runtime error")):
            result = format_response("test response")
            assert result == "test response"


class TestCallApiEdgeCases:
    """Tests for call_api edge cases."""

    @patch("collect_responses.requests.post")
    @patch("collect_responses.log_warning")
    @patch("collect_responses.log_info")
    def test_call_api_unexpected_response_format(self, mock_info, mock_warning, mock_post):
        """Test call_api handles unexpected response format."""
        from collect_responses import call_api

        # Mock response with unexpected format (not dict or list)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "plain text response"
        mock_response.json.return_value = "not a dict or list"
        mock_post.return_value = mock_response

        result = call_api("test question", "http://example.com/api", "model", verbose=True)

        # Should return response.text
        assert result == "plain text response"
        mock_warning.assert_called()

    @patch("collect_responses.requests.post")
    @patch("collect_responses.log_warning")
    def test_call_api_missing_answer_field(self, mock_warning, mock_post):
        """Test call_api handles missing answer field."""
        from collect_responses import call_api

        # Mock response without answer field
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "response text"
        mock_response.json.return_value = {"other_field": "value"}
        mock_post.return_value = mock_response

        result = call_api("test question", "http://example.com/api", "model", verbose=True)

        # Should return response.text when answer field is missing
        assert result == "response text"
        mock_warning.assert_called()

    @patch("collect_responses.requests.post")
    @patch("collect_responses.log_processing_time_entry")
    @patch("collect_responses.log_info")
    def test_call_api_with_urls_field(self, mock_info, mock_log_time, mock_post):
        """Test call_api handles urls field in response."""
        from collect_responses import call_api

        # Mock response with urls field
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "Test answer",
            "urls": ["http://example.com/1", "http://example.com/2"],
        }
        mock_post.return_value = mock_response

        result = call_api("test question", "http://example.com/api", "model", verbose=True)

        assert result == "Test answer"
        # Check that URL count was logged
        assert any("url" in str(call).lower() for call in mock_info.call_args_list)

    @patch("collect_responses.requests.post")
    @patch("collect_responses.log_error")
    def test_call_api_json_decode_error_with_attribute_error(self, mock_error, mock_post):
        """Test call_api handles AttributeError when accessing response.text."""
        from collect_responses import call_api

        # Mock response that raises AttributeError
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        # Make response.text raise AttributeError
        del mock_response.text
        mock_post.return_value = mock_response

        result = call_api("test question", "http://example.com/api", "model")

        # Should handle AttributeError gracefully
        assert result is not None or result == ""


class TestCollectResponsesVerbose:
    """Tests for collect_responses verbose mode."""

    @patch("collect_responses.call_api")
    @patch("collect_responses.format_response", side_effect=lambda x: x)
    @patch("collect_responses.log_section")
    @patch("collect_responses.log_info")
    @patch("collect_responses.log_success")
    @patch("collect_responses.log_warning")
    @patch("collect_responses.time.sleep")
    def test_collect_responses_verbose_mode(
        self,
        mock_sleep,
        mock_warning,
        mock_success,
        mock_info,
        mock_section,
        mock_format,
        mock_call_api,
    ):
        """Test collect_responses with verbose=True."""
        from collect_responses import collect_responses
        import pandas as pd

        mock_call_api.side_effect = ["Answer A1", "Answer B1"]

        df = collect_responses(
            questions=["Q1"],
            api_url="http://example.com/api",
            model_a="model-a",
            model_b="model-b",
            verbose=True,
            delay=0.0,
        )

        assert len(df) == 1
        # Check that verbose logging was called
        assert mock_section.called
        assert mock_info.called

    @patch("collect_responses.call_api")
    @patch("collect_responses.format_response", side_effect=lambda x: x)
    @patch("collect_responses.log_section")
    @patch("collect_responses.log_info")
    @patch("collect_responses.time.sleep")
    def test_collect_responses_non_verbose_mode(
        self,
        mock_sleep,
        mock_info,
        mock_section,
        mock_format,
        mock_call_api,
    ):
        """Test collect_responses with verbose=False."""
        from collect_responses import collect_responses

        mock_call_api.side_effect = ["Answer A1", "Answer B1"]

        df = collect_responses(
            questions=["Q1"],
            api_url="http://example.com/api",
            model_a="model-a",
            model_b="model-b",
            verbose=False,
            delay=0.0,
        )

        assert len(df) == 1
        # With verbose=False, fewer log calls should be made
        # (but some are still needed for errors)

    @patch("collect_responses.call_api")
    @patch("collect_responses.format_response", side_effect=lambda x: x)
    @patch("collect_responses.log_info")
    def test_collect_responses_with_failed_responses_verbose(
        self,
        mock_info,
        mock_format,
        mock_call_api,
    ):
        """Test collect_responses verbose mode with failed responses."""
        from collect_responses import collect_responses

        # Mock call_api to return None (failed)
        mock_call_api.side_effect = [None, "Answer B1"]

        df = collect_responses(
            questions=["Q1"],
            api_url="http://example.com/api",
            model_a="model-a",
            model_b="model-b",
            verbose=True,
            delay=0.0,
        )

        assert len(df) == 1
        assert df.iloc[0]["Model_A_Response"] == ""
        assert df.iloc[0]["Model_B_Response"] == "Answer B1"


class TestReadQuestionsErrorHandling:
    """Tests for read_questions error handling."""

    def test_read_questions_permission_error(self, tmp_path):
        """Test read_questions handles PermissionError."""
        from collect_responses import read_questions

        # Create a file that will cause PermissionError
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Mock open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(SystemExit):
                read_questions(str(test_file))

    def test_read_questions_unicode_decode_error(self, tmp_path):
        """Test read_questions handles UnicodeDecodeError."""
        from collect_responses import read_questions

        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"\xff\xfe")  # Invalid UTF-8

        # Mock open to raise UnicodeDecodeError
        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")):
            with pytest.raises(SystemExit):
                read_questions(str(test_file))

    def test_read_questions_generic_exception(self, tmp_path):
        """Test read_questions handles generic Exception."""
        from collect_responses import read_questions

        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Mock open to raise generic exception
        with patch("builtins.open", side_effect=Exception("Unexpected error")):
            with pytest.raises(SystemExit):
                read_questions(str(test_file))


class TestCollectResponsesConfigDefaults:
    """Tests for collect_responses config defaults."""

    @patch("collect_responses.get_default_identity")
    @patch("collect_responses.get_timeout")
    @patch("collect_responses.get_api_delay")
    @patch("collect_responses.call_api")
    @patch("collect_responses.format_response", side_effect=lambda x: x)
    def test_collect_responses_uses_config_defaults(
        self,
        mock_format,
        mock_call_api,
        mock_get_delay,
        mock_get_timeout,
        mock_get_identity,
    ):
        """Test collect_responses uses config defaults when None."""
        from collect_responses import collect_responses

        mock_get_identity.return_value = "default-identity"
        mock_get_timeout.return_value = 120
        mock_get_delay.return_value = 1.0
        mock_call_api.side_effect = ["Answer A1", "Answer B1"]

        df = collect_responses(
            questions=["Q1"],
            api_url="http://example.com/api",
            model_a="model-a",
            model_b="model-b",
            identity=None,
            timeout=None,
            delay=None,
            verbose=False,
        )

        assert len(df) == 1
        mock_get_identity.assert_called_once()
        mock_get_timeout.assert_called_once()
        mock_get_delay.assert_called_once()

