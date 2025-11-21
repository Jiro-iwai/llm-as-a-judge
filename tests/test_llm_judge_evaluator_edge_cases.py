"""
Unit tests for llm_judge_evaluator.py edge cases and error handling.

This module tests edge cases and error handling paths that are not
covered by other tests.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProcessCsvEdgeCases:
    """Tests for process_csv edge cases.
    
    Note: Comprehensive process_csv tests are in test_process_csv.py.
    These tests focus on error handling edge cases.
    """

    @patch("scripts.llm_judge_evaluator.log_error")
    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    def test_process_csv_file_not_found(self, mock_log_error):
        """Test process_csv handles FileNotFoundError."""
        from scripts.llm_judge_evaluator import process_csv

        # Call with non-existent file
        # sys.exit(1) raises SystemExit exception
        with pytest.raises(SystemExit) as exc_info:
            process_csv("nonexistent_file.csv", "output.csv", non_interactive=True)

        # Should exit with code 1
        assert exc_info.value.code == 1
        # Should log error
        assert mock_log_error.called, "log_error should be called for FileNotFoundError"

    @patch("scripts.llm_judge_evaluator.log_error")
    @patch("scripts.llm_judge_evaluator.pd.read_csv")
    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    def test_process_csv_empty_data_error(self, mock_read_csv, mock_log_error):
        """Test process_csv handles EmptyDataError."""
        import pandas as pd
        from scripts.llm_judge_evaluator import process_csv

        # Mock pd.read_csv to raise EmptyDataError
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No columns to parse from file")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_file = f.name

        try:
            # sys.exit(1) raises SystemExit exception
            with pytest.raises(SystemExit) as exc_info:
                process_csv(temp_file, "output.csv", non_interactive=True)

            # Should exit with code 1
            assert exc_info.value.code == 1
            # Should log error
            assert mock_log_error.called, "log_error should be called for EmptyDataError"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("scripts.llm_judge_evaluator.log_error")
    @patch("scripts.llm_judge_evaluator.pd.read_csv")
    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    def test_process_csv_parser_error(self, mock_read_csv, mock_log_error):
        """Test process_csv handles ParserError."""
        import pandas as pd
        from scripts.llm_judge_evaluator import process_csv

        # Mock pd.read_csv to raise ParserError
        mock_read_csv.side_effect = pd.errors.ParserError("Error tokenizing data")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_file = f.name

        try:
            # sys.exit(1) raises SystemExit exception
            with pytest.raises(SystemExit) as exc_info:
                process_csv(temp_file, "output.csv", non_interactive=True)

            # Should exit with code 1
            assert exc_info.value.code == 1
            # Should log error
            assert mock_log_error.called, "log_error should be called for ParserError"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("scripts.llm_judge_evaluator.log_error")
    @patch("builtins.open")
    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    def test_process_csv_permission_error(self, mock_open, mock_log_error):
        """Test process_csv handles PermissionError."""
        from scripts.llm_judge_evaluator import process_csv

        # Mock open to raise PermissionError
        mock_open.side_effect = PermissionError("Permission denied")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_file = f.name

        try:
            # sys.exit(1) raises SystemExit exception
            with pytest.raises(SystemExit) as exc_info:
                process_csv(temp_file, "output.csv", non_interactive=True)

            # Should exit with code 1
            assert exc_info.value.code == 1
            # Should log error
            assert mock_log_error.called, "log_error should be called for PermissionError"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestExtractScoresEdgeCases:
    """Tests for extract_scores_from_evaluation edge cases."""

    def test_extract_scores_with_empty_evaluation(self):
        """Test extract_scores_from_evaluation handles empty evaluation."""
        from scripts.llm_judge_evaluator import extract_scores_from_evaluation

        result = extract_scores_from_evaluation({}, "model_a_evaluation")

        # Should return empty dict or default values
        assert isinstance(result, dict)

    def test_extract_scores_with_missing_key(self):
        """Test extract_scores_from_evaluation handles missing model key."""
        from scripts.llm_judge_evaluator import extract_scores_from_evaluation

        result = extract_scores_from_evaluation({"other_key": "value"}, "model_a_evaluation")

        # Should handle missing key gracefully
        assert isinstance(result, dict)


class TestCallJudgeModelEdgeCases:
    """Tests for call_judge_model edge cases."""

    def test_call_judge_model_with_none_client(self):
        """Test call_judge_model handles None client."""
        from scripts.llm_judge_evaluator import call_judge_model

        # This should raise an error or handle gracefully
        # The actual behavior depends on implementation
        with pytest.raises((AttributeError, TypeError)):
            # call_judge_model signature: (client, prompt, model_name, timeout, model_a_response, model_b_response)
            call_judge_model(None, "prompt", "model", timeout=10, model_a_response="response_a", model_b_response="response_b")  # type: ignore[call-arg]

