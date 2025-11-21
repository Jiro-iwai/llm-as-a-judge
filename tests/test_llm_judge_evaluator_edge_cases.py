"""
Unit tests for llm_judge_evaluator.py edge cases and error handling.

This module tests edge cases and error handling paths that are not
covered by other tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProcessCsvEdgeCases:
    """Tests for process_csv edge cases."""

    @patch("scripts.llm_judge_evaluator.AzureOpenAI")
    @patch("scripts.llm_judge_evaluator.OpenAI")
    def test_process_csv_file_not_found(self, mock_openai, mock_azure):
        """Test process_csv handles FileNotFoundError."""
        from scripts.llm_judge_evaluator import process_csv

        with pytest.raises(SystemExit):
            process_csv("nonexistent_file.csv", "output.csv")

    @patch("scripts.llm_judge_evaluator.AzureOpenAI")
    @patch("scripts.llm_judge_evaluator.OpenAI")
    @patch("scripts.llm_judge_evaluator.pd.read_csv")
    def test_process_csv_empty_data_error(self, mock_read_csv, mock_openai, mock_azure):
        """Test process_csv handles EmptyDataError."""
        from scripts.llm_judge_evaluator import process_csv

        # Mock pd.read_csv to raise EmptyDataError
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No columns to parse from file")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_file = f.name

        try:
            with pytest.raises(SystemExit):
                process_csv(temp_file, "output.csv")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("scripts.llm_judge_evaluator.AzureOpenAI")
    @patch("scripts.llm_judge_evaluator.OpenAI")
    @patch("scripts.llm_judge_evaluator.pd.read_csv")
    def test_process_csv_parser_error(self, mock_read_csv, mock_openai, mock_azure):
        """Test process_csv handles ParserError."""
        from scripts.llm_judge_evaluator import process_csv

        # Mock pd.read_csv to raise ParserError
        mock_read_csv.side_effect = pd.errors.ParserError("Error tokenizing data")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_file = f.name

        try:
            with pytest.raises(SystemExit):
                process_csv(temp_file, "output.csv")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("scripts.llm_judge_evaluator.AzureOpenAI")
    @patch("scripts.llm_judge_evaluator.OpenAI")
    @patch("builtins.open")
    def test_process_csv_permission_error(self, mock_open, mock_openai, mock_azure):
        """Test process_csv handles PermissionError."""
        from scripts.llm_judge_evaluator import process_csv

        # Mock open to raise PermissionError
        mock_open.side_effect = PermissionError("Permission denied")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_file = f.name

        try:
            with pytest.raises(SystemExit):
                process_csv(temp_file, "output.csv")
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

    @patch("scripts.llm_judge_evaluator.time.sleep")
    def test_call_judge_model_with_none_client(self, mock_sleep):
        """Test call_judge_model handles None client."""
        from scripts.llm_judge_evaluator import call_judge_model

        # call_judge_model signature: (client, question, model_a_response, model_b_response, model_name, ...)
        # When client is None, it will fail after retries and return None
        # Mock time.sleep to avoid actual delays during retries
        result = call_judge_model(
            None, "question", "response_a", "response_b", "model", timeout=10
        )  # type: ignore[call-arg]

        # Should return None after all retries fail
        assert result is None
        # Verify that time.sleep was called during retries
        assert mock_sleep.called

