"""
Tests for helper functions extracted from process_csv in llm_judge_evaluator.py.

These tests ensure that helper functions work correctly before refactoring process_csv.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.llm_judge_evaluator import (
    process_csv,
    extract_scores_from_evaluation,
)


class TestProcessCsvHelperFunctions:
    """Tests for helper functions that will be extracted from process_csv"""

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("scripts.llm_judge_evaluator.log_error")
    @patch("scripts.llm_judge_evaluator.log_success")
    @patch("scripts.llm_judge_evaluator.log_warning")
    @patch("scripts.llm_judge_evaluator.log_info")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.tqdm")
    @patch("scripts.llm_judge_evaluator.extract_scores_from_evaluation")
    @patch("scripts.llm_judge_evaluator.call_judge_model")
    @patch("scripts.llm_judge_evaluator.OpenAI")
    @patch("scripts.llm_judge_evaluator.AzureOpenAI")
    def test_process_csv_still_works_after_refactoring(
        self,
        mock_azure,
        mock_openai,
        mock_call_judge,
        mock_extract_scores,
        mock_tqdm,
        mock_log_section,
        mock_log_info,
        mock_log_warning,
        mock_log_success,
        mock_log_error,
    ):
        """Test that process_csv still works correctly after refactoring"""
        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x

        # Mock extract_scores_from_evaluation
        mock_extract_scores.side_effect = lambda eval_dict, key: {
            "citation_score": 4,
            "citation_justification": "Good",
            "relevance_score": 5,
            "relevance_justification": "Excellent",
            "react_performance_thought_score": 4,
            "react_performance_thought_justification": "Good",
            "rag_retrieval_observation_score": 5,
            "rag_retrieval_observation_justification": "Excellent",
            "information_integration_score": 4,
            "information_integration_justification": "Good",
        }

        # Mock call_judge_model
        mock_call_judge.return_value = {
            "model_a_evaluation": {"citation_score": {"score": 4}},
            "model_b_evaluation": {"citation_score": {"score": 5}},
        }

        # Mock Azure OpenAI client
        mock_azure_client = Mock()
        mock_azure.return_value = mock_azure_client

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Question,Model_A_Response,Model_B_Response\n")
            f.write("Q1,Answer A1,Answer B1\n")
            f.write("Q2,Answer A2,Answer B2\n")
            temp_csv = f.name

        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                temp_output = f.name

            # Call process_csv
            process_csv(
                temp_csv,
                temp_output,
                limit_rows=2,
                model_name="gpt-4.1",
                non_interactive=True,
            )

            # Verify that output file was created
            assert os.path.exists(temp_output)

            # Verify that output file has correct structure
            output_df = pd.read_csv(temp_output)
            assert "Question" in output_df.columns
            assert "Model_A_Response" in output_df.columns
            assert "Model_B_Response" in output_df.columns
            assert "Model_A_Citation_Score" in output_df.columns
            assert "Model_B_Citation_Score" in output_df.columns
            assert len(output_df) == 2

            # Verify that call_judge_model was called for each row
            assert mock_call_judge.call_count == 2

        finally:
            # Clean up
            if os.path.exists(temp_csv):
                os.unlink(temp_csv)
            if os.path.exists(temp_output):
                os.unlink(temp_output)

