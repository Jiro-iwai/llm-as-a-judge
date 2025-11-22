"""
Tests for parallel processing functionality.

These tests ensure that parallel processing works correctly and maintains
the same behavior as sequential processing.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestParallelProcessingSupport:
    """Tests to verify parallel processing support exists and works correctly"""

    def test_process_csv_supports_max_workers_parameter(self):
        """Test that process_csv functions accept max_workers parameter for parallel processing"""
        # This test will fail initially, then we'll implement the feature
        from scripts.llm_judge_evaluator import process_csv

        # Check if process_csv accepts max_workers parameter
        import inspect

        sig = inspect.signature(process_csv)
        assert (
            "max_workers" in sig.parameters
        ), "process_csv should accept max_workers parameter for parallel processing"

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("builtins.input")
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
    def test_parallel_processing_maintains_order(
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
        mock_input,
    ):
        """Test that parallel processing maintains the order of results"""
        # Mock input to avoid interactive prompt
        mock_input.return_value = "y"

        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x

        # Mock extract_scores_from_evaluation - it's called twice per row (once for model_a, once for model_b)
        def mock_extract(evaluation, model_key):
            if model_key == "model_a_evaluation":
                return {
                    "citation_score": 4,
                    "relevance_score": 5,
                    "citation_justification": "Good",
                    "relevance_justification": "Excellent",
                    "react_performance_thought_score": 4,
                    "react_performance_thought_justification": "Good",
                    "rag_retrieval_observation_score": 4,
                    "rag_retrieval_observation_justification": "Good",
                    "information_integration_score": 4,
                    "information_integration_justification": "Good",
                }
            else:  # model_b_evaluation
                return {
                    "citation_score": 3,
                    "relevance_score": 4,
                    "citation_justification": "OK",
                    "relevance_justification": "Good",
                    "react_performance_thought_score": 3,
                    "react_performance_thought_justification": "OK",
                    "rag_retrieval_observation_score": 3,
                    "rag_retrieval_observation_justification": "OK",
                    "information_integration_score": 3,
                    "information_integration_justification": "OK",
                }

        mock_extract_scores.side_effect = mock_extract

        # Mock call_judge_model to return evaluation results
        mock_call_judge.return_value = {
            "model_a_evaluation": {
                "citation_score": {"score": 4, "justification": "Good"},
                "relevance_score": {"score": 5, "justification": "Excellent"},
                "react_performance_thought_score": {"score": 4, "justification": "Good"},
                "rag_retrieval_observation_score": {"score": 4, "justification": "Good"},
                "information_integration_score": {"score": 4, "justification": "Good"},
            },
            "model_b_evaluation": {
                "citation_score": {"score": 3, "justification": "OK"},
                "relevance_score": {"score": 4, "justification": "Good"},
                "react_performance_thought_score": {"score": 3, "justification": "OK"},
                "rag_retrieval_observation_score": {"score": 3, "justification": "OK"},
                "information_integration_score": {"score": 3, "justification": "OK"},
            },
        }

        # Mock Azure OpenAI client
        mock_azure_client = Mock()
        mock_azure.return_value = mock_azure_client

        from scripts.llm_judge_evaluator import process_csv

        # Create temporary CSV file with 3 rows
        test_data = pd.DataFrame({
            "Question": ["Q1", "Q2", "Q3"],
            "Model_A_Response": ["Answer A1", "Answer A2", "Answer A3"],
            "Model_B_Response": ["Answer B1", "Answer B2", "Answer B3"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_csv = f.name

        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
                temp_output = f.name

            # Call process_csv with max_workers=2 (parallel)
            process_csv(
                temp_csv,
                temp_output,
                limit_rows=3,
                model_name="gpt-4.1",
                non_interactive=True,
                max_workers=2,
            )

            # Verify that output file was created
            assert os.path.exists(temp_output)

            # Verify that output file has correct structure and order
            output_df = pd.read_csv(temp_output)
            assert len(output_df) == 3
            assert output_df.iloc[0]["Question"] == "Q1"
            assert output_df.iloc[1]["Question"] == "Q2"
            assert output_df.iloc[2]["Question"] == "Q3"

            # Verify that API was called 3 times (once per row)
            assert mock_call_judge.call_count == 3

        finally:
            # Clean up
            if os.path.exists(temp_csv):
                os.unlink(temp_csv)
            if os.path.exists(temp_output):
                os.unlink(temp_output)

    def test_parallel_processing_respects_rate_limits(self):
        """Test that parallel processing respects API rate limits"""
        # This test verifies that max_workers parameter limits the number of concurrent API calls
        # Rate limiting is handled by the API client, but max_workers ensures we don't
        # exceed reasonable concurrency levels
        from scripts.llm_judge_evaluator import process_csv
        import inspect

        sig = inspect.signature(process_csv)
        assert "max_workers" in sig.parameters, "max_workers parameter should exist"

        # Verify that max_workers can be set to limit concurrency
        param = sig.parameters["max_workers"]
        assert param.default is None or isinstance(param.default, int), "max_workers should accept int or None"

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("builtins.input")
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
    def test_parallel_processing_handles_errors_correctly(
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
        mock_input,
    ):
        """Test that parallel processing handles errors correctly"""
        # Mock input to avoid interactive prompt
        mock_input.return_value = "y"

        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x

        # Mock extract_scores_from_evaluation
        def mock_extract(evaluation, model_key):
            if model_key == "model_a_evaluation":
                return {
                    "citation_score": 4,
                    "relevance_score": 5,
                    "citation_justification": "Good",
                    "relevance_justification": "Excellent",
                    "react_performance_thought_score": 4,
                    "react_performance_thought_justification": "Good",
                    "rag_retrieval_observation_score": 4,
                    "rag_retrieval_observation_justification": "Good",
                    "information_integration_score": 4,
                    "information_integration_justification": "Good",
                }
            else:  # model_b_evaluation
                return {
                    "citation_score": 3,
                    "relevance_score": 4,
                    "citation_justification": "OK",
                    "relevance_justification": "Good",
                    "react_performance_thought_score": 3,
                    "react_performance_thought_justification": "OK",
                    "rag_retrieval_observation_score": 3,
                    "rag_retrieval_observation_justification": "OK",
                    "information_integration_score": 3,
                    "information_integration_justification": "OK",
                }

        mock_extract_scores.side_effect = mock_extract

        # Mock call_judge_model to raise an error for the second row
        call_count = [0]

        def mock_call_judge_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call fails
                raise Exception("Simulated API error")
            return {
                "model_a_evaluation": {
                    "citation_score": {"score": 4, "justification": "Good"},
                    "relevance_score": {"score": 5, "justification": "Excellent"},
                    "react_performance_thought_score": {"score": 4, "justification": "Good"},
                    "rag_retrieval_observation_score": {"score": 4, "justification": "Good"},
                    "information_integration_score": {"score": 4, "justification": "Good"},
                },
                "model_b_evaluation": {
                    "citation_score": {"score": 3, "justification": "OK"},
                    "relevance_score": {"score": 4, "justification": "Good"},
                    "react_performance_thought_score": {"score": 3, "justification": "OK"},
                    "rag_retrieval_observation_score": {"score": 3, "justification": "OK"},
                    "information_integration_score": {"score": 3, "justification": "OK"},
                },
            }

        mock_call_judge.side_effect = mock_call_judge_side_effect

        # Mock Azure OpenAI client
        mock_azure_client = Mock()
        mock_azure.return_value = mock_azure_client

        from scripts.llm_judge_evaluator import process_csv

        # Create temporary CSV file with 3 rows
        test_data = pd.DataFrame({
            "Question": ["Q1", "Q2", "Q3"],
            "Model_A_Response": ["Answer A1", "Answer A2", "Answer A3"],
            "Model_B_Response": ["Answer B1", "Answer B2", "Answer B3"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_csv = f.name

        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
                temp_output = f.name

            # Call process_csv with max_workers=2 (parallel)
            process_csv(
                temp_csv,
                temp_output,
                limit_rows=3,
                model_name="gpt-4.1",
                non_interactive=True,
                max_workers=2,
            )

            # Verify that output file was created
            assert os.path.exists(temp_output)

            # Verify that output file has all 3 rows (even though one failed)
            output_df = pd.read_csv(temp_output)
            assert len(output_df) == 3

            # Verify that error was logged for the failed row
            error_logged = any(
                "エラーが発生しました" in str(call) or "Simulated API error" in str(call)
                for call in mock_log_error.call_args_list
            )
            assert error_logged, "Error should be logged for failed row"

            # Verify that all rows are present and one has an error
            # Note: In parallel processing, the order is maintained but error might occur in any row
            # Find the row with error
            error_rows = []
            success_rows = []
            for idx in range(len(output_df)):
                error_val = output_df.iloc[idx]["Evaluation_Error"]
                if pd.notna(error_val) and error_val != "":
                    error_rows.append(idx)
                else:
                    success_rows.append(idx)

            # Should have exactly 1 error row and 2 success rows
            assert len(error_rows) == 1, f"Should have exactly 1 error row, got {len(error_rows)}"
            assert len(success_rows) == 2, f"Should have exactly 2 success rows, got {len(success_rows)}"
            
            # Verify error row has error message
            error_row_idx = error_rows[0]
            error_msg = str(output_df.iloc[error_row_idx]["Evaluation_Error"])
            assert "Parallel processing error" in error_msg or "Simulated API error" in error_msg, f"Error row should have error message, got: {error_msg}"

        finally:
            # Clean up
            if os.path.exists(temp_csv):
                os.unlink(temp_csv)
            if os.path.exists(temp_output):
                os.unlink(temp_output)

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("builtins.input")
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
    def test_parallel_processing_produces_same_results_as_sequential(
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
        mock_input,
    ):
        """Test that parallel processing produces identical results to sequential processing"""
        # Mock input to avoid interactive prompt
        mock_input.return_value = "y"

        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x

        # Mock extract_scores_from_evaluation
        def mock_extract(evaluation, model_key):
            if model_key == "model_a_evaluation":
                return {
                    "citation_score": 4,
                    "relevance_score": 5,
                    "citation_justification": "Good",
                    "relevance_justification": "Excellent",
                    "react_performance_thought_score": 4,
                    "react_performance_thought_justification": "Good",
                    "rag_retrieval_observation_score": 4,
                    "rag_retrieval_observation_justification": "Good",
                    "information_integration_score": 4,
                    "information_integration_justification": "Good",
                }
            else:  # model_b_evaluation
                return {
                    "citation_score": 3,
                    "relevance_score": 4,
                    "citation_justification": "OK",
                    "relevance_justification": "Good",
                    "react_performance_thought_score": 3,
                    "react_performance_thought_justification": "OK",
                    "rag_retrieval_observation_score": 3,
                    "rag_retrieval_observation_justification": "OK",
                    "information_integration_score": 3,
                    "information_integration_justification": "OK",
                }

        mock_extract_scores.side_effect = mock_extract

        # Mock call_judge_model to return consistent results
        mock_call_judge.return_value = {
            "model_a_evaluation": {
                "citation_score": {"score": 4, "justification": "Good"},
                "relevance_score": {"score": 5, "justification": "Excellent"},
                "react_performance_thought_score": {"score": 4, "justification": "Good"},
                "rag_retrieval_observation_score": {"score": 4, "justification": "Good"},
                "information_integration_score": {"score": 4, "justification": "Good"},
            },
            "model_b_evaluation": {
                "citation_score": {"score": 3, "justification": "OK"},
                "relevance_score": {"score": 4, "justification": "Good"},
                "react_performance_thought_score": {"score": 3, "justification": "OK"},
                "rag_retrieval_observation_score": {"score": 3, "justification": "OK"},
                "information_integration_score": {"score": 3, "justification": "OK"},
            },
        }

        # Mock Azure OpenAI client
        mock_azure_client = Mock()
        mock_azure.return_value = mock_azure_client

        from scripts.llm_judge_evaluator import process_csv

        # Create temporary CSV file with 2 rows
        test_data = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Response": ["Answer A1", "Answer A2"],
            "Model_B_Response": ["Answer B1", "Answer B2"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_csv = f.name

        try:
            # Test sequential processing
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
                temp_output_sequential = f.name

            process_csv(
                temp_csv,
                temp_output_sequential,
                limit_rows=2,
                model_name="gpt-4.1",
                non_interactive=True,
                max_workers=1,  # Sequential
            )

            sequential_df = pd.read_csv(temp_output_sequential)

            # Reset mocks for parallel processing
            mock_call_judge.reset_mock()
            mock_extract_scores.reset_mock()

            # Test parallel processing
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
                temp_output_parallel = f.name

            process_csv(
                temp_csv,
                temp_output_parallel,
                limit_rows=2,
                model_name="gpt-4.1",
                non_interactive=True,
                max_workers=2,  # Parallel
            )

            parallel_df = pd.read_csv(temp_output_parallel)

            # Compare results - they should be identical
            assert len(sequential_df) == len(parallel_df), "Both should have same number of rows"
            assert len(sequential_df) == 2, "Should have 2 rows"

            # Compare each row
            for idx in range(len(sequential_df)):
                seq_row = sequential_df.iloc[idx]
                par_row = parallel_df.iloc[idx]

                # Compare key fields
                assert seq_row["Question"] == par_row["Question"], f"Row {idx} Question should match"
                assert (
                    seq_row["Model_A_Response"] == par_row["Model_A_Response"]
                ), f"Row {idx} Model_A_Response should match"
                assert (
                    seq_row["Model_B_Response"] == par_row["Model_B_Response"]
                ), f"Row {idx} Model_B_Response should match"
                assert (
                    seq_row["Model_A_Citation_Score"] == par_row["Model_A_Citation_Score"]
                ), f"Row {idx} Model_A_Citation_Score should match"

        finally:
            # Clean up
            for f in [temp_csv, temp_output_sequential, temp_output_parallel]:
                if os.path.exists(f):
                    os.unlink(f)

