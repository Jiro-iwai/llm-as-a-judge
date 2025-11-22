"""
Unit tests for process_csv functions in evaluator scripts.

This module tests the main processing functions that read CSV files,
call APIs, and write results.
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

from scripts.format_clarity_evaluator import process_csv as format_process_csv
from scripts.llm_judge_evaluator import process_csv as llm_process_csv
from scripts.ragas_llm_judge_evaluator import process_csv as ragas_process_csv


class TestFormatClarityProcessCsv:
    """Tests for format_clarity_evaluator.process_csv"""

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("builtins.input")
    @patch("scripts.format_clarity_evaluator.log_error")
    @patch("scripts.format_clarity_evaluator.log_success")
    @patch("scripts.format_clarity_evaluator.log_warning")
    @patch("scripts.format_clarity_evaluator.log_info")
    @patch("scripts.format_clarity_evaluator.tqdm")
    @patch("scripts.format_clarity_evaluator.parse_final_answer")
    @patch("scripts.format_clarity_evaluator.extract_scores_from_evaluation")
    @patch("scripts.format_clarity_evaluator.call_judge_model")
    @patch("scripts.format_clarity_evaluator.OpenAI")
    @patch("scripts.format_clarity_evaluator.AzureOpenAI")
    def test_process_csv_with_header(self, mock_azure, mock_openai, mock_call_judge, mock_extract_scores, mock_parse_final, mock_tqdm, mock_log_info, mock_log_warning, mock_log_success, mock_log_error, mock_input):
        """Test process_csv with CSV file that has header row"""
        # Mock input to avoid interactive prompt
        mock_input.return_value = "y"
        
        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Mock parse_final_answer to return simple answers
        mock_parse_final.side_effect = lambda x: f"Parsed: {x[:10]}"
        
        # Mock extract_scores_from_evaluation
        mock_extract_scores.return_value = (4, "Good formatting")
        
        # Create test CSV with header
        test_data = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Response": ["Answer A1", "Answer A2"],
            "Model_B_Response": ["Answer B1", "Answer B2"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            output_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv").name

            # Mock API response
            mock_call_judge.return_value = {
                "format_clarity_score": 4,
                "format_clarity_justification": "Good formatting"
            }

            # Mock Azure client
            mock_client = MagicMock()
            mock_azure.return_value = mock_client

            format_process_csv(temp_file, output_file, limit_rows=2)

            # Verify output file was created
            assert os.path.exists(output_file)

            # Verify API was called (limit_rows=2 so should be called twice)
            assert mock_call_judge.call_count == 2

        finally:
            for f in [temp_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)

    @patch("scripts.format_clarity_evaluator.AzureOpenAI")
    @patch("scripts.format_clarity_evaluator.OpenAI")
    @patch("scripts.format_clarity_evaluator.call_judge_model")
    @patch("scripts.format_clarity_evaluator.tqdm")
    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    def test_process_csv_without_header(self, mock_tqdm, mock_openai, mock_azure, mock_call_judge):
        """Test process_csv with CSV file without header row"""
        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Create test CSV without header
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("Q1,Answer A1,Answer B1\n")
            f.write("Q2,Answer A2,Answer B2\n")
            temp_file = f.name

        try:
            output_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv").name

            # Mock API response
            mock_call_judge.return_value = {
                "format_clarity_score": 4,
                "format_clarity_justification": "Good formatting"
            }

            # Mock Azure client
            mock_client = MagicMock()
            mock_azure.return_value = mock_client

            format_process_csv(temp_file, output_file, limit_rows=2)

            # Verify output file was created
            assert os.path.exists(output_file)

        finally:
            for f in [temp_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)

    @patch("scripts.format_clarity_evaluator.AzureOpenAI")
    @patch("scripts.format_clarity_evaluator.OpenAI")
    def test_process_csv_file_not_found(self, mock_openai, mock_azure):
        """Test process_csv with non-existent file"""
        with pytest.raises(SystemExit):
            format_process_csv("nonexistent_file.csv", "output.csv")

    @patch("scripts.format_clarity_evaluator.AzureOpenAI")
    @patch("scripts.format_clarity_evaluator.OpenAI")
    @patch("scripts.format_clarity_evaluator.call_judge_model")
    @patch("scripts.format_clarity_evaluator.tqdm")
    @patch("scripts.format_clarity_evaluator.log_error")
    @patch("scripts.format_clarity_evaluator.log_info")
    @patch("scripts.format_clarity_evaluator.log_warning")
    @patch("builtins.input")
    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    def test_process_csv_missing_columns(self, mock_input, mock_log_warning, mock_log_info, mock_log_error, mock_tqdm, mock_call_judge, mock_openai, mock_azure):
        """Test process_csv with CSV missing required columns after mapping"""
        # Mock input to avoid interactive prompt
        mock_input.return_value = "y"
        
        # Mock tqdm to return the iterable directly to avoid progress bar delays
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Mock call_judge_model to avoid actual API calls - this should not be called
        mock_call_judge.return_value = {
            "format_clarity_score": 4,
            "format_clarity_justification": "Good formatting"
        }
        
        # Mock Azure client to avoid actual initialization
        mock_client = MagicMock()
        mock_azure.return_value = mock_client
        
        # Create test CSV with header that triggers header detection but columns don't map correctly
        # and after mapping, we still don't have all required columns
        # Use column names that trigger header detection but don't map to required columns
        test_data = pd.DataFrame({
            "Question": ["Q1", "Q2"],  # This maps correctly
            "Wrong_Column": ["A1", "A2"],  # This doesn't map to Model_A_Response or Model_B_Response
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            with pytest.raises(SystemExit):
                format_process_csv(temp_file, "output.csv")
            
            # Verify error was logged about missing columns
            assert mock_log_error.called
            # Verify API was not called since validation failed early
            assert mock_call_judge.call_count == 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("builtins.input")
    @patch("scripts.format_clarity_evaluator.log_error")
    @patch("scripts.format_clarity_evaluator.log_success")
    @patch("scripts.format_clarity_evaluator.log_warning")
    @patch("scripts.format_clarity_evaluator.log_info")
    @patch("scripts.format_clarity_evaluator.tqdm")
    @patch("scripts.format_clarity_evaluator.parse_final_answer")
    @patch("scripts.format_clarity_evaluator.extract_scores_from_evaluation")
    @patch("scripts.format_clarity_evaluator.call_judge_model")
    @patch("scripts.format_clarity_evaluator.OpenAI")
    @patch("scripts.format_clarity_evaluator.AzureOpenAI")
    @patch("scripts.format_clarity_evaluator.pd.DataFrame.to_csv")
    def test_process_csv_no_cost_estimate_message(self, mock_to_csv, mock_azure, mock_openai, mock_call_judge, mock_extract_scores, mock_parse_final, mock_tqdm, mock_log_info, mock_log_warning, mock_log_success, mock_log_error, mock_input):
        """Test that cost estimate message is removed and replaced with guidance message"""
        # Mock input to avoid interactive prompt
        mock_input.return_value = "y"
        
        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Mock parse_final_answer
        mock_parse_final.side_effect = lambda x: f"Parsed: {x[:10]}"
        
        # Mock extract_scores_from_evaluation
        mock_extract_scores.return_value = (4, "Good formatting")
        
        # Mock call_judge_model
        mock_call_judge.return_value = {"format_clarity_evaluation": {"score": 4, "justification": "Good"}}
        
        # Mock Azure client
        mock_client = MagicMock()
        mock_azure.return_value = mock_client
        
        # Create test CSV with 15 rows (should trigger cost message area)
        test_data = pd.DataFrame({
            "Question": [f"Q{i}" for i in range(15)],
            "Model_A_Response": [f"A{i}" for i in range(15)],
            "Model_B_Response": [f"B{i}" for i in range(15)],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            format_process_csv(temp_file, "output.csv", non_interactive=True)
            
            # Verify that cost estimate message (with $) is NOT logged
            warning_calls = [str(call) for call in mock_log_warning.call_args_list]
            cost_estimate_found = any("Estimated cost" in str(call) or "$" in str(call) for call in warning_calls)
            assert not cost_estimate_found, "Cost estimate message should be removed"
            
            # Verify that guidance message is logged instead
            guidance_found = any("API costs" in str(call) or "-n flag" in str(call) for call in warning_calls)
            assert guidance_found, "Guidance message should be logged"
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("scripts.format_clarity_evaluator.AzureOpenAI")
    @patch("scripts.format_clarity_evaluator.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_process_csv_standard_openai(self, mock_openai, mock_azure):
        """Test process_csv with standard OpenAI client"""
        test_data = pd.DataFrame({
            "Question": ["Q1"],
            "Model_A_Response": ["Answer A1"],
            "Model_B_Response": ["Answer B1"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            output_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv").name

            # Mock standard OpenAI client
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            with patch("scripts.format_clarity_evaluator.call_judge_model") as mock_call:
                mock_call.return_value = {
                    "format_clarity_score": 4,
                    "format_clarity_justification": "Good"
                }

                format_process_csv(temp_file, output_file, limit_rows=1)

                # Verify standard OpenAI was used
                mock_openai.assert_called_once()
                assert os.path.exists(output_file)

        finally:
            for f in [temp_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)


class TestLLMJudgeProcessCsv:
    """Tests for llm_judge_evaluator.process_csv"""

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("builtins.input")
    @patch("scripts.llm_judge_evaluator.log_error")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.log_success")
    @patch("scripts.llm_judge_evaluator.log_warning")
    @patch("scripts.llm_judge_evaluator.log_info")
    @patch("scripts.llm_judge_evaluator.tqdm")
    @patch("scripts.llm_judge_evaluator.extract_scores_from_evaluation")
    @patch("scripts.llm_judge_evaluator.call_judge_model")
    @patch("scripts.llm_judge_evaluator.OpenAI")
    @patch("scripts.llm_judge_evaluator.AzureOpenAI")
    def test_process_csv_with_header(self, mock_azure, mock_openai, mock_call_judge, mock_extract_scores, mock_tqdm, mock_log_info, mock_log_warning, mock_log_success, mock_log_section, mock_log_error, mock_input):
        """Test process_csv with CSV file that has header row"""
        # Mock input to avoid interactive prompt
        mock_input.return_value = "y"
        
        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Mock extract_scores_from_evaluation - it's called twice (once for model_a, once for model_b)
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
        
        test_data = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Response": ["Answer A1", "Answer A2"],
            "Model_B_Response": ["Answer B1", "Answer B2"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            output_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv").name

            # Mock API response
            mock_call_judge.return_value = {
                "model_a_evaluation": {
                    "citation_score": {"score": 4, "justification": "Good"},
                    "relevance_score": {"score": 5, "justification": "Excellent"},
                },
                "model_b_evaluation": {
                    "citation_score": {"score": 3, "justification": "OK"},
                    "relevance_score": {"score": 4, "justification": "Good"},
                },
            }

            # Mock Azure client
            mock_client = MagicMock()
            mock_azure.return_value = mock_client

            llm_process_csv(temp_file, output_file, limit_rows=2)

            # Verify output file was created
            assert os.path.exists(output_file)

            # Verify API was called
            assert mock_call_judge.call_count == 2

        finally:
            for f in [temp_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("builtins.input")
    @patch("scripts.llm_judge_evaluator.log_error")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.log_success")
    @patch("scripts.llm_judge_evaluator.log_warning")
    @patch("scripts.llm_judge_evaluator.log_info")
    @patch("scripts.llm_judge_evaluator.tqdm")
    @patch("scripts.llm_judge_evaluator.extract_scores_from_evaluation")
    @patch("scripts.llm_judge_evaluator.call_judge_model")
    @patch("scripts.llm_judge_evaluator.OpenAI")
    @patch("scripts.llm_judge_evaluator.AzureOpenAI")
    @patch("scripts.llm_judge_evaluator.pd.DataFrame.to_csv")
    def test_process_csv_no_cost_estimate_message(self, mock_to_csv, mock_azure, mock_openai, mock_call_judge, mock_extract_scores, mock_tqdm, mock_log_info, mock_log_warning, mock_log_success, mock_log_section, mock_log_error, mock_input):
        """Test that cost estimate message is removed and replaced with guidance message"""
        # Mock input to avoid interactive prompt
        mock_input.return_value = "y"
        
        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Mock extract_scores_from_evaluation
        def mock_extract(evaluation, model_key):
            return {
                "citation_score": 5,
                "relevance_score": 5,
                "citation_justification": "Good",
                "relevance_justification": "Excellent",
                "react_performance_thought_score": 5,
                "react_performance_thought_justification": "Good",
                "rag_retrieval_observation_score": 5,
                "rag_retrieval_observation_justification": "Good",
                "information_integration_score": 5,
                "information_integration_justification": "Good",
            }
        mock_extract_scores.side_effect = mock_extract
        
        # Mock call_judge_model
        mock_call_judge.return_value = {
            "model_a_evaluation": {"citation_score": {"score": 5}},
            "model_b_evaluation": {"citation_score": {"score": 5}},
        }
        
        # Mock Azure client
        mock_client = MagicMock()
        mock_azure.return_value = mock_client
        
        # Create test CSV with 15 rows (should trigger cost message area)
        test_data = pd.DataFrame({
            "Question": [f"Q{i}" for i in range(15)],
            "Model_A_Response": [f"A{i}" for i in range(15)],
            "Model_B_Response": [f"B{i}" for i in range(15)],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            llm_process_csv(temp_file, "output.csv", non_interactive=True)
            
            # Verify that cost estimate message (with $) is NOT logged
            warning_calls = [str(call) for call in mock_log_warning.call_args_list]
            cost_estimate_found = any("推定コスト" in str(call) or "$" in str(call) for call in warning_calls)
            assert not cost_estimate_found, "Cost estimate message should be removed"
            
            # Verify that guidance message is logged instead
            guidance_found = any("APIコストがかかる" in str(call) or "-nフラグ" in str(call) for call in warning_calls)
            assert guidance_found, "Guidance message should be logged"
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("scripts.llm_judge_evaluator.AzureOpenAI")
    @patch("scripts.llm_judge_evaluator.OpenAI")
    def test_process_csv_file_not_found(self, mock_openai, mock_azure):
        """Test process_csv with non-existent file"""
        with pytest.raises(SystemExit):
            llm_process_csv("nonexistent_file.csv", "output.csv")


class TestRagasProcessCsv:
    """Tests for ragas_llm_judge_evaluator.process_csv"""

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("scripts.ragas_llm_judge_evaluator.log_error")
    @patch("scripts.ragas_llm_judge_evaluator.log_section")
    @patch("scripts.ragas_llm_judge_evaluator.log_success")
    @patch("scripts.ragas_llm_judge_evaluator.log_warning")
    @patch("scripts.ragas_llm_judge_evaluator.log_info")
    @patch("scripts.ragas_llm_judge_evaluator.tqdm")
    @patch("scripts.ragas_llm_judge_evaluator.parse_react_log")
    @patch("scripts.ragas_llm_judge_evaluator.evaluate_with_ragas")
    @patch("scripts.ragas_llm_judge_evaluator.initialize_azure_openai_for_ragas")
    def test_process_csv_with_header(self, mock_init, mock_evaluate, mock_parse_react, mock_tqdm, mock_log_info, mock_log_warning, mock_log_success, mock_log_section, mock_log_error):
        """Test process_csv with CSV file that has header row"""
        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Mock parse_react_log to return simple parsed data
        mock_parse_react.return_value = ("Final Answer", ["Context1", "Context2"])
        
        test_data = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Response": ["Answer A1", "Answer A2"],
            "Model_B_Response": ["Answer B1", "Answer B2"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            output_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv").name

            # Mock Ragas evaluation results
            mock_result_a = pd.DataFrame({
                "question": ["Q1", "Q2"],
                "Model_A_faithfulness_score": [0.8, 0.9],
            })
            mock_result_b = pd.DataFrame({
                "question": ["Q1", "Q2"],
                "Model_B_faithfulness_score": [0.7, 0.85],
            })

            mock_evaluate.side_effect = [mock_result_a, mock_result_b]
            mock_llm = MagicMock()
            mock_client = MagicMock()
            mock_embeddings = MagicMock()
            mock_init.return_value = (mock_llm, mock_client, mock_embeddings)

            ragas_process_csv(temp_file, output_file, limit_rows=2)

            # Verify output file was created
            assert os.path.exists(output_file)

            # Verify evaluation was called
            assert mock_evaluate.call_count == 2

        finally:
            for f in [temp_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)

    @patch("scripts.ragas_llm_judge_evaluator.initialize_azure_openai_for_ragas")
    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    def test_process_csv_file_not_found(self, mock_init):
        """Test process_csv with non-existent file"""
        # Mock initialize to return tuple
        mock_llm = MagicMock()
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_init.return_value = (mock_llm, mock_client, mock_embeddings)
        
        with pytest.raises(SystemExit):
            ragas_process_csv("nonexistent_file.csv", "output.csv")

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/", "AZURE_OPENAI_API_KEY": "test-key"})
    @patch("scripts.ragas_llm_judge_evaluator.log_error")
    @patch("scripts.ragas_llm_judge_evaluator.log_section")
    @patch("scripts.ragas_llm_judge_evaluator.log_success")
    @patch("scripts.ragas_llm_judge_evaluator.log_warning")
    @patch("scripts.ragas_llm_judge_evaluator.log_info")
    @patch("scripts.ragas_llm_judge_evaluator.tqdm")
    @patch("scripts.ragas_llm_judge_evaluator.parse_react_log")
    @patch("scripts.ragas_llm_judge_evaluator.evaluate_with_ragas")
    @patch("scripts.ragas_llm_judge_evaluator.initialize_azure_openai_for_ragas")
    def test_process_csv_with_four_metrics(self, mock_init, mock_evaluate, mock_parse_react, mock_tqdm, mock_log_info, mock_log_warning, mock_log_success, mock_log_section, mock_log_error):
        """Test process_csv with all four metrics (faithfulness, answer_relevance, context_precision, context_recall)"""
        # Mock tqdm to return the iterable directly
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Mock parse_react_log to return simple parsed data
        mock_parse_react.return_value = ("Final Answer", ["Context1", "Context2"])
        
        test_data = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Response": ["Answer A1", "Answer A2"],
            "Model_B_Response": ["Answer B1", "Answer B2"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            output_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv").name

            # Mock Ragas evaluation results with all four metrics
            metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
            mock_result_a = pd.DataFrame({
                "question": ["Q1", "Q2"],
                **{f"Model_A_{metric}_score": [0.8 + i * 0.05, 0.9 + i * 0.05] for i, metric in enumerate(metrics)}
            })
            mock_result_b = pd.DataFrame({
                "question": ["Q1", "Q2"],
                **{f"Model_B_{metric}_score": [0.7 + i * 0.05, 0.85 + i * 0.05] for i, metric in enumerate(metrics)}
            })

            mock_evaluate.side_effect = [mock_result_a, mock_result_b]
            mock_llm = MagicMock()
            mock_client = MagicMock()
            mock_embeddings = MagicMock()
            mock_init.return_value = (mock_llm, mock_client, mock_embeddings)

            ragas_process_csv(temp_file, output_file, limit_rows=2, metric_names=metrics)

            # Verify evaluate_with_ragas was called twice (once for each model) with correct metrics
            assert mock_evaluate.call_count == 2
            
            # Check that both calls used the correct metrics
            call_a = mock_evaluate.call_args_list[0]
            call_b = mock_evaluate.call_args_list[1]
            assert call_a.kwargs["metric_names"] == metrics
            assert call_b.kwargs["metric_names"] == metrics

            # Verify output file was created
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            
            # Verify all four metric columns are present for both models
            for metric in metrics:
                assert f"Model_A_{metric}_score" in output_df.columns
                assert f"Model_B_{metric}_score" in output_df.columns
        finally:
            for f in [temp_file, output_file]:
                if os.path.exists(f):
                    os.unlink(f)

