"""Unit tests for ragas_llm_judge_evaluator.py"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ragas_llm_judge_evaluator import (
    parse_react_log,
    get_model_config,
    initialize_azure_openai_for_ragas,
    evaluate_with_ragas,
)


class TestParseReactLog:
    """Tests for parse_react_log function"""

    def test_parse_react_log_with_final_answer(self):
        """Test parsing ReAct log with final answer"""
        log = """## 📝 Task タスク
---
情報検索

## 🤖 LLM Thought Process 思考
---
思考プロセスです

## 📚 Raw Search Results (Cleaned) 観察
---
検索結果1
検索結果2

## ✅ Final Answer 回答
---
最終回答です

## 🔗 URLs URL
---
https://example.com"""
        final_answer, contexts = parse_react_log(log)

        assert "最終回答です" in final_answer
        # Contexts should include thought process and search results
        contexts_text = " ".join(contexts)
        assert "思考" in contexts_text or "検索結果" in contexts_text

    def test_parse_react_log_without_final_answer(self):
        """Test parsing ReAct log without final answer"""
        log = """## 📝 Task タスク
---
情報検索

## 🤖 LLM Thought Process 思考
---
思考プロセスです"""
        final_answer, contexts = parse_react_log(log)

        # parse_react_log returns "No answer provided" when no final answer found
        assert final_answer == "No answer provided" or final_answer == ""
        assert len(contexts) > 0

    def test_parse_react_log_empty_string(self):
        """Test parsing empty string"""
        final_answer, contexts = parse_react_log("")
        # parse_react_log returns "No answer provided" when no final answer found
        assert final_answer == "No answer provided" or final_answer == ""
        assert isinstance(contexts, list)

    def test_parse_react_log_extracts_contexts(self):
        """Test that contexts are extracted correctly"""
        log = """## 🤖 LLM Thought Process 思考
---
思考1

## 📚 Raw Search Results (Cleaned) 観察
---
結果1

## ✅ Final Answer 回答
---
回答"""
        final_answer, contexts = parse_react_log(log)

        assert "回答" in final_answer
        # Should extract thought process and search results as contexts
        assert len(contexts) >= 1


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

    def test_get_model_config_partial_match(self):
        """Test partial matching (e.g., 'gpt5' -> 'gpt-5')"""
        config1 = get_model_config("gpt5")
        config2 = get_model_config("gpt-5")
        assert config1 == config2

    def test_get_model_config_unknown_model(self):
        """Test getting config for unknown model (should return default)"""
        config = get_model_config("unknown-model")
        # Should return default (gpt-4.1) config
        assert config["temperature"] == 0.7


class TestInitializeAzureOpenAIForRagas:
    """Tests for initialize_azure_openai_for_ragas function"""

    @patch("langchain_openai.AzureChatOpenAI")
    @patch("scripts.ragas_llm_judge_evaluator.os.getenv")
    def test_initialize_azure_openai_for_ragas_success(self, mock_getenv, mock_azure_chat):
        """Test successful Azure OpenAI initialization"""
        mock_getenv.side_effect = lambda key, default=None: {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4.1",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "text-embedding-3-large",
        }.get(key, default)
        
        mock_client = Mock()
        mock_ragas_llm = Mock()
        mock_embeddings = Mock()
        mock_azure_chat.return_value = mock_client

        result = initialize_azure_openai_for_ragas(model_name="gpt-4.1")

        # Function returns a tuple (ragas_llm, client, embeddings)
        assert result is not None
        assert len(result) == 3
        mock_azure_chat.assert_called_once()

    @patch("langchain_openai.AzureChatOpenAI")
    @patch("scripts.ragas_llm_judge_evaluator.os.getenv")
    def test_initialize_azure_openai_for_ragas_with_model_config(self, mock_getenv, mock_azure_chat):
        """Test initialization with model config"""
        mock_getenv.side_effect = lambda key, default=None: {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-5",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "text-embedding-3-large",
        }.get(key, default)
        
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_azure_chat.return_value = mock_client

        result = initialize_azure_openai_for_ragas(model_name="gpt-5")

        assert result is not None
        assert len(result) == 3
        # Verify that model config is used (gpt-5 has temperature 1.0)
        call_kwargs = mock_azure_chat.call_args[1]
        assert call_kwargs["temperature"] == 1.0


class TestEvaluateWithRagas:
    """Tests for evaluate_with_ragas function"""

    @patch("scripts.ragas_llm_judge_evaluator.evaluate")
    @patch("scripts.ragas_llm_judge_evaluator.log_error")
    @patch("scripts.ragas_llm_judge_evaluator.log_warning")
    @patch("scripts.ragas_llm_judge_evaluator.log_info")
    def test_evaluate_with_ragas_error_maintains_column_structure(self, mock_log_info, mock_log_warning, mock_log_error, mock_evaluate):
        """Test that error handling maintains expected column structure"""
        mock_llm = Mock()
        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        contexts_list = [["C1"], ["C2"]]
        metrics_to_run = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

        # Simulate an error during evaluation
        mock_evaluate.side_effect = ValueError("Evaluation failed")

        result = evaluate_with_ragas(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            llm=mock_llm,
            model_name="Model_A",
            metric_names=metrics_to_run,
        )

        # Verify that result is a DataFrame
        import pandas as pd
        assert isinstance(result, pd.DataFrame)

        # Note: context_precision and context_recall are skipped when reference is not provided
        # So only faithfulness and answer_relevance columns should be present
        expected_columns = ["question", "Model_A_faithfulness_score", "Model_A_answer_relevance_score"]

        assert set(result.columns) == set(expected_columns)

        # Verify that all score columns have None values (only for metrics that weren't skipped)
        metrics_that_were_run = ["faithfulness", "answer_relevance"]
        for metric in metrics_that_were_run:
            col_name = f"Model_A_{metric}_score"
            assert col_name in result.columns
            # Check that all values are None/NaN
            col_values = result[col_name].values
            assert all(pd.isna(val) or val is None for val in col_values)

        # Verify that question column has correct values
        assert list(result["question"]) == questions

        # Verify that error was logged
        assert mock_log_error.called

    @patch("scripts.ragas_llm_judge_evaluator.evaluate")
    @patch("scripts.ragas_llm_judge_evaluator.log_warning")
    @patch("scripts.ragas_llm_judge_evaluator.log_info")
    def test_evaluate_with_ragas_skips_metrics_requiring_reference(self, mock_log_info, mock_log_warning, mock_evaluate):
        """Test that metrics requiring reference are skipped when reference is not provided"""
        import pandas as pd
        mock_llm = Mock()
        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        contexts_list = [["C1"], ["C2"]]
        metrics_to_run = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

        # Mock successful evaluation result (only for metrics that don't require reference)
        # Note: Ragas returns column names based on metric object names (answer_relevancy, not answer_relevance)
        mock_result = pd.DataFrame({
            "faithfulness": [0.8, 0.9],
            "answer_relevancy": [0.7, 0.85],  # Ragas uses "answer_relevancy" as column name
        })
        mock_dataset = Mock()
        mock_dataset.to_pandas = Mock(return_value=mock_result)
        mock_evaluate.return_value = mock_dataset

        result = evaluate_with_ragas(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            llm=mock_llm,
            model_name="Model_A",
            metric_names=metrics_to_run,
            references=None,  # No reference provided
        )

        # Verify that warning was logged about skipping metrics requiring reference
        assert mock_log_warning.called
        warning_calls = [str(call) for call in mock_log_warning.call_args_list]
        assert any("context_precision" in str(call) or "context_recall" in str(call) for call in warning_calls)

        # Verify that result only contains metrics that don't require reference
        assert isinstance(result, pd.DataFrame)
        # Should only have faithfulness and answer_relevance columns (Ragas returns scores without question column)
        expected_columns = ["Model_A_faithfulness_score", "Model_A_answer_relevance_score"]
        assert set(result.columns) == set(expected_columns)
        
        # Verify that evaluate was called with only metrics that don't require reference
        assert mock_evaluate.called
        # Check that evaluate was called with correct metrics (should not include context_precision/context_recall)
        call_args = mock_evaluate.call_args
        metrics_passed = call_args.kwargs.get("metrics", [])
        # Should only have 2 metrics (faithfulness and answer_relevance)
        assert len(metrics_passed) == 2


