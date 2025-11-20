"""Unit tests for ragas_llm_judge_evaluator.py"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas_llm_judge_evaluator import (
    parse_react_log,
    get_model_config,
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

