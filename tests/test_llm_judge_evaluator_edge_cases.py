"""
Unit tests for llm_judge_evaluator.py edge cases and error handling.

This module tests edge cases and error handling paths that are not
covered by other tests.
"""

import sys
from pathlib import Path


# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


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

        # call_judge_model signature: (client, question, model_a_response, model_b_response, model_name, ...)
        # When client is None, it will fail after retries and return None
        result = call_judge_model(
            None, "question", "response_a", "response_b", "model", timeout=10
        )  # type: ignore[call-arg]

        # Should return None after all retries fail
        assert result is None

