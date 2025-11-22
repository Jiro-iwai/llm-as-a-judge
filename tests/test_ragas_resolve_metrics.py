"""
Unit tests for resolve_metrics function in ragas_llm_judge_evaluator.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ragas_llm_judge_evaluator import resolve_metrics, BASIC_METRICS, METRICS_WITH_REFERENCE, DEFAULT_METRICS


class TestResolveMetrics:
    """Tests for resolve_metrics function"""

    def test_resolve_metrics_with_explicit_metrics(self):
        """Test resolve_metrics with explicit metrics list"""
        metrics = ["faithfulness", "context_precision"]
        result = resolve_metrics(metrics=metrics, preset=None)
        assert result == metrics
        assert isinstance(result, list)

    def test_resolve_metrics_with_basic_preset(self):
        """Test resolve_metrics with basic preset"""
        result = resolve_metrics(metrics=None, preset="basic")
        assert result == BASIC_METRICS
        assert isinstance(result, list)

    def test_resolve_metrics_with_with_reference_preset(self):
        """Test resolve_metrics with with_reference preset"""
        result = resolve_metrics(metrics=None, preset="with_reference")
        assert result == METRICS_WITH_REFERENCE
        assert isinstance(result, list)

    def test_resolve_metrics_with_default(self):
        """Test resolve_metrics with no arguments (default behavior)"""
        result = resolve_metrics(metrics=None, preset=None)
        assert result == list(DEFAULT_METRICS)  # DEFAULT_METRICS is BASIC_METRICS
        assert isinstance(result, list)

    def test_resolve_metrics_metrics_takes_precedence(self):
        """Test that explicit metrics take precedence over preset"""
        metrics = ["faithfulness"]
        result = resolve_metrics(metrics=metrics, preset="with_reference")
        assert result == metrics
        assert len(result) == 1

    def test_resolve_metrics_returns_list(self):
        """Test that resolve_metrics always returns a list"""
        result1 = resolve_metrics(metrics=["faithfulness"], preset=None)
        result2 = resolve_metrics(metrics=None, preset="basic")
        result3 = resolve_metrics(metrics=None, preset=None)
        
        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert isinstance(result3, list)

