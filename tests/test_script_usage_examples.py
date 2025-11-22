"""
Tests for script usage examples consistency.

This test ensures that all script docstrings and log messages
use the correct 'scripts/' prefix in usage examples.
"""

import re
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestScriptUsageExamples:
    """Tests for script usage examples consistency"""

    def test_llm_judge_evaluator_usage_has_scripts_prefix(self):
        """Test that llm_judge_evaluator.py docstring uses scripts/ prefix"""
        script_path = Path(__file__).parent.parent / "scripts" / "llm_judge_evaluator.py"
        content = script_path.read_text(encoding="utf-8")
        
        # Check docstring Usage section
        usage_pattern = r"Usage:\s*\n\s*python\s+llm_judge_evaluator\.py"
        assert not re.search(usage_pattern, content), (
            "llm_judge_evaluator.py docstring should use 'python scripts/llm_judge_evaluator.py'"
        )
        
        # Should use scripts/ prefix
        correct_pattern = r"Usage:\s*\n\s*python\s+scripts/llm_judge_evaluator\.py"
        assert re.search(correct_pattern, content), (
            "llm_judge_evaluator.py docstring should use 'python scripts/llm_judge_evaluator.py'"
        )

    def test_format_clarity_evaluator_usage_has_scripts_prefix(self):
        """Test that format_clarity_evaluator.py docstring uses scripts/ prefix"""
        script_path = Path(__file__).parent.parent / "scripts" / "format_clarity_evaluator.py"
        content = script_path.read_text(encoding="utf-8")
        
        # Check docstring Usage section
        usage_pattern = r"Usage:\s*\n\s*python\s+format_clarity_evaluator\.py"
        assert not re.search(usage_pattern, content), (
            "format_clarity_evaluator.py docstring should use 'python scripts/format_clarity_evaluator.py'"
        )
        
        # Should use scripts/ prefix
        correct_pattern = r"Usage:\s*\n\s*python\s+scripts/format_clarity_evaluator\.py"
        assert re.search(correct_pattern, content), (
            "format_clarity_evaluator.py docstring should use 'python scripts/format_clarity_evaluator.py'"
        )

    def test_ragas_llm_judge_evaluator_usage_has_scripts_prefix(self):
        """Test that ragas_llm_judge_evaluator.py docstring uses scripts/ prefix"""
        script_path = Path(__file__).parent.parent / "scripts" / "ragas_llm_judge_evaluator.py"
        content = script_path.read_text(encoding="utf-8")
        
        # Check docstring Usage section
        usage_pattern = r"Usage:\s*\n\s*python\s+ragas_llm_judge_evaluator\.py"
        assert not re.search(usage_pattern, content), (
            "ragas_llm_judge_evaluator.py docstring should use 'python scripts/ragas_llm_judge_evaluator.py'"
        )
        
        # Should use scripts/ prefix
        correct_pattern = r"Usage:\s*\n\s*python\s+scripts/ragas_llm_judge_evaluator\.py"
        assert re.search(correct_pattern, content), (
            "ragas_llm_judge_evaluator.py docstring should use 'python scripts/ragas_llm_judge_evaluator.py'"
        )

    def test_collect_responses_usage_has_scripts_prefix(self):
        """Test that collect_responses.py docstring uses scripts/ prefix"""
        script_path = Path(__file__).parent.parent / "scripts" / "collect_responses.py"
        content = script_path.read_text(encoding="utf-8")
        
        # Check docstring Usage section
        usage_pattern = r"Usage:\s*\n\s*python\s+collect_responses\.py"
        assert not re.search(usage_pattern, content), (
            "collect_responses.py docstring should use 'python scripts/collect_responses.py'"
        )
        
        # Should use scripts/ prefix
        correct_pattern = r"Usage:\s*\n\s*python\s+scripts/collect_responses\.py"
        assert re.search(correct_pattern, content), (
            "collect_responses.py docstring should use 'python scripts/collect_responses.py'"
        )

    def test_collect_responses_log_messages_have_scripts_prefix(self):
        """Test that collect_responses.py log messages use scripts/ prefix"""
        script_path = Path(__file__).parent.parent / "scripts" / "collect_responses.py"
        content = script_path.read_text(encoding="utf-8")
        
        # Check log messages for next steps
        incorrect_patterns = [
            r'python llm_judge_evaluator\.py',
            r'python ragas_llm_judge_evaluator\.py',
        ]
        
        for pattern in incorrect_patterns:
            assert not re.search(pattern, content), (
                f"collect_responses.py log messages should use 'python scripts/...' instead of 'python {pattern}'"
            )
        
        # Should use scripts/ prefix
        correct_patterns = [
            r'python scripts/llm_judge_evaluator\.py',
            r'python scripts/ragas_llm_judge_evaluator\.py',
        ]
        
        for pattern in correct_patterns:
            assert re.search(pattern, content), (
                f"collect_responses.py log messages should use '{pattern}'"
            )

