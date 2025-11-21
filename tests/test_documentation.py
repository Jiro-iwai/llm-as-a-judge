"""
Unit tests for documentation completeness.

This module tests that all public functions have proper docstrings
with Args, Returns, and Examples sections where appropriate.
"""

import inspect
import sys
from pathlib import Path

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.collect_responses as collect_responses
import scripts.compare_processing_time as compare_processing_time
import scripts.format_clarity_evaluator as format_clarity_evaluator
import scripts.llm_judge_evaluator as llm_judge_evaluator
import scripts.ragas_llm_judge_evaluator as ragas_llm_judge_evaluator
import scripts.visualize_results as visualize_results


class TestDocstringCompleteness:
    """Tests for docstring completeness across all modules."""

    def check_docstring(self, func, module_name: str, func_name: str) -> list[str]:
        """Check if a function has a complete docstring."""
        issues = []
        
        if func.__doc__ is None:
            issues.append(f"Missing docstring")
            return issues
        
        docstring = func.__doc__.strip()
        
        # Check for Args section (if function has parameters)
        sig = inspect.signature(func)
        params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
        
        if params:
            if "Args:" not in docstring and "Parameters:" not in docstring:
                issues.append(f"Missing Args/Parameters section")
        
        # Check for Returns section (if function has return type annotation)
        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation != type(None):
                if "Returns:" not in docstring and "Return:" not in docstring:
                    issues.append(f"Missing Returns section")
        
        # Check docstring is not too short (at least 20 characters)
        if len(docstring) < 20:
            issues.append(f"Docstring too short (less than 20 characters)")
        
        return issues

    def test_collect_responses_functions_have_docstrings(self):
        """Test that collect_responses.py functions have docstrings."""
        functions_to_check = [
            (collect_responses.format_response, "format_response"),
            (collect_responses.call_api, "call_api"),
            (collect_responses.collect_responses, "collect_responses"),
            (collect_responses.read_questions, "read_questions"),
        ]
        
        all_issues = []
        for func, name in functions_to_check:
            issues = self.check_docstring(func, "collect_responses", name)
            if issues:
                all_issues.append(f"collect_responses.{name}: {', '.join(issues)}")
        
        assert not all_issues, f"Docstring issues found:\n" + "\n".join(all_issues)

    def test_compare_processing_time_functions_have_docstrings(self):
        """Test that compare_processing_time.py functions have docstrings."""
        functions_to_check = [
            (compare_processing_time.extract_processing_times, "extract_processing_times"),
            (compare_processing_time.create_comparison_chart, "create_comparison_chart"),
            (compare_processing_time.create_statistics_chart, "create_statistics_chart"),
            (compare_processing_time.create_summary_table, "create_summary_table"),
        ]
        
        all_issues = []
        for func, name in functions_to_check:
            issues = self.check_docstring(func, "compare_processing_time", name)
            if issues:
                all_issues.append(f"compare_processing_time.{name}: {', '.join(issues)}")
        
        assert not all_issues, f"Docstring issues found:\n" + "\n".join(all_issues)

    def test_visualize_results_functions_have_docstrings(self):
        """Test that visualize_results.py functions have docstrings."""
        functions_to_check = [
            (visualize_results.load_data, "load_data"),
            (visualize_results.prepare_data, "prepare_data"),
            (visualize_results.create_score_comparison_chart, "create_score_comparison_chart"),
            (visualize_results.create_score_distribution_chart, "create_score_distribution_chart"),
            (visualize_results.create_boxplot_chart, "create_boxplot_chart"),
            (visualize_results.create_summary_table, "create_summary_table"),
        ]
        
        all_issues = []
        for func, name in functions_to_check:
            issues = self.check_docstring(func, "visualize_results", name)
            if issues:
                all_issues.append(f"visualize_results.{name}: {', '.join(issues)}")
        
        assert not all_issues, f"Docstring issues found:\n" + "\n".join(all_issues)

    def test_evaluator_functions_have_docstrings(self):
        """Test that evaluator functions have docstrings."""
        functions_to_check = [
            (llm_judge_evaluator.is_gpt5, "is_gpt5"),
            (llm_judge_evaluator.create_user_prompt, "create_user_prompt"),
            (llm_judge_evaluator.call_judge_model, "call_judge_model"),
            (llm_judge_evaluator.extract_scores_from_evaluation, "extract_scores_from_evaluation"),
            (format_clarity_evaluator.parse_final_answer, "parse_final_answer"),
            (format_clarity_evaluator.create_user_prompt, "create_user_prompt"),
            (format_clarity_evaluator.call_judge_model, "call_judge_model"),
            (format_clarity_evaluator.extract_scores_from_evaluation, "extract_scores_from_evaluation"),
            (ragas_llm_judge_evaluator.parse_react_log, "parse_react_log"),
        ]
        
        all_issues = []
        for func, name in functions_to_check:
            issues = self.check_docstring(func, "evaluator", name)
            if issues:
                all_issues.append(f"{name}: {', '.join(issues)}")
        
        assert not all_issues, f"Docstring issues found:\n" + "\n".join(all_issues)


class TestDocstringQuality:
    """Tests for docstring quality (Args, Returns sections)."""

    def check_docstring_args(self, func, module_name: str, func_name: str) -> list[str]:
        """Check if docstring has proper Args section with all parameters."""
        issues = []
        
        if func.__doc__ is None:
            return ["Missing docstring"]
        
        docstring = func.__doc__
        sig = inspect.signature(func)
        params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
        
        if not params:
            return []  # No parameters, Args section not needed
        
        # Check if Args section exists
        if "Args:" not in docstring and "Parameters:" not in docstring:
            issues.append("Missing Args/Parameters section")
            return issues
        
        # Extract Args section
        args_section_start = docstring.find("Args:")
        if args_section_start == -1:
            args_section_start = docstring.find("Parameters:")
        
        if args_section_start == -1:
            return issues
        
        # Find end of Args section (next section or end of docstring)
        args_section = docstring[args_section_start:]
        # Look for next section markers
        next_section_markers = ["Returns:", "Return:", "Raises:", "Examples:", "Example:", "Note:"]
        args_section_end = len(args_section)
        for marker in next_section_markers:
            marker_pos = args_section.find(marker)
            if marker_pos != -1 and marker_pos < args_section_end:
                args_section_end = marker_pos
        
        args_section = args_section[:args_section_end]
        
        # Check if all parameters are documented
        for param in params:
            param_name = param.name
            # Check if parameter name appears in Args section
            if param_name not in args_section:
                issues.append(f"Parameter '{param_name}' not documented in Args section")
        
        return issues

    def test_collect_responses_docstring_args(self):
        """Test that collect_responses.py functions document all parameters."""
        functions_to_check = [
            (collect_responses.call_api, "call_api"),
            (collect_responses.collect_responses, "collect_responses"),
            (collect_responses.read_questions, "read_questions"),
        ]
        
        all_issues = []
        for func, name in functions_to_check:
            issues = self.check_docstring_args(func, "collect_responses", name)
            if issues:
                all_issues.append(f"collect_responses.{name}: {', '.join(issues)}")
        
        assert not all_issues, f"Docstring Args issues found:\n" + "\n".join(all_issues)

    def test_compare_processing_time_docstring_args(self):
        """Test that compare_processing_time.py functions document all parameters."""
        functions_to_check = [
            (compare_processing_time.extract_processing_times, "extract_processing_times"),
            (compare_processing_time.create_comparison_chart, "create_comparison_chart"),
            (compare_processing_time.create_statistics_chart, "create_statistics_chart"),
            (compare_processing_time.create_summary_table, "create_summary_table"),
        ]
        
        all_issues = []
        for func, name in functions_to_check:
            issues = self.check_docstring_args(func, "compare_processing_time", name)
            if issues:
                all_issues.append(f"compare_processing_time.{name}: {', '.join(issues)}")
        
        assert not all_issues, f"Docstring Args issues found:\n" + "\n".join(all_issues)

    def test_visualize_results_docstring_args(self):
        """Test that visualize_results.py functions document all parameters."""
        functions_to_check = [
            (visualize_results.load_data, "load_data"),
            (visualize_results.prepare_data, "prepare_data"),
            (visualize_results.create_score_comparison_chart, "create_score_comparison_chart"),
            (visualize_results.create_score_distribution_chart, "create_score_distribution_chart"),
            (visualize_results.create_boxplot_chart, "create_boxplot_chart"),
            (visualize_results.create_summary_table, "create_summary_table"),
        ]
        
        all_issues = []
        for func, name in functions_to_check:
            issues = self.check_docstring_args(func, "visualize_results", name)
            if issues:
                all_issues.append(f"visualize_results.{name}: {', '.join(issues)}")
        
        assert not all_issues, f"Docstring Args issues found:\n" + "\n".join(all_issues)

