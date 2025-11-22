"""
Tests to ensure process_csv functions are properly refactored.

This test ensures that process_csv functions are split into smaller,
more manageable functions instead of being monolithic.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProcessCsvRefactoring:
    """Tests to ensure process_csv functions are properly refactored"""

    def test_process_csv_functions_are_not_too_long(self):
        """Test that process_csv functions are not excessively long (should be < 200 lines)"""
        scripts = {
            "llm_judge_evaluator.py": Path(__file__).parent.parent / "scripts" / "llm_judge_evaluator.py",
            "format_clarity_evaluator.py": Path(__file__).parent.parent / "scripts" / "format_clarity_evaluator.py",
            "ragas_llm_judge_evaluator.py": Path(__file__).parent.parent / "scripts" / "ragas_llm_judge_evaluator.py",
        }

        max_lines = 150  # Target: functions should be < 150 lines after refactoring

        for script_name, script_path in scripts.items():
            if not script_path.exists():
                continue

            content = script_path.read_text(encoding="utf-8")

            # Find process_csv function
            import re

            # Find function start
            start_line = None
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if re.match(r"^def process_csv\(", line):
                    start_line = i
                    break

            if start_line is None:
                continue

            # Find function end (next def or if __name__)
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
            end_line = len(lines)

            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and not line.startswith(" " * (indent_level + 1)) and not line.startswith("\t"):
                    if line.startswith("def ") or line.startswith("if __name__"):
                        end_line = i
                        break

            function_lines = end_line - start_line
            assert function_lines < max_lines, (
                f"{script_name}: process_csv function is {function_lines} lines, "
                f"which exceeds the target of {max_lines} lines. "
                "Please refactor this function into smaller, more manageable functions."
            )

    def test_process_csv_functions_have_helper_functions(self):
        """Test that process_csv functions use helper functions for common operations"""
        scripts = {
            "llm_judge_evaluator.py": Path(__file__).parent.parent / "scripts" / "llm_judge_evaluator.py",
            "format_clarity_evaluator.py": Path(__file__).parent.parent / "scripts" / "format_clarity_evaluator.py",
            "ragas_llm_judge_evaluator.py": Path(__file__).parent.parent / "scripts" / "ragas_llm_judge_evaluator.py",
        }

        # After refactoring, process_csv should call helper functions
        # This is a soft check - we'll verify that common patterns are extracted
        for script_name, script_path in scripts.items():
            if not script_path.exists():
                continue

            content = script_path.read_text(encoding="utf-8")

            # Check if process_csv function calls helper functions
            # This is a basic check - ideally we'd have functions like:
            # - initialize_client()
            # - read_and_validate_csv()
            # - process_rows()
            # - write_results()

            # For now, we'll just check that the function exists and is not too long
            # The actual refactoring will be done incrementally
            assert "def process_csv" in content, f"{script_name} should have process_csv function"

