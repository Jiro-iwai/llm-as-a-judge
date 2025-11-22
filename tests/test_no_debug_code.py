"""
Tests to ensure no debug code remains in production code.

This test ensures that debug-related code and comments
are not present in production scripts.
"""

import re
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNoDebugCode:
    """Tests to ensure no debug code remains in production code"""

    def test_llm_judge_evaluator_no_debug_code(self):
        """Test that llm_judge_evaluator.py has no debug code"""
        script_path = Path(__file__).parent.parent / "scripts" / "llm_judge_evaluator.py"
        content = script_path.read_text(encoding="utf-8")
        
        # Check for debug-related patterns
        debug_patterns = [
            r"content_for_debug",
            r"# Debug:",
            r"# Debug ",
        ]
        
        for pattern in debug_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert len(matches) == 0, (
                f"llm_judge_evaluator.py contains debug code: found '{pattern}' "
                f"({len(matches)} occurrences)"
            )

    def test_format_clarity_evaluator_no_debug_code(self):
        """Test that format_clarity_evaluator.py has no debug code"""
        script_path = Path(__file__).parent.parent / "scripts" / "format_clarity_evaluator.py"
        content = script_path.read_text(encoding="utf-8")
        
        # Check for debug-related patterns
        debug_patterns = [
            r"content_for_debug",
            r"response_for_debug",
            r"# Debug:",
            r"# Debug ",
        ]
        
        for pattern in debug_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert len(matches) == 0, (
                f"format_clarity_evaluator.py contains debug code: found '{pattern}' "
                f"({len(matches)} occurrences)"
            )

