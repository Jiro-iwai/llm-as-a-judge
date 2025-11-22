"""
Tests for token estimation constant usage.

This test ensures that token estimation uses a named constant
instead of a magic number.
"""

import re
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenEstimationConstant:
    """Tests for token estimation constant usage"""

    def test_llm_judge_evaluator_uses_constant_for_token_estimation(self):
        """Test that llm_judge_evaluator.py uses a constant instead of magic number 4"""
        script_path = Path(__file__).parent.parent / "scripts" / "llm_judge_evaluator.py"
        content = script_path.read_text(encoding="utf-8")
        
        # Check that magic number / 4 is not used directly
        magic_number_pattern = r"estimated_input_tokens\s*=\s*\([^)]+\)\s*/\s*4\b"
        matches = re.findall(magic_number_pattern, content)
        assert len(matches) == 0, (
            "llm_judge_evaluator.py should use a constant instead of magic number 4 "
            f"for token estimation. Found: {matches}"
        )
        
        # Check that a constant is defined for token estimation
        constant_patterns = [
            r"TOKEN_ESTIMATION",
            r"CHARS_PER_TOKEN",
            r"TOKEN.*ESTIMATION",
        ]
        has_constant = any(re.search(pattern, content, re.IGNORECASE) for pattern in constant_patterns)
        assert has_constant, (
            "llm_judge_evaluator.py should define a constant for token estimation "
            "(e.g., TOKEN_ESTIMATION_CHARS_PER_TOKEN)"
        )
        
        # Check that the constant is used in token estimation
        usage_pattern = r"estimated_input_tokens\s*=\s*\([^)]+\)\s*/\s*[A-Z_]+"
        assert re.search(usage_pattern, content), (
            "llm_judge_evaluator.py should use the constant in token estimation calculation"
        )

