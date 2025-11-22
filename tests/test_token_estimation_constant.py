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
        """Test that token estimation uses a constant instead of magic number 4"""
        # After refactoring, llm_judge_evaluator.py uses call_judge_model_common
        # which defines TOKEN_ESTIMATION_CHARS_PER_TOKEN in src/utils/judge_model_common.py
        common_module_path = Path(__file__).parent.parent / "src" / "utils" / "judge_model_common.py"
        common_content = common_module_path.read_text(encoding="utf-8")
        
        # Check that magic number / 4 is not used directly in common module
        magic_number_pattern = r"estimated_input_tokens\s*=\s*\([^)]+\)\s*/\s*4\b"
        matches = re.findall(magic_number_pattern, common_content)
        assert len(matches) == 0, (
            "judge_model_common.py should use a constant instead of magic number 4 "
            f"for token estimation. Found: {matches}"
        )
        
        # Check that a constant is defined for token estimation
        constant_patterns = [
            r"TOKEN_ESTIMATION_CHARS_PER_TOKEN",
        ]
        has_constant = any(re.search(pattern, common_content, re.IGNORECASE) for pattern in constant_patterns)
        assert has_constant, (
            "judge_model_common.py should define TOKEN_ESTIMATION_CHARS_PER_TOKEN constant "
            "for token estimation"
        )
        
        # Check that the constant is used in token estimation
        usage_pattern = r"estimated_input_tokens\s*=\s*\([^)]+\)\s*/\s*TOKEN_ESTIMATION_CHARS_PER_TOKEN"
        assert re.search(usage_pattern, common_content), (
            "judge_model_common.py should use TOKEN_ESTIMATION_CHARS_PER_TOKEN "
            "in token estimation calculation"
        )

