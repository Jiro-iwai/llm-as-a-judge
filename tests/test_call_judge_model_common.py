"""
Tests for common call_judge_model function.

This test ensures that both llm_judge_evaluator.py and format_clarity_evaluator.py
use a common call_judge_model function instead of duplicating code.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCallJudgeModelCommon:
    """Tests to ensure call_judge_model is not duplicated"""

    def test_call_judge_model_uses_common_function(self):
        """Test that call_judge_model wrapper functions use the common function"""
        llm_judge_path = Path(__file__).parent.parent / "scripts" / "llm_judge_evaluator.py"
        format_clarity_path = Path(__file__).parent.parent / "scripts" / "format_clarity_evaluator.py"
        
        llm_judge_content = llm_judge_path.read_text(encoding="utf-8")
        format_clarity_content = format_clarity_path.read_text(encoding="utf-8")
        
        # Check that both files import from a common module
        has_common_import_llm = "from src.utils.judge_model_common import" in llm_judge_content
        has_common_import_format = "from src.utils.judge_model_common import" in format_clarity_content
        
        assert has_common_import_llm, (
            "llm_judge_evaluator.py should import call_judge_model_common from common module"
        )
        assert has_common_import_format, (
            "format_clarity_evaluator.py should import call_judge_model_common from common module"
        )
        
        # Check that wrapper functions call the common function
        # llm_judge_evaluator.py should call call_judge_model_common with enable_token_estimation=True
        assert "call_judge_model_common(" in llm_judge_content, (
            "llm_judge_evaluator.py's call_judge_model should call call_judge_model_common"
        )
        assert "enable_token_estimation=True" in llm_judge_content, (
            "llm_judge_evaluator.py should use token estimation"
        )
        
        # format_clarity_evaluator.py should call call_judge_model_common with enable_token_estimation=False
        assert "call_judge_model_common(" in format_clarity_content, (
            "format_clarity_evaluator.py's call_judge_model should call call_judge_model_common"
        )
        assert "enable_token_estimation=False" in format_clarity_content, (
            "format_clarity_evaluator.py should not use token estimation"
        )

