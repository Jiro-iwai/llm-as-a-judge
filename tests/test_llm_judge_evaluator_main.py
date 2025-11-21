"""
Unit tests for llm_judge_evaluator.py main() function.

This module tests the main() function and command-line argument handling.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLLMJudgeMain:
    """Tests for llm_judge_evaluator.py main() function."""

    @patch("llm_judge_evaluator.process_csv")
    @patch("llm_judge_evaluator.log_section")
    @patch("llm_judge_evaluator.log_info")
    @patch("llm_judge_evaluator.log_warning")
    @patch("os.getenv")
    def test_main_with_model_argument(
        self,
        mock_getenv,
        mock_warning,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function with --model argument."""
        from llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        mock_getenv.return_value = None

        with patch("sys.argv", ["llm_judge_evaluator.py", str(input_csv), "-m", "gpt-5"]):
            main()

        mock_process_csv.assert_called_once()
        # Check that model name was normalized
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["model_name"] == "gpt-5"

    @patch("llm_judge_evaluator.process_csv")
    @patch("llm_judge_evaluator.log_section")
    @patch("llm_judge_evaluator.log_info")
    @patch("llm_judge_evaluator.log_warning")
    @patch("os.getenv")
    def test_main_with_env_var_model(
        self,
        mock_getenv,
        mock_warning,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function uses MODEL_NAME env var when no --model argument."""
        from llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        mock_getenv.return_value = "gpt-4.1"

        with patch("sys.argv", ["llm_judge_evaluator.py", str(input_csv)]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["model_name"] == "gpt-4.1"

    @patch("llm_judge_evaluator.process_csv")
    @patch("llm_judge_evaluator.log_section")
    @patch("llm_judge_evaluator.log_info")
    @patch("llm_judge_evaluator.log_warning")
    @patch("os.getenv")
    def test_main_with_unsupported_model_warning(
        self,
        mock_getenv,
        mock_warning,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function warns about unsupported model."""
        from llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        mock_getenv.return_value = None

        with patch("sys.argv", ["llm_judge_evaluator.py", str(input_csv), "-m", "unsupported-model"]):
            main()

        # Should warn about unsupported model
        assert any("サポート" in str(call) for call in mock_warning.call_args_list)

    @patch("llm_judge_evaluator.process_csv")
    @patch("llm_judge_evaluator.log_section")
    @patch("llm_judge_evaluator.log_info")
    def test_main_with_limit_argument(
        self,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function with --limit argument."""
        from llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        with patch("sys.argv", ["llm_judge_evaluator.py", str(input_csv), "-n", "5"]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["limit_rows"] == 5

    @patch("llm_judge_evaluator.process_csv")
    @patch("llm_judge_evaluator.log_section")
    @patch("llm_judge_evaluator.log_info")
    def test_main_model_name_normalization(
        self,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function normalizes model names (gpt5 -> gpt-5)."""
        from llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        with patch("sys.argv", ["llm_judge_evaluator.py", str(input_csv), "-m", "gpt5"]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        # Should normalize to gpt-5
        assert call_args.kwargs["model_name"] == "gpt-5"

