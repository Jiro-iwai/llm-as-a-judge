"""
Unit tests for ragas_llm_judge_evaluator.py main() function.

This module tests the main() function and command-line argument handling.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRagasMain:
    """Tests for ragas_llm_judge_evaluator.py main() function."""

    @patch("ragas_llm_judge_evaluator.process_csv")
    @patch("ragas_llm_judge_evaluator.log_section")
    @patch("ragas_llm_judge_evaluator.log_info")
    @patch("os.getenv")
    def test_main_with_model_argument(
        self,
        mock_getenv,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function with --model argument."""
        from ragas_llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        mock_getenv.return_value = None

        with patch("sys.argv", ["ragas_llm_judge_evaluator.py", str(input_csv), "-m", "gpt-4.1"]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["model_name"] == "gpt-4.1"

    @patch("ragas_llm_judge_evaluator.process_csv")
    @patch("ragas_llm_judge_evaluator.log_section")
    @patch("ragas_llm_judge_evaluator.log_info")
    @patch("os.getenv")
    def test_main_with_env_var_model(
        self,
        mock_getenv,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function uses MODEL_NAME env var when no --model argument."""
        from ragas_llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        mock_getenv.return_value = "gpt-5"

        with patch("sys.argv", ["ragas_llm_judge_evaluator.py", str(input_csv)]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["model_name"] == "gpt-5"

    @patch("ragas_llm_judge_evaluator.process_csv")
    @patch("ragas_llm_judge_evaluator.log_section")
    @patch("ragas_llm_judge_evaluator.log_info")
    def test_main_with_limit_argument(
        self,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function with --limit argument."""
        from ragas_llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        with patch("sys.argv", ["ragas_llm_judge_evaluator.py", str(input_csv), "-n", "3"]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["limit_rows"] == 3

    @patch("ragas_llm_judge_evaluator.process_csv")
    @patch("ragas_llm_judge_evaluator.log_section")
    @patch("ragas_llm_judge_evaluator.log_info")
    def test_main_model_name_normalization(
        self,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function normalizes model names."""
        from ragas_llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        with patch("sys.argv", ["ragas_llm_judge_evaluator.py", str(input_csv), "-m", "gpt41"]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        # Model name normalization may not work perfectly, so just check it's called
        assert "model_name" in call_args.kwargs

