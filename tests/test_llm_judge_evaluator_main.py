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

    @patch("scripts.llm_judge_evaluator.process_csv")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.log_info")
    @patch("scripts.llm_judge_evaluator.log_warning")
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
        from scripts.llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        mock_getenv.return_value = None

        with patch("sys.argv", ["scripts/llm_judge_evaluator.py", str(input_csv), "-m", "gpt-5"]):
            main()

        mock_process_csv.assert_called_once()
        # Check that model name was normalized
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["model_name"] == "gpt-5"

    @patch("scripts.llm_judge_evaluator.process_csv")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.log_info")
    @patch("scripts.llm_judge_evaluator.log_warning")
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
        from scripts.llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        mock_getenv.return_value = "gpt-4.1"

        with patch("sys.argv", ["scripts/llm_judge_evaluator.py", str(input_csv)]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["model_name"] == "gpt-4.1"

    @patch("scripts.llm_judge_evaluator.process_csv")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.log_info")
    @patch("scripts.llm_judge_evaluator.log_warning")
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
        from scripts.llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        mock_getenv.return_value = None

        with patch("sys.argv", ["scripts/llm_judge_evaluator.py", str(input_csv), "-m", "unsupported-model"]):
            main()

        # Should warn about unsupported model
        assert any("サポート" in str(call) for call in mock_warning.call_args_list)

    @patch("scripts.llm_judge_evaluator.process_csv")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.log_info")
    def test_main_with_limit_argument(
        self,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function with --limit argument."""
        from scripts.llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        with patch("sys.argv", ["scripts/llm_judge_evaluator.py", str(input_csv), "-n", "5"]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        assert call_args.kwargs["limit_rows"] == 5

    @patch("scripts.llm_judge_evaluator.process_csv")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.log_info")
    def test_main_model_name_normalization(
        self,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test main() function normalizes model names (gpt5 -> gpt-5)."""
        from scripts.llm_judge_evaluator import main

        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Question,Model_A_Response,Model_B_Response\nQ1,A1,B1\n")

        with patch("sys.argv", ["scripts/llm_judge_evaluator.py", str(input_csv), "-m", "gpt5"]):
            main()

        mock_process_csv.assert_called_once()
        call_args = mock_process_csv.call_args
        # Should normalize to gpt-5
        assert call_args.kwargs["model_name"] == "gpt-5"

    @patch("scripts.llm_judge_evaluator.process_csv")
    @patch("scripts.llm_judge_evaluator.log_section")
    @patch("scripts.llm_judge_evaluator.log_info")
    @patch("builtins.input")
    def test_main_with_yes_flag_skips_confirmation(
        self,
        mock_input,
        mock_info,
        mock_section,
        mock_process_csv,
        tmp_path,
    ):
        """Test that --yes flag skips confirmation prompt even for >10 rows."""
        from scripts.llm_judge_evaluator import main

        # Create CSV with 15 rows (should trigger confirmation without --yes)
        input_csv = tmp_path / "input.csv"
        rows = ["Question,Model_A_Response,Model_B_Response"]
        for i in range(15):
            rows.append(f"Q{i},A{i},B{i}")
        input_csv.write_text("\n".join(rows))

        with patch("sys.argv", ["scripts/llm_judge_evaluator.py", str(input_csv), "--yes"]):
            main()

        # input() should not be called when --yes flag is present
        mock_input.assert_not_called()
        mock_process_csv.assert_called_once()

    @patch("builtins.input")
    @patch("scripts.llm_judge_evaluator.call_judge_model")
    @patch("scripts.llm_judge_evaluator.tqdm")
    def test_main_without_yes_flag_shows_confirmation_for_many_rows(
        self,
        mock_tqdm,
        mock_call_judge,
        mock_input,
        tmp_path,
    ):
        """Test that confirmation prompt is shown for >10 rows without --yes flag."""
        from scripts.llm_judge_evaluator import process_csv

        # Create CSV with 15 data rows (plus header = 16 total lines)
        input_csv = tmp_path / "input.csv"
        rows = ["Question,Model_A_Response,Model_B_Response"]
        for i in range(15):
            rows.append(f"Q{i},A{i},B{i}")
        input_csv.write_text("\n".join(rows))

        mock_input.return_value = "y"  # User confirms
        mock_call_judge.return_value = {"citation_score": 5}  # Mock API response
        # Mock tqdm to return iterator directly without progress bar
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        # Mock API client to avoid actual API calls
        with patch("scripts.llm_judge_evaluator.AzureOpenAI") as mock_azure_class, patch(
            "scripts.llm_judge_evaluator.OpenAI"
        ) as mock_openai_class, patch("os.getenv") as mock_getenv, patch(
            "scripts.llm_judge_evaluator.pd.DataFrame.to_csv"
        ) as mock_to_csv:
            # Set up mock Azure client
            mock_client = Mock()
            mock_azure_class.return_value = mock_client
            
            mock_getenv.side_effect = lambda key, default=None: (
                "https://test.openai.azure.com/" if key == "AZURE_OPENAI_ENDPOINT" else (
                    "test-key" if key == "AZURE_OPENAI_API_KEY" else (
                        "gpt-4.1" if key == "MODEL_NAME" else default
                    )
                )
            )
            # Call process_csv directly to test confirmation prompt
            try:
                process_csv(str(input_csv), non_interactive=False)
            except (SystemExit, Exception):
                pass  # Expected when API credentials are invalid or other errors

        # input() should be called when >10 rows and non_interactive=False
        # Note: CSV has header, so 15 data rows = len(df) = 15, which is > 10
        assert mock_input.call_count >= 1, "input() should be called for >10 rows without --yes flag"

    @patch("builtins.input")
    def test_main_without_yes_flag_cancels_on_n(
        self,
        mock_input,
        tmp_path,
    ):
        """Test that script exits when user answers 'n' to confirmation."""
        from scripts.llm_judge_evaluator import process_csv

        # Create CSV with 15 rows
        input_csv = tmp_path / "input.csv"
        rows = ["Question,Model_A_Response,Model_B_Response"]
        for i in range(15):
            rows.append(f"Q{i},A{i},B{i}")
        input_csv.write_text("\n".join(rows))

        mock_input.return_value = "n"  # User cancels

        # Mock API client to avoid actual API calls
        with patch("scripts.llm_judge_evaluator.AzureOpenAI") as mock_azure_class, patch(
            "llm_judge_evaluator.OpenAI"
        ) as mock_openai_class, patch("os.getenv") as mock_getenv:
            mock_client = Mock()
            mock_azure_class.return_value = mock_client
            
            mock_getenv.side_effect = lambda key, default=None: (
                "https://test.openai.azure.com/" if key == "AZURE_OPENAI_ENDPOINT" else (
                    "test-key" if key == "AZURE_OPENAI_API_KEY" else (
                        "gpt-4.1" if key == "MODEL_NAME" else default
                    )
                )
            )
            # Call process_csv directly to test cancellation
            with pytest.raises(SystemExit) as exc_info:
                process_csv(str(input_csv), non_interactive=False)
            # Should exit with code 0 (cancelled)
            assert exc_info.value.code == 0

        # input() should be called
        mock_input.assert_called_once()

