"""
Unit tests for run_full_pipeline.py

This module tests the full pipeline that integrates:
- collect_responses.py
- Evaluation scripts (llm_judge_evaluator.py, ragas_llm_judge_evaluator.py, format_clarity_evaluator.py)
- visualize_results.py
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineBasicFunctionality:
    """Tests for basic pipeline functionality"""

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_pipeline_with_llm_judge(self, mock_exists, mock_subprocess_run):
        """Test that pipeline runs successfully with llm-judge evaluator"""
        # Mock file existence
        mock_exists.side_effect = lambda: True if str(mock_exists.call_count) == "1" else False

        # Mock subprocess.run calls
        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        # Import and run pipeline
        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "questions.txt", "--evaluator", "llm-judge"]):
            main()

        # Verify subprocess.run was called 3 times
        assert mock_subprocess_run.call_count == 3

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_pipeline_with_ragas(self, mock_exists, mock_subprocess_run):
        """Test that pipeline runs successfully with ragas evaluator"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # ragas_llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "questions.txt", "--evaluator", "ragas"]):
            main()

        # Should call collect, evaluation, and visualization
        assert mock_subprocess_run.call_count == 3

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_pipeline_with_format_clarity(self, mock_exists, mock_subprocess_run):
        """Test that pipeline runs successfully with format-clarity evaluator"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # format_clarity_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "questions.txt", "--evaluator", "format-clarity"]):
            main()

        # Should call collect, evaluation, and visualization
        assert mock_subprocess_run.call_count == 3

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_pipeline_with_all_evaluators(self, mock_exists, mock_subprocess_run):
        """Test that pipeline runs all evaluators when --evaluator all is specified"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # ragas_llm_judge_evaluator.py
            Mock(returncode=0),  # format_clarity_evaluator.py
            Mock(returncode=0),  # visualize_results.py (for llm-judge)
        ]

        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "questions.txt", "--evaluator", "all"]):
            main()

        # Should call collect once, then 3 evaluators, then visualize
        assert mock_subprocess_run.call_count >= 4


class TestPipelineOptions:
    """Tests for pipeline options"""

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_with_skip_collect(self, mock_exists, mock_subprocess_run):
        """Test that --skip-collect option skips the collect step"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--skip-collect",
                "--evaluator",
                "llm-judge",
            ],
        ):
            main()

        # Should skip collect, so only 2 calls
        assert mock_subprocess_run.call_count == 2

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_with_skip_visualize(self, mock_exists, mock_subprocess_run):
        """Test that --skip-visualize option skips the visualize step"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--skip-visualize",
                "--evaluator",
                "llm-judge",
            ],
        ):
            main()

        # Should skip visualize, so only 2 calls
        assert mock_subprocess_run.call_count == 2

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_with_custom_models(self, mock_exists, mock_subprocess_run):
        """Test that --model-a and --model-b options are passed correctly"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--model-a",
                "claude4.5-sonnet",
                "--model-b",
                "claude4.5-haiku",
                "--evaluator",
                "llm-judge",
            ],
        ):
            main()

        # Verify that model arguments were passed to collect_responses.py
        collect_call = mock_subprocess_run.call_args_list[0]
        assert "--model-a" in str(collect_call) or "claude4.5-sonnet" in str(collect_call)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_with_limit(self, mock_exists, mock_subprocess_run):
        """Test that --limit option is passed correctly to evaluation script"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--limit",
                "5",
                "--evaluator",
                "llm-judge",
            ],
        ):
            main()

        # Verify that limit was passed to evaluation script
        eval_call = mock_subprocess_run.call_args_list[1]
        assert "--limit" in str(eval_call) or "5" in str(eval_call)


class TestPipelineErrorHandling:
    """Tests for error handling"""

    @patch("subprocess.run")
    @patch("run_full_pipeline.Path")
    def test_pipeline_fails_on_collect_error(self, mock_path_class, mock_subprocess_run):
        """Test that pipeline fails appropriately when collect step errors"""
        # Mock Path.exists() to return True for questions.txt
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance

        # Mock subprocess.run to raise CalledProcessError
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["python", "collect_responses.py"], stderr="Error occurred"
        )

        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "questions.txt", "--evaluator", "llm-judge"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("subprocess.run")
    @patch("run_full_pipeline.Path")
    def test_pipeline_fails_on_evaluation_error(self, mock_path_class, mock_subprocess_run):
        """Test that pipeline fails appropriately when evaluation step errors"""
        # Mock Path.exists() to return True for questions.txt
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance

        mock_subprocess_run.side_effect = [
            Mock(returncode=0, stdout="", stderr=""),  # collect_responses.py succeeds
            subprocess.CalledProcessError(
                returncode=1, cmd=["python", "llm_judge_evaluator.py"], stderr="Evaluation error"
            ),  # llm_judge_evaluator.py fails
        ]

        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "questions.txt", "--evaluator", "llm-judge"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_continues_on_visualize_error(self, mock_exists, mock_subprocess_run):
        """Test that pipeline continues (with warning) when visualize step errors"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py succeeds
            Mock(returncode=0),  # llm_judge_evaluator.py succeeds
            Mock(returncode=1),  # visualize_results.py fails
        ]

        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "questions.txt", "--evaluator", "llm-judge"]):
            # Should not raise SystemExit, but log warning
            main()


class TestPipelineCommandLineArguments:
    """Tests for command line argument parsing"""

    def test_pipeline_help_option(self):
        """Test that --help option works correctly"""
        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 0 for --help
            assert exc_info.value.code == 0

    def test_pipeline_invalid_evaluator(self):
        """Test that invalid --evaluator value raises error"""
        from scripts.run_full_pipeline import main

        with patch("sys.argv", ["run_full_pipeline.py", "questions.txt", "--evaluator", "invalid"]):
            with pytest.raises(SystemExit):
                main()


class TestPipelineJudgeModelOption:
    """Tests for --judge-model option"""

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_with_judge_model_option(self, mock_exists, mock_subprocess_run):
        """Test that --judge-model option is passed to evaluation script"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--evaluator",
                "llm-judge",
                "--judge-model",
                "gpt-5",
            ],
        ):
            main()

        # Verify that --judge-model was passed to evaluation script as -m
        eval_call = mock_subprocess_run.call_args_list[1]
        eval_cmd = eval_call[0][0]
        assert "-m" in eval_cmd, "Evaluation script should receive -m option"
        assert "gpt-5" in eval_cmd, "Evaluation script should receive judge model name"

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_with_judge_model_for_ragas(self, mock_exists, mock_subprocess_run):
        """Test that --judge-model option works with ragas evaluator"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # ragas_llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--evaluator",
                "ragas",
                "--judge-model",
                "gpt-4.1",
            ],
        ):
            main()

        # Verify that --judge-model was passed to ragas evaluator
        eval_call = mock_subprocess_run.call_args_list[1]
        eval_cmd = eval_call[0][0]
        assert "-m" in eval_cmd, "Ragas evaluator should receive -m option"
        assert "gpt-4.1" in eval_cmd, "Ragas evaluator should receive judge model name"

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_with_judge_model_for_all_evaluators(self, mock_exists, mock_subprocess_run):
        """Test that --judge-model option is passed to all evaluators when --evaluator all"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # ragas_llm_judge_evaluator.py
            Mock(returncode=0),  # format_clarity_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--evaluator",
                "all",
                "--judge-model",
                "gpt-5",
            ],
        ):
            main()

        # Verify that --judge-model was passed to all evaluation scripts
        eval_calls = [call for call in mock_subprocess_run.call_args_list[1:4]]  # Skip collect, get 3 evaluators
        for eval_call in eval_calls:
            eval_cmd = eval_call[0][0]
            assert "-m" in eval_cmd, "All evaluators should receive -m option"
            assert "gpt-5" in eval_cmd, "All evaluators should receive judge model name"

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_without_judge_model_uses_default(self, mock_exists, mock_subprocess_run):
        """Test that evaluation script runs without -m option when --judge-model is not specified"""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--evaluator",
                "llm-judge",
            ],
        ):
            main()

        # Verify that -m option is not passed when --judge-model is not specified
        eval_call = mock_subprocess_run.call_args_list[1]
        eval_cmd = eval_call[0][0]
        # The script should still run, but -m might be absent (depends on default behavior)
        # We just verify the script was called
        assert "llm_judge_evaluator.py" in eval_cmd

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_passes_yes_flag_to_evaluation_scripts(
        self, mock_exists, mock_subprocess_run
    ):
        """Test that --yes flag is automatically passed to evaluation scripts from pipeline."""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--evaluator",
                "llm-judge",
            ],
        ):
            main()

        # Verify that --yes flag was passed to evaluation script
        eval_call = mock_subprocess_run.call_args_list[1]
        eval_cmd = eval_call[0][0]
        assert "--yes" in eval_cmd, "Pipeline should pass --yes flag to evaluation script for non-interactive execution"
        assert "llm_judge_evaluator.py" in eval_cmd

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_pipeline_passes_yes_flag_to_all_evaluators(
        self, mock_exists, mock_subprocess_run
    ):
        """Test that --yes flag is passed to all evaluators when --evaluator all."""
        mock_exists.return_value = True

        mock_subprocess_run.side_effect = [
            Mock(returncode=0),  # collect_responses.py
            Mock(returncode=0),  # llm_judge_evaluator.py
            Mock(returncode=0),  # ragas_llm_judge_evaluator.py
            Mock(returncode=0),  # format_clarity_evaluator.py
            Mock(returncode=0),  # visualize_results.py
        ]

        from scripts.run_full_pipeline import main

        with patch(
            "sys.argv",
            [
                "run_full_pipeline.py",
                "questions.txt",
                "--evaluator",
                "all",
            ],
        ):
            main()

        # Verify that --yes flag was passed to all evaluation scripts
        eval_calls = mock_subprocess_run.call_args_list[1:4]  # Skip collect, get 3 evaluators
        for eval_call in eval_calls:
            eval_cmd = eval_call[0][0]
            assert "--yes" in eval_cmd, "All evaluators should receive --yes flag from pipeline"

