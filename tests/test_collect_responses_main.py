"""
Unit tests for collect_responses.py main() function and error handling.

This module tests the main() function and error handling paths that are not
covered by other tests.
"""

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInitializeProcessingTimeLog:
    """Tests for initialize_processing_time_log error handling."""

    def test_initialize_processing_time_log_empty_log_file(self, tmp_path):
        """Test that initialize_processing_time_log handles empty log_file."""
        from scripts.collect_responses import initialize_processing_time_log

        # Should not raise an error
        initialize_processing_time_log("")

    def test_initialize_processing_time_log_creates_directory(self, tmp_path):
        """Test that initialize_processing_time_log creates parent directories."""
        from scripts.collect_responses import initialize_processing_time_log

        log_file = tmp_path / "subdir" / "time.log"
        initialize_processing_time_log(str(log_file))

        assert log_file.exists()
        assert log_file.read_text().startswith("# Processing time log")

    def test_initialize_processing_time_log_handles_oserror(self, tmp_path):
        """Test that initialize_processing_time_log handles OSError gracefully."""
        from scripts.collect_responses import initialize_processing_time_log

        # Create a path that will cause OSError (e.g., invalid path)
        invalid_path = "/invalid/path/to/file.log"

        # Should not raise, but log a warning
        with patch("scripts.collect_responses.log_warning") as mock_warning:
            initialize_processing_time_log(invalid_path)
            mock_warning.assert_called_once()


class TestLogProcessingTimeEntry:
    """Tests for log_processing_time_entry error handling."""

    def test_log_processing_time_entry_empty_log_file(self):
        """Test that log_processing_time_entry handles empty log_file."""
        from scripts.collect_responses import log_processing_time_entry

        # Should not raise an error
        log_processing_time_entry("model", 1.0, "")

    def test_log_processing_time_entry_writes_to_file(self, tmp_path):
        """Test that log_processing_time_entry writes to file correctly."""
        from scripts.collect_responses import log_processing_time_entry

        log_file = tmp_path / "time.log"
        log_file.write_text("# Header\n")

        log_processing_time_entry(
            "claude3.5-sonnet",
            1.23,
            str(log_file),
            question_number=1,
            model_label="Model A",
        )

        content = log_file.read_text()
        assert "📥 [claude3.5-sonnet]" in content
        assert "Model A" in content
        assert "質問1" in content
        assert "1.23秒" in content

    def test_log_processing_time_entry_handles_oserror(self, tmp_path):
        """Test that log_processing_time_entry handles OSError gracefully."""
        from scripts.collect_responses import log_processing_time_entry

        invalid_path = "/invalid/path/to/file.log"

        with patch("scripts.collect_responses.log_warning") as mock_warning:
            log_processing_time_entry("model", 1.0, invalid_path)
            mock_warning.assert_called_once()


class TestCollectResponsesMain:
    """Tests for collect_responses.py main() function."""

    @patch("scripts.collect_responses.collect_responses")
    @patch("scripts.collect_responses.read_questions")
    @patch("scripts.collect_responses.log_section")
    @patch("scripts.collect_responses.log_info")
    @patch("scripts.collect_responses.log_success")
    @patch("scripts.collect_responses.log_error")
    @patch("scripts.collect_responses.log_warning")
    def test_main_success(
        self,
        mock_warning,
        mock_error,
        mock_success,
        mock_info,
        mock_section,
        mock_read_questions,
        mock_collect_responses,
        tmp_path,
    ):
        """Test main() function with successful execution."""
        from scripts.collect_responses import main
        import pandas as pd

        # Setup mocks
        questions_file = tmp_path / "questions.txt"
        questions_file.write_text("Question 1\nQuestion 2\n")
        output_file = tmp_path / "output.csv"

        mock_read_questions.return_value = ["Question 1", "Question 2"]
        mock_collect_responses.return_value = pd.DataFrame(
            {
                "Question": ["Q1", "Q2"],
                "Model_A_Response": ["A1", "A2"],
                "Model_B_Response": ["B1", "B2"],
            }
        )

        with patch("sys.argv", ["scripts/collect_responses.py", str(questions_file), "-o", str(output_file)]):
            main()

        mock_read_questions.assert_called_once()
        mock_collect_responses.assert_called_once()
        assert output_file.exists()

    @patch("scripts.collect_responses.read_questions")
    @patch("scripts.collect_responses.log_error")
    def test_main_empty_questions_file(self, mock_error, mock_read_questions, tmp_path):
        """Test main() function exits when no questions found."""
        from scripts.collect_responses import main

        questions_file = tmp_path / "questions.txt"
        questions_file.write_text("")

        mock_read_questions.return_value = []

        with patch("sys.argv", ["scripts/collect_responses.py", str(questions_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        mock_error.assert_called()

    @patch("scripts.collect_responses.collect_responses")
    @patch("scripts.collect_responses.read_questions")
    @patch("scripts.collect_responses.log_section")
    @patch("scripts.collect_responses.log_info")
    @patch("scripts.collect_responses.log_success")
    @patch("scripts.collect_responses.log_warning")
    @patch("scripts.collect_responses.log_error")
    def test_main_with_failed_responses(
        self,
        mock_error,
        mock_warning,
        mock_success,
        mock_info,
        mock_section,
        mock_read_questions,
        mock_collect_responses,
        tmp_path,
    ):
        """Test main() function handles failed responses correctly."""
        from scripts.collect_responses import main
        import pandas as pd

        questions_file = tmp_path / "questions.txt"
        questions_file.write_text("Question 1\n")
        output_file = tmp_path / "output.csv"

        mock_read_questions.return_value = ["Question 1"]
        # Create DataFrame with empty responses (failed)
        mock_collect_responses.return_value = pd.DataFrame(
            {
                "Question": ["Q1"],
                "Model_A_Response": [""],  # Failed
                "Model_B_Response": ["B1"],  # Success
            }
        )

        with patch("sys.argv", ["scripts/collect_responses.py", str(questions_file), "-o", str(output_file)]):
            main()

        # Should log warnings about failed responses
        # Check that warnings were called (may be in Japanese or English)
        assert mock_warning.called or mock_error.called

    @patch("scripts.collect_responses.collect_responses")
    @patch("scripts.collect_responses.read_questions")
    @patch("scripts.collect_responses.log_section")
    @patch("scripts.collect_responses.log_info")
    @patch("scripts.collect_responses.log_success")
    def test_main_saves_csv_with_quoting(
        self,
        mock_success,
        mock_info,
        mock_section,
        mock_read_questions,
        mock_collect_responses,
        tmp_path,
    ):
        """Test main() function saves CSV with proper quoting."""
        from scripts.collect_responses import main
        import pandas as pd

        questions_file = tmp_path / "questions.txt"
        questions_file.write_text("Question 1\n")
        output_file = tmp_path / "output.csv"

        mock_read_questions.return_value = ["Question 1"]
        df = pd.DataFrame(
            {
                "Question": ["Q1"],
                "Model_A_Response": ['Response with "quotes"'],
                "Model_B_Response": ["B1"],
            }
        )
        mock_collect_responses.return_value = df

        with patch("sys.argv", ["scripts/collect_responses.py", str(questions_file), "-o", str(output_file)]):
            main()

        # Verify CSV was saved with proper quoting
        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) > 0
            # Check that quotes are properly handled
            assert '"' in rows[1][1] or rows[1][1].startswith('"')

