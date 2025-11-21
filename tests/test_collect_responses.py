"""Unit tests for collect_responses.py"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_responses import (
    format_response,
    call_api,
    read_questions,
)
from src.utils.log_output_simplifier import (
    clean_html,
    clean_and_format_llm_log,
)


class TestCleanHtml:
    """Tests for clean_html function"""

    def test_clean_html_with_links(self):
        """Test cleaning HTML links"""
        html = '<a href="https://example.com">Example</a>'
        result = clean_html(html)
        assert "Example (https://example.com)" in result
        assert "<a" not in result

    def test_clean_html_with_tags(self):
        """Test removing HTML tags"""
        html = "<p>Text</p><div>More text</div>"
        result = clean_html(html)
        assert "Text" in result
        assert "More text" in result
        assert "<p>" not in result
        assert "<div>" not in result

    def test_clean_html_empty_string(self):
        """Test cleaning empty string"""
        result = clean_html("")
        assert result == ""

    def test_clean_html_none(self):
        """Test cleaning None"""
        result = clean_html(None)
        assert result == ""


class TestCleanAndFormatLlmLog:
    """Tests for clean_and_format_llm_log function"""

    def test_clean_and_format_llm_log_with_json(self):
        """Test formatting JSON log"""
        json_log = json.dumps({"answer": "タスク：テストタスク\n回答：テスト回答"})
        result = clean_and_format_llm_log(json_log)
        assert "## 📝 Task タスク" in result
        assert "テストタスク" in result
        assert "## ✅ Final Answer 回答" in result
        assert "テスト回答" in result

    def test_clean_and_format_llm_log_without_json(self):
        """Test formatting non-JSON log"""
        log = "タスク：テストタスク\n回答：テスト回答"
        result = clean_and_format_llm_log(log)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_and_format_llm_log_no_markers(self):
        """Test formatting log without known markers"""
        log = "Plain text without markers"
        result = clean_and_format_llm_log(log)
        assert "No known section markers" in result or len(result) > 0

    def test_clean_and_format_llm_log_with_thought_process(self):
        """Test extracting thought process section"""
        log = json.dumps({"answer": "思考：これは思考プロセスです\n回答：最終回答"})
        result = clean_and_format_llm_log(log)
        assert "## 🤖 LLM Thought Process 思考" in result
        assert "思考プロセス" in result

    def test_clean_and_format_llm_log_json_parse_error(self):
        """Test handling JSON parse error with exception"""
        # Create invalid JSON that causes exception
        invalid_json = '{"answer": "test"'
        # This should trigger the Exception handler
        result = clean_and_format_llm_log(invalid_json)
        # Should return error message or fallback to plain text processing
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_and_format_llm_log_with_raw_search_results(self):
        """Test formatting log with Raw Search Results section"""
        log = json.dumps({
            "answer": "観察：結果1\n################################################\n結果2\n回答：最終回答"
        })
        result = clean_and_format_llm_log(log)
        assert "## 📚 Raw Search Results" in result
        assert "Result 1" in result or "結果1" in result


class TestFormatResponse:
    """Tests for format_response function"""

    def test_format_response_valid(self):
        """Test formatting valid response"""
        response = json.dumps({"answer": "タスク：テスト\n回答：回答"})
        result = format_response(response)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_response_empty_string(self):
        """Test formatting empty string"""
        result = format_response("")
        assert result == ""

    def test_format_response_none(self):
        """Test formatting None"""
        result = format_response(None)
        assert result == ""

    def test_format_response_formatting_error(self):
        """Test handling formatting error"""
        # Create a response that will cause an error
        with patch("collect_responses.clean_and_format_llm_log", side_effect=Exception("Error")):
            result = format_response("test")
            assert result == "test"  # Should return original text on error


class TestCallApi:
    """Tests for call_api function"""

    @patch("collect_responses.requests.post")
    def test_call_api_success(self, mock_post):
        """Test successful API call"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"answer": "Test answer"}
        mock_post.return_value = mock_response

        result = call_api("Test question", "http://example.com/api", "model-name")

        assert result == "Test answer"
        mock_post.assert_called_once()

    @patch("collect_responses.requests.post")
    def test_call_api_http_error(self, mock_post):
        """Test handling HTTP error"""
        import requests
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("HTTP Error")
        mock_post.return_value = mock_response

        result = call_api("Test question", "http://example.com/api", "model-name", verbose=False)

        assert result is None

    @patch("collect_responses.requests.post")
    def test_call_api_request_exception(self, mock_post):
        """Test handling request exception"""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        result = call_api("Test question", "http://example.com/api", "model-name", verbose=False)

        assert result is None

    @patch("collect_responses.requests.post")
    def test_call_api_json_decode_error(self, mock_post):
        """Test handling JSON decode error"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Error", "", 0)
        mock_response.text = "Invalid JSON"
        mock_post.return_value = mock_response

        result = call_api("Test question", "http://example.com/api", "model-name", verbose=False)

        # Should return text content when JSON decode fails
        assert result == "Invalid JSON"

    @patch("collect_responses.requests.post")
    def test_call_api_timeout(self, mock_post):
        """Test handling timeout error"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")

        result = call_api("Test question", "http://example.com/api", "model-name", verbose=False)

        assert result is None

    @patch("collect_responses.requests.post")
    def test_call_api_array_response(self, mock_post):
        """Test handling array format response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = ['{"answer": "Test answer", "urls": []}', 200]
        mock_post.return_value = mock_response

        result = call_api("Test question", "http://example.com/api", "model-name", verbose=False)

        assert result == "Test answer"

    @patch("collect_responses.requests.post")
    def test_call_api_no_answer_field(self, mock_post):
        """Test handling response without answer field"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"urls": ["http://example.com"]}
        mock_response.text = "Response text"
        mock_post.return_value = mock_response

        result = call_api("Test question", "http://example.com/api", "model-name", verbose=False)

        # Should return response.text when answer field is missing
        assert result == "Response text"

    @patch("collect_responses.requests.post")
    def test_call_api_logs_processing_time(self, mock_post, tmp_path):
        """Test that processing time entries are written to the log file."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"answer": "Logged answer"}
        mock_post.return_value = mock_response

        log_file = tmp_path / "processing_time.log"
        result = call_api(
            "Test question",
            "http://example.com/api",
            "claude3.5-sonnet",
            time_log_path=str(log_file),
            question_number=2,
            model_label="Model A",
            verbose=False,
        )

        assert result == "Logged answer"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "claude3.5-sonnet" in content
        assert "質問2" in content
        assert "Model A" in content
        assert "経過時間" in content


class TestReadQuestions:
    """Tests for read_questions function"""

    def test_read_questions_text_file(self, tmp_path):
        """Test reading questions from text file"""
        test_file = tmp_path / "questions.txt"
        test_file.write_text("Question 1\nQuestion 2\n# Comment line\nQuestion 3", encoding="utf-8")

        result = read_questions(str(test_file))

        assert len(result) == 3
        assert "Question 1" in result
        assert "Question 2" in result
        assert "Question 3" in result
        assert "# Comment line" not in result

    def test_read_questions_csv_file(self, tmp_path):
        """Test reading questions from CSV file"""
        import pandas as pd
        test_file = tmp_path / "questions.csv"
        df = pd.DataFrame({"Question": ["Q1", "Q2", "Q3"]})
        df.to_csv(test_file, index=False)

        result = read_questions(str(test_file))

        assert len(result) == 3
        assert "Q1" in result
        assert "Q2" in result
        assert "Q3" in result

    def test_read_questions_csv_with_header(self, tmp_path):
        """Test reading CSV with header row"""
        import pandas as pd
        test_file = tmp_path / "questions.csv"
        df = pd.DataFrame({"Question": ["Question", "Q1", "Q2"]})
        df.to_csv(test_file, index=False)

        result = read_questions(str(test_file))

        # Header should be removed
        assert "Question" not in result or result.count("Question") == 0
        assert "Q1" in result
        assert "Q2" in result

    def test_read_questions_file_not_found(self):
        """Test handling file not found error"""
        with pytest.raises(SystemExit):
            read_questions("nonexistent_file.txt")

    def test_read_questions_empty_file(self, tmp_path):
        """Test reading empty file"""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        result = read_questions(str(test_file))

        assert len(result) == 0

    def test_read_questions_csv_empty_after_header(self, tmp_path):
        """Test reading CSV with only header"""
        import pandas as pd
        test_file = tmp_path / "questions.csv"
        df = pd.DataFrame({"Question": ["Question"]})
        df.to_csv(test_file, index=False)

        result = read_questions(str(test_file))

        # Should return empty list after removing header
        assert len(result) == 0

    def test_read_questions_with_comments(self, tmp_path):
        """Test reading text file with comment lines"""
        test_file = tmp_path / "questions.txt"
        test_file.write_text("# Comment 1\nQuestion 1\n  # Comment 2\nQuestion 2", encoding="utf-8")

        result = read_questions(str(test_file))

        assert len(result) == 2
        assert "Question 1" in result
        assert "Question 2" in result
        assert "# Comment" not in " ".join(result)


class TestCollectResponsesIntegration:
    """Tests for collect_responses helper integrations."""

    @patch("collect_responses.generate_processing_time_reports")
    @patch("collect_responses.format_response", side_effect=lambda x: x)
    @patch(
        "collect_responses.call_api",
        side_effect=[
            "Answer A1",
            "Answer B1",
            "Answer A2",
            "Answer B2",
        ],
    )
    def test_collect_responses_generates_reports(
        self, mock_call_api, mock_format_response, mock_generate_reports, tmp_path
    ):
        """Test that collect_responses triggers processing time reports."""
        from scripts.collect_responses import collect_responses

        log_file = tmp_path / "time.log"
        df = collect_responses(
            questions=["Q1", "Q2"],
            api_url="http://example.com/api",
            model_a="claude3.5-sonnet",
            model_b="claude4.5-haiku",
            time_log_path=str(log_file),
            verbose=False,
            delay=0.0,
        )

        assert len(df) == 2
        assert mock_call_api.call_count == 4  # two models per question
        mock_generate_reports.assert_called_once_with(
            str(log_file),
            model_a_name="claude3.5-sonnet",
            model_b_name="claude4.5-haiku",
        )

