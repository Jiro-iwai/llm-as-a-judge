"""Unit tests for compare_processing_time.py"""

import pytest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compare_processing_time import (
    extract_processing_times,
    create_summary_table,
    create_comparison_chart,
    create_statistics_chart,
)


class TestExtractProcessingTimes:
    """Tests for extract_processing_times function"""

    def test_extract_processing_times_success(self):
        """Test successful extraction of processing times"""
        log_content = """📥 [claude3.5-sonnet] HTTPステータス: 200 (経過時間: 10.5秒)
📥 [claude4.5-haiku] HTTPステータス: 200 (経過時間: 8.3秒)
📥 [claude3.5-sonnet] HTTPステータス: 200 (経過時間: 12.1秒)
📥 [claude4.5-haiku] HTTPステータス: 200 (経過時間: 9.2秒)"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write(log_content)
            temp_file = f.name

        try:
            question_numbers, model_a_times, model_b_times = extract_processing_times(temp_file)

            assert len(model_a_times) == 2
            assert len(model_b_times) == 2
            assert model_a_times == [10.5, 12.1]
            assert model_b_times == [8.3, 9.2]
            assert question_numbers == [1, 2]
        finally:
            os.unlink(temp_file)

    def test_extract_processing_times_file_not_found(self):
        """Test handling file not found error"""
        with pytest.raises(SystemExit):
            extract_processing_times("nonexistent_file.txt")

    def test_extract_processing_times_mismatched_counts(self):
        """Test handling mismatched data counts"""
        log_content = """📥 [claude3.5-sonnet] HTTPステータス: 200 (経過時間: 10.5秒)
📥 [claude4.5-haiku] HTTPステータス: 200 (経過時間: 8.3秒)
📥 [claude3.5-sonnet] HTTPステータス: 200 (経過時間: 12.1秒)"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write(log_content)
            temp_file = f.name

        try:
            question_numbers, model_a_times, model_b_times = extract_processing_times(temp_file)

            # Should truncate to minimum length
            assert len(model_a_times) == len(model_b_times)
            assert len(model_a_times) == 1
        finally:
            os.unlink(temp_file)

    def test_extract_processing_times_no_matches(self):
        """Test handling no matches found"""
        log_content = "No processing times here"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write(log_content)
            temp_file = f.name

        try:
            question_numbers, model_a_times, model_b_times = extract_processing_times(temp_file)

            assert len(model_a_times) == 0
            assert len(model_b_times) == 0
            assert len(question_numbers) == 0
        finally:
            os.unlink(temp_file)


class TestCreateSummaryTable:
    """Tests for create_summary_table function"""

    def test_create_summary_table_success(self):
        """Test successful creation of summary table"""
        question_numbers = [1, 2, 3]
        model_a_times = [10.5, 12.1, 11.8]
        model_b_times = [8.3, 9.2, 9.5]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as f:
            temp_file = f.name

        try:
            create_summary_table(question_numbers, model_a_times, model_b_times, temp_file)

            assert os.path.exists(temp_file)
            with open(temp_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert "処理時間比較サマリー" in content
                assert "Model A" in content
                assert "Model B" in content
                assert "平均" in content
                assert "統計情報" in content
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_create_summary_table_empty_data(self):
        """Test creating summary table with empty data"""
        question_numbers = []
        model_a_times = []
        model_b_times = []

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as f:
            temp_file = f.name

        try:
            create_summary_table(question_numbers, model_a_times, model_b_times, temp_file)

            assert os.path.exists(temp_file)
            with open(temp_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert "処理時間比較サマリー" in content
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestCreateComparisonChart:
    """Tests for create_comparison_chart function"""

    @patch("scripts.compare_processing_time.plt")
    def test_create_comparison_chart_success(self, mock_plt):
        """Test successful creation of comparison chart"""
        # Mock plt.subplots to return figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        question_numbers = [1, 2, 3]
        model_a_times = [10.5, 12.1, 11.8]
        model_b_times = [8.3, 9.2, 9.5]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".png") as f:
            temp_file = f.name

        try:
            create_comparison_chart(question_numbers, model_a_times, model_b_times, temp_file)

            # Verify plt.savefig was called
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestCreateStatisticsChart:
    """Tests for create_statistics_chart function"""

    @patch("scripts.compare_processing_time.plt")
    def test_create_statistics_chart_success(self, mock_plt):
        """Test successful creation of statistics chart"""
        # Mock plt.subplots to return figure and axes (2x2 grid)
        # axes should be a numpy array-like structure that supports [0, 0] indexing
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax3 = MagicMock()
        mock_ax4 = MagicMock()
        # Create a mock array that supports [0, 0] indexing
        class MockAxesArray:
            def __init__(self):
                self._axes = [[mock_ax1, mock_ax2], [mock_ax3, mock_ax4]]
            def __getitem__(self, key):
                if isinstance(key, tuple):
                    return self._axes[key[0]][key[1]]
                return self._axes[key]
        mock_axes = MockAxesArray()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        question_numbers = [1, 2, 3]
        model_a_times = [10.5, 12.1, 11.8]
        model_b_times = [8.3, 9.2, 9.5]

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".png") as f:
            temp_file = f.name

        try:
            create_statistics_chart(question_numbers, model_a_times, model_b_times, temp_file)

            # Verify plt.savefig was called
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

