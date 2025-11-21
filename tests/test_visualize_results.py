"""Unit tests for visualize_results.py"""

import pytest
import sys
import tempfile
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualize_results import (
    load_data,
    prepare_data,
    create_summary_table,
    create_score_comparison_chart,
    create_score_distribution_chart,
    create_boxplot_chart,
)


class TestLoadData:
    """Tests for load_data function"""

    def test_load_data_success(self):
        """Test successful CSV loading"""
        df_data = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Citation_Score": [4, 5],
            "Model_B_Citation_Score": [3, 4],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            df_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            result = load_data(temp_file)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "Question" in result.columns
        finally:
            os.unlink(temp_file)

    def test_load_data_file_not_found(self):
        """Test handling file not found error"""
        with pytest.raises(SystemExit):
            load_data("nonexistent_file.csv")


class TestPrepareData:
    """Tests for prepare_data function"""

    def test_prepare_data_without_errors(self):
        """Test preparing data without error column"""
        df = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Citation_Score": [4, 5],
        })
        result = prepare_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_prepare_data_with_errors(self):
        """Test preparing data with error column"""
        df = pd.DataFrame({
            "Question": ["Q1", "Q2", "Q3"],
            "Model_A_Citation_Score": [4, 5, 3],
            "Evaluation_Error": [None, None, "Some error"],
        })
        result = prepare_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Error row should be excluded

    def test_prepare_data_all_errors(self):
        """Test preparing data when all rows have errors"""
        df = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Citation_Score": [4, 5],
            "Evaluation_Error": ["Error1", "Error2"],
        })
        result = prepare_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_prepare_data_with_empty_string_errors(self):
        """Test preparing data with empty string in Evaluation_Error (normal rows from llm_judge_evaluator.py)"""
        # llm_judge_evaluator.py outputs empty string ("") for normal rows, not NaN
        df = pd.DataFrame({
            "Question": ["Q1", "Q2", "Q3"],
            "Model_A_Citation_Score": [4, 5, 3],
            "Evaluation_Error": ["", "", "Some error"],  # Empty string means normal row
        })
        result = prepare_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Empty string rows should be kept, error row should be excluded

    def test_prepare_data_with_mixed_empty_string_and_nan(self):
        """Test preparing data with both empty string and NaN in Evaluation_Error"""
        df = pd.DataFrame({
            "Question": ["Q1", "Q2", "Q3", "Q4"],
            "Model_A_Citation_Score": [4, 5, 3, 2],
            "Evaluation_Error": ["", pd.NA, "Some error", None],  # Empty string and NaN are both normal
        })
        result = prepare_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Empty string, NaN, and None rows should be kept, error row excluded

    def test_prepare_data_empty_dataframe(self):
        """Test preparing empty dataframe"""
        df = pd.DataFrame()
        result = prepare_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestCreateSummaryTable:
    """Tests for create_summary_table function"""

    def test_create_summary_table_success(self):
        """Test successful creation of summary table"""
        df = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Citation_Score": [4.0, 5.0],
            "Model_B_Citation_Score": [3.0, 4.0],
            "Model_A_Relevance_Score": [4.5, 4.8],
            "Model_B_Relevance_Score": [3.5, 4.2],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as f:
            temp_file = f.name

        try:
            create_summary_table(df, temp_file)

            assert os.path.exists(temp_file)
            with open(temp_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert "評価結果サマリー" in content
                assert "Citation" in content
                assert "Relevance" in content
                assert "Model A" in content
                assert "Model B" in content
                assert "詳細統計" in content
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_create_summary_table_empty_dataframe(self):
        """Test creating summary table with empty dataframe"""
        df = pd.DataFrame()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as f:
            temp_file = f.name

        try:
            create_summary_table(df, temp_file)

            assert os.path.exists(temp_file)
            with open(temp_file, "r", encoding="utf-8") as f:
                content = f.read()
                assert "評価結果サマリー" in content
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_create_summary_table_missing_columns(self):
        """Test creating summary table with missing score columns"""
        df = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Citation_Score": [4.0, 5.0],
            # Missing Model_B_Citation_Score
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as f:
            temp_file = f.name

        try:
            create_summary_table(df, temp_file)

            assert os.path.exists(temp_file)
            # Should not crash even if columns are missing
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestCreateScoreComparisonChart:
    """Tests for create_score_comparison_chart function"""

    @patch("visualize_results.plt")
    def test_create_score_comparison_chart_success(self, mock_plt):
        """Test successful creation of score comparison chart"""
        # Mock plt.subplots to return figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        df = pd.DataFrame({
            "Question": ["Q1", "Q2"],
            "Model_A_Citation_Score": [4.0, 5.0],
            "Model_B_Citation_Score": [3.0, 4.0],
            "Model_A_Relevance_Score": [4.5, 4.8],
            "Model_B_Relevance_Score": [3.5, 4.2],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".png") as f:
            temp_file = f.name

        try:
            create_score_comparison_chart(df, temp_file)

            # Verify plt.savefig was called
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch("visualize_results.plt")
    def test_create_score_comparison_chart_no_scores(self, mock_plt):
        """Test creating chart with no score columns"""
        # Mock plt.subplots to return figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        df = pd.DataFrame({
            "Question": ["Q1", "Q2"],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".png") as f:
            temp_file = f.name

        try:
            create_score_comparison_chart(df, temp_file)

            # Should not crash even if no score columns
            mock_plt.close.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestCreateScoreDistributionChart:
    """Tests for create_score_distribution_chart function"""

    @patch("visualize_results.plt")
    def test_create_score_distribution_chart_success(self, mock_plt):
        """Test successful creation of score distribution chart"""
        # Mock plt.subplots to return figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        df = pd.DataFrame({
            "Question": ["Q1", "Q2", "Q3"],
            "Model_A_Citation_Score": [4.0, 5.0, 3.0],
            "Model_B_Citation_Score": [3.0, 4.0, 4.0],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".png") as f:
            temp_file = f.name

        try:
            create_score_distribution_chart(df, temp_file)

            # Verify plt.savefig was called
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestCreateBoxplotChart:
    """Tests for create_boxplot_chart function"""

    @patch("visualize_results.plt")
    def test_create_boxplot_chart_success(self, mock_plt):
        """Test successful creation of boxplot chart"""
        # Mock plt.subplots to return figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        df = pd.DataFrame({
            "Question": ["Q1", "Q2", "Q3"],
            "Model_A_Citation_Score": [4.0, 5.0, 3.0],
            "Model_B_Citation_Score": [3.0, 4.0, 4.0],
        })

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".png") as f:
            temp_file = f.name

        try:
            create_boxplot_chart(df, temp_file)

            # Verify plt.savefig was called
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once()
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

