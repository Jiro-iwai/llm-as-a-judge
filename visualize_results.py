#!/usr/bin/env python3
"""
評価結果を可視化するスクリプト
evaluation_output.csvの評価結果をグラフで表示します。
"""

import argparse
import platform
import sys
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from config.app_config import get_output_file_names
from utils.logging_config import (
    log_error,
    log_info,
    log_section,
    log_success,
    log_warning,
    setup_logging,
)

# Set up logging system
setup_logging()

# 日本語フォントの設定（macOS用）

if platform.system() == "Darwin":  # macOS
    # macOSの日本語フォントを試す
    try:
        matplotlib.rcParams["font.family"] = [
            "Hiragino Sans",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    except (OSError, ImportError, ValueError) as e:
        # フォント設定に失敗した場合はデフォルトフォントを使用
        log_warning(f"日本語フォントの設定に失敗しました: {e}")
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.figsize"] = (14, 10)


def load_data(csv_file: str) -> pd.DataFrame:
    """
    Load evaluation results from a CSV file.

    Args:
        csv_file: Path to the CSV file containing evaluation results.

    Returns:
        DataFrame containing the evaluation results.

    Raises:
        SystemExit: If the file is not found, cannot be parsed, or
            permission is denied.
    """
    try:
        df = pd.read_csv(csv_file)
        log_success(f"データを読み込みました: {len(df)}行")
        return df
    except FileNotFoundError:
        log_error(f"ファイル '{csv_file}' が見つかりません")
        sys.exit(1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        log_error(f"ファイルの読み込みに失敗しました: {e}")
        log_error("ファイル形式が正しいか、文字エンコーディングを確認してください。")
        sys.exit(1)
    except PermissionError as e:
        log_error(f"ファイルへのアクセス権限がありません: {e}")
        sys.exit(1)
    except Exception as e:
        # 予期しないエラーの場合は詳細な情報を記録
        log_error(f"予期しないエラーが発生しました: {type(e).__name__}: {e}")
        sys.exit(1)


def detect_evaluator_type(df: pd.DataFrame) -> str:
    """
    Detect the type of evaluator based on the columns in the DataFrame.

    Args:
        df: DataFrame containing evaluation results.

    Returns:
        Evaluator type: 'llm-judge', 'ragas', 'format-clarity', or 'unknown'.
    """
    columns = set(df.columns)

    # Check for llm-judge format (multiple score columns)
    if "Model_A_Citation_Score" in columns and "Model_B_Citation_Score" in columns:
        return "llm-judge"

    # Check for ragas format (faithfulness_score columns)
    if (
        "Model_A_faithfulness_score" in columns
        and "Model_B_faithfulness_score" in columns
    ):
        return "ragas"

    # Check for format-clarity format (single Format_Clarity_Score column)
    if "Format_Clarity_Score" in columns:
        return "format-clarity"

    return "unknown"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare evaluation data by removing rows with errors.

    Args:
        df: DataFrame containing evaluation results, potentially with
            error rows marked in the "Evaluation_Error" column.

    Returns:
        Cleaned DataFrame with error rows removed. If no "Evaluation_Error"
        column exists, returns the original DataFrame unchanged.
    """
    # エラーが発生した行を除外
    # llm_judge_evaluator.py outputs empty string ("") for normal rows, not NaN
    # So we need to keep rows where Evaluation_Error is empty string, NaN, or None
    if "Evaluation_Error" in df.columns:
        # Keep rows where Evaluation_Error is NaN, None, or empty string
        # Exclude rows where Evaluation_Error has a non-empty error message
        df_clean = df[
            df["Evaluation_Error"].isna() | (df["Evaluation_Error"] == "")
        ].copy()
        error_count = len(df) - len(df_clean)
        if error_count > 0:
            log_warning(f"エラー行を除外: {error_count}行")
    else:
        df_clean = df.copy()

    # Ensure return type is DataFrame
    if isinstance(df_clean, pd.DataFrame):
        return df_clean
    else:
        return pd.DataFrame(df_clean)


def create_score_comparison_chart(
    df: pd.DataFrame,
    output_file: str = "evaluation_comparison.png",
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None,
    evaluator_type: str = "llm-judge",
) -> None:
    """
    Create a bar chart comparing average scores between Model A and Model B.

    Supports multiple evaluator types:
    - llm-judge: Multiple metrics (Citation, Relevance, etc.)
    - ragas: Faithfulness score (0-1 range)
    - format-clarity: Single Format Clarity Score (1-5 range)

    Args:
        df: DataFrame containing evaluation results.
        output_file: Path to save the output PNG file. Defaults to
            "evaluation_comparison.png".
        model_a_name: Optional name for Model A. If None, uses "Model A".
        model_b_name: Optional name for Model B. If None, uses "Model B".
        evaluator_type: Type of evaluator ('llm-judge', 'ragas', 'format-clarity').

    Returns:
        None. The chart is saved to the specified output file.
    """

    # モデル名を決定（指定されない場合は汎用的なラベルを使用）
    label_a = model_a_name if model_a_name else "Model A"
    label_b = model_b_name if model_b_name else "Model B"

    if evaluator_type == "llm-judge":
        metrics = [
            ("Citation", "Citation_Score"),
            ("Relevance", "Relevance_Score"),
            ("ReAct Performance\nThought", "ReAct_Performance_Thought_Score"),
            ("RAG Retrieval\nObservation", "RAG_Retrieval_Observation_Score"),
            ("Information\nIntegration", "Information_Integration_Score"),
        ]
        y_max = 5.5
        y_ticks = range(0, 6)
    elif evaluator_type == "ragas":
        metrics = [
            ("Faithfulness", "faithfulness_score"),
        ]
        y_max = 1.1
        y_ticks = [i / 10 for i in range(0, 12, 2)]
    elif evaluator_type == "format-clarity":
        # Format clarity has a single score column, not Model_A/Model_B format
        # We'll create a simple comparison showing the score distribution
        if "Format_Clarity_Score" not in df.columns:
            log_error("Format_Clarity_Score column not found")
            plt.close()
            return

        # Convert to numeric, handling empty strings
        scores = pd.to_numeric(df["Format_Clarity_Score"], errors="coerce").dropna()
        if len(scores) == 0:
            log_error("No valid Format_Clarity_Score values found")
            plt.close()
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            ["Format Clarity Score"],
            [scores.mean()],
            alpha=0.8,
            color="#3498db",
        )
        ax.set_ylabel("平均スコア", fontsize=12, fontweight="bold")
        ax.set_title("Format Clarity Score", fontsize=14, fontweight="bold", pad=20)
        ax.set_ylim(0, 5.5)
        ax.set_yticks(range(0, 6))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.text(
            0,
            scores.mean(),
            f"{scores.mean():.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        log_success(f"スコア比較チャートを保存: {output_file}")
        plt.close()
        return
    else:
        log_error(f"Unknown evaluator type: {evaluator_type}")
        return

    model_a_scores = []
    model_b_scores = []
    metric_names = []

    for metric_name, score_col in metrics:
        model_a_col = f"Model_A_{score_col}"
        model_b_col = f"Model_B_{score_col}"

        if model_a_col in df.columns and model_b_col in df.columns:
            # Convert to numeric, handling empty strings
            model_a_values = pd.to_numeric(df[model_a_col], errors="coerce").dropna()
            model_b_values = pd.to_numeric(df[model_b_col], errors="coerce").dropna()

            if len(model_a_values) > 0 and len(model_b_values) > 0:
                model_a_avg = float(model_a_values.mean())
                model_b_avg = float(model_b_values.mean())

                model_a_scores.append(model_a_avg)
                model_b_scores.append(model_b_avg)
                metric_names.append(metric_name)

    if len(metric_names) == 0:
        log_error("No valid score columns found for comparison")
        plt.close()
        return

    # バーチャートを作成
    x = range(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        model_a_scores,
        width,
        label=label_a,
        alpha=0.8,
        color="#3498db",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        model_b_scores,
        width,
        label=label_b,
        alpha=0.8,
        color="#e74c3c",
    )

    ax.set_xlabel("評価メトリクス", fontsize=12, fontweight="bold")
    ax.set_ylabel("平均スコア", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{label_a} vs {label_b} スコア比較", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=0, ha="center")
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    # スコアをバーの上に表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    log_success(f"スコア比較チャートを保存: {output_file}")
    plt.close()


def create_score_distribution_chart(
    df: pd.DataFrame,
    output_file: str = "evaluation_distribution.png",
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None,
    evaluator_type: str = "llm-judge",
) -> None:
    """
    Create histogram charts showing score distributions for each metric.

    Supports multiple evaluator types:
    - llm-judge: Multiple metrics (Citation, Relevance, etc.)
    - ragas: Faithfulness score (0-1 range)
    - format-clarity: Single Format Clarity Score (1-5 range)

    Args:
        df: DataFrame containing evaluation results.
        output_file: Path to save the output PNG file. Defaults to
            "evaluation_distribution.png".
        model_a_name: Optional name for Model A. If None, uses "Model A".
        model_b_name: Optional name for Model B. If None, uses "Model B".
        evaluator_type: Type of evaluator ('llm-judge', 'ragas', 'format-clarity').

    Returns:
        None. The chart is saved to the specified output file.
    """

    # モデル名を決定（指定されない場合は汎用的なラベルを使用）
    label_a = model_a_name if model_a_name else "Model A"
    label_b = model_b_name if model_b_name else "Model B"

    if evaluator_type == "llm-judge":
        metrics = [
            ("Citation", "Citation_Score"),
            ("Relevance", "Relevance_Score"),
            ("ReAct Performance Thought", "ReAct_Performance_Thought_Score"),
            ("RAG Retrieval Observation", "RAG_Retrieval_Observation_Score"),
            ("Information Integration", "Information_Integration_Score"),
        ]
        x_ticks = range(1, 6)
        x_lim = (0.5, 5.5)
        bins = 5
        fig_size = (15, 10)
        subplot_layout = (2, 3)
    elif evaluator_type == "ragas":
        metrics = [
            ("Faithfulness", "faithfulness_score"),
        ]
        x_ticks = [i / 10 for i in range(0, 12, 2)]
        x_lim = (-0.05, 1.05)
        bins = 10
        fig_size = (8, 6)
        subplot_layout = (1, 1)
    elif evaluator_type == "format-clarity":
        if "Format_Clarity_Score" not in df.columns:
            log_error("Format_Clarity_Score column not found")
            return

        scores = pd.to_numeric(df["Format_Clarity_Score"], errors="coerce").dropna()
        if len(scores) == 0:
            log_error("No valid Format_Clarity_Score values found")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(scores, bins=5, alpha=0.7, color="#3498db", edgecolor="black")
        ax.set_xlabel("スコア", fontsize=12)
        ax.set_ylabel("頻度", fontsize=12)
        ax.set_title("Format Clarity Score 分布", fontsize=14, fontweight="bold")
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.5, 5.5)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        log_success(f"スコア分布チャートを保存: {output_file}")
        plt.close()
        return
    else:
        log_error(f"Unknown evaluator type: {evaluator_type}")
        return

    fig, axes = plt.subplots(*subplot_layout, figsize=fig_size)
    if evaluator_type == "ragas":
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (metric_name, score_col) in enumerate(metrics):
        ax = axes[idx]
        model_a_col = f"Model_A_{score_col}"
        model_b_col = f"Model_B_{score_col}"

        if model_a_col in df.columns and model_b_col in df.columns:
            model_a_scores = pd.to_numeric(df[model_a_col], errors="coerce").dropna()
            model_b_scores = pd.to_numeric(df[model_b_col], errors="coerce").dropna()

            if len(model_a_scores) > 0 and len(model_b_scores) > 0:
                ax.hist(
                    [model_a_scores, model_b_scores],
                    bins=bins,
                    alpha=0.7,
                    label=[label_a, label_b],
                    color=["#3498db", "#e74c3c"],
                    edgecolor="black",
                )
                ax.set_xlabel("スコア", fontsize=10)
                ax.set_ylabel("頻度", fontsize=10)
                ax.set_title(metric_name, fontsize=11, fontweight="bold")
                ax.set_xticks(x_ticks)
                ax.set_xlim(x_lim)
                ax.legend(fontsize=9)
                ax.grid(axis="y", alpha=0.3)

    # 最後のサブプロットを非表示（llm-judgeの場合のみ）
    if evaluator_type == "llm-judge" and len(metrics) < len(axes):
        axes[5].axis("off")

    plt.suptitle(
        f"スコア分布（{label_a} vs {label_b}）", fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    log_success(f"スコア分布チャートを保存: {output_file}")
    plt.close()


def create_boxplot_chart(
    df: pd.DataFrame,
    output_file: str = "evaluation_boxplot.png",
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None,
    evaluator_type: str = "llm-judge",
) -> None:
    """
    Create box plots comparing score distributions between Model A and Model B.

    Supports multiple evaluator types:
    - llm-judge: Multiple metrics (Citation, Relevance, etc.)
    - ragas: Faithfulness score (0-1 range)
    - format-clarity: Single Format Clarity Score (1-5 range)

    Args:
        df: DataFrame containing evaluation results.
        output_file: Path to save the output PNG file. Defaults to
            "evaluation_boxplot.png".
        model_a_name: Optional name for Model A. If None, uses "Model A".
        model_b_name: Optional name for Model B. If None, uses "Model B".
        evaluator_type: Type of evaluator ('llm-judge', 'ragas', 'format-clarity').

    Returns:
        None. The chart is saved to the specified output file.
    """

    # モデル名を決定（指定されない場合は汎用的なラベルを使用）
    label_a = model_a_name if model_a_name else "Model A"
    label_b = model_b_name if model_b_name else "Model B"

    if evaluator_type == "llm-judge":
        metrics = [
            ("Citation", "Citation_Score"),
            ("Relevance", "Relevance_Score"),
            ("ReAct Performance\nThought", "ReAct_Performance_Thought_Score"),
            ("RAG Retrieval\nObservation", "RAG_Retrieval_Observation_Score"),
            ("Information\nIntegration", "Information_Integration_Score"),
        ]
        y_lim = (0.5, 5.5)
        y_ticks = range(1, 6)
    elif evaluator_type == "ragas":
        metrics = [
            ("Faithfulness", "faithfulness_score"),
        ]
        y_lim = (-0.05, 1.05)
        y_ticks = [i / 10 for i in range(0, 12, 2)]
    elif evaluator_type == "format-clarity":
        if "Format_Clarity_Score" not in df.columns:
            log_error("Format_Clarity_Score column not found")
            return

        scores = pd.to_numeric(df["Format_Clarity_Score"], errors="coerce").dropna()
        if len(scores) == 0:
            log_error("No valid Format_Clarity_Score values found")
            return

        fig, ax = plt.subplots(figsize=(6, 8))
        bp = ax.boxplot(
            [scores.values],
            tick_labels=["Format Clarity Score"],
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][0].set_alpha(0.7)
        ax.set_ylabel("スコア", fontsize=12, fontweight="bold")
        ax.set_title(
            "Format Clarity Score 分布", fontsize=14, fontweight="bold", pad=20
        )
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks(range(1, 6))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        log_success(f"箱ひげ図を保存: {output_file}")
        plt.close()
        return
    else:
        log_error(f"Unknown evaluator type: {evaluator_type}")
        return

    # データを準備
    plot_data = []
    labels = []

    for metric_name, score_col in metrics:
        model_a_col = f"Model_A_{score_col}"
        model_b_col = f"Model_B_{score_col}"

        if model_a_col in df.columns and model_b_col in df.columns:
            model_a_values = pd.to_numeric(df[model_a_col], errors="coerce").dropna()
            model_b_values = pd.to_numeric(df[model_b_col], errors="coerce").dropna()

            if len(model_a_values) > 0 and len(model_b_values) > 0:
                plot_data.append(model_a_values.values)
                plot_data.append(model_b_values.values)
                labels.append(f"{metric_name}\n{label_a}")
                labels.append(f"{metric_name}\n{label_b}")

    if len(plot_data) == 0:
        log_error("No valid score columns found for boxplot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(
        plot_data, tick_labels=labels, patch_artist=True, showmeans=True, meanline=True
    )

    # 色を設定
    colors = ["#3498db", "#e74c3c"] * len(metrics)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("スコア", fontsize=12, fontweight="bold")
    ax.set_title(
        f"スコア分布の箱ひげ図（{label_a} vs {label_b}）",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    log_success(f"箱ひげ図を保存: {output_file}")
    plt.close()


def create_summary_table(
    df: pd.DataFrame,
    output_file: str = "evaluation_summary.txt",
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None,
    evaluator_type: str = "llm-judge",
) -> None:
    """
    Create a summary table with evaluation statistics in text format.

    Supports multiple evaluator types:
    - llm-judge: Multiple metrics (Citation, Relevance, etc.)
    - ragas: Faithfulness score (0-1 range)
    - format-clarity: Single Format Clarity Score (1-5 range)

    Args:
        df: DataFrame containing evaluation results.
        output_file: Path to save the output text file. Defaults to
            "evaluation_summary.txt".
        model_a_name: Optional name for Model A. If None, uses "Model A".
        model_b_name: Optional name for Model B. If None, uses "Model B".
        evaluator_type: Type of evaluator ('llm-judge', 'ragas', 'format-clarity').

    Returns:
        None. The summary table is saved to the specified output file.
    """

    # モデル名を決定（指定されない場合は汎用的なラベルを使用）
    label_a = model_a_name if model_a_name else "Model A"
    label_b = model_b_name if model_b_name else "Model B"

    if evaluator_type == "llm-judge":
        metrics = [
            ("Citation", "Citation_Score"),
            ("Relevance", "Relevance_Score"),
            ("ReAct Performance Thought", "ReAct_Performance_Thought_Score"),
            ("RAG Retrieval Observation", "RAG_Retrieval_Observation_Score"),
            ("Information Integration", "Information_Integration_Score"),
        ]
    elif evaluator_type == "ragas":
        metrics = [
            ("Faithfulness", "faithfulness_score"),
        ]
    elif evaluator_type == "format-clarity":
        if "Format_Clarity_Score" not in df.columns:
            log_error("Format_Clarity_Score column not found")
            return

        scores = pd.to_numeric(df["Format_Clarity_Score"], errors="coerce").dropna()
        if len(scores) == 0:
            log_error("No valid Format_Clarity_Score values found")
            return

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("評価結果サマリー (Format Clarity)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"評価対象行数: {len(df)}行\n\n")

            f.write("-" * 70 + "\n")
            f.write("Format Clarity Score 統計\n")
            f.write("-" * 70 + "\n")
            f.write(f"平均: {scores.mean():.2f}\n")
            f.write(f"最小: {scores.min():.0f}\n")
            f.write(f"最大: {scores.max():.0f}\n")
            f.write(f"標準偏差: {scores.std():.2f}\n")
            f.write(f"中央値: {scores.median():.2f}\n")
            f.write("-" * 70 + "\n")

        log_success(f"サマリーテーブルを保存: {output_file}")
        return
    else:
        log_error(f"Unknown evaluator type: {evaluator_type}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("評価結果サマリー\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"評価対象行数: {len(df)}行\n")
        f.write(f"評価メトリクス数: {len(metrics)}個\n\n")

        f.write("-" * 70 + "\n")
        f.write(
            f"{'メトリクス':<30} {label_a + '平均':<15} {label_b + '平均':<15} {'差分':<10}\n"
        )
        f.write("-" * 70 + "\n")

        for metric_name, score_col in metrics:
            model_a_col = f"Model_A_{score_col}"
            model_b_col = f"Model_B_{score_col}"

            if model_a_col in df.columns and model_b_col in df.columns:
                model_a_values = pd.to_numeric(
                    df[model_a_col], errors="coerce"
                ).dropna()
                model_b_values = pd.to_numeric(
                    df[model_b_col], errors="coerce"
                ).dropna()

                if len(model_a_values) > 0 and len(model_b_values) > 0:
                    model_a_avg = model_a_values.mean()
                    model_b_avg = model_b_values.mean()
                    diff = model_b_avg - model_a_avg

                    f.write(
                        f"{metric_name:<30} {model_a_avg:<15.2f} {model_b_avg:<15.2f} {diff:+.2f}\n"
                    )

        f.write("-" * 70 + "\n\n")

        # 各メトリクスの詳細統計
        f.write("詳細統計\n")
        f.write("=" * 70 + "\n\n")

        for metric_name, score_col in metrics:
            model_a_col = f"Model_A_{score_col}"
            model_b_col = f"Model_B_{score_col}"

            if model_a_col in df.columns and model_b_col in df.columns:
                model_a_values = pd.to_numeric(
                    df[model_a_col], errors="coerce"
                ).dropna()
                model_b_values = pd.to_numeric(
                    df[model_b_col], errors="coerce"
                ).dropna()

                if len(model_a_values) > 0 and len(model_b_values) > 0:
                    f.write(f"{metric_name}:\n")
                    f.write(
                        f"  {label_a}: 平均={model_a_values.mean():.2f}, "
                        f"最小={model_a_values.min():.2f}, "
                        f"最大={model_a_values.max():.2f}, "
                        f"標準偏差={model_a_values.std():.2f}\n"
                    )
                    f.write(
                        f"  {label_b}: 平均={model_b_values.mean():.2f}, "
                        f"最小={model_b_values.min():.2f}, "
                        f"最大={model_b_values.max():.2f}, "
                        f"標準偏差={model_b_values.std():.2f}\n\n"
                    )

    log_success(f"サマリーテーブルを保存: {output_file}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="評価結果を可視化するスクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # デフォルトのevaluation_output.csvを使用
    python visualize_results.py
    
    # カスタムCSVファイルを指定
    python visualize_results.py my_evaluation_results.csv
    
    # モデル名を指定して可視化
    python visualize_results.py evaluation_output.csv --model-a claude4.5-sonnet --model-b claude4.5-haiku
    
    # ragas_evaluation_output.csvを可視化
    python visualize_results.py ragas_evaluation_output.csv

入力CSV形式:
    以下の評価スクリプトの出力CSVに対応しています:
    
    1. llm-judge (llm_judge_evaluator.py):
       - Question
       - Model_A_Citation_Score, Model_B_Citation_Score
       - Model_A_Relevance_Score, Model_B_Relevance_Score
       - Model_A_ReAct_Performance_Thought_Score, Model_B_ReAct_Performance_Thought_Score
       - Model_A_RAG_Retrieval_Observation_Score, Model_B_RAG_Retrieval_Observation_Score
       - Model_A_Information_Integration_Score, Model_B_Information_Integration_Score
       - Evaluation_Error (オプション)
    
    2. ragas (ragas_llm_judge_evaluator.py):
       - Question
       - Model_A_faithfulness_score, Model_B_faithfulness_score
       - Evaluation_Error (オプション)
    
    3. format-clarity (format_clarity_evaluator.py):
       - Question
       - Format_Clarity_Score
       - Evaluation_Error (オプション)
    
    評価スクリプトの種類は自動検出されます。

出力ファイル:
    - evaluation_comparison.png: Model AとModel Bのスコア比較チャート
    - evaluation_distribution.png: スコア分布のヒストグラム
    - evaluation_boxplot.png: スコア分布の箱ひげ図
    - evaluation_summary.txt: 統計サマリーテーブル
        """,
    )

    parser.add_argument(
        "input_csv",
        nargs="?",
        default="evaluation_output.csv",
        help="評価結果のCSVファイル（デフォルト: evaluation_output.csv）",
    )

    parser.add_argument(
        "--model-a",
        type=str,
        default=None,
        help="Model Aの名前（例: claude4.5-sonnet）。指定しない場合は「Model A」と表示されます。",
    )

    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Model Bの名前（例: claude4.5-haiku）。指定しない場合は「Model B」と表示されます。",
    )

    args = parser.parse_args()
    csv_file = args.input_csv

    log_section("評価結果の可視化")

    # データを読み込む
    df = load_data(csv_file)
    df_clean = prepare_data(df)

    if len(df_clean) == 0:
        log_error("有効な評価データがありません")
        sys.exit(1)

    log_success(f"有効な評価データ: {len(df_clean)}行")
    log_info("")

    # 評価スクリプトの種類を自動検出
    evaluator_type = detect_evaluator_type(df_clean)
    log_info(f"評価スクリプトタイプを検出: {evaluator_type}")

    if evaluator_type == "unknown":
        log_warning(
            "評価スクリプトタイプを検出できませんでした。llm-judge形式として処理します。"
        )
        evaluator_type = "llm-judge"

    log_info("")

    # Get output file names from config
    output_files = get_output_file_names()

    # グラフを作成
    log_info("グラフを作成中...")
    create_score_comparison_chart(
        df_clean,
        output_files["evaluation_comparison"],
        model_a_name=args.model_a,
        model_b_name=args.model_b,
        evaluator_type=evaluator_type,
    )
    create_score_distribution_chart(
        df_clean,
        output_files["evaluation_distribution"],
        model_a_name=args.model_a,
        model_b_name=args.model_b,
        evaluator_type=evaluator_type,
    )
    create_boxplot_chart(
        df_clean,
        output_files["evaluation_boxplot"],
        model_a_name=args.model_a,
        model_b_name=args.model_b,
        evaluator_type=evaluator_type,
    )
    create_summary_table(
        df_clean,
        output_files["evaluation_summary"],
        model_a_name=args.model_a,
        model_b_name=args.model_b,
        evaluator_type=evaluator_type,
    )

    log_info("")
    log_section("✓ 可視化完了!")
    log_info("生成されたファイル:")
    log_info(
        f"  - {output_files['evaluation_comparison']}: スコア比較チャート", indent=1
    )
    log_info(
        f"  - {output_files['evaluation_distribution']}: スコア分布チャート", indent=1
    )
    log_info(f"  - {output_files['evaluation_boxplot']}: 箱ひげ図", indent=1)
    log_info(f"  - {output_files['evaluation_summary']}: サマリーテーブル", indent=1)


if __name__ == "__main__":
    main()
