#!/usr/bin/env python3
"""
評価結果を可視化するスクリプト
evaluation_output.csvの評価結果をグラフで表示します。
"""

import argparse
import platform
import sys

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
    if "Evaluation_Error" in df.columns:
        df_clean = df[df["Evaluation_Error"].isna()].copy()
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
    df: pd.DataFrame, output_file: str = "evaluation_comparison.png"
) -> None:
    """
    Create a bar chart comparing average scores between Model A and Model B.

    The chart compares the following metrics:
    - Citation Score
    - Relevance Score
    - ReAct Performance Thought Score
    - RAG Retrieval Observation Score
    - Information Integration Score

    Args:
        df: DataFrame containing evaluation results with columns like
            "Model_A_Citation_Score", "Model_B_Citation_Score", etc.
        output_file: Path to save the output PNG file. Defaults to
            "evaluation_comparison.png".

    Returns:
        None. The chart is saved to the specified output file.
    """

    metrics = [
        ("Citation", "Citation_Score"),
        ("Relevance", "Relevance_Score"),
        ("ReAct Performance\nThought", "ReAct_Performance_Thought_Score"),
        ("RAG Retrieval\nObservation", "RAG_Retrieval_Observation_Score"),
        ("Information\nIntegration", "Information_Integration_Score"),
    ]

    model_a_scores = []
    model_b_scores = []
    metric_names = []

    for metric_name, score_col in metrics:
        model_a_col = f"Model_A_{score_col}"
        model_b_col = f"Model_B_{score_col}"

        if model_a_col in df.columns and model_b_col in df.columns:
            model_a_avg = df[model_a_col].mean()
            model_b_avg = df[model_b_col].mean()

            model_a_scores.append(model_a_avg)
            model_b_scores.append(model_b_avg)
            metric_names.append(metric_name)

    # バーチャートを作成
    x = range(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        model_a_scores,
        width,
        label="Model A (claude3.5-sonnet)",
        alpha=0.8,
        color="#3498db",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        model_b_scores,
        width,
        label="Model B (claude4.5-haiku)",
        alpha=0.8,
        color="#e74c3c",
    )

    ax.set_xlabel("評価メトリクス", fontsize=12, fontweight="bold")
    ax.set_ylabel("平均スコア", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model A vs Model B スコア比較", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=0, ha="center")
    ax.set_ylim(0, 5.5)
    ax.set_yticks(range(0, 6))
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
    df: pd.DataFrame, output_file: str = "evaluation_distribution.png"
) -> None:
    """
    Create histogram charts showing score distributions for each metric.

    The chart displays histograms for the following metrics:
    - Citation Score
    - Relevance Score
    - ReAct Performance Thought Score
    - RAG Retrieval Observation Score
    - Information Integration Score

    Args:
        df: DataFrame containing evaluation results with columns like
            "Model_A_Citation_Score", "Model_B_Citation_Score", etc.
        output_file: Path to save the output PNG file. Defaults to
            "evaluation_distribution.png".

    Returns:
        None. The chart is saved to the specified output file.
    """

    metrics = [
        ("Citation", "Citation_Score"),
        ("Relevance", "Relevance_Score"),
        ("ReAct Performance Thought", "ReAct_Performance_Thought_Score"),
        ("RAG Retrieval Observation", "RAG_Retrieval_Observation_Score"),
        ("Information Integration", "Information_Integration_Score"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (metric_name, score_col) in enumerate(metrics):
        ax = axes[idx]
        model_a_col = f"Model_A_{score_col}"
        model_b_col = f"Model_B_{score_col}"

        if model_a_col in df.columns and model_b_col in df.columns:
            model_a_scores = df[model_a_col].dropna()
            model_b_scores = df[model_b_col].dropna()

            ax.hist(
                [model_a_scores, model_b_scores],
                bins=5,
                alpha=0.7,
                label=["Model A", "Model B"],
                color=["#3498db", "#e74c3c"],
                edgecolor="black",
            )
            ax.set_xlabel("スコア", fontsize=10)
            ax.set_ylabel("頻度", fontsize=10)
            ax.set_title(metric_name, fontsize=11, fontweight="bold")
            ax.set_xticks(range(1, 6))
            ax.set_xlim(0.5, 5.5)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

    # 最後のサブプロットを非表示
    axes[5].axis("off")

    plt.suptitle(
        "スコア分布（Model A vs Model B）", fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    log_success(f"スコア分布チャートを保存: {output_file}")
    plt.close()


def create_boxplot_chart(
    df: pd.DataFrame, output_file: str = "evaluation_boxplot.png"
) -> None:
    """
    Create box plots comparing score distributions between Model A and Model B.

    The chart displays box plots for the following metrics:
    - Citation Score
    - Relevance Score
    - ReAct Performance Thought Score
    - RAG Retrieval Observation Score
    - Information Integration Score

    Args:
        df: DataFrame containing evaluation results with columns like
            "Model_A_Citation_Score", "Model_B_Citation_Score", etc.
        output_file: Path to save the output PNG file. Defaults to
            "evaluation_boxplot.png".

    Returns:
        None. The chart is saved to the specified output file.
    """

    metrics = [
        ("Citation", "Citation_Score"),
        ("Relevance", "Relevance_Score"),
        ("ReAct Performance\nThought", "ReAct_Performance_Thought_Score"),
        ("RAG Retrieval\nObservation", "RAG_Retrieval_Observation_Score"),
        ("Information\nIntegration", "Information_Integration_Score"),
    ]

    # データを準備
    plot_data = []
    labels = []

    for metric_name, score_col in metrics:
        model_a_col = f"Model_A_{score_col}"
        model_b_col = f"Model_B_{score_col}"

        if model_a_col in df.columns and model_b_col in df.columns:
            plot_data.append(df[model_a_col].dropna().values)
            plot_data.append(df[model_b_col].dropna().values)
            labels.append(f"{metric_name}\nModel A")
            labels.append(f"{metric_name}\nModel B")

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
        "スコア分布の箱ひげ図（Model A vs Model B）",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(range(1, 6))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    log_success(f"箱ひげ図を保存: {output_file}")
    plt.close()


def create_summary_table(
    df: pd.DataFrame, output_file: str = "evaluation_summary.txt"
) -> None:
    """
    Create a summary table with evaluation statistics in text format.

    The table includes:
    - Average scores for each metric (Model A vs Model B)
    - Difference between Model A and Model B scores
    - Overall statistics (mean, min, max, standard deviation)

    Args:
        df: DataFrame containing evaluation results with columns like
            "Model_A_Citation_Score", "Model_B_Citation_Score", etc.
        output_file: Path to save the output text file. Defaults to
            "evaluation_summary.txt".

    Returns:
        None. The summary table is saved to the specified output file.
    """

    metrics = [
        ("Citation", "Citation_Score"),
        ("Relevance", "Relevance_Score"),
        ("ReAct Performance Thought", "ReAct_Performance_Thought_Score"),
        ("RAG Retrieval Observation", "RAG_Retrieval_Observation_Score"),
        ("Information Integration", "Information_Integration_Score"),
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("評価結果サマリー\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"評価対象行数: {len(df)}行\n")
        f.write(f"評価メトリクス数: {len(metrics)}個\n\n")

        f.write("-" * 70 + "\n")
        f.write(
            f"{'メトリクス':<30} {'Model A平均':<15} {'Model B平均':<15} {'差分':<10}\n"
        )
        f.write("-" * 70 + "\n")

        for metric_name, score_col in metrics:
            model_a_col = f"Model_A_{score_col}"
            model_b_col = f"Model_B_{score_col}"

            if model_a_col in df.columns and model_b_col in df.columns:
                model_a_avg = df[model_a_col].mean()
                model_b_avg = df[model_b_col].mean()
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
                f.write(f"{metric_name}:\n")
                f.write(
                    f"  Model A: 平均={df[model_a_col].mean():.2f}, "
                    f"最小={df[model_a_col].min():.0f}, "
                    f"最大={df[model_a_col].max():.0f}, "
                    f"標準偏差={df[model_a_col].std():.2f}\n"
                )
                f.write(
                    f"  Model B: 平均={df[model_b_col].mean():.2f}, "
                    f"最小={df[model_b_col].min():.0f}, "
                    f"最大={df[model_b_col].max():.0f}, "
                    f"標準偏差={df[model_b_col].std():.2f}\n\n"
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
    
    # ragas_evaluation_output.csvを可視化
    python visualize_results.py ragas_evaluation_output.csv

入力CSV形式:
    llm_judge_evaluator.pyの出力CSV（evaluation_output.csv）を想定しています。
    以下の列が必要です:
    - Question
    - Model_A_Citation_Score, Model_B_Citation_Score
    - Model_A_Relevance_Score, Model_B_Relevance_Score
    - Model_A_ReAct_Performance_Thought_Score, Model_B_ReAct_Performance_Thought_Score
    - Model_A_RAG_Retrieval_Observation_Score, Model_B_RAG_Retrieval_Observation_Score
    - Model_A_Information_Integration_Score, Model_B_Information_Integration_Score
    - Evaluation_Error (オプション)

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

    # Get output file names from config
    output_files = get_output_file_names()

    # グラフを作成
    log_info("グラフを作成中...")
    create_score_comparison_chart(df_clean, output_files["evaluation_comparison"])
    create_score_distribution_chart(df_clean, output_files["evaluation_distribution"])
    create_boxplot_chart(df_clean, output_files["evaluation_boxplot"])
    create_summary_table(df_clean, output_files["evaluation_summary"])

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
