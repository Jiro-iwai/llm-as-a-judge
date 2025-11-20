#!/usr/bin/env python3
"""
処理時間比較スクリプト
tmp.txtから処理時間を抽出して2つのモデルを比較します。
"""

import platform
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from config.app_config import get_output_file_names, get_regex_patterns
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
    try:
        matplotlib.rcParams["font.family"] = [
            "Hiragino Sans",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    except Exception:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["figure.figsize"] = (14, 8)


def extract_processing_times(log_file: str):
    """ログファイルから処理時間を抽出"""

    model_a_times = []
    model_b_times = []
    question_numbers = []

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        log_error(f"ファイル '{log_file}' が見つかりません")
        sys.exit(1)

    # Get regex patterns from config
    patterns = get_regex_patterns()
    pattern_a = patterns["model_a_pattern"]
    pattern_b = patterns["model_b_pattern"]

    # Model A (claude3.5-sonnet) の処理時間を抽出
    matches_a = re.findall(pattern_a, content)
    model_a_times = [float(t) for t in matches_a]

    # Model B (claude4.5-haiku) の処理時間を抽出
    matches_b = re.findall(pattern_b, content)
    model_b_times = [float(t) for t in matches_b]

    # 質問番号を生成
    question_numbers = list(range(1, len(model_a_times) + 1))

    log_success(f"Model A (claude3.5-sonnet) の処理時間: {len(model_a_times)}件")
    log_success(f"Model B (claude4.5-haiku) の処理時間: {len(model_b_times)}件")

    if len(model_a_times) != len(model_b_times):
        log_warning("Model AとModel Bのデータ数が一致しません")
        min_len = min(len(model_a_times), len(model_b_times))
        model_a_times = model_a_times[:min_len]
        model_b_times = model_b_times[:min_len]
        question_numbers = question_numbers[:min_len]

    return question_numbers, model_a_times, model_b_times


def create_comparison_chart(
    question_numbers,
    model_a_times,
    model_b_times,
    output_file: str = "processing_time_comparison.png",
):
    """処理時間比較チャートを作成"""

    # バーチャートを作成
    x = question_numbers
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(
        [i - width / 2 for i in x],
        model_a_times,
        width,
        label="Model A (claude3.5-sonnet)",
        alpha=0.8,
        color="#3498db",
    )
    ax.bar(
        [i + width / 2 for i in x],
        model_b_times,
        width,
        label="Model B (claude4.5-haiku)",
        alpha=0.8,
        color="#e74c3c",
    )

    ax.set_xlabel("質問番号", fontsize=12, fontweight="bold")
    ax.set_ylabel("処理時間 (秒)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model A vs Model B 処理時間比較", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    log_success(f"処理時間比較チャートを保存: {output_file}")
    plt.close()


def create_statistics_chart(
    question_numbers,
    model_a_times,
    model_b_times,
    output_file: str = "processing_time_statistics.png",
):
    """統計チャートを作成"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 平均処理時間の比較
    ax1 = axes[0, 0]
    avg_a = sum(model_a_times) / len(model_a_times)
    avg_b = sum(model_b_times) / len(model_b_times)
    bars = ax1.bar(
        ["Model A\n(claude3.5-sonnet)", "Model B\n(claude4.5-haiku)"],
        [avg_a, avg_b],
        color=["#3498db", "#e74c3c"],
        alpha=0.8,
    )
    ax1.set_ylabel("平均処理時間 (秒)", fontsize=11, fontweight="bold")
    ax1.set_title("平均処理時間の比較", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}秒",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 2. 処理時間の分布（ヒストグラム）
    ax2 = axes[0, 1]
    ax2.hist(
        [model_a_times, model_b_times],
        bins=15,
        alpha=0.7,
        label=["Model A", "Model B"],
        color=["#3498db", "#e74c3c"],
        edgecolor="black",
    )
    ax2.set_xlabel("処理時間 (秒)", fontsize=11)
    ax2.set_ylabel("頻度", fontsize=11)
    ax2.set_title("処理時間の分布", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # 3. 箱ひげ図
    ax3 = axes[1, 0]
    bp = ax3.boxplot(
        [model_a_times, model_b_times],
        labels=["Model A", "Model B"],
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    colors = ["#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel("処理時間 (秒)", fontsize=11, fontweight="bold")
    ax3.set_title("処理時間の箱ひげ図", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 4. 時系列プロット
    ax4 = axes[1, 1]
    ax4.plot(
        question_numbers,
        model_a_times,
        marker="o",
        label="Model A",
        color="#3498db",
        linewidth=2,
        markersize=4,
    )
    ax4.plot(
        question_numbers,
        model_b_times,
        marker="s",
        label="Model B",
        color="#e74c3c",
        linewidth=2,
        markersize=4,
    )
    ax4.set_xlabel("質問番号", fontsize=11, fontweight="bold")
    ax4.set_ylabel("処理時間 (秒)", fontsize=11, fontweight="bold")
    ax4.set_title("処理時間の推移", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    plt.suptitle("処理時間統計分析", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    log_success(f"統計チャートを保存: {output_file}")
    plt.close()


def create_summary_table(
    question_numbers,
    model_a_times,
    model_b_times,
    output_file: str = "processing_time_summary.txt",
):
    """サマリーテーブルを作成"""

    df = pd.DataFrame(
        {
            "Question": question_numbers,
            "Model_A_Time": model_a_times,
            "Model_B_Time": model_b_times,
            "Difference": [b - a for a, b in zip(model_a_times, model_b_times)],
            "Speedup": [
                a / b if b > 0 else 0 for a, b in zip(model_a_times, model_b_times)
            ],
        }
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("処理時間比較サマリー\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"評価対象質問数: {len(question_numbers)}件\n\n")

        f.write("-" * 70 + "\n")
        f.write(
            f"{'質問':<6} {'Model A (秒)':<15} {'Model B (秒)':<15} {'差分 (秒)':<12} {'速度比':<10}\n"
        )
        f.write("-" * 70 + "\n")

        for _, row in df.iterrows():
            f.write(
                f"{row['Question']:<6} {row['Model_A_Time']:<15.2f} {row['Model_B_Time']:<15.2f} "
                f"{row['Difference']:<12.2f} {row['Speedup']:<10.2f}x\n"
            )

        f.write("-" * 70 + "\n\n")

        # 統計情報
        f.write("統計情報\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model A (claude3.5-sonnet):\n")
        f.write(f"  平均: {df['Model_A_Time'].mean():.2f}秒\n")
        f.write(f"  最小: {df['Model_A_Time'].min():.2f}秒\n")
        f.write(f"  最大: {df['Model_A_Time'].max():.2f}秒\n")
        f.write(f"  標準偏差: {df['Model_A_Time'].std():.2f}秒\n")
        f.write(
            f"  合計: {df['Model_A_Time'].sum():.2f}秒 ({df['Model_A_Time'].sum() / 60:.2f}分)\n\n"
        )

        f.write("Model B (claude4.5-haiku):\n")
        f.write(f"  平均: {df['Model_B_Time'].mean():.2f}秒\n")
        f.write(f"  最小: {df['Model_B_Time'].min():.2f}秒\n")
        f.write(f"  最大: {df['Model_B_Time'].max():.2f}秒\n")
        f.write(f"  標準偏差: {df['Model_B_Time'].std():.2f}秒\n")
        f.write(
            f"  合計: {df['Model_B_Time'].sum():.2f}秒 ({df['Model_B_Time'].sum() / 60:.2f}分)\n\n"
        )

        avg_diff = df["Difference"].mean()
        total_diff = df["Model_A_Time"].sum() - df["Model_B_Time"].sum()
        avg_speedup = df["Speedup"].mean()
        model_a_total = df["Model_A_Time"].sum()

        f.write("比較結果:\n")
        f.write(
            f"  平均処理時間差: {avg_diff:.2f}秒 (Model Bが{'速い' if avg_diff < 0 else '遅い'})\n"
        )
        f.write(f"  合計処理時間差: {total_diff:.2f}秒 ({total_diff / 60:.2f}分)\n")
        f.write(
            f"  平均速度比: {avg_speedup:.2f}x (Model Bが{'速い' if avg_speedup > 1 else '遅い'})\n"
        )
        if model_a_total > 0:
            f.write(f"  時間削減率: {abs(total_diff) / model_a_total * 100:.1f}%\n")
        else:
            f.write("  時間削減率: 計算不可（データがありません）\n")

    log_success(f"サマリーテーブルを保存: {output_file}")


def main():
    """メイン処理"""
    log_file = "tmp.txt"

    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    log_section("処理時間比較分析")

    # 処理時間を抽出
    question_numbers, model_a_times, model_b_times = extract_processing_times(log_file)

    if len(model_a_times) == 0 or len(model_b_times) == 0:
        log_error("処理時間データが見つかりません")
        sys.exit(1)

    log_info("")

    # Get output file names from config
    output_files = get_output_file_names()

    # グラフを作成
    log_info("グラフを作成中...")
    create_comparison_chart(
        question_numbers,
        model_a_times,
        model_b_times,
        output_files["processing_time_comparison"],
    )
    create_statistics_chart(
        question_numbers,
        model_a_times,
        model_b_times,
        output_files["processing_time_statistics"],
    )
    create_summary_table(
        question_numbers,
        model_a_times,
        model_b_times,
        output_files["processing_time_summary"],
    )

    log_info("")
    log_section("✓ 分析完了!")
    log_info("生成されたファイル:")
    log_info(
        f"  - {output_files['processing_time_comparison']}: 処理時間比較チャート",
        indent=1,
    )
    log_info(
        f"  - {output_files['processing_time_statistics']}: 統計チャート", indent=1
    )
    log_info(
        f"  - {output_files['processing_time_summary']}: サマリーテーブル", indent=1
    )


if __name__ == "__main__":
    main()
