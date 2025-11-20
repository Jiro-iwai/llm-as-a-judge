#!/usr/bin/env python3
"""
評価結果を可視化するスクリプト
evaluation_output.csvの評価結果をグラフで表示します。
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys

# 日本語フォントの設定（macOS用）
import platform

if platform.system() == "Darwin":  # macOS
    # macOSの日本語フォントを試す
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
plt.rcParams["figure.figsize"] = (14, 10)


def load_data(csv_file: str) -> pd.DataFrame:
    """CSVファイルを読み込む"""
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ データを読み込みました: {len(df)}行")
        return df
    except FileNotFoundError:
        print(f"❌ エラー: ファイル '{csv_file}' が見つかりません", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ エラー: ファイルの読み込みに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """評価データを準備（エラー行を除外）"""
    # エラーが発生した行を除外
    if "Evaluation_Error" in df.columns:
        df_clean = df[df["Evaluation_Error"].isna()].copy()
        error_count = len(df) - len(df_clean)
        if error_count > 0:
            print(f"⚠️  エラー行を除外: {error_count}行")
    else:
        df_clean = df.copy()

    # Ensure return type is DataFrame
    if isinstance(df_clean, pd.DataFrame):
        return df_clean
    else:
        return pd.DataFrame(df_clean)


def create_score_comparison_chart(
    df: pd.DataFrame, output_file: str = "evaluation_comparison.png"
):
    """Model AとModel Bのスコア比較チャートを作成"""

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
    print(f"✓ スコア比較チャートを保存: {output_file}")
    plt.close()


def create_score_distribution_chart(
    df: pd.DataFrame, output_file: str = "evaluation_distribution.png"
):
    """スコアの分布を表示"""

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
    print(f"✓ スコア分布チャートを保存: {output_file}")
    plt.close()


def create_boxplot_chart(df: pd.DataFrame, output_file: str = "evaluation_boxplot.png"):
    """箱ひげ図を作成"""

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
    print(f"✓ 箱ひげ図を保存: {output_file}")
    plt.close()


def create_summary_table(df: pd.DataFrame, output_file: str = "evaluation_summary.txt"):
    """サマリーテーブルをテキストファイルに出力"""

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

    print(f"✓ サマリーテーブルを保存: {output_file}")


def main():
    """メイン処理"""
    csv_file = "evaluation_output.csv"

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    print("=" * 70)
    print("評価結果の可視化")
    print("=" * 70)
    print()

    # データを読み込む
    df = load_data(csv_file)
    df_clean = prepare_data(df)

    if len(df_clean) == 0:
        print("❌ エラー: 有効な評価データがありません", file=sys.stderr)
        sys.exit(1)

    print(f"✓ 有効な評価データ: {len(df_clean)}行")
    print()

    # グラフを作成
    print("グラフを作成中...")
    create_score_comparison_chart(df_clean, "evaluation_comparison.png")
    create_score_distribution_chart(df_clean, "evaluation_distribution.png")
    create_boxplot_chart(df_clean, "evaluation_boxplot.png")
    create_summary_table(df_clean, "evaluation_summary.txt")

    print()
    print("=" * 70)
    print("✓ 可視化完了!")
    print("=" * 70)
    print("生成されたファイル:")
    print("  - evaluation_comparison.png: スコア比較チャート")
    print("  - evaluation_distribution.png: スコア分布チャート")
    print("  - evaluation_boxplot.png: 箱ひげ図")
    print("  - evaluation_summary.txt: サマリーテーブル")


if __name__ == "__main__":
    main()
