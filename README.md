# LLM-as-a-Judge 評価スクリプト

LLM-as-a-JudgeとRagasフレームワークを使用した、CyChat SDの応答を自動評価するためのPythonスクリプトです。

## 概要

このリポジトリには、**3つの異なる評価ツール**が含まれています：

1.  **`llm_judge_evaluator.py`**: 詳細な5メトリックのルーブリックを持つカスタムLLM-as-a-Judge
2.  **`ragas_llm_judge_evaluator.py`**: Ragasフレームワークベースの評価
3.  **`format_clarity_evaluator.py`**: 2つのモデルの応答のフォーマット/スタイルの類似性を比較

すべてのスクリプトは、LLM-as-a-Judgeとして**Azure OpenAI (GPT-5, GPT-4.1)とStandard OpenAI (GPT-4 Turbo)**をサポートしています。

## 目次

  - [インストール](#インストール)
  - [Makefileの使い方](#makefileの使い方)
  - [プログラム別のAPI要件](#プログラム別のapi要件)
  - [Custom LLM Judge Evaluator](#custom-llm-judge-evaluator)
  - [Ragas-Based Evaluator](#ragas-based-evaluator)
  - [Format Clarity Evaluator](#format-clarity-evaluator)
  - [データ収集スクリプト](#データ収集スクリプト)
  - [処理時間比較スクリプト](#処理時間比較スクリプト)
  - [結果可視化スクリプト](#結果可視化スクリプト)
  - [エラーハンドリング](#エラーハンドリング)
  - [パフォーマンスに関する考慮事項](#パフォーマンスに関する考慮事項)

-----

## インストール

このリポジトリ内のすべてのスクリプトは、同じ依存関係を共有しています。

### 方法1: Makefileを使用（推奨）

1.  このリポジトリをクローンまたはダウンロードします
2.  仮想環境を作成して依存関係をインストールします：

```bash
# 仮想環境の作成と依存関係のインストールを一度に実行
make setup
```

または、個別に実行する場合：

```bash
# 仮想環境を作成
make venv

# 依存関係をインストール
make install-deps
```

3.  API認証情報を安全に設定します（いずれかの方法を選択）：

### 方法2: 手動インストール

1.  このリポジトリをクローンまたはダウンロードします
2.  仮想環境を作成します：

```bash
# uvを使用する場合（推奨）
uv venv

# または標準のvenvを使用する場合
python3 -m venv .venv
```

3.  仮想環境を有効化します：

```bash
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate  # Windows
```

4.  依存関係をインストールします：

```bash
# uvを使用する場合（推奨）
uv pip install -r requirements.txt

# または標準のpipを使用する場合
pip install -r requirements.txt
```

5.  API認証情報を安全に設定します（いずれかの方法を選択）：

### `.env` ファイル

プロジェクトのルートディレクトリに `.env` ファイルを作成します：

**Azure OpenAI の場合：**

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-azure-key
MODEL_NAME=gpt-5  # または gpt-4.1
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

**Standard OpenAI の場合：**

```env
OPENAI_API_KEY=your-actual-openai-key
MODEL_NAME=gpt-4-turbo  # または gpt-4.1
```

**注意**: `MODEL_NAME`環境変数は任意です。コマンドライン引数 `-m` で指定することもできます。

### アプリケーション設定（設定値の外部化）

このプロジェクトでは、多くの設定値を環境変数やYAML設定ファイルから読み込むことができます。これにより、コードを編集せずに設定を変更できます。

#### 環境変数での設定

以下の環境変数を使用して設定値を変更できます：

```bash
# API設定
export APP_TIMEOUT=150              # リクエストタイムアウト（秒、デフォルト: 120）
export APP_MAX_RETRIES=5           # 最大リトライ回数（デフォルト: 3）
export APP_RETRY_DELAY=3           # リトライ間隔（秒、デフォルト: 2）
export APP_API_DELAY=2.0           # API呼び出し間隔（秒、デフォルト: 1.0）
export APP_DEFAULT_IDENTITY="USER" # デフォルトidentity（デフォルト値は設定ファイルまたはコード内のデフォルト値を参照）

# 出力ファイル名（オプション）
export APP_OUTPUT_FILE_PROCESSING_TIME_LOG="custom_time_log.txt"  # 処理時間ログファイル名
export APP_OUTPUT_FILE_EVALUATION_COMPARISON="custom_comparison.png"  # 評価比較チャートのファイル名
export APP_OUTPUT_FILE_EVALUATION_DISTRIBUTION="custom_distribution.png"  # 評価分布チャートのファイル名
export APP_OUTPUT_FILE_EVALUATION_BOXPLOT="custom_boxplot.png"  # 評価箱ひげ図のファイル名
export APP_OUTPUT_FILE_EVALUATION_SUMMARY="custom_summary.txt"  # 評価サマリーテーブルのファイル名
export APP_OUTPUT_FILE_PROCESSING_TIME_COMPARISON="custom_time_comparison.png"  # 処理時間比較チャートのファイル名
export APP_OUTPUT_FILE_PROCESSING_TIME_STATISTICS="custom_time_statistics.png"  # 処理時間統計チャートのファイル名
export APP_OUTPUT_FILE_PROCESSING_TIME_SUMMARY="custom_time_summary.txt"  # 処理時間サマリーテーブルのファイル名

# 設定ファイルのパス
export APP_CONFIG_FILE=config.yaml  # YAML設定ファイルのパス（オプション）
```

#### YAML設定ファイルでの設定

プロジェクトルートに `config.yaml` ファイルを作成して設定を管理できます：

```yaml
# API設定
timeout: 180
max_retries: 5
retry_delay: 3
api_delay: 2.0
default_identity: "CONFIG_USER"

# 出力ファイル名
output_files:
  evaluation_comparison: "output/custom_comparison.png"
  evaluation_distribution: "output/evaluation_distribution.png"
  evaluation_boxplot: "output/evaluation_boxplot.png"
  evaluation_summary: "output/evaluation_summary.txt"
  ragas_evaluation_comparison: "output/ragas_evaluation_comparison.png"
  ragas_evaluation_distribution: "output/ragas_evaluation_distribution.png"
  ragas_evaluation_boxplot: "output/ragas_evaluation_boxplot.png"
  ragas_evaluation_summary: "output/ragas_evaluation_summary.txt"
  processing_time_comparison: "output/processing_time_comparison.png"
  processing_time_statistics: "output/processing_time_statistics.png"
  processing_time_summary: "output/processing_time_summary.txt"
  processing_time_log: "output/processing_time_log.txt"

# 正規表現パターン（compare_processing_time.py用）
regex_patterns:
  model_a_pattern: "📥 \[claude3\.5-sonnet\].*?経過時間: ([\\d.]+)秒"
  model_b_pattern: "📥 \[claude4\.5-haiku\].*?経過時間: ([\\d.]+)秒"
```

設定ファイルは `APP_CONFIG_FILE` 環境変数で指定できます：

```bash
export APP_CONFIG_FILE=/path/to/config.yaml
```

#### 優先順位

設定値は以下の優先順位で読み込まれます：

1. **環境変数**（最優先）
2. **設定ファイル**（YAML）
3. **デフォルト値**（コード内）

環境変数は設定ファイルの値を上書きします。

#### 使用例

```bash
# 環境変数でタイムアウトを変更
export APP_TIMEOUT=200
python scripts/llm_judge_evaluator.py data.csv

# 設定ファイルを使用
export APP_CONFIG_FILE=my_config.yaml
python scripts/collect_responses.py questions.txt

# コマンドライン引数で上書き（一部のスクリプトのみ）
python scripts/collect_responses.py questions.txt --timeout 150 --delay 2.5
```

**注意**: コマンドライン引数が指定された場合は、それが環境変数や設定ファイルよりも優先されます。

## Makefileの使い方

このプロジェクトには、開発作業を効率化するためのMakefileが含まれています。

### 基本的な使い方

```bash
# ヘルプを表示
make help

# 仮想環境と依存関係をセットアップ（初回のみ）
make setup

# すべてのテストを実行（フォーマット、リンター、型チェック、スクリプトテスト）
make test

# コードをフォーマット
make format

# リンターを実行
make lint

# 型チェックを実行
make typecheck

# スクリプトのインターフェースをテスト
make test-scripts

# 生成ファイルをクリーンアップ
make clean
```

### スクリプトの使い方を確認

各スクリプトの使い方をMakefile経由で確認できます：

```bash
# llm_judge_evaluator.pyの使い方を表示
make help-llm-judge

# format_clarity_evaluator.pyの使い方を表示
make help-format-clarity

# ragas_llm_judge_evaluator.pyの使い方を表示
make help-ragas

# collect_responses.pyの使い方を表示
make help-collect

# すべてのスクリプトの使い方を一度に表示
make help-all
```

### 利用可能なターゲット

#### セットアップ関連

| ターゲット | 説明 |
|-----------|------|
| `make setup` | 仮想環境を作成して依存関係をインストール（初回セットアップ用） |
| `make venv` | 仮想環境（.venv）を作成 |
| `make install-deps` | 依存関係をインストール（仮想環境が必要） |
| `make update-deps` | 依存関係を最新バージョンに更新（仮想環境が必要） |
| `make clean-venv` | 仮想環境（.venv）を削除 |

#### テスト関連

| ターゲット | 説明 |
|-----------|------|
| `make test` | すべてのテストを実行（format, lint, typecheck, test-scripts, test-unit） |
| `make format` | ruffでコードをフォーマット |
| `make lint` | ruffでリンターを実行 |
| `make typecheck` | pyrightで型チェックを実行 |
| `make test-scripts` | スクリプトのインターフェース（--help）をテスト |
| `make test-unit` | pytestでユニットテストを実行（テストファイルがある場合） |
| `make test-coverage` | pytestでカバレッジレポートを生成（HTMLレポート: htmlcov/index.html） |
| `make clean` | 生成ファイルとキャッシュを削除 |

#### ヘルプ関連

| ターゲット | 説明 |
|-----------|------|
| `make help-llm-judge` | `llm_judge_evaluator.py`の使い方を表示 |
| `make help-format-clarity` | `format_clarity_evaluator.py`の使い方を表示 |
| `make help-ragas` | `ragas_llm_judge_evaluator.py`の使い方を表示 |
| `make help-collect` | `collect_responses.py`の使い方を表示 |
| `make help-all` | すべてのスクリプトの使い方を表示 |

**注意**: 
- `make setup`は`uv`を使用して仮想環境と依存関係をセットアップします。`uv`がインストールされていない場合は、方法2の手動インストール手順を参照してください。
- `ruff`、`pyright`、`pytest`などの開発依存関係は`requirements.txt`に含まれており、`make setup`または`make install-deps`でインストールされます。
- 仮想環境がセットアップされていない場合、`format`、`lint`、`typecheck`ターゲットは警告を表示してスキップしますが、スクリプトの動作確認テスト（`test-scripts`）は実行されます。

## プログラム別のAPI要件

各評価スクリプトは設定オプションに重要な違いがあります。

### モデルの互換性と推奨事項

| スクリプト | 推奨モデル | デフォルトモデル |
|--------|------------------|------------------|
| `llm_judge_evaluator.py` | **GPT-5** | GPT-4.1 |
| `ragas_llm_judge_evaluator.py` | **GPT-4.1** | GPT-4.1 |
| `format_clarity_evaluator.py` | **GPT-5** | GPT-4-turbo |

### Temperature（温度）設定

3つのスクリプトすべてに、モデルに基づいてTemperatureを異なる方法で処理する条件付きロジックがあります：

#### **GPT-5** (Azure OpenAI)

  -  **Temperatureは1.0に固定**（設定不可）
  - 環境変数で `MODEL_NAME=gpt-5` を設定

#### **GPT-4 モデル** (Azure OpenAI または Standard OpenAI)

  - **Temperatureは設定可能**（デフォルト：0.7）
  - `MODEL_NAME=gpt-4.1`（またはお好みのGPT-4バリアント）を設定

###  推奨理由

#### **GPT-5を使用する場合：**

  -  **`llm_judge_evaluator.py`** - カスタムルーブリック評価は、GPT-5の強化された機能の恩恵を受けます
  -  **`format_clarity_evaluator.py`** - スタイル/フォーマットの比較は、GPT-5の推論能力と相性が良いです

#### **GPT-4を使用する場合：**

  - **`ragas_llm_judge_evaluator.py`** - Ragasフレームワークは、一貫したメトリクスのために**Temperature制御**を必要とします
      - GPT-4.1はTemperatureを設定できます（デフォルト0.7）
      - GPT-5の固定Temperature=1.0は、コード実行時に問題を引き起こす可能性があります
      - **重要**：このスクリプトには `gpt-4.1` の使用を強く推奨します（デフォルトモデル）

## Custom LLM Judge Evaluator

`llm_judge_evaluator.py` スクリプトは、カスタムの5メトリックルーブリックを使用して、包括的でReAct固有の評価を提供します。

### 主な機能

  - **ReAct固有のメトリクス**：思考プロセス、検索品質、引用の正確性を評価
  - **詳細な根拠**：各スコアに対する qualitative な説明を提供
  - **カスタムルーブリック**：ReActチャットボット評価用に調整された採点システム
  - **構造化された出力**：スコアと根拠をCSV形式で整理

### 入力CSVの形式

入力CSVファイルには、この順序で正確に3つの列が含まれている必要があります：

1.  **Question**: 元のユーザーの質問
2.  **Model_A_Response**: モデルAの完全な応答
3.  **Model_B_Response**: モデルBの完全な応答

**注意**: ヘッダー行は任意です。スクリプトは自動的にヘッダー行を検出し、ヘッダー行がない場合は最初の行をデータとして扱います。カラム名は`Model_A_Response`/`Model_B_Response`形式で指定してください。

### 使用方法

####  コストに関する警告

**このスクリプトはCSVの1行ごとに1回のAPIコールを行います。** 

**常に少量のサンプルで最初にテストしてください：**

```bash
# 最初に5行だけでテストします（推奨！）
python scripts/llm_judge_evaluator.py my_test_data.csv -n 5
```

**基本的な使用方法：**

```bash
python scripts/llm_judge_evaluator.py my_test_data.csv
```

**モデルを指定して実行：**

```bash
# GPT-5を使用
python scripts/llm_judge_evaluator.py my_test_data.csv -m gpt-5

# GPT-4.1を使用（デフォルト）
python scripts/llm_judge_evaluator.py my_test_data.csv -m gpt-4.1

# GPT-4 Turboを使用
python scripts/llm_judge_evaluator.py my_test_data.csv -m gpt-4-turbo
```

**カスタム出力ファイルを指定：**

```bash
python scripts/llm_judge_evaluator.py my_test_data.csv -o my_results.csv
```

**最初のN行のみを処理（コスト管理）：**

```bash
# 最初の10行のみを処理
python scripts/llm_judge_evaluator.py my_test_data.csv -n 10

# 最初の50行をカスタム出力で処理
python scripts/llm_judge_evaluator.py my_test_data.csv -n 50 -o test_results.csv

# モデルを指定して最初の10行を処理
python scripts/llm_judge_evaluator.py my_test_data.csv -n 10 -m gpt-5
```

**非対話実行（CI/バッチ環境用）：**

```bash
# --yesフラグで確認プロンプトをスキップ（10行超でも自動実行）
python scripts/llm_judge_evaluator.py my_test_data.csv --yes

# 通常実行（10行超の場合は確認プロンプトが表示される）
python scripts/llm_judge_evaluator.py my_test_data.csv
```

**注意**: 10行を超えるCSVファイルを処理する場合、デフォルトでは確認プロンプトが表示されます。CI/バッチ環境や自動実行の場合は`--yes`フラグを使用してください。`run_full_pipeline.py`から実行する場合は自動的に`--yes`フラグが付与されます。

### モデル指定オプション

モデルは以下の3つの方法で指定できます：

1. **コマンドライン引数**（推奨）: `-m gpt-5` または `--model gpt-4.1`
2. **環境変数**: `export MODEL_NAME='gpt-5'`
3. **デフォルト**: `gpt-4.1`（指定がない場合）

**サポートされているモデル：**
- `gpt-5`: GPT-5（`max_completion_tokens`使用、temperature=1.0）
- `gpt-4.1`: GPT-4.1（`max_tokens`使用、temperature=0.7）**デフォルト**
- `gpt-4-turbo`: GPT-4 Turbo（`max_tokens`使用、temperature=0.7）

### 出力CSVの形式

出力ファイル（デフォルトは `evaluation_output.csv`、`run_full_pipeline.py`経由の場合は `output/evaluation_output.csv`）には以下が含まれます：

  - すべての元の列（Question, Model_A_Response, Model_B_Response）
  - 各モデル（AとB）について：
      - Citation Score & Justification
      - Relevance Score & Justification
      - ReAct Performance Thought Score & Justification
      - RAG Retrieval Observation Score & Justification
      - Information Integration Score & Justification
  - Evaluation\_Error 列（評価が失敗した場合のエラーメッセージを含む）

### 採点ルーブリック

各応答は、5つの側面について1～5のスケールで評価されます：

#### 1\. RAG Generation - Citation (1-5)

引用の品質、正確性、必要性を評価します。

#### 2\. Relevance (1-5)

回答がユーザーのクエリ全体にどれだけうまく対応しているかを評価します。

#### 3\. ReAct Performance - Thought (1-5)

推論プロセスの論理的な品質と効率を評価します。

#### 4\. RAG Retrieval - Observation (1-5)

検索されたソース資料の品質と関連性を評価します。

#### 5\. RAG Generation - Information Integration (1-5)

モデルが検索された情報をどれだけ正確に統合しているかを評価します。

-----

## Ragas-Based Evaluator

`ragas_llm_judge_evaluator.py` スクリプトは、Ragasフレームワークと自動ReActログ解析を使用して、最新の標準化された評価アプローチを提供します。

**推奨モデル：GPT-4.1** - **一貫したRagasメトリクスのためにTemperature制御が必要です！**（詳細は[プログラム別のAPI要件](#プログラム別のapi要件)を参照）

### 主な機能

  - **自動ログ解析**：生のReActログから最終回答（Final Answer）とコンテキスト（Contexts）を自動的に抽出
  - **標準化されたメトリクス**：Ragasのメトリクスの1つである faithfulness を使用
  - **手動でのデータ準備不要**：生のチャットボット出力ログを直接処理
  - **並列比較**：カスタムジャッジ評価との比較が可能

### Ragas用の入力CSV形式

入力CSVには、正確に3つの列が含まれている必要があります：

1.  **Question**: 元のユーザーの質問
2.  **Model\_A\_Response**: モデルAの完全な**フォーマット済み**ReActログ
3.  **Model\_B\_Response**: モデルBの完全な**フォーマット済み**ReActログ

**注意**: ヘッダー行は任意です。スクリプトは自動的にヘッダー行を検出します。

#### 期待されるReActログの構造（フォーマット済み - `src/utils/log_output_simplifier.py` を使用）

スクリプトは、これらのセクションを持つログを期待します：

```
## 📝 Task タスク
---
情報検索

## 💬 Reaction 反応
---
None

## 📂 Classification 分類
---
社内

## 📊 Status 状態
---
独立

## 🤖 LLM Thought Process 思考
---
[Reasoning about the task...]

## ⚡ Action 行動
---
社内検索

## ⌨️ Action Input 行動入力
---
[Search query]

## 📚 Raw Search Results (Cleaned) 観察
---
### Result 1
[First search result content...]

## 🔗 URLs URL
---
https://example.com/doc1
関連度：0.95
################################################
### Result 2
[Second search result content...]

## 🔗 URLs URL
---
https://example.com/doc2
関連度：0.90
################################################

## ✅ Final Answer 回答
---
[The final answer to the user's question...]
```

### 使用方法

**基本的な使用方法：**

```bash
python scripts/ragas_llm_judge_evaluator.py test_5_rows.csv
```

**モデルを指定して実行：**

```bash
# GPT-4.1を使用（デフォルト、推奨）
python scripts/ragas_llm_judge_evaluator.py my_data.csv -m gpt-4.1

# GPT-5を使用（非推奨：Temperature制御の問題）
python scripts/ragas_llm_judge_evaluator.py my_data.csv -m gpt-5
```

**カスタム出力ファイルを指定：**

```bash
python scripts/ragas_llm_judge_evaluator.py my_data.csv -o output/ragas_results.csv
```

**最初のN行でテスト：**

```bash
# 最初に3行だけでテスト
python scripts/ragas_llm_judge_evaluator.py my_data.csv -n 3

# モデルを指定して最初の3行をテスト
python scripts/ragas_llm_judge_evaluator.py my_data.csv -n 3 -m gpt-4.1
```

**メトリクスを切り替える：**

```bash
# faithfulness / answer_relevance / context_precision / context_recall の中から選択
python scripts/ragas_llm_judge_evaluator.py my_data.csv --metrics faithfulness context_precision

# プリセット（basic: faithfulness+answer_relevance, extended: 全メトリクス）
python scripts/ragas_llm_judge_evaluator.py my_data.csv --metrics-preset basic
```

### モデル指定オプション

モデルは以下の3つの方法で指定できます：

1. **コマンドライン引数**（推奨）: `-m gpt-4.1` または `--model gpt-4.1`
2. **環境変数**: `export MODEL_NAME='gpt-4.1'`
3. **デフォルト**: `gpt-4.1`（指定がない場合）

**サポートされているモデル：**
- `gpt-5`: GPT-5（`max_completion_tokens`使用、temperature=1.0）**非推奨**
- `gpt-4.1`: GPT-4.1（`max_tokens`使用、temperature=0.7）**デフォルト・推奨**
- `gpt-4-turbo`: GPT-4 Turbo（`max_tokens`使用、temperature=0.7）

**重要**: Ragasフレームワークは一貫したメトリクスのためにTemperature制御が必要です。GPT-5の固定Temperature=1.0は問題を引き起こす可能性があるため、**GPT-4.1の使用を強く推奨**します。

### 出力CSVの形式

出力ファイル（デフォルトは `output/ragas_evaluation_output.csv`）には以下が含まれます：

  - **元の列**：Question, Model\_A\_Response, Model\_B\_Response
  - **解析された列**：
      - `model_A_answer`: モデルAから抽出された最終回答
      - `model_A_contexts`: モデルAのログから抽出されたコンテキストのリスト
      - `model_B_answer`: モデルBから抽出された最終回答
      - `model_B_contexts`: モデルBのログから抽出されたコンテキストのリスト
  - **Ragasスコア（Model\_A\_*/Model\_B\_* プレフィックスでペアを構成）**：
      - `*_faithfulness_score`: コンテキストとの事実上一貫性 (0-1)
      - `*_answer_relevance_score`: 質問への回答適合度 (0-1)
      - `*_context_precision_score`: 取得したコンテキストのうち回答に寄与した割合 (0-1)
      - `*_context_recall_score`: 回答に必要な情報がどれだけ取得できたか (0-1)

### Ragasメトリクスの説明

#### [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) (0-1)

回答が、与えられたコンテキストと事実上整合しているかを測定します。スコアが高いほど、回答がハルシネーション（幻覚）を起こさず、検索された情報に忠実であることを示します。

#### [Answer Relevance](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) (0-1)

質問に対してどれだけ適切に回答しているかを評価します。高スコアの回答は、ユーザーの質問意図を正確に捉えています。

#### [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) (0-1)

検索・取得した複数のコンテキストのうち、回答に実際に貢献した情報の割合を測定します。余計な検索結果が多い場合はスコアが下がります。

#### [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) (0-1)

回答に必要な情報が、取得できたコンテキストにどれだけ含まれていたかを測定します。重要な証拠が欠落している場合はスコアが下がります。

-----

## Format Clarity Evaluator

`format_clarity_evaluator.py` スクリプトは、2つのモデルの応答フォーマットの類似性を評価するための専用ツールです。モデルAの応答フォーマットが、ゴールデンスタンダードとして使用されるモデルBのフォーマットにどれだけ近いかを評価します。

### 主な機能

  - **フォーマット比較**：マークダウン、リスト、構造の類似性を評価
  - **単一スコア**：包括的なフォーマット/明瞭性スコア（1-5）を1つ提供
  - **自動解析**：生のReActログから "Final Answer" セクションを抽出
  - **詳細な根拠**：フォーマットが一致する、または異なる理由を説明

### 評価対象

エバリュエーターは以下に焦点を当てます：

  - マークダウンの使用（`##` のような見出し、`**text**` のような太字）
  - リスト構造（`-` の箇条書き vs `1.` の番号付き）
  - アイデアの論理的な分離（段落、セクション）
  - 全体的な構造の類似性

### 入力CSVの形式

入力CSVファイルには、この順序で正確に3つの列が含まれている必要があります：

1.  **Question**: 元のユーザーの質問
2.  **Model_A_Response**: モデルAの完全な応答（ReActログ形式）
3.  **Model_B_Response**: モデルBの完全な応答（ReActログ形式）

**注意**: ヘッダー行は任意です。スクリプトは自動的にヘッダー行を検出し、ヘッダー行がない場合は最初の行をデータとして扱います。カラム名は`Model_A_Response`/`Model_B_Response`または`Claude_35_Raw_Log`/`Claude_45_Raw_Log`のいずれもサポートされます（後者は互換性のためのエイリアスで、実際のモデル名とは無関係です）。評価するモデルは固定されておらず、任意の2つのモデルを比較できます。

### 使用方法

**基本的な使用方法：**

```bash
python scripts/format_clarity_evaluator.py input.csv
```

**モデルを指定して実行：**

```bash
# GPT-5を使用
python scripts/format_clarity_evaluator.py input.csv -m gpt-5

# GPT-4-turboを使用（デフォルト）
python scripts/format_clarity_evaluator.py input.csv -m gpt-4-turbo

# GPT-4.1を使用
python scripts/format_clarity_evaluator.py input.csv -m gpt-4.1
```

**行数を制限してテスト：**

```bash
# 最初の5行のみを処理
python scripts/format_clarity_evaluator.py input.csv -n 5

# 最初の10行をカスタム出力で処理
python scripts/format_clarity_evaluator.py input.csv -n 10 -o test_results.csv

# モデルを指定して最初の5行を処理
python scripts/format_clarity_evaluator.py input.csv -n 5 -m gpt-5
```

**カスタム出力ファイルを指定：**

```bash
python scripts/format_clarity_evaluator.py input.csv -o my_format_results.csv
```

**非対話実行（CI/バッチ環境用）：**

```bash
# --yesフラグで確認プロンプトをスキップ（10行超でも自動実行）
python scripts/format_clarity_evaluator.py input.csv --yes

# 通常実行（10行超の場合は確認プロンプトが表示される）
python scripts/format_clarity_evaluator.py input.csv
```

**注意**: 10行を超えるCSVファイルを処理する場合、デフォルトでは確認プロンプトが表示されます。CI/バッチ環境や自動実行の場合は`--yes`フラグを使用してください。`run_full_pipeline.py`から実行する場合は自動的に`--yes`フラグが付与されます。

### モデル指定オプション

モデルは以下の3つの方法で指定できます：

1. **コマンドライン引数**（推奨）: `-m gpt-5` または `--model gpt-4-turbo`
2. **環境変数**: `export MODEL_NAME='gpt-5'`
3. **デフォルト**: `gpt-4-turbo`（指定がない場合）

**サポートされているモデル：**
- `gpt-5`: GPT-5（`max_completion_tokens`使用、temperature=1.0）
- `gpt-4.1`: GPT-4.1（`max_tokens`使用、temperature=0.7）
- `gpt-4-turbo`: GPT-4 Turbo（`max_tokens`使用、temperature=0.7）**デフォルト**

### 出力CSVの形式

出力ファイル（デフォルトは `format_clarity_output.csv`、`run_full_pipeline.py`経由の場合は `output/format_clarity_output.csv`）には以下が含まれます：

| 列 | 説明 |
|--------|-------------|
| `Question` | 元の質問 |
| `Model_A_Final_Answer` | モデルAのログから解析された最終回答 |
| `Model_B_Final_Answer` | モデルBのログから解析された最終回答 |
| `Format_Clarity_Score` | 1～5のスコア |
| `Format_Clarity_Justification` | スコアの詳細な説明 |
| `Evaluation_Error` | 評価が失敗した場合のエラーメッセージ |

### 採点ルーブリック

LLMジャッジは詳細な5段階のスケールを使用します：

  - **5 (優秀)**：ほぼ同一のフォーマット。マークダウン、リスト、構造を完璧に反映している
  - **4 (良い)**：ほとんど同様だが、わずかな逸脱がある（例：箇条書き vs 番号付き）
  - **3 (許容)**：いくつかの類似点はあるが、重大な違いもある（例：リスト vs 段落）
  - **2 (悪い)**：ほとんどが異なり、ゴールデンスタンダード（モデルB）の構造に似ていない
  - **1 (非常に悪い)**：完全に異なるフォーマット（例：構造化 vs 単一のテキストブロック）

### 出力例

処理後、スクリプトは要約統計を表示します：

```
✓ 評価完了！
✓ 結果を output/format_clarity_output.csv に書き込みました
✓ 50行を処理しました

📊 平均フォーマット明瞭性スコア： 3.84/5.0
📊 スコア分布：
5    12
4    21
3    15
2     2
1     0
```

-----

## データ収集スクリプト

### `collect_responses.py` - LLM応答収集スクリプト

評価用のデータを収集するためのスクリプトです。2つのモデルにAPI経由で質問を送信し、応答を収集してCSVファイルに保存します。評価するモデルは任意に指定できます（例: Claude 3.5 SonnetとClaude 4.5 Haiku）。

**主な機能：**
- APIから応答を収集（2つのモデルに同じ質問を送信）
- ログの整形・フォーマット（ReActログ形式に変換）
- CSVファイルの生成（評価スクリプト用の形式）
- 処理時間ログの自動記録と比較チャートの自動生成

**使用方法：**

```bash
# 基本的な使用方法
python scripts/collect_responses.py questions.txt -o responses.csv

# カスタムAPI URLを指定
python scripts/collect_responses.py questions.txt --api-url http://localhost:8080/api/v1/urls

# カスタムモデルを指定
python scripts/collect_responses.py questions.txt --model-a claude3.5-sonnet --model-b claude4.5-haiku

# カスタムidentity、timeout、delayを指定（デフォルト値は設定ファイルまたは環境変数から読み込まれます）
python scripts/collect_responses.py questions.txt --identity YOUR_IDENTITY --timeout 150 --delay 2.5

# カスタム処理時間ログファイルを指定
python scripts/collect_responses.py questions.txt --time-log custom_time_log.txt
```

**設定値の外部化：**

`identity`、`timeout`、`delay`のデフォルト値は、環境変数（`APP_DEFAULT_IDENTITY`、`APP_TIMEOUT`、`APP_API_DELAY`）または設定ファイルから読み込まれます。コマンドライン引数で明示的に指定した場合は、それが優先されます。

処理時間ログの出力先は `APP_OUTPUT_FILE_PROCESSING_TIME_LOG` または `--time-log` オプションで変更できます。

**入力ファイル形式：**
- **テキストファイル（.txt）**: 1行に1つの質問。`#`で始まる行はコメントとして無視されます
- **CSVファイル（.csv）**: 最初の列に質問（ヘッダー行は任意）

**出力：**
- `output/collected_responses.csv`（デフォルト）: `Question`, `Model_A_Response`, `Model_B_Response`の列を持つCSV
- `output/processing_time_log.txt`: 各API呼び出しの処理時間ログ（デフォルト）
- `output/processing_time_comparison.png`, `output/processing_time_statistics.png`, `output/processing_time_summary.txt`: 処理時間比較チャートと統計サマリー

**注意：**
- このスクリプトの出力は`llm_judge_evaluator.py`、`ragas_llm_judge_evaluator.py`、`format_clarity_evaluator.py`の入力として使用可能です
- 出力されるログは「## ✅ Final Answer 回答」セクションを含む整形済みReActログ形式です

**使用例：**

```bash
# 1. 質問ファイルを準備
echo "会社の休暇制度について教えてください" > questions.txt

# 2. 応答を収集（処理時間ログとチャートが自動生成されます）
python scripts/collect_responses.py questions.txt -o responses.csv

# 3. 処理時間レポートを確認
ls -la output/processing_time_*.png output/processing_time_summary.txt output/processing_time_log.txt

# 4. 収集したデータを評価
python scripts/llm_judge_evaluator.py responses.csv -n 5
```

-----

## 処理時間比較スクリプト

### `compare_processing_time.py` - 処理時間比較スクリプト

ログファイルから2つのモデルの処理時間を抽出し、比較チャートや統計情報を生成するスクリプトです。`collect_responses.py`を実行すると処理時間ログとレポートが自動生成されますが、既存のログを再分析したい場合やカスタムログを扱いたい場合には本スクリプトを直接実行できます。

**主な機能：**
- ログファイルから処理時間を抽出（正規表現パターンを使用）
- 処理時間比較チャート（バーチャート）
- 統計チャート（平均、分布、トレンド、速度比）
- サマリーテーブル（詳細な統計情報）

**使用方法：**

```bash
# 基本的な使用方法（デフォルトのログファイル名: output/processing_time_log.txt）
python scripts/compare_processing_time.py

# カスタムログファイルを指定
python scripts/compare_processing_time.py output/processing_time_log.txt
```

**設定値の外部化：**

正規表現パターンと出力ファイル名は、環境変数または設定ファイルから読み込まれます：

- `APP_REGEX_MODEL_A_PATTERN`: Model Aの処理時間を抽出する正規表現パターン
- `APP_REGEX_MODEL_B_PATTERN`: Model Bの処理時間を抽出する正規表現パターン
- `APP_OUTPUT_FILE_PROCESSING_TIME_COMPARISON`: 比較チャートの出力ファイル名
- `APP_OUTPUT_FILE_PROCESSING_TIME_STATISTICS`: 統計チャートの出力ファイル名
- `APP_OUTPUT_FILE_PROCESSING_TIME_SUMMARY`: サマリーテーブルの出力ファイル名

**出力ファイル名のカスタマイズ：**

出力ファイル名（`processing_time_*.png`、`processing_time_summary.txt`など）は`src/config/app_config.py`の`output_files`設定、または環境変数（`APP_OUTPUT_FILE_PROCESSING_TIME_COMPARISON`など）で変更できます。詳細は[アプリケーション設定](#アプリケーション設定設定値の外部化)セクションを参照してください。

**入力ファイル形式：**

ログファイルには、以下のような形式で処理時間情報が含まれている必要があります：

```
📥 [claude3.5-sonnet] ... 経過時間: 12.34秒
📥 [claude4.5-haiku] ... 経過時間: 8.90秒
```

**出力ファイル：**

- `output/processing_time_comparison.png`: Model AとModel Bの処理時間比較チャート（バーチャート）
- `output/processing_time_statistics.png`: 統計チャート（平均、分布、トレンド、速度比の4つのサブプロット）
- `output/processing_time_summary.txt`: 詳細な統計サマリーテーブル（各質問の処理時間、差分、速度比、全体統計）

**使用例：**

```bash
# 1. ログファイルを準備（API呼び出しのログなど）
# 2. 処理時間を抽出して比較
python scripts/compare_processing_time.py api_logs.txt

# 3. 生成されたファイルを確認
ls -la output/processing_time_*.png output/processing_time_summary.txt
```

**注意：**
- Model AとModel Bのデータ数が一致しない場合、少ない方に合わせて調整されます
- 正規表現パターンは設定ファイルまたは環境変数でカスタマイズ可能です

-----

## 結果可視化スクリプト

### `visualize_results.py` - 評価結果の可視化スクリプト

すべての評価スクリプト（`llm_judge_evaluator.py`、`ragas_llm_judge_evaluator.py`、`format_clarity_evaluator.py`）の評価結果をグラフやチャートで可視化するスクリプトです。評価スクリプトの種類は自動検出されます。

**主な機能：**
- Model AとModel Bのスコア比較チャート
- スコア分布のヒストグラム
- スコア分布の箱ひげ図
- 統計サマリーテーブル

**使用方法：**

```bash
# デフォルトのoutput/evaluation_output.csvを使用（llm-judge形式）
python scripts/visualize_results.py

# カスタムCSVファイルを指定
python scripts/visualize_results.py my_evaluation_results.csv

# モデル名を指定して可視化（PNGファイルに実際のモデル名が表示されます）
python scripts/visualize_results.py output/evaluation_output.csv --model-a claude4.5-sonnet --model-b claude4.5-haiku

# ragas_evaluation_output.csvを可視化
python scripts/visualize_results.py output/ragas_evaluation_output.csv

# output/format_clarity_output.csvを可視化
python scripts/visualize_results.py output/format_clarity_output.csv
```

**入力CSV形式：**

以下の評価スクリプトの出力CSVに対応しています：

1. **llm-judge** (`llm_judge_evaluator.py`の出力):
   - `Question`
   - `Model_A_Citation_Score`, `Model_B_Citation_Score`
   - `Model_A_Relevance_Score`, `Model_B_Relevance_Score`
   - `Model_A_ReAct_Performance_Thought_Score`, `Model_B_ReAct_Performance_Thought_Score`
   - `Model_A_RAG_Retrieval_Observation_Score`, `Model_B_RAG_Retrieval_Observation_Score`
   - `Model_A_Information_Integration_Score`, `Model_B_Information_Integration_Score`
   - `Evaluation_Error` (オプション)

2. **ragas** (`ragas_llm_judge_evaluator.py`の出力):
   - `Question`
   - `Model_A_faithfulness_score`, `Model_B_faithfulness_score`
   - `Model_A_answer_relevance_score`, `Model_B_answer_relevance_score`
   - `Model_A_context_precision_score`, `Model_B_context_precision_score`
   - `Model_A_context_recall_score`, `Model_B_context_recall_score`
   - `Evaluation_Error` (オプション)

3. **format-clarity** (`format_clarity_evaluator.py`の出力):
   - `Question`
   - `Format_Clarity_Score`
   - `Evaluation_Error` (オプション)

**出力ファイル：**

- `output/evaluation_comparison.png`: Model AとModel Bのスコア比較チャート（バーチャート）
- `output/evaluation_distribution.png`: スコア分布のヒストグラム
- `output/evaluation_boxplot.png`: スコア分布の箱ひげ図
- `output/evaluation_summary.txt`: 統計サマリーテーブル（平均、最小、最大、標準偏差など）
- `output/ragas_evaluation_comparison.png`: Ragasメトリクスのスコア比較チャート（ragas結果を可視化した場合）
- `output/ragas_evaluation_distribution.png`: Ragasメトリクスのスコア分布ヒストグラム
- `output/ragas_evaluation_boxplot.png`: Ragasメトリクスの箱ひげ図
- `output/ragas_evaluation_summary.txt`: Ragasメトリクス用の統計サマリーテーブル

**使用例：**

```bash
# 1. 評価を実行
python scripts/llm_judge_evaluator.py responses.csv -o output/evaluation_output.csv

# 2. 結果を可視化（モデル名を指定するとPNGファイルに実際のモデル名が表示されます）
python scripts/visualize_results.py output/evaluation_output.csv --model-a claude4.5-sonnet --model-b claude4.5-haiku

# または、モデル名を指定しない場合（デフォルトで「Model A」と「Model B」と表示）
python scripts/visualize_results.py output/evaluation_output.csv

# 3. 生成されたファイルを確認
ls -la output/evaluation_*.png output/evaluation_summary.txt
```

**注意：**
- **エラー行のフィルタリング**: `Evaluation_Error`列に非空のエラーメッセージがある行は自動的に除外されます。`Evaluation_Error`が空文字列（`""`）または`NaN`の行は正常行として扱われ、可視化に含まれます。
- 日本語フォントが正しく表示されない場合は、システムの日本語フォント設定を確認してください
- `--model-a`と`--model-b`オプションでモデル名を指定すると、PNGファイルのタイトル、凡例、サマリーテーブルに実際のモデル名が表示されます。指定しない場合は「Model A」と「Model B」と表示されます
- **出力ファイル名のカスタマイズ**: 出力ファイル名（`evaluation_*.png`、`evaluation_summary.txt`など）は`src/config/app_config.py`の`output_files`設定、または環境変数（`APP_OUTPUT_FILE_EVALUATION_COMPARISON`など）で変更できます。詳細は[アプリケーション設定](#アプリケーション設定設定値の外部化)セクションを参照してください。

-----

## 一気通貫パイプライン

### `run_full_pipeline.py` - 一気通貫パイプラインスクリプト

モデルのAPIからの情報収集、評価、可視化を一気通貫で実行する統合スクリプトです。

**主な機能：**
- `collect_responses.py`を実行してAPIから応答を収集
- 評価スクリプトを実行（`llm_judge_evaluator.py`、`ragas_llm_judge_evaluator.py`、`format_clarity_evaluator.py`から選択）
- `visualize_results.py`を実行して結果を可視化
- 各ステップの成功/失敗を追跡し、エラー時は適切に処理

**使用方法：**

```bash
# デフォルトでllm-judge評価を使用してパイプラインを実行
python scripts/run_full_pipeline.py questions.txt

# ragas評価を使用
python scripts/run_full_pipeline.py questions.txt --evaluator ragas

# 全ての評価スクリプトを実行
python scripts/run_full_pipeline.py questions.txt --evaluator all

# ragas評価でメトリクスを指定
python scripts/run_full_pipeline.py questions.txt --evaluator ragas --ragas-metrics faithfulness context_precision

# ragas評価でプリセットを指定（basic / extended）
python scripts/run_full_pipeline.py questions.txt --evaluator ragas --ragas-metrics-preset basic

# 収集ステップをスキップ（既存のCSVファイルを使用）
python scripts/run_full_pipeline.py questions.txt --skip-collect

# 可視化ステップをスキップ
python scripts/run_full_pipeline.py questions.txt --skip-visualize

# カスタムモデルとAPI URLを指定
python scripts/run_full_pipeline.py questions.txt --model-a claude4.5-sonnet --model-b claude4.5-haiku --api-url http://localhost:8080/api/v2/questions

# 評価行数を制限
python scripts/run_full_pipeline.py questions.txt --limit 5

# 評価用のモデルを指定（評価スクリプトの--modelオプションに渡される）
python scripts/run_full_pipeline.py questions.txt --judge-model gpt-5
```

**コマンドライン引数：**

- `questions_file`: 質問ファイル（必須、`.txt`または`.csv`形式）
- `--evaluator`: 評価スクリプトの選択
  - `llm-judge`: `llm_judge_evaluator.py`を実行（デフォルト）
  - `ragas`: `ragas_llm_judge_evaluator.py`を実行
  - `format-clarity`: `format_clarity_evaluator.py`を実行
  - `all`: 全ての評価スクリプトを実行
- `--model-a`, `--model-b`: モデル名（`collect_responses.py`に渡される）
- `--api-url`: API URL（`collect_responses.py`に渡される）
- `--limit`: 評価行数の制限（評価スクリプトに渡される）
- `--judge-model`: 評価用のモデル名（評価スクリプトの`--model`オプションに渡される）
- `--skip-collect`: 収集ステップをスキップ（既存の`output/collected_responses.csv`を使用）
- `--skip-visualize`: 可視化ステップをスキップ
- `--collect-output`: 収集ステップの出力ファイル名（デフォルト: `output/collected_responses.csv`）

**実行フロー：**

1. **Step 1: 応答収集**（`--skip-collect`が指定されていない場合）
   - `collect_responses.py`を実行
   - 出力: `output/collected_responses.csv`（デフォルト）

2. **Step 2: 評価実行**
   - 選択された評価スクリプトを実行
   - 出力: `output/evaluation_output.csv`（llm-judge）、`output/ragas_evaluation_output.csv`（ragas）、`output/format_clarity_output.csv`（format-clarity）

3. **Step 3: 結果可視化**（`--skip-visualize`が指定されていない場合）
   - `visualize_results.py`を実行
   - すべての評価スクリプト（llm-judge、ragas、format-clarity）の出力に対応
   - 評価スクリプトの種類は自動検出されます
   - 出力: `evaluation_comparison.png`、`evaluation_distribution.png`、`evaluation_boxplot.png`、`evaluation_summary.txt`

**使用例：**

```bash
# 1. 基本的な使用方法（llm-judge評価）
python scripts/run_full_pipeline.py questions.txt

# 2. ragas評価を使用
python scripts/run_full_pipeline.py questions.txt --evaluator ragas

# 3. 全ての評価スクリプトを実行
python scripts/run_full_pipeline.py questions.txt --evaluator all

# 4. 既存のCSVファイルを使用して評価と可視化のみ実行
python scripts/run_full_pipeline.py questions.txt --skip-collect

# 5. 評価用モデルを指定して実行
python scripts/run_full_pipeline.py questions.txt --judge-model gpt-5

# 6. Makefileを使用
make pipeline ARGS="questions.txt --evaluator llm-judge --judge-model gpt-5"
```

**注意：**
- 可視化は現在`llm-judge`評価の結果のみサポートされています
- 各ステップでエラーが発生した場合、パイプラインは適切に停止します
- 可視化ステップでエラーが発生した場合、警告を表示してパイプラインは続行します
- **非対話実行**: `run_full_pipeline.py`から実行する場合、評価スクリプト（`llm_judge_evaluator.py`、`format_clarity_evaluator.py`）には自動的に`--yes`フラグが付与され、10行を超える場合でも確認プロンプトが表示されずに実行されます。これにより、CI/バッチ環境での自動実行が可能です。個別に評価スクリプトを実行する場合は、`--yes`フラグを明示的に指定することで非対話実行できます。
- **出力ファイル名のカスタマイズ**: 評価結果や処理時間レポートの出力ファイル名（`evaluation_*.png`、`processing_time_*.png`など）は`src/config/app_config.py`の`output_files`設定、または環境変数（`APP_OUTPUT_FILE_EVALUATION_COMPARISON`、`APP_OUTPUT_FILE_PROCESSING_TIME_COMPARISON`など）で変更できます。詳細は[アプリケーション設定](#アプリケーション設定設定値の外部化)セクションを参照してください。

-----

## エラーハンドリング

すべてのスクリプトには、堅牢なエラーハンドリングが含まれています：

  - **APIエラー**：指数関数的バックオフによる自動リトライ（デフォルト: 最大3回、`APP_MAX_RETRIES`で変更可能）
  - **JSON解析エラー**：応答の形式が不正な場合に検証し、リトライします
  - **APIキーの欠落**：セットアップ手順を含む明確なエラーメッセージ
  - **ファイルが見つからない**：入力ファイルが見つからない場合のグレースフルなエラーハンドリング

すべてのエラーは、出力CSVのそれぞれのエラー列に記録されます。

**リトライ設定のカスタマイズ：**

リトライ設定は環境変数または設定ファイルで変更できます：
- `APP_MAX_RETRIES`: 最大リトライ回数（デフォルト: 3）
- `APP_RETRY_DELAY`: リトライ間隔（秒、デフォルト: 2）

詳細は[アプリケーション設定](#アプリケーション設定設定値の外部化)セクションを参照してください。

-----

## パフォーマンスに関する考慮事項

  - APIコールはリトライロジックと共に順次実行されます
  - 進捗はtqdmプログレスバーを介して表示されます
  - Temperatureは一貫した評価のために最適化されています
  - Max tokensは各エバリュエーターに適切に設定されています
