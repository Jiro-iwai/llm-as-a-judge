# LLM-as-a-Judge 評価スクリプト

LLM-as-a-JudgeとRagasフレームワークを使用した、CyChat SDの応答を自動評価するためのPythonスクリプトです。

## 概要

このリポジトリには、**3つの異なる評価ツール**が含まれています：

1.  **`llm_judge_evaluator.py`**: 詳細な5メトリックのルーブリックを持つカスタムLLM-as-a-Judge
2.  **`ragas_llm_judge_evaluator.py`**: Ragasフレームワークベースの評価
3.  **`format_clarity_evaluator.py`**: Claude 3.5とClaude 4.5 Sonnetの応答のフォーマット/スタイルの類似性を比較

すべてのスクリプトは、LLM-as-a-Judgeとして**Azure OpenAI (GPT-5, GPT-4.1)とStandard OpenAI (GPT-4 Turbo)**をサポートしています。

## 目次

  - [インストール](#インストール)
  - [Makefileの使い方](#makefileの使い方)
  - [プログラム別のAPI要件](#プログラム別のapi要件)
  - [Custom LLM Judge Evaluator](#custom-llm-judge-evaluator)
  - [Ragas-Based Evaluator](#ragas-based-evaluator)
  - [Format Clarity Evaluator](#format-clarity-evaluator)
  - [データ収集スクリプト](#データ収集スクリプト)
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
python llm_judge_evaluator.py my_test_data.csv -n 5
```

**基本的な使用方法：**

```bash
python llm_judge_evaluator.py my_test_data.csv
```

**モデルを指定して実行：**

```bash
# GPT-5を使用
python llm_judge_evaluator.py my_test_data.csv -m gpt-5

# GPT-4.1を使用（デフォルト）
python llm_judge_evaluator.py my_test_data.csv -m gpt-4.1

# GPT-4 Turboを使用
python llm_judge_evaluator.py my_test_data.csv -m gpt-4-turbo
```

**カスタム出力ファイルを指定：**

```bash
python llm_judge_evaluator.py my_test_data.csv -o my_results.csv
```

**最初のN行のみを処理（コスト管理）：**

```bash
# 最初の10行のみを処理
python llm_judge_evaluator.py my_test_data.csv -n 10

# 最初の50行をカスタム出力で処理
python llm_judge_evaluator.py my_test_data.csv -n 50 -o test_results.csv

# モデルを指定して最初の10行を処理
python llm_judge_evaluator.py my_test_data.csv -n 10 -m gpt-5
```

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

出力ファイル（デフォルトは `evaluation_output.csv`）には以下が含まれます：

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

#### 期待されるReActログの構造（フォーマット済み - `log-output-simplifier.py` を使用）

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
python ragas_llm_judge_evaluator.py test_5_rows.csv
```

**モデルを指定して実行：**

```bash
# GPT-4.1を使用（デフォルト、推奨）
python ragas_llm_judge_evaluator.py my_data.csv -m gpt-4.1

# GPT-5を使用（非推奨：Temperature制御の問題）
python ragas_llm_judge_evaluator.py my_data.csv -m gpt-5
```

**カスタム出力ファイルを指定：**

```bash
python ragas_llm_judge_evaluator.py my_data.csv -o ragas_results.csv
```

**最初のN行でテスト：**

```bash
# 最初に3行だけでテスト
python ragas_llm_judge_evaluator.py my_data.csv -n 3

# モデルを指定して最初の3行をテスト
python ragas_llm_judge_evaluator.py my_data.csv -n 3 -m gpt-4.1
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

出力ファイル（デフォルトは `ragas_evaluation_output.csv`）には以下が含まれます：

  - **元の列**：Question, Model\_A\_Response, Model\_B\_Response
  - **解析された列**：
      - `model_A_answer`: モデルAから抽出された最終回答
      - `model_A_contexts`: モデルAのログから抽出されたコンテキストのリスト
      - `model_B_answer`: モデルBから抽出された最終回答
      - `model_B_contexts`: モデルBのログから抽出されたコンテキストのリスト
  - **モデルAのRagasスコア**：
      - `Model_A_faithfulness_score`: コンテキストとの事実上の一貫性 (0-1)
  - **モデルBのRagasスコア**：Model\_Bプレフィックス付きの同じメトリクス
      - `Model_A_faithfulness_score`: コンテキストとの事実上の一貫性 (0-1)

### Ragasメトリクスの説明

#### [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) (0-1)

回答が、与えられたコンテキストと事実上整合しているかを測定します。スコアが高いほど、回答がハルシネーション（幻覚）を起こさず、検索された情報に忠実であることを示します。

-----

## Format Clarity Evaluator

`format_clarity_evaluator.py` スクリプトは、Claude 4.5 Sonnetの応答フォーマットが（ゴールデンスタンダードとして使用される）Claude 3.5 Sonnetのフォーマットにどれだけ近いかを評価するための専用ツールです。

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
2.  **Model_A_Response**: モデルAの完全な応答（Claude 3.5 Sonnetの完全なReActログ）
3.  **Model_B_Response**: モデルBの完全な応答（Claude 4.5 Sonnetの完全なReActログ）

**注意**: ヘッダー行は任意です。スクリプトは自動的にヘッダー行を検出し、ヘッダー行がない場合は最初の行をデータとして扱います。カラム名は`Model_A_Response`/`Model_B_Response`または`Claude_35_Raw_Log`/`Claude_45_Raw_Log`のいずれもサポートされます。

### 使用方法

**基本的な使用方法：**

```bash
python format_clarity_evaluator.py input.csv
```

**モデルを指定して実行：**

```bash
# GPT-5を使用
python format_clarity_evaluator.py input.csv -m gpt-5

# GPT-4-turboを使用（デフォルト）
python format_clarity_evaluator.py input.csv -m gpt-4-turbo

# GPT-4.1を使用
python format_clarity_evaluator.py input.csv -m gpt-4.1
```

**行数を制限してテスト：**

```bash
# 最初の5行のみを処理
python format_clarity_evaluator.py input.csv -n 5

# 最初の10行をカスタム出力で処理
python format_clarity_evaluator.py input.csv -n 10 -o test_results.csv

# モデルを指定して最初の5行を処理
python format_clarity_evaluator.py input.csv -n 5 -m gpt-5
```

**カスタム出力ファイルを指定：**

```bash
python format_clarity_evaluator.py input.csv -o my_format_results.csv
```

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

出力ファイル（デフォルトは `format_clarity_output.csv`）には以下が含まれます：

| 列 | 説明 |
|--------|-------------|
| `Question` | 元の質問 |
| `Claude_3.5_Final_Answer` | Claude 3.5のログから解析された最終回答 |
| `Claude_4.5_Final_Answer` | Claude 4.5のログから解析された最終回答 |
| `Format_Clarity_Score` | 1～5のスコア |
| `Format_Clarity_Justification` | スコアの詳細な説明 |
| `Evaluation_Error` | 評価が失敗した場合のエラーメッセージ |

### 採点ルーブリック

LLMジャッジは詳細な5段階のスケールを使用します：

  - **5 (優秀)**：ほぼ同一のフォーマット。マークダウン、リスト、構造を完璧に反映している
  - **4 (良い)**：ほとんど同様だが、わずかな逸脱がある（例：箇条書き vs 番号付き）
  - **3 (許容)**：いくつかの類似点はあるが、重大な違いもある（例：リスト vs 段落）
  - **2 (悪い)**：ほとんどが異なり、3.5モデルの構造に似ていない
  - **1 (非常に悪い)**：完全に異なるフォーマット（例：構造化 vs 単一のテキストブロック）

### 出力例

処理後、スクリプトは要約統計を表示します：

```
✓ 評価完了！
✓ 結果を format_clarity_output.csv に書き込みました
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

評価用のデータを収集するためのスクリプトです。2つのモデル（Claude 3.5 SonnetとClaude 4.5 Haiku）にAPI経由で質問を送信し、応答を収集してCSVファイルに保存します。

**主な機能：**
- APIから応答を収集（2つのモデルに同じ質問を送信）
- ログの整形・フォーマット（ReActログ形式に変換）
- CSVファイルの生成（評価スクリプト用の形式）

**使用方法：**

```bash
# 基本的な使用方法
python collect_responses.py questions.txt -o responses.csv

# カスタムAPI URLを指定
python collect_responses.py questions.txt --api-url http://localhost:8080/api/v1/urls

# カスタムモデルを指定
python collect_responses.py questions.txt --model-a claude3.5-sonnet --model-b claude4.5-haiku

# カスタムidentityを指定
python collect_responses.py questions.txt --identity YOUR_IDENTITY
```

**入力ファイル形式：**
- **テキストファイル（.txt）**: 1行に1つの質問。`#`で始まる行はコメントとして無視されます
- **CSVファイル（.csv）**: 最初の列に質問（ヘッダー行は任意）

**出力：**
- `collected_responses.csv`（デフォルト）: `Question`, `Model_A_Response`, `Model_B_Response`の列を持つCSV

**注意：**
- このスクリプトの出力は`llm_judge_evaluator.py`、`ragas_llm_judge_evaluator.py`、`format_clarity_evaluator.py`の入力として使用可能です
- 出力されるログは「## ✅ Final Answer 回答」セクションを含む整形済みReActログ形式です

**使用例：**

```bash
# 1. 質問ファイルを準備
echo "会社の休暇制度について教えてください" > questions.txt

# 2. 応答を収集
python collect_responses.py questions.txt -o responses.csv

# 3. 収集したデータを評価
python llm_judge_evaluator.py responses.csv -n 5
```

-----

## エラーハンドリング

すべてのスクリプトには、堅牢なエラーハンドリングが含まれています：

  - **APIエラー**：指数関数的バックオフによる自動リトライ（最大3回）
  - **JSON解析エラー**：応答の形式が不正な場合に検証し、リトライします
  - **APIキーの欠落**：セットアップ手順を含む明確なエラーメッセージ
  - **ファイルが見つからない**：入力ファイルが見つからない場合のグレースフルなエラーハンドリング

すべてのエラーは、出力CSVのそれぞれのエラー列に記録されます。

-----

## パフォーマンスに関する考慮事項

  - APIコールはリトライロジックと共に順次実行されます
  - 進捗はtqdmプログレスバーを介して表示されます
  - Temperatureは一貫した評価のために最適化されています
  - Max tokensは各エバリュエーターに適切に設定されています
