# サンプルファイル

このディレクトリには、各評価スクリプトで使用できるサンプル入力ファイルが含まれています。

## ファイル一覧

### sample_input_llm_judge.csv

LLM-as-a-Judge評価用のサンプルファイルです。

**使用方法:**
```bash
python scripts/llm_judge_evaluator.py examples/sample_input_llm_judge.csv
```

**データ形式:**
- `Question`: 評価対象の質問
- `Model_A_Response`: モデルAの回答
- `Model_B_Response`: モデルBの回答

**注意**: ヘッダー行は任意です。スクリプトは自動的にヘッダー行を検出します。

### sample_input_format_clarity.csv

フォーマット明確性評価用のサンプルファイルです。

**使用方法:**
```bash
python scripts/format_clarity_evaluator.py examples/sample_input_format_clarity.csv
```

**データ形式:**
- `Question`: 評価対象の質問
- `Model_A_Response`: モデルAのReActログ（フルログ）
- `Model_B_Response`: モデルBのReActログ（フルログ）

**注意**: ヘッダー行は任意です。スクリプトは自動的にヘッダー行を検出します。

### sample_input_ragas.csv

Ragas評価用のサンプルファイルです。

**使用方法:**
```bash
python scripts/ragas_llm_judge_evaluator.py examples/sample_input_ragas.csv
```

**データ形式:**
- `Question`: 評価対象の質問
- `Model_A_Response`: モデルAのReActログ
- `Model_B_Response`: モデルBのReActログ

**注意**: 
- ヘッダー行は任意です。スクリプトは自動的にヘッダー行を検出します。
- Ragas評価では、ReActログから以下が抽出されます：
  - Final Answer（回答）
  - Contexts（検索結果とLLM思考プロセス）

## カスタムデータの作成

自分のデータを評価する場合は、これらのサンプルファイルを参考にして、同じ形式のCSVファイルを作成してください。

### 必須列

- すべての評価スクリプトで`Question`列は必須です
- 各評価スクリプトに応じて、以下の列が必要です：
  - **LLM Judge Evaluator**: `Model_A_Response`, `Model_B_Response`
  - **Format Clarity Evaluator**: `Model_A_Response`, `Model_B_Response`（ReActログ形式）
  - **Ragas Evaluator**: `Model_A_Response`, `Model_B_Response`（ReActログ形式）

### ヘッダー行

ヘッダー行はオプションです。ヘッダー行がない場合、スクリプトは自動的に最初の行がヘッダーかどうかを判定します。

### データ形式の詳細

各評価スクリプトの詳細な入力形式については、メインの[README.md](../README.md)を参照してください。

