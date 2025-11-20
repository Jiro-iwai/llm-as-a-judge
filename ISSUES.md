# コードベース調査結果 - 改善提案

このドキュメントは、コードベースを徹底的に調査した結果、発見された課題と改善提案をまとめたものです。
各課題は1つのissueとして扱うことを想定しています。

---

## Issue 1: ロギングシステムの統一: print()とsys.stderrの混在を解消

### 問題

現在、コードベース全体でロギング方法が統一されていません：
- `print()`を直接使用している箇所
- `sys.stderr`への出力を使用している箇所
- `llm_judge_evaluator.py`には`log_info()`, `log_error()`, `log_warning()`などのヘルパー関数があるが、他のスクリプトでは使用されていない

### 影響

- ログレベルの制御ができない
- ログの出力先を変更できない
- デバッグ時にログをフィルタリングできない
- コードの一貫性が損なわれる

### 提案

Python標準の`logging`モジュールを使用して、統一されたロギングシステムを実装する。

#### 実装方針

1. 共通のロギング設定モジュールを作成（例: `utils/logging_config.py`）
2. すべてのスクリプトでこのモジュールを使用
3. 環境変数やコマンドライン引数でログレベルを制御可能にする
4. 既存の`log_info()`, `log_error()`などの関数を`logging`モジュールのラッパーに置き換え

#### 対象ファイル

- `llm_judge_evaluator.py`
- `format_clarity_evaluator.py`
- `ragas_llm_judge_evaluator.py`
- `collect_responses.py`
- `compare_processing_time.py`
- `visualize_results.py`

### 期待される効果

- ログレベルの制御が可能になる
- ログの出力先を変更可能になる（ファイル、コンソールなど）
- デバッグ時のログフィルタリングが容易になる
- コードの一貫性が向上する

---

## Issue 2: 設定値の外部化: ハードコードされた値を設定ファイルや環境変数に移行

### 問題

現在、多くの設定値がコード内にハードコードされています：

1. **タイムアウト値**: `timeout=120`が複数箇所にハードコード
2. **リトライ設定**: `max_retries=3`, `retry_delay=2`がハードコード
3. **API遅延**: `delay=1.0`が`collect_responses.py`にハードコード
4. **デフォルトidentity**: `identity="A14804"`が`collect_responses.py`にハードコード
5. **正規表現パターン**: `compare_processing_time.py`にモデル名がハードコード
6. **出力ファイル名**: `visualize_results.py`に出力ファイル名がハードコード

### 影響

- 設定を変更するためにコードを編集する必要がある
- 環境ごとに異なる設定を使いにくい
- テスト時に設定を変更しにくい

### 提案

設定値を外部化する方法を検討する：

#### オプション1: 設定ファイル（config.yamlまたはconfig.json）
- プロジェクトルートに設定ファイルを作成
- 環境ごとに異なる設定ファイルを用意可能

#### オプション2: 環境変数
- 既存の`.env`ファイルを拡張
- デフォルト値をコードに保持

#### オプション3: コマンドライン引数
- 主要な設定値をコマンドライン引数で指定可能にする

#### 推奨アプローチ

設定ファイル（YAML）を基本とし、環境変数で上書き可能にする。

### 対象箇所

- `collect_responses.py`: `identity`, `timeout`, `delay`
- `compare_processing_time.py`: 正規表現パターン
- `visualize_results.py`: 出力ファイル名
- すべてのスクリプト: `max_retries`, `retry_delay`, `timeout`

### 期待される効果

- 設定の変更が容易になる
- 環境ごとの設定管理が可能になる
- テスト時の設定変更が容易になる

---

## Issue 3: MODEL_CONFIGSの重複定義を解消: 共通設定モジュールの作成

### 問題

`MODEL_CONFIGS`が複数のファイルに重複定義されています：

1. `llm_judge_evaluator.py`: 詳細な設定（max_total_tokens, safety_margin等）
2. `format_clarity_evaluator.py`: 簡易設定（max_tokens, temperature等）
3. `ragas_llm_judge_evaluator.py`: 簡易設定（max_tokens, temperature等）

各ファイルで設定値が異なり、保守性が低下しています。

### 影響

- 設定値の変更時に複数ファイルを編集する必要がある
- 設定値の不整合が発生する可能性がある
- 新しいモデルを追加する際に複数箇所を更新する必要がある

### 提案

共通の設定モジュールを作成する：

#### 実装方針

1. `config/model_configs.py`を作成
2. すべてのモデル設定を一元管理
3. 各スクリプトでこのモジュールをインポート
4. スクリプトごとに必要な設定のみを取得する関数を提供

#### 構造案

```python
# config/model_configs.py
MODEL_CONFIGS = {
    "gpt-5": {
        "max_total_tokens": 128000,
        "max_tokens": 2000,
        "max_completion_tokens": 2000,
        "min_output_tokens": 800,
        "max_output_tokens": 4000,
        "safety_margin": 2000,
        "temperature": 1.0,
        "use_max_completion_tokens": True,
        "timeout": 120,
    },
    # ...
}

def get_model_config(model_name: str, config_type: str = "full") -> Dict[str, Any]:
    """config_type: 'full', 'simple', 'ragas'"""
    # ...
```

### 期待される効果

- 設定値の一元管理が可能になる
- 設定値の不整合を防げる
- 新しいモデルの追加が容易になる
- コードの重複が削減される

---

## Issue 4: compare_processing_time.pyの正規表現パターンを設定可能にする

### 問題

`compare_processing_time.py`で、モデル名が正規表現パターンにハードコードされています：

```python
pattern_a = r"📥 \[claude3\.5-sonnet\].*?経過時間: ([\d.]+)秒"
pattern_b = r"📥 \[claude4\.5-haiku\].*?経過時間: ([\d.]+)秒"
```

### 影響

- 異なるモデル名を使用する場合にコードを編集する必要がある
- ログフォーマットが変更された場合に対応できない
- テスト時に異なるパターンをテストしにくい

### 提案

1. コマンドライン引数でモデル名を指定可能にする
2. 設定ファイルまたは環境変数でパターンをカスタマイズ可能にする
3. より柔軟な正規表現パターンを生成する関数を作成

#### 実装案

```python
def create_pattern(model_name: str) -> str:
    """モデル名から正規表現パターンを生成"""
    escaped_name = re.escape(model_name)
    return rf"📥 \[{escaped_name}\].*?経過時間: ([\d.]+)秒"
```

### 期待される効果

- 異なるモデル名に対応可能になる
- ログフォーマットの変更に対応しやすくなる
- コードの再利用性が向上する

---

## Issue 5: visualize_results.pyの出力ファイル名を設定可能にする

### 問題

`visualize_results.py`で、出力ファイル名がハードコードされています：

```python
create_score_comparison_chart(df_clean, "evaluation_comparison.png")
create_score_distribution_chart(df_clean, "evaluation_distribution.png")
create_boxplot_chart(df_clean, "evaluation_boxplot.png")
create_summary_table(df_clean, "evaluation_summary.txt")
```

### 影響

- 出力先ディレクトリを変更できない
- ファイル名をカスタマイズできない
- 複数の可視化結果を同時に保持しにくい

### 提案

1. コマンドライン引数で出力ディレクトリとファイル名を指定可能にする
2. デフォルト値は現在の動作を維持
3. 出力ディレクトリが存在しない場合は自動作成

#### 実装案

```python
parser.add_argument(
    "--output-dir",
    default=".",
    help="出力ディレクトリ（デフォルト: カレントディレクトリ）"
)
parser.add_argument(
    "--prefix",
    default="evaluation",
    help="出力ファイル名のプレフィックス（デフォルト: evaluation）"
)
```

### 期待される効果

- 出力先を柔軟に指定可能になる
- 複数の可視化結果を管理しやすくなる
- ユーザビリティが向上する

---

## Issue 6: collect_responses.pyのデフォルトidentityを環境変数から取得する

### 問題

`collect_responses.py`で、デフォルトの`identity`がハードコードされています：

```python
def call_api(
    question: str,
    api_url: str,
    model_name: str,
    identity: str = "A14804",  # ハードコード
    ...
)
```

### 影響

- 異なるユーザーが使用する際にコードを編集する必要がある
- 環境ごとに異なるidentityを使いにくい
- セキュリティ上の懸念（identityがコードに含まれる）

### 提案

環境変数から`identity`を取得し、デフォルト値として使用する：

```python
identity: str = os.getenv("API_IDENTITY", "A14804")
```

または、コマンドライン引数で必須にする：

```python
parser.add_argument(
    "--identity",
    required=True,
    help="API identity (環境変数API_IDENTITYからも取得可能)"
)
```

### 期待される効果

- コードを編集せずにidentityを変更可能になる
- 環境ごとの設定管理が容易になる
- セキュリティが向上する（identityがコードに含まれない）

---

## Issue 7: エラーハンドリングの改善: より具体的な例外処理

### 問題

一部の箇所で`except Exception:`が使用されており、エラーの種類を区別できません：

- `visualize_results.py`: `except Exception:`でフォント設定をキャッチ
- その他の箇所でも汎用的な例外処理が散在

### 影響

- 予期しないエラーが適切に処理されない可能性がある
- デバッグが困難になる
- エラーメッセージが不十分

### 提案

1. より具体的な例外クラスを使用する
2. 例外の種類に応じた適切な処理を実装する
3. エラーログに詳細な情報を含める

#### 改善例

```python
# Before
except Exception:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

# After
except (OSError, ImportError) as e:
    log_warning(f"日本語フォントの設定に失敗しました: {e}")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
```

### 対象ファイル

- `visualize_results.py`
- `compare_processing_time.py`
- その他の例外処理箇所

### 期待される効果

- エラーの原因を特定しやすくなる
- デバッグが容易になる
- より適切なエラーメッセージを提供できる

---

## Issue 8: 型ヒントの完全性向上: すべての関数に適切な型ヒントを追加

### 問題

一部の関数で型ヒントが不完全または欠落しています：

1. `ragas_llm_judge_evaluator.py`の`get_model_config()`: 戻り値の型が明示されていない
2. `call_judge_model()`などの関数で`client`パラメータの型が`Any`または未指定
3. 一部の関数で戻り値の型が`Optional[...]`として明示されていない

### 影響

- 型チェッカー（pyright）の効果が低下する
- IDEの補完機能が十分に機能しない
- コードの可読性が低下する

### 提案

1. すべての関数に適切な型ヒントを追加
2. `client`パラメータに`Union[OpenAI, AzureOpenAI]`などの適切な型を指定
3. `typing`モジュールを適切に使用

#### 改善例

```python
# Before
def get_model_config(model_name: str):
    # ...

# After
def get_model_config(model_name: str) -> Dict[str, Any]:
    # ...

# Before
def call_judge_model(client, ...):
    # ...

# After
from typing import Union
def call_judge_model(
    client: Union[OpenAI, AzureOpenAI],
    ...
) -> Optional[Dict[str, Any]]:
    # ...
```

### 対象ファイル

- `llm_judge_evaluator.py`
- `format_clarity_evaluator.py`
- `ragas_llm_judge_evaluator.py`
- `collect_responses.py`
- `compare_processing_time.py`
- `visualize_results.py`

### 期待される効果

- 型安全性が向上する
- IDEの補完機能が改善する
- コードの可読性が向上する
- バグの早期発見が可能になる

---

## Issue 9: テストカバレッジの向上: 44%から60%以上を目指す

### 現状

現在のテストカバレッジは約44%です。まだ改善の余地があります。

### 不足しているテストケース

1. **エッジケースのテスト**
   - 空のCSVファイル
   - 不正なCSV形式
   - 非常に長い入力データ
   - 特殊文字を含むデータ

2. **エラーハンドリングのテスト**
   - ネットワークエラー
   - APIレート制限エラー
   - タイムアウトエラー
   - 不正なAPIレスポンス

3. **統合テスト**
   - スクリプト間の連携テスト
   - エンドツーエンドのテスト

4. **パフォーマンステスト**
   - 大量データの処理
   - メモリ使用量の確認

### 提案

1. カバレッジレポートを確認し、未カバーの行を特定
2. エッジケースのテストを追加
3. モックを使用したエラーハンドリングのテストを追加
4. 統合テストの追加を検討

### 目標

- カバレッジを60%以上に向上
- すべての主要な関数をテスト
- エラーハンドリングのテストを充実

### 期待される効果

- バグの早期発見
- リファクタリング時の安全性向上
- コードの品質向上

---

## Issue 10: ドキュメントの充実: docstringとREADMEの改善

### 問題

一部の関数にdocstringが不足している、または不十分な可能性があります。また、README.mdに一部のスクリプトの説明が不足しています。

### 提案

1. すべての公開関数に適切なdocstringを追加
2. パラメータと戻り値の説明を充実させる
3. 使用例を含める
4. README.mdに不足している情報を追加

### 期待される効果

- コードの理解が容易になる
- 新しい開発者のオンボーディングが容易になる
- APIの使用方法が明確になる

---

## まとめ

上記の10個の課題を解決することで、コードベースの品質、保守性、使いやすさが大幅に向上することが期待されます。
各課題は独立して対応可能ですが、一部は関連しているため、優先順位を付けて段階的に実装することを推奨します。

