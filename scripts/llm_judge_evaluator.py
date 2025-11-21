#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Script

This script automates the evaluation of two models' responses using an LLM judge.
It reads questions and responses from a CSV file, sends them to OpenAI's API (standard or Azure)
for evaluation, and writes the scored results to an output CSV file.

Usage:
    python llm_judge_evaluator.py <input_csv_file>

Requirements:
    - For Azure OpenAI: Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and MODEL_NAME
    - For Standard OpenAI: Set OPENAI_API_KEY (and optionally MODEL_NAME)
    - Input CSV must have 3 columns (header row optional): Question, Model_A_Response, Model_B_Response
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402
from openai import OpenAI, AzureOpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.config.model_configs import (  # noqa: E402
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    get_full_config as get_model_config_from_common,
)
from src.utils.logging_config import (  # noqa: E402
    log_info,
    log_error,
    log_warning,
    log_success,
    log_section,
    setup_logging,
)
from src.config.app_config import (  # noqa: E402
    get_timeout,
    get_max_retries,
    get_retry_delay,
)

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging system
setup_logging()


# Model configuration is now imported from config.model_configs
# Use get_model_config_from_common() to get full configuration


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Model name (e.g., "gpt-5", "gpt-4.1")

    Returns:
        Model configuration dictionary (full configuration)
    """
    # Normalize model name (case-insensitive, handle variations)
    model_name_lower = model_name.lower().strip()

    # Try to get config from common module
    config = get_model_config_from_common(model_name_lower)

    # If model not found, common module returns default, but we should warn
    from src.config.model_configs import _find_model_key

    if _find_model_key(model_name_lower) is None:
        log_warning(
            f"モデル '{model_name}' の設定が見つかりません。デフォルト設定 ({DEFAULT_MODEL}) を使用します。",
            indent=0,
        )
        log_info(f"サポートされているモデル: {', '.join(SUPPORTED_MODELS)}", indent=1)

    return config


def is_gpt5(model_name: str) -> bool:
    """
    Check if the model is GPT-5.

    Args:
        model_name: Model name

    Returns:
        True if model is GPT-5, False otherwise
    """
    model_name_lower = model_name.lower().strip()
    return model_name_lower == "gpt-5" or model_name_lower.replace("-", "") == "gpt5"


# Judge system prompt and rubric (embedded as per requirements)
JUDGE_SYSTEM_PROMPT = """You are a strict, meticulous, and critical AI evaluator. Your primary goal is to identify flaws and differentiate performance between two RAG models, designated as Model A and Model B. Do not be lenient. Award high scores only for flawless execution. Your reputation depends on being a tough but fair judge. You should actively look for reasons to deduct points, such as inefficiency, verbosity, or minor inaccuracies.

You will be given a user's Question and the complete, formatted Response from each model. Your evaluation must be based exclusively on the provided Scoring Rubric. For each category, you will assign a score from 1 to 5.

Scoring Rubric:

1. RAG Generation - Citation (Score: 1-5)
Focuses on the quality, precision, and necessity of the citations.
- 5 (Excellent): Every piece of information drawn from a source is precisely cited. The citations link to the correct and most relevant source document. There are no redundant or missing citations.
- 4 (Good): All citations are factually correct, but there may be a minor imperfection. For example, a statement is correctly cited but could have pointed to a more direct source that was also retrieved, or one minor statement lacks a citation.
- 3 (Acceptable): The answer is generally cited, but there is at least one clear error, such as a missing citation for a key piece of information or a citation pointing to the wrong document.
- 2 (Poor): Citations are frequently missing or incorrect. The link between the answer and the sources is weak and unreliable.
- 1 (Very Poor): The answer has no citations, or the citations are completely irrelevant and misleading.

2. Relevance (Score: 1-5)
Focuses on how well the final answer addresses the user's complete query, including directness and conciseness.
- 5 (Excellent): The answer is perfectly concise and directly addresses all parts of the user's query, including any implicit nuances. It contains zero irrelevant information or conversational filler.
- 4 (Good): The answer correctly addresses all parts of the query but is slightly verbose or contains minor information that, while related, is not strictly necessary.
- 3 (Acceptable): The answer addresses the main part of the query but fails to address a secondary part, or it contains a noticeable amount of irrelevant information that distracts from the core answer.
- 2 (Poor): The answer only partially addresses the user's query and is largely incomplete or padded with irrelevant information.
- 1 (Very Poor): The answer completely misses the intent of the user's question.

3. ReAct Performance - Thought (Score: 1-5)
Focuses on the logical quality, efficiency, and strategy of the model's reasoning process.
- 5 (Excellent): The thought process is optimal and efficient. It correctly identifies the problem, formulates the best possible search query or tool use on the first attempt, and follows a direct path to the solution.
- 4 (Good): The logic is correct and effective but slightly inefficient. It may take an extra, slightly redundant step or refine its search query once to get the needed information, but it reaches the correct conclusion without major detours.
- 3 (Acceptable): The logic is mostly correct but contains a noticeable flaw or a suboptimal plan. For instance, it uses a vague search query that returns noisy results before correcting itself, or it misunderstands a part of the problem temporarily.
- 2 (Poor): The reasoning has significant flaws. It struggles to form a coherent plan, makes incorrect assumptions, or repeatedly uses the wrong tool or query.
- 1 (Very Poor): The entire thought process is illogical, unrelated to the question, or gets stuck in a loop of incorrect actions.

4. RAG Retrieval - Observation/観察 (Score: 1-5)
Focuses on the quality and relevance of the retrieved source material.
- 5 (Excellent): Retrieves the minimal and most relevant set of sources needed to answer the question completely. The information is perfectly focused and contains no noise.
- 4 (Good): Retrieves all necessary information but also includes one or two extra sources that are only tangentially relevant, indicating a slightly inefficient retrieval process.
- 3 (Acceptable): Retrieves most of the necessary information but also includes distracting or irrelevant sources that add significant noise to the context.
- 2 (Poor): Fails to retrieve a key source or piece of information that is essential for a complete and accurate answer.
- 1 (Very Poor): Retrieves completely incorrect, irrelevant, or no sources at all.

5. RAG Generation - Information Integration (Score: 1-5)
Focuses on how accurately the model synthesizes the retrieved information into its final answer.
- 5 (Excellent): Perfectly and accurately synthesizes information from the retrieved sources. The final answer is factually flawless and contains no information that wasn't present in the context. If sources conflict, it notes the discrepancy.
- 4 (Good): Synthesizes information correctly for the most part but may misinterpret a minor detail or phrase something awkwardly. The answer is factually correct according to the sources but lacks polish.
- 3 (Acceptable): The answer is mostly based on the sources but contains one clear factual error or introduces a small piece of outside information not supported by the context (a minor hallucination).
- 2 (Poor): The answer struggles to combine information, presenting it as a disjointed list rather than a coherent response, or it contains significant factual errors based on the sources.
- 1 (Very Poor): The final answer is a clear hallucination or completely misrepresents the information found in the retrieved sources.

Important Scoring Instruction: The rubric provides definitions for scores 1, 3, and 5. You may use the intermediate scores of 2 and 4 if you assess a model's performance to fall between these defined levels. For example, a score of 4 can be used if the performance is better than the description for 3 points but does not fully meet the criteria for 5 points. You must provide a brief justification for each score you assign.

Required JSON Output Format:
{
  "model_a_evaluation": {
    "citation_score": { "score": <score>, "justification": "<justification>" },
    "relevance_score": { "score": <score>, "justification": "<justification>" },
    "react_performance_thought_score": { "score": <score>, "justification": "<justification>" },
    "rag_retrieval_observation_score": { "score": <score>, "justification": "<justification>" },
    "information_integration_score": { "score": <score>, "justification": "<justification>" }
  },
  "model_b_evaluation": {
    "citation_score": { "score": <score>, "justification": "<justification>" },
    "relevance_score": { "score": <score>, "justification": "<justification>" },
    "react_performance_thought_score": { "score": <score>, "justification": "<justification>" },
    "rag_retrieval_observation_score": { "score": <score>, "justification": "<justification>" },
    "information_integration_score": { "score": <score>, "justification": "<justification>" }
  }
}"""


def create_user_prompt(
    question: str, model_a_response: str, model_b_response: str
) -> str:
    """
    Create the user prompt that will be sent to the judge model.

    Args:
        question: The original user question
        model_a_response: Response from Model A
        model_b_response: Response from Model B

    Returns:
        Formatted prompt string
    """
    return f"""Please evaluate the following two model responses to the given question.

**Question:**
{question}

**Model A Response:**
{model_a_response}

**Model B Response:**
{model_b_response}

Provide your evaluation as a JSON object following the specified format."""


def call_judge_model(
    client: Union[OpenAI, AzureOpenAI],
    question: str,
    model_a_response: str,
    model_b_response: str,
    model_name: str = DEFAULT_MODEL,
    is_azure: bool = False,
    max_retries: Optional[int] = None,
    retry_delay: Optional[int] = None,
    timeout: Optional[int] = None,  # If None, will use model config default
) -> Optional[Dict[str, Any]]:
    """
    Call the OpenAI API to evaluate the two model responses.

    Args:
        client: OpenAI or AzureOpenAI client instance
        question: The original user question
        model_a_response: Response from Model A
        model_b_response: Response from Model B
        model_name: Model name or Azure deployment name
        is_azure: Whether using Azure OpenAI (affects parameter naming)
        max_retries: Maximum number of retry attempts on failure (defaults to config value)
        retry_delay: Delay in seconds between retries (defaults to config value)
        timeout: Request timeout in seconds (defaults to model config or app config)

    Returns:
        Parsed JSON response from the judge model, or None if all retries fail
    """
    # Use config values if not provided
    if max_retries is None:
        max_retries = get_max_retries()
    if retry_delay is None:
        retry_delay = get_retry_delay()

    user_prompt = create_user_prompt(question, model_a_response, model_b_response)

    # Get model configuration
    model_config = get_model_config(model_name)

    # Use timeout from parameter, model config, or app config
    if timeout is None:
        timeout = model_config.get("timeout", get_timeout())

    response: Any = None
    for attempt in range(max_retries):
        try:
            # Prepare API call parameters
            api_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }

            # 入力トークン数に応じて動的に出力トークン数を調整
            user_prompt_len = len(user_prompt)
            system_prompt_len = len(JUDGE_SYSTEM_PROMPT)
            estimated_input_tokens = (user_prompt_len + system_prompt_len) / 4

            # モデル設定から値を取得
            max_total_tokens = model_config["max_total_tokens"]
            min_output_tokens = model_config["min_output_tokens"]
            max_output_tokens_limit = model_config["max_output_tokens"]
            safety_margin = model_config["safety_margin"]
            temperature = model_config["temperature"]
            use_max_completion_tokens = model_config["use_max_completion_tokens"]

            # 入力が長すぎる場合は事前にエラーを出す
            max_input_tokens = max_total_tokens - min_output_tokens - safety_margin
            if estimated_input_tokens > max_input_tokens:
                error_msg = (
                    f"入力トークン数が長すぎます（約{estimated_input_tokens:.0f}トークン）。"
                    f"最大{max_input_tokens:.0f}トークンまで対応可能です。"
                    f"入力データを短縮するか、分割してください。"
                )
                raise ValueError(error_msg)

            # 入力トークン数に応じて出力トークン数を動的に調整
            # 合計がmax_total_tokensを超えないようにする（安全マージンを含む）
            calculated_max_output_tokens = max(
                min_output_tokens,
                min(
                    max_output_tokens_limit,
                    int(max_total_tokens - estimated_input_tokens - safety_margin),
                ),
            )

            # モデルに応じて適切なパラメータを設定
            if use_max_completion_tokens:
                api_params["max_completion_tokens"] = calculated_max_output_tokens
            else:
                api_params["max_tokens"] = calculated_max_output_tokens

            api_params["temperature"] = temperature

            # Log token information only on first attempt
            if attempt == 0:
                log_info("📊 トークン情報:", indent=1)
                log_info(f"  モデル: {model_name}", indent=2)
                log_info(
                    f"  入力トークン（推定）: 約{estimated_input_tokens:.0f}トークン",
                    indent=2,
                )
                log_info(f"  - User prompt: {user_prompt_len:,}文字", indent=3)
                log_info(f"  - System prompt: {system_prompt_len:,}文字", indent=3)
                log_info(
                    f"  出力トークン（最大）: {calculated_max_output_tokens}トークン",
                    indent=2,
                )
                log_info(
                    f"  合計トークン（最大）: 約{estimated_input_tokens + calculated_max_output_tokens:.0f}トークン（制限: {max_total_tokens:,}トークン）",
                    indent=2,
                )
                log_info(f"⏱️  タイムアウト: {timeout}秒", indent=1)

            # Add timeout to API call with progress indication
            import time as time_module
            import threading

            start_time = time_module.time()

            # Progress indicator
            progress_stop = threading.Event()

            def show_progress():
                while not progress_stop.is_set():
                    elapsed = time_module.time() - start_time
                    if timeout is not None and elapsed < timeout:
                        print(
                            f"  ⏳ API処理中... {elapsed:.0f}秒経過",
                            file=sys.stderr,
                            end="\r",
                        )
                        time_module.sleep(5)  # Update every 5 seconds
                    else:
                        break

            progress_thread = threading.Thread(target=show_progress, daemon=True)
            progress_thread.start()

            try:
                response = client.chat.completions.create(**api_params, timeout=timeout)
                progress_stop.set()
                elapsed_time = time_module.time() - start_time
                if attempt == 0:
                    print("", file=sys.stderr)  # New line after progress
                    log_info(f"✓ API呼び出し成功（{elapsed_time:.1f}秒）", indent=1)
            except TimeoutError:
                progress_stop.set()
                elapsed_time = time_module.time() - start_time
                print("", file=sys.stderr)  # New line after progress
                if timeout is not None:
                    raise TimeoutError(
                        f"API呼び出しがタイムアウトしました（{timeout}秒経過）"
                    )
                raise
            except Exception as api_error:
                # OpenAI APIのエラーは様々な例外タイプを投げる可能性があるため、
                # ここでは一般的なExceptionをキャッチして再スロー
                progress_stop.set()
                elapsed_time = time_module.time() - start_time
                print("", file=sys.stderr)  # New line after progress
                if timeout is not None and elapsed_time >= timeout:
                    raise TimeoutError(
                        f"API呼び出しがタイムアウトしました（{timeout}秒経過）"
                    )
                else:
                    raise api_error

            # Extract the response content (response is guaranteed to be set here)
            if response is None:
                raise ValueError("Response is None after API call")
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Log response details only on first attempt
            if attempt == 0:
                content_length = len(content) if content else 0
                log_info(
                    f"📥 レスポンス受信: {content_length:,}文字, finish_reason={finish_reason}",
                    indent=1,
                )

            # Check if response was truncated
            if finish_reason == "length":
                content_length = len(content) if content else 0

                # If content is empty or very short, this indicates input+output exceeds API limit
                # This should have been caught by the pre-check, but if it happens, don't retry
                if content_length == 0:
                    log_error(
                        "レスポンスが空です（max_tokens制限に達しました）", indent=1
                    )
                    log_warning(
                        "入力+出力の合計がAPIの制限を超えています。リトライしても同じ結果になります。処理をスキップします。",
                        indent=2,
                    )
                    return None  # Don't retry - input+output exceeds limit
                elif content_length < 100:
                    # Very short response - likely incomplete, but may be retryable with lower output tokens
                    log_warning(
                        f"レスポンスが非常に短いです（{content_length}文字）。max_tokens制限に達した可能性があります。",
                        indent=1,
                    )
                    log_warning(
                        "リトライしても同じ結果になる可能性があります。", indent=2
                    )
                    # Continue to retry logic - may succeed with lower output tokens
                else:
                    # Response was truncated but has content - this is acceptable, just warn
                    log_warning(
                        f"レスポンスが途中で切れました（{content_length:,}文字）。max_tokens制限に達しましたが、処理を続行します。",
                        indent=1,
                    )
                    # Continue processing - truncated response may still be usable

            # Debug: Check if content is empty or None
            if not content:
                raise ValueError(
                    f"Empty response from API. Finish reason: {finish_reason}"
                )

            # Parse and validate JSON
            evaluation = json.loads(content)

            # Basic validation of the response structure
            if (
                "model_a_evaluation" not in evaluation
                or "model_b_evaluation" not in evaluation
            ):
                raise ValueError("Response missing required evaluation keys")

            return evaluation

        except json.JSONDecodeError as e:
            log_error(
                f"JSON解析エラー (試行 {attempt + 1}/{max_retries}): {e}", indent=1
            )

            # Debug: Print what we actually received
            content_for_debug: Optional[str] = None
            if response is not None:
                try:
                    content_for_debug = (
                        response.choices[0].message.content
                        if response.choices
                        else None
                    )
                except (AttributeError, IndexError):
                    pass

            if content_for_debug:
                log_info(
                    f"受信したコンテンツ（最初の500文字）: {content_for_debug[:500]}",
                    indent=2,
                )
            else:
                log_info("受信したコンテンツが空または取得できませんでした", indent=2)

            if attempt == max_retries - 1:
                return None
            time.sleep(retry_delay)

        except ValueError as e:
            error_msg = str(e)
            # 入力が長すぎる場合は事前に検出済み - リトライ不要
            if "入力トークン数が長すぎます" in error_msg:
                log_error(error_msg, indent=1)
                return None  # Don't retry - input is too long
            # Empty response due to input being too long - retrying won't help
            elif (
                "Empty or too short response" in error_msg
                and "Content length: 0" in error_msg
            ):
                log_error(error_msg, indent=1)
                log_warning(
                    "入力が長すぎるため、リトライしても同じ結果になります。処理をスキップします。",
                    indent=2,
                )
                return None  # Don't retry - it won't help
            else:
                # Other ValueError - may be retryable
                log_error(
                    f"ValueError (試行 {attempt + 1}/{max_retries}): {e}", indent=1
                )
                if attempt == max_retries - 1:
                    return None
                time.sleep(retry_delay)

        except TimeoutError as e:
            log_error(
                f"タイムアウトエラー (試行 {attempt + 1}/{max_retries}): {e}", indent=1
            )
            if attempt == max_retries - 1:
                return None
            time.sleep(retry_delay)

        except Exception as e:
            error_msg = str(e)
            # APIがmax_tokensのエラーを返した場合（入力+出力の合計が制限を超えている）
            if (
                "max_tokens" in error_msg
                or "max_completion_tokens" in error_msg
                or "output limit" in error_msg.lower()
            ):
                log_error(
                    f"APIエラー (試行 {attempt + 1}/{max_retries}): {error_msg}",
                    indent=1,
                )
                log_warning(
                    "入力+出力の合計がAPIの制限を超えています。リトライしても同じ結果になります。処理をスキップします。",
                    indent=2,
                )
                return None  # Don't retry - input+output exceeds limit
            else:
                # その他のAPIエラー（接続エラー、認証エラーなど）は再試行
                log_error(
                    f"APIエラー (試行 {attempt + 1}/{max_retries}): {type(e).__name__}: {e}",
                    indent=1,
                )
                if attempt == max_retries - 1:
                    return None
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

    return None


def extract_scores_from_evaluation(
    evaluation: Dict[str, Any], model_key: str
) -> Dict[str, Any]:
    """
    Extract scores and justifications from the evaluation JSON for a specific model.

    Args:
        evaluation: The full evaluation JSON object
        model_key: Either "model_a_evaluation" or "model_b_evaluation"

    Returns:
        Dictionary containing all scores and justifications with standardized keys
    """
    model_eval = evaluation.get(model_key, {})

    result = {}

    # Extract citation score
    citation = model_eval.get("citation_score", {})
    result["citation_score"] = citation.get("score", None)
    result["citation_justification"] = citation.get("justification", "")

    # Extract relevance score
    relevance = model_eval.get("relevance_score", {})
    result["relevance_score"] = relevance.get("score", None)
    result["relevance_justification"] = relevance.get("justification", "")

    # Extract react performance thought score
    react = model_eval.get("react_performance_thought_score", {})
    result["react_performance_thought_score"] = react.get("score", None)
    result["react_performance_thought_justification"] = react.get("justification", "")

    # Extract RAG retrieval observation score
    rag_retrieval = model_eval.get("rag_retrieval_observation_score", {})
    result["rag_retrieval_observation_score"] = rag_retrieval.get("score", None)
    result["rag_retrieval_observation_justification"] = rag_retrieval.get(
        "justification", ""
    )

    # Extract information integration score
    info_integration = model_eval.get("information_integration_score", {})
    result["information_integration_score"] = info_integration.get("score", None)
    result["information_integration_justification"] = info_integration.get(
        "justification", ""
    )

    return result


def process_csv(
    input_file: str,
    output_file: str = "evaluation_output.csv",
    limit_rows: Optional[int] = None,
    model_name: Optional[str] = None,
    non_interactive: bool = False,
) -> None:
    """
    Main processing function that reads the input CSV, evaluates each row,
    and writes the results to the output CSV.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (default: evaluation_output.csv)
        limit_rows: Optional limit on number of rows to process (for cost control)
        model_name: Model name for evaluation. If None, uses MODEL_NAME environment variable or default model.
        non_interactive: If True, skips confirmation prompt even for >10 rows. Default is False.
    """
    # Check if using Azure OpenAI or standard OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    # Model name can be set via command line argument, environment variable, or default
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)

    # Validate model name
    if model_name not in SUPPORTED_MODELS:
        # Try to get config (will warn if not found)
        get_model_config(model_name)

    is_azure = bool(azure_endpoint and azure_api_key)

    if is_azure:
        # Initialize Azure OpenAI client
        if azure_endpoint is None:
            raise ValueError("azure_endpoint is required for Azure OpenAI")
        log_section("Azure OpenAI 設定")
        log_info(f"Endpoint: {azure_endpoint}")
        log_info(f"Model/Deployment: {model_name}")
        log_info(f"API Version: {azure_api_version}")
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
    else:
        # Initialize standard OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log_error(
                "Azure OpenAI または Standard OpenAI の認証情報が見つかりません。",
                indent=0,
            )
            print("\nAzure OpenAI を使用する場合:", file=sys.stderr)
            print(
                "  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'",
                file=sys.stderr,
            )
            print("  export AZURE_OPENAI_API_KEY='your-api-key'", file=sys.stderr)
            print(
                "  export MODEL_NAME='gpt-4.1'  # or your deployment name",
                file=sys.stderr,
            )
            print("\nStandard OpenAI を使用する場合:", file=sys.stderr)
            print("  export OPENAI_API_KEY='your-api-key-here'", file=sys.stderr)
            sys.exit(1)
        log_section("Standard OpenAI 設定")
        log_info(f"Model: {model_name}")
        client = OpenAI(api_key=api_key)

    # Read input CSV
    log_section("入力ファイルの読み込み")
    log_info(f"ファイル: {input_file}")
    try:
        # Try to detect if there's a header row by reading first line
        with open(input_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().lower()

        # If first line looks like headers, use it as header
        if any(
            keyword in first_line
            for keyword in ["question", "model", "answer", "response"]
        ):
            log_info("CSVヘッダー行を検出しました")
            df = pd.read_csv(input_file)
            # Rename columns to standard names
            df.columns = ["Question", "Model_A_Response", "Model_B_Response"]
        else:
            log_info("ヘッダー行なし。最初の行をデータとして扱います")
            df = pd.read_csv(
                input_file,
                header=None,
                names=["Question", "Model_A_Response", "Model_B_Response"],
            )
    except FileNotFoundError:
        log_error(f"入力ファイル '{input_file}' が見つかりません。", indent=0)
        sys.exit(1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        log_error(f"入力ファイルの読み込みに失敗しました: {e}", indent=0)
        log_error(
            "ファイル形式が正しいか、文字エンコーディングを確認してください。", indent=0
        )
        sys.exit(1)
    except PermissionError as e:
        log_error(f"ファイルへのアクセス権限がありません: {e}", indent=0)
        sys.exit(1)
    except Exception as e:
        # 予期しないエラーの場合は詳細な情報を記録
        log_error(f"予期しないエラーが発生しました: {type(e).__name__}: {e}", indent=0)
        sys.exit(1)

    log_success(f"{len(df)}行を読み込みました")

    # Apply row limit if specified (for cost control during testing)
    log_section("処理設定")
    if limit_rows is not None and limit_rows < len(df):
        df = df.head(limit_rows)
        log_warning(
            f"テスト用に最初の{limit_rows}行に制限しています（-nフラグで変更可能）",
            indent=0,
        )
        log_warning(f"API呼び出し回数: {limit_rows}回", indent=0)
    else:
        log_warning(f"API呼び出し回数: {len(df)}回（モデル: {model_name}）", indent=0)
        log_warning(
            "APIコストがかかるため、まずは-nフラグで少量から試すことを推奨します",
            indent=0,
        )

        # Prompt for confirmation if processing many rows (unless non-interactive mode)
        if len(df) > 10 and not non_interactive:
            try:
                response = (
                    input(f"\n🤔 {len(df)}回のAPI呼び出しを実行しますか？ [y/N]: ")
                    .strip()
                    .lower()
                )
                if response != "y" and response != "yes":
                    print(
                        "キャンセルしました。少ない行数でテストする場合は -n フラグを使用してください: python llm_judge_evaluator.py input.csv -n 5"
                    )
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\nキャンセルしました。")
                sys.exit(0)

    # Prepare output columns
    output_columns = [
        "Question",
        "Model_A_Response",
        "Model_B_Response",
        # Model A scores and justifications
        "Model_A_Citation_Score",
        "Model_A_Citation_Justification",
        "Model_A_Relevance_Score",
        "Model_A_Relevance_Justification",
        "Model_A_ReAct_Performance_Thought_Score",
        "Model_A_ReAct_Performance_Thought_Justification",
        "Model_A_RAG_Retrieval_Observation_Score",
        "Model_A_RAG_Retrieval_Observation_Justification",
        "Model_A_Information_Integration_Score",
        "Model_A_Information_Integration_Justification",
        # Model B scores and justifications
        "Model_B_Citation_Score",
        "Model_B_Citation_Justification",
        "Model_B_Relevance_Score",
        "Model_B_Relevance_Justification",
        "Model_B_ReAct_Performance_Thought_Score",
        "Model_B_ReAct_Performance_Thought_Justification",
        "Model_B_RAG_Retrieval_Observation_Score",
        "Model_B_RAG_Retrieval_Observation_Justification",
        "Model_B_Information_Integration_Score",
        "Model_B_Information_Integration_Justification",
        # Error tracking
        "Evaluation_Error",
    ]

    results = []

    # Process each row with progress bar
    log_section("評価処理の開始")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="評価中"):
        # Convert pandas Series to str if needed
        question_val = row["Question"]
        model_a_val = row["Model_A_Response"]
        model_b_val = row["Model_B_Response"]

        # Use isinstance check to avoid Series condition operator error
        question = (
            str(question_val)
            if not (isinstance(question_val, float) and pd.isna(question_val))
            else ""
        )
        model_a_response = (
            str(model_a_val)
            if not (isinstance(model_a_val, float) and pd.isna(model_a_val))
            else ""
        )
        model_b_response = (
            str(model_b_val)
            if not (isinstance(model_b_val, float) and pd.isna(model_b_val))
            else ""
        )

        # Initialize result row with original data
        result_row = {
            "Question": question,
            "Model_A_Response": model_a_response,
            "Model_B_Response": model_b_response,
        }

        # Call judge model (timeout will be automatically set from model config)
        evaluation = call_judge_model(
            client,
            question,
            model_a_response,
            model_b_response,
            model_name=model_name,
            is_azure=is_azure,
            timeout=None,  # None means use model config default
        )

        if evaluation is None:
            # If evaluation failed, record error and set all scores to None
            result_row["Evaluation_Error"] = (
                "Failed to get valid evaluation from judge model"
            )
            for col in output_columns:
                if col not in result_row:
                    result_row[col] = ""
        else:
            # Extract scores for Model A
            model_a_scores = extract_scores_from_evaluation(
                evaluation, "model_a_evaluation"
            )
            result_row["Model_A_Citation_Score"] = model_a_scores["citation_score"]
            result_row["Model_A_Citation_Justification"] = model_a_scores[
                "citation_justification"
            ]
            result_row["Model_A_Relevance_Score"] = model_a_scores["relevance_score"]
            result_row["Model_A_Relevance_Justification"] = model_a_scores[
                "relevance_justification"
            ]
            result_row["Model_A_ReAct_Performance_Thought_Score"] = model_a_scores[
                "react_performance_thought_score"
            ]
            result_row["Model_A_ReAct_Performance_Thought_Justification"] = (
                model_a_scores["react_performance_thought_justification"]
            )
            result_row["Model_A_RAG_Retrieval_Observation_Score"] = model_a_scores[
                "rag_retrieval_observation_score"
            ]
            result_row["Model_A_RAG_Retrieval_Observation_Justification"] = (
                model_a_scores["rag_retrieval_observation_justification"]
            )
            result_row["Model_A_Information_Integration_Score"] = model_a_scores[
                "information_integration_score"
            ]
            result_row["Model_A_Information_Integration_Justification"] = (
                model_a_scores["information_integration_justification"]
            )

            # Extract scores for Model B
            model_b_scores = extract_scores_from_evaluation(
                evaluation, "model_b_evaluation"
            )
            result_row["Model_B_Citation_Score"] = model_b_scores["citation_score"]
            result_row["Model_B_Citation_Justification"] = model_b_scores[
                "citation_justification"
            ]
            result_row["Model_B_Relevance_Score"] = model_b_scores["relevance_score"]
            result_row["Model_B_Relevance_Justification"] = model_b_scores[
                "relevance_justification"
            ]
            result_row["Model_B_ReAct_Performance_Thought_Score"] = model_b_scores[
                "react_performance_thought_score"
            ]
            result_row["Model_B_ReAct_Performance_Thought_Justification"] = (
                model_b_scores["react_performance_thought_justification"]
            )
            result_row["Model_B_RAG_Retrieval_Observation_Score"] = model_b_scores[
                "rag_retrieval_observation_score"
            ]
            result_row["Model_B_RAG_Retrieval_Observation_Justification"] = (
                model_b_scores["rag_retrieval_observation_justification"]
            )
            result_row["Model_B_Information_Integration_Score"] = model_b_scores[
                "information_integration_score"
            ]
            result_row["Model_B_Information_Integration_Justification"] = (
                model_b_scores["information_integration_justification"]
            )

            result_row["Evaluation_Error"] = ""

        results.append(result_row)

    # Create output DataFrame and write to CSV
    if results:
        output_df = pd.DataFrame(results)
        # Ensure all columns exist
        for col in output_columns:
            if col not in output_df.columns:
                output_df[col] = ""
        # Reorder columns
        output_df = output_df.reindex(columns=output_columns)
        output_df.to_csv(output_file, index=False)
    else:
        # Create empty DataFrame with correct columns
        output_df = pd.DataFrame({col: [] for col in output_columns})
        output_df.to_csv(output_file, index=False)

    # Print summary
    log_section("処理完了")
    log_success("評価が完了しました")
    log_success(f"結果ファイル: {output_file}")
    log_success(f"処理行数: {len(results)}行")

    # Print summary statistics
    errors = output_df[output_df["Evaluation_Error"] != ""].shape[0]
    if errors > 0:
        log_warning(f"{errors}行で評価エラーが発生しました", indent=0)
    else:
        log_success("すべての行が正常に処理されました", indent=0)


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python llm_judge_evaluator.py my_test_data.csv
    python llm_judge_evaluator.py /path/to/input.csv

Setup for Azure OpenAI:
    1. Install dependencies: pip install -r requirements.txt
    2. Set environment variables:
       export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'
       export AZURE_OPENAI_API_KEY='your-api-key'
       export MODEL_NAME='gpt-4.1'  # or 'gpt-5', 'gpt-4-turbo'
       export AZURE_OPENAI_API_VERSION='2024-08-01-preview'  # optional, defaults to this
    3. Run the script with your input CSV file:
       python llm_judge_evaluator.py input.csv -m gpt-5  # Use -m to specify model

Setup for Standard OpenAI:
    1. Install dependencies: pip install -r requirements.txt
    2. Set API key: export OPENAI_API_KEY='your-api-key-here'
    3. Optionally set model: export MODEL_NAME='gpt-4-turbo'
    4. Run the script with your input CSV file:
       python llm_judge_evaluator.py input.csv -m gpt-4-turbo  # Use -m to specify model

Supported Models:
    - gpt-5: GPT-5 (uses max_completion_tokens, temperature=1.0)
    - gpt-4.1: GPT-4.1 (uses max_tokens, temperature=0.7)
    - gpt-4-turbo: GPT-4 Turbo (uses max_tokens, temperature=0.7)

You can specify the model via:
    - Command line: -m gpt-5 or --model gpt-4.1
    - Environment variable: export MODEL_NAME='gpt-5'
    - Default: gpt-4.1
        """,
    )

    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file (header row optional, columns: Question, Model_A_Response, Model_B_Response)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="evaluation_output.csv",
        help="Path to the output CSV file (default: evaluation_output.csv)",
    )

    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N rows (useful for testing to avoid high API costs)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=f"Model to use for evaluation (default: {DEFAULT_MODEL}). Supported models: {', '.join(SUPPORTED_MODELS)}",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and run non-interactively (useful for CI/batch execution)",
    )

    args = parser.parse_args()

    # Normalize model name if provided
    if args.model:
        # Convert "gpt5" to "gpt-5", etc.
        model_name_normalized = args.model.lower().strip()
        for supported_model in SUPPORTED_MODELS:
            if (
                supported_model.lower() == model_name_normalized
                or supported_model.replace("-", "").lower()
                == model_name_normalized.replace("-", "")
            ):
                args.model = supported_model
                break
        else:
            # If not found, keep original but will warn later
            pass

    log_section("LLM-as-a-Judge 評価スクリプト")

    # Determine model name
    model_name = args.model or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    if model_name:
        log_info(f"使用モデル: {model_name}")
        if model_name not in SUPPORTED_MODELS:
            log_warning(
                f"モデル '{model_name}' は正式にサポートされていませんが、試行します。",
                indent=0,
            )
            log_info(
                f"サポートされているモデル: {', '.join(SUPPORTED_MODELS)}", indent=1
            )

    process_csv(
        args.input_csv,
        args.output,
        limit_rows=args.limit,
        model_name=model_name,
        non_interactive=args.yes,
    )


if __name__ == "__main__":
    main()
