"""
Common function for calling judge model API.

This module provides a common implementation of call_judge_model that can be used
by both llm_judge_evaluator.py and format_clarity_evaluator.py to eliminate code duplication.
"""

import json
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Union

from openai import AzureOpenAI, OpenAI

from src.config.app_config import get_max_retries, get_retry_delay, get_timeout
from src.utils.logging_config import log_error, log_info, log_warning

# Token estimation constant
# OpenAI tokenizer: approximately 4 characters per token for English
# Note: This is an approximation. For accurate token counting, use tiktoken library.
# Japanese text typically requires 2-3 characters per token.
TOKEN_ESTIMATION_CHARS_PER_TOKEN = 4


def call_judge_model_common(
    client: Union[OpenAI, AzureOpenAI],
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    model_config: Dict[str, Any],
    response_validation_keys: List[str],
    enable_token_estimation: bool = False,
    max_retries: Optional[int] = None,
    retry_delay: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Common function to call the OpenAI API for judge model evaluation.

    This function handles API calls, retries, error handling, and response validation
    for both llm_judge_evaluator and format_clarity_evaluator use cases.

    Args:
        client: OpenAI or AzureOpenAI client instance
        system_prompt: System prompt for the judge model
        user_prompt: User prompt containing the evaluation request
        model_name: Model name or Azure deployment name
        model_config: Model configuration dictionary
        response_validation_keys: List of keys that must be present in the response
        enable_token_estimation: Whether to enable token estimation and dynamic adjustment
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }

            # Token estimation and dynamic adjustment (if enabled)
            if enable_token_estimation:
                user_prompt_len = len(user_prompt)
                system_prompt_len = len(system_prompt)
                estimated_input_tokens = (
                    user_prompt_len + system_prompt_len
                ) / TOKEN_ESTIMATION_CHARS_PER_TOKEN

                # Get model configuration values
                max_total_tokens = model_config["max_total_tokens"]
                min_output_tokens = model_config["min_output_tokens"]
                max_output_tokens_limit = model_config["max_output_tokens"]
                safety_margin = model_config["safety_margin"]
                temperature = model_config["temperature"]
                use_max_completion_tokens = model_config["use_max_completion_tokens"]

                # Check if input is too long
                max_input_tokens = max_total_tokens - min_output_tokens - safety_margin
                if estimated_input_tokens > max_input_tokens:
                    error_msg = (
                        f"入力トークン数が長すぎます（約{estimated_input_tokens:.0f}トークン）。"
                        f"最大{max_input_tokens:.0f}トークンまで対応可能です。"
                        f"入力データを短縮するか、分割してください。"
                    )
                    raise ValueError(error_msg)

                # Calculate max output tokens dynamically
                calculated_max_output_tokens = max(
                    min_output_tokens,
                    min(
                        max_output_tokens_limit,
                        int(max_total_tokens - estimated_input_tokens - safety_margin),
                    ),
                )

                # Set token parameters
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
            else:
                # Use model config values directly
                temperature = model_config["temperature"]
                use_max_completion_tokens = model_config["use_max_completion_tokens"]

                if use_max_completion_tokens:
                    api_params["max_completion_tokens"] = model_config.get(
                        "max_completion_tokens", model_config.get("max_tokens", 2000)
                    )
                else:
                    api_params["max_tokens"] = model_config.get("max_tokens", 2000)

                api_params["temperature"] = temperature

            # Add timeout to API call with progress indication (if timeout is set)
            if timeout is not None:
                import time as time_module

                start_time = time_module.time()

                # Progress indicator
                progress_stop = threading.Event()

                def show_progress():
                    while not progress_stop.is_set():
                        elapsed = time_module.time() - start_time
                        if elapsed < timeout:
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
                    progress_stop.set()
                    elapsed_time = time_module.time() - start_time
                    print("", file=sys.stderr)  # New line after progress
                    if timeout is not None and elapsed_time >= timeout:
                        raise TimeoutError(
                            f"API呼び出しがタイムアウトしました（{timeout}秒経過）"
                        )
                    else:
                        raise api_error
            else:
                # No timeout specified
                response = client.chat.completions.create(**api_params)
                if attempt == 0:
                    log_info("✓ API呼び出し成功", indent=1)

            # Extract the response content
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

                if enable_token_estimation:
                    # More detailed handling for token estimation mode
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
                        log_warning(
                            f"レスポンスが非常に短いです（{content_length}文字）。max_tokens制限に達した可能性があります。",
                            indent=1,
                        )
                        log_warning(
                            "リトライしても同じ結果になる可能性があります。", indent=2
                        )
                        # Continue to retry logic
                    else:
                        log_warning(
                            f"レスポンスが途中で切れました（{content_length:,}文字）。max_tokens制限に達しましたが、処理を続行します。",
                            indent=1,
                        )
                else:
                    # Simple warning for non-token-estimation mode
                    log_warning("Response was truncated (hit max_completion_tokens limit)")

            if not content:
                raise ValueError(
                    f"Empty response from API. Finish reason: {finish_reason}"
                )

            # Parse and validate JSON
            evaluation = json.loads(content)

            # Basic validation of the response structure
            for key in response_validation_keys:
                if key not in evaluation:
                    raise ValueError(f"Response missing required key: {key}")

            return evaluation

        except json.JSONDecodeError as e:
            log_error(
                f"JSON解析エラー (試行 {attempt + 1}/{max_retries}): {e}", indent=1
            )

            # Log error details for debugging
            if response is not None:
                try:
                    received_content = (
                        response.choices[0].message.content
                        if response.choices
                        else None
                    )
                    if received_content:
                        log_info(
                            f"受信したコンテンツ（最初の500文字）: {received_content[:500]}",
                            indent=2,
                        )
                    else:
                        log_info(
                            "受信したコンテンツが空または取得できませんでした", indent=2
                        )
                except (AttributeError, IndexError):
                    log_info("レスポンスの解析に失敗しました", indent=2)

            if attempt == max_retries - 1:
                return None
            time.sleep(retry_delay)

        except ValueError as e:
            error_msg = str(e)
            # Input too long - don't retry
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
            # API returned max_tokens error (input+output exceeds limit)
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
                # Other API errors (connection errors, auth errors, etc.) - retry
                log_error(
                    f"APIエラー (試行 {attempt + 1}/{max_retries}): {type(e).__name__}: {e}",
                    indent=1,
                )
                if attempt == max_retries - 1:
                    return None
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

    return None

