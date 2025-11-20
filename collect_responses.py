#!/usr/bin/env python3
"""
LLM Response Collector Script

This script collects responses from the LLM API for two models (claude3.5-sonnet and claude4.5-haiku)
and creates a CSV file suitable for evaluation.

Usage:
    python collect_responses.py questions.txt -o output.csv
    python collect_responses.py questions.txt --api-url http://localhost:8080/api/v1/urls
"""

import argparse
import csv
import json
import re
import sys
import time
import uuid
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


def clean_html(text_block: str) -> str:
    """
    Replaces HTML links with a 'Text (URL)' format and removes other HTML tags.
    """
    if not text_block:
        return ""

    # Pattern to find <a href="...">...</a> and capture the URL and the link text.
    link_pattern = re.compile(r'<a.*?href="([^"]*)".*?>(.*?)</a>', re.DOTALL)
    cleaned_text = link_pattern.sub(r"\2 (\1)", text_block)

    # Pattern to find and remove any other leftover HTML tags.
    tag_pattern = re.compile(r"<[^>]+>")
    cleaned_text = tag_pattern.sub("", cleaned_text)

    # Normalize whitespace for better readability.
    cleaned_text = re.sub(r"[\r\n]{2,}", "\n\n", cleaned_text).strip()

    return cleaned_text


def clean_and_format_llm_log(messy_text: str) -> str:
    """
    Parses, cleans, and formats a messy LLM log string, ensuring no sections are lost.
    This function is adapted from ../log-output-simplifier/main.py

    Args:
        messy_text: The raw JSON string from the LLM log file or the answer field content.

    Returns:
        A clean, formatted string with all available sections included.
    """
    try:
        # Try to parse as JSON first
        log_data = json.loads(messy_text)
        # The main content is usually inside the "answer" key.
        answer_content = log_data.get("answer", messy_text)
        # Normalize escaped newlines to actual newlines.
        answer_content = answer_content.replace("\\n", "\n")
    except json.JSONDecodeError:
        # If not JSON, treat as plain text
        answer_content = messy_text.replace("\\n", "\n")
    except Exception as e:
        return f"An unexpected error occurred during JSON parsing: {e}"

    # Define all sections we want to extract from the log.
    sections_to_find = [
        {"title": "## 📝 Task タスク", "marker": "タスク："},
        {"title": "## 💬 Reaction 反応", "marker": "反応："},
        {"title": "## 📂 Classification 分類", "marker": "分類："},
        {"title": "## 📊 Status 状態", "marker": "状態："},
        {"title": "## 🤖 LLM Thought Process 思考", "marker": "思考："},
        {"title": "## ⚡ Action 行動", "marker": "行動："},
        {"title": "## ⌨️ Action Input 行動入力", "marker": "行動入力："},
        {"title": "## 📚 Raw Search Results (Cleaned) 観察", "marker": "観察："},
        {"title": "## ✅ Final Answer 回答", "marker": "回答："},
        {"title": "## 🔗 URLs URL", "marker": "URL："},
    ]

    # Find the starting position of each section marker in the text.
    found_sections = []
    for section in sections_to_find:
        # Use a loop to find all occurrences of a marker (like '思考：')
        start_index = -1
        while True:
            start_index = answer_content.find(section["marker"], start_index + 1)
            if start_index == -1:
                break
            found_sections.append(
                {
                    "start": start_index,
                    "title": section["title"],
                    "marker_len": len(section["marker"]),
                }
            )

    # If no markers are found, clean the entire text as a fallback.
    if not found_sections:
        return (
            "No known section markers found. Performing a full clean:\n\n"
            + clean_html(answer_content)
        )

    # Sort the found sections by their starting position to process them in order.
    found_sections.sort(key=lambda x: x["start"])

    output_parts = []
    # Extract the content for each section.
    for i, section in enumerate(found_sections):
        content_start = section["start"] + section["marker_len"]

        # Determine the end of the current section's content.
        # It's either the start of the next section or the end of the string.
        if i + 1 < len(found_sections):
            content_end = found_sections[i + 1]["start"]
        else:
            # For the last section, go all the way to the end of the content.
            content_end = len(answer_content)

        content = answer_content[content_start:content_end]

        # Clean the extracted content block.
        cleaned_content = clean_html(content)

        # Avoid adding empty sections
        if not cleaned_content.strip():
            continue

        output_parts.append(section["title"])
        output_parts.append("---")

        # Special formatting for the "Observation" (Raw Search Results) section.
        if "Raw Search Results" in section["title"]:
            results = cleaned_content.split(
                "################################################"
            )
            for j, result in enumerate(results, 1):
                if result.strip():
                    output_parts.append(f"### Result {j}\n{result.strip()}")
        else:
            output_parts.append(cleaned_content.strip())

        output_parts.append("\n")

    return "\n".join(output_parts)


def format_response(response_text: str) -> str:
    """
    Format the API response using the log simplifier.

    Args:
        response_text: The raw response text from API (answer field content)

    Returns:
        Formatted response string
    """
    if not response_text:
        return ""

    # Wrap the response in a JSON-like structure for the formatter
    # The formatter expects JSON with an "answer" field, but we already have the answer content
    try:
        # Try to format as if it's already the answer content
        formatted = clean_and_format_llm_log(response_text)
        return formatted
    except Exception as e:
        # If formatting fails, return original text
        print(f"  ⚠️  ログ整形エラー: {e}", file=sys.stderr)
        return response_text


def call_api(
    question: str,
    api_url: str,
    model_name: str,
    identity: str = "A14804",
    timeout: int = 120,
    verbose: bool = True,
) -> Optional[str]:
    """
    Call the LLM API and return the response.

    Args:
        question: The question to ask
        api_url: The API endpoint URL
        model_name: The model name (claude3.5-sonnet or claude4.5-haiku)
        identity: The x-amzn-oidc-identity header value
        timeout: Request timeout in seconds
        verbose: Whether to print detailed logs

    Returns:
        The response text, or None if failed
    """
    question_uuid = str(uuid.uuid4())

    # Prepare the request
    url = f"{api_url}?llm_model_name={model_name}&rag_enabled=auto"
    headers = {"x-amzn-oidc-identity": identity, "Content-Type": "application/json"}
    data = {
        "question_uuid": question_uuid,
        "messages": [{"role": "user", "content": question}],
    }

    if verbose:
        print(f"  📤 [{model_name}] API呼び出し開始")
        print(f"     URL: {url}")
        print(
            f"     質問: {question[:60]}..."
            if len(question) > 60
            else f"     質問: {question}"
        )

    start_time = time.time()
    response: Optional[requests.Response] = None

    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        elapsed_time = time.time() - start_time

        if verbose:
            print(
                f"  📥 [{model_name}] HTTPステータス: {response.status_code} (経過時間: {elapsed_time:.2f}秒)"
            )

        response.raise_for_status()

        # Parse response - API returns object format: {"answer": "...", "urls": [...], ...}
        response_data = response.json()

        # Handle both array and object formats
        if isinstance(response_data, list) and len(response_data) >= 1:
            # Array format: ["{...}", status_code]
            json_str = response_data[0]
            if isinstance(json_str, str):
                parsed = json.loads(json_str)
            else:
                parsed = json_str
        elif isinstance(response_data, dict):
            # Direct object format: {"answer": "...", ...}
            parsed = response_data
        else:
            if verbose:
                print(f"  ⚠️  [{model_name}] 予期しないレスポンス形式")
            return response.text

        # Extract the answer field (this contains the LLM response)
        if "answer" in parsed:
            answer = parsed["answer"]
            answer_length = len(answer)
            if verbose:
                print(
                    f"  ✅ [{model_name}] レスポンス取得成功 (answer長さ: {answer_length:,}文字)"
                )
                if "urls" in parsed and isinstance(parsed["urls"], list):
                    print(f"     検索結果URL数: {len(parsed['urls'])}")
            return answer
        else:
            if verbose:
                print(f"  ⚠️  [{model_name}] 'answer'フィールドが見つかりません")
            return response.text

    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print(
            f"\n  ❌ [{model_name}] API呼び出しタイムアウト (経過時間: {elapsed_time:.2f}秒)",
            file=sys.stderr,
        )
        print(f"     URL: {url}", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        elapsed_time = time.time() - start_time
        print(
            f"\n  ❌ [{model_name}] API呼び出しエラー (経過時間: {elapsed_time:.2f}秒)",
            file=sys.stderr,
        )
        print(f"     エラー: {e}", file=sys.stderr)
        print(f"     URL: {url}", file=sys.stderr)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        elapsed_time = time.time() - start_time
        print(
            f"\n  ❌ [{model_name}] レスポンス解析エラー (経過時間: {elapsed_time:.2f}秒)",
            file=sys.stderr,
        )
        print(f"     エラー: {e}", file=sys.stderr)
        response_text: Optional[str] = None
        if response is not None:
            try:
                response_text = (
                    response.text[:200]
                    if hasattr(response, "text")
                    else str(response)[:200]
                )
                print(f"     レスポンスプレビュー: {response_text}", file=sys.stderr)
            except AttributeError:
                pass
        return response_text


def collect_responses(
    questions: List[str],
    api_url: str,
    model_a: str,
    model_b: str,
    identity: str = "A14804",
    timeout: int = 120,
    delay: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Collect responses from both models for all questions.

    Args:
        questions: List of questions to ask
        api_url: The API endpoint URL
        model_a: Model name for Model A (e.g., claude3.5-sonnet)
        model_b: Model name for Model B (e.g., claude4.5-haiku)
        identity: The x-amzn-oidc-identity header value
        timeout: Request timeout in seconds
        delay: Delay between API calls in seconds
        verbose: Whether to print detailed logs

    Returns:
        DataFrame with Question, Model_A_Response, Model_B_Response columns
        (Each response contains only the "answer" field from API)
    """
    results = []
    total_start_time = time.time()

    print("=" * 70)
    print("📋 収集設定")
    print("=" * 70)
    print(f"  質問数: {len(questions)}")
    print(f"  Model A: {model_a}")
    print(f"  Model B: {model_b}")
    print(f"  API URL: {api_url}")
    print(f"  リクエスト間隔: {delay}秒")
    print(f"  タイムアウト: {timeout}秒")
    print(
        f"  予想処理時間: 約{len(questions) * 2 * (delay + 15):.0f}秒 (各API呼び出し15秒想定)"
    )
    print("=" * 70)
    print()

    success_count_a = 0
    success_count_b = 0
    failed_count_a = 0
    failed_count_b = 0

    for idx, question in enumerate(tqdm(questions, desc="📊 進捗"), 1):
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"📝 質問 {idx}/{len(questions)}")
            print(f"{'=' * 70}")
            print(f"質問: {question}")
            print()

        question_start_time = time.time()

        # Call Model A
        if verbose:
            print(f"[{idx}/{len(questions)}] Model A ({model_a}) を呼び出し中...")
        response_a_raw = call_api(
            question, api_url, model_a, identity, timeout, verbose=verbose
        )

        # Format the response using log simplifier
        if response_a_raw:
            if verbose:
                print("  🔧 Model A レスポンスを整形中...")
            response_a = format_response(response_a_raw)
            success_count_a += 1
        else:
            response_a = ""
            failed_count_a += 1

        # Wait between Model A and Model B calls
        if verbose:
            print(f"  ⏸️  Model B呼び出しまで{delay}秒待機中...")
        time.sleep(delay)  # Rate limiting

        # Call Model B
        if verbose:
            print(f"[{idx}/{len(questions)}] Model B ({model_b}) を呼び出し中...")
        response_b_raw = call_api(
            question, api_url, model_b, identity, timeout, verbose=verbose
        )

        # Format the response using log simplifier
        if response_b_raw:
            if verbose:
                print("  🔧 Model B レスポンスを整形中...")
            response_b = format_response(response_b_raw)
            success_count_b += 1
        else:
            response_b = ""
            failed_count_b += 1

        # Store formatted responses
        # Column names compatible with both llm_judge_evaluator.py and ragas_llm_judge_evaluator.py
        results.append(
            {
                "Question": question,
                "Model_A_Response": response_a,
                "Model_B_Response": response_b,
            }
        )

        question_elapsed = time.time() - question_start_time

        if verbose:
            status_a = "✅" if response_a else "❌"
            status_b = "✅" if response_b else "❌"
            print(f"\n  📊 質問 {idx} 完了 (経過時間: {question_elapsed:.2f}秒)")
            print(f"     Model A: {status_a} | Model B: {status_b}")
            print(f"     成功数: A={success_count_a}/{idx}, B={success_count_b}/{idx}")

        # Wait before next question (if not the last question)
        if idx < len(questions):
            if verbose:
                print(f"  ⏸️  次の質問まで{delay}秒待機中...")
            time.sleep(delay)  # Rate limiting

    total_elapsed = time.time() - total_start_time

    if verbose:
        print("\n" + "=" * 70)
        print("📊 収集完了統計")
        print("=" * 70)
        print(f"  総処理時間: {total_elapsed:.2f}秒 ({total_elapsed / 60:.2f}分)")
        print(f"  質問数: {len(questions)}")
        print(f"  Model A ({model_a}):")
        print(
            f"    ✅ 成功: {success_count_a}/{len(questions)} ({success_count_a / len(questions) * 100:.1f}%)"
        )
        print(f"    ❌ 失敗: {failed_count_a}/{len(questions)}")
        print(f"  Model B ({model_b}):")
        print(
            f"    ✅ 成功: {success_count_b}/{len(questions)} ({success_count_b / len(questions) * 100:.1f}%)"
        )
        print(f"    ❌ 失敗: {failed_count_b}/{len(questions)}")
        print("=" * 70)

    return pd.DataFrame(results)


def read_questions(input_file: str) -> List[str]:
    """
    Read questions from a text file or CSV file.

    Supports:
    - Text file: One question per line
    - CSV file: First column contains questions (with or without header)

    Args:
        input_file: Path to the input file

    Returns:
        List of questions
    """
    questions = []
    try:
        # Check if file is CSV by extension or try to read as CSV first
        if input_file.lower().endswith(".csv"):
            # Try to read as CSV
            df = pd.read_csv(input_file)
            # Get first column (usually "Questions" or "Question")
            first_col = df.columns[0]
            questions = df[first_col].dropna().astype(str).tolist()
            # Remove header if it looks like a header (common header names)
            if questions and questions[0].lower() in [
                "question",
                "questions",
                "q",
                "query",
                "queries",
            ]:
                questions = questions[1:]
            # Filter out empty strings
            questions = [q.strip() for q in questions if q.strip()]
        else:
            # Read as text file (one question per line)
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(
                        "#"
                    ):  # Skip empty lines and comments
                        questions.append(line)
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)

    return questions


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Collect LLM responses from API for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Collect responses from questions.txt
    python collect_responses.py questions.txt -o responses.csv
    
    # Use custom API URL
    python collect_responses.py questions.txt --api-url http://localhost:8080/api/v1/urls
    
    # Use custom models
    python collect_responses.py questions.txt --model-a claude3.5-sonnet --model-b claude4.5-haiku
    
    # Use custom identity
    python collect_responses.py questions.txt --identity YOUR_IDENTITY

Input file format:
    - Text file (.txt): One question per line. Lines starting with # are treated as comments.
    - CSV file (.csv): First column contains questions (with or without header row)
    
    Text file example:
        AIオペ室の相談窓口はどこ？
        会社の休暇制度について教えてください
        # This is a comment
        社内のWiFiパスワードは？
    
    CSV file example:
        Questions
        AIオペ室の相談窓口はどこ？
        会社の休暇制度について教えてください
        """,
    )

    parser.add_argument(
        "input_file",
        help="Path to the input file containing questions (.txt or .csv format)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="collected_responses.csv",
        help="Path to the output CSV file (default: collected_responses.csv)",
    )

    parser.add_argument(
        "--api-url",
        default="http://0.0.0.0:8080/api/v2/questions",
        help="API endpoint URL (default: http://0.0.0.0:8080/api/v2/questions)",
    )

    parser.add_argument(
        "--model-a",
        default="claude3.5-sonnet",
        help="Model name for Model A (default: claude3.5-sonnet)",
    )

    parser.add_argument(
        "--model-b",
        default="claude4.5-haiku",
        help="Model name for Model B (default: claude4.5-haiku)",
    )

    parser.add_argument(
        "--identity",
        default="A14804",
        help="x-amzn-oidc-identity header value (default: A14804)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LLM Response Collector")
    print("=" * 70)

    # Read questions
    print(f"\nReading questions from: {args.input_file}")
    questions = read_questions(args.input_file)
    print(f"✓ Loaded {len(questions)} questions")

    if len(questions) == 0:
        print("ERROR: No questions found in input file.", file=sys.stderr)
        sys.exit(1)

    # Collect responses
    df = collect_responses(
        questions=questions,
        api_url=args.api_url,
        model_a=args.model_a,
        model_b=args.model_b,
        identity=args.identity,
        timeout=args.timeout,
        delay=args.delay,
        verbose=True,
    )

    # Save to CSV
    print("\n" + "=" * 70)
    print("💾 CSVファイルに保存中...")
    print("=" * 70)
    df.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL)

    print(f"✅ ファイル保存完了: {args.output}")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   列名: {', '.join(df.columns)}")

    # Check for errors
    failed_a = df[df["Model_A_Response"] == ""].shape[0]
    failed_b = df[df["Model_B_Response"] == ""].shape[0]

    print("\n" + "=" * 70)
    print("✅ 収集完了!")
    print("=" * 70)
    print(f"📄 出力ファイル: {args.output}")
    print(f"📊 収集したレスポンス数: {len(df)}")

    if failed_a > 0 or failed_b > 0:
        print("\n⚠️  警告:")
        if failed_a > 0:
            print(f"  ❌ Model A ({args.model_a}): {failed_a}件のレスポンス取得に失敗")
        if failed_b > 0:
            print(f"  ❌ Model B ({args.model_b}): {failed_b}件のレスポンス取得に失敗")
    else:
        print("\n✅ すべてのレスポンスが正常に取得されました!")

    print("\n" + "=" * 70)
    print("📝 次のステップ")
    print("=" * 70)
    print("評価スクリプトを実行:")
    print(f"  python llm_judge_evaluator.py {args.output} -n 5")
    print("\nまたは:")
    print(f"  python ragas_llm_judge_evaluator.py {args.output} -n 5")
    print("=" * 70)


if __name__ == "__main__":
    main()
