#!/usr/bin/env python3
"""
Format Clarity Evaluator - LLM-as-a-Judge for Stylistic Similarity

This script evaluates how closely the formatting and style of Model A's responses
match Model B's responses (the baseline/golden standard).

It reads raw ReAct logs from a CSV file, parses the Final Answer sections,
and uses an LLM judge (GPT-5 or GPT-4-turbo) to score format similarity on a 1-5 scale.

Usage:
    python scripts/format_clarity_evaluator.py <input_csv_file>

Requirements:
    - For Azure OpenAI: Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and MODEL_NAME
    - For Standard OpenAI: Set OPENAI_API_KEY (and optionally MODEL_NAME)
    - Input CSV must have 3 columns (header row optional): Question, Model_A_Response, Model_B_Response
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402
from openai import OpenAI, AzureOpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.config.model_configs import (  # noqa: E402
    SUPPORTED_MODELS,
    get_simple_config as get_model_config_from_common,
)
from src.utils.logging_config import (  # noqa: E402
    log_info,
    log_error,
    log_warning,
    log_success,
    log_section,
    setup_logging,
)
from src.utils.judge_model_common import call_judge_model_common  # noqa: E402
from src.config.app_config import get_max_workers  # noqa: E402

# Format clarity evaluator uses gpt-4-turbo as default (different from common default)
DEFAULT_MODEL = "gpt-4-turbo"

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging system
setup_logging()


# Model configuration is now imported from config.model_configs
# Use get_model_config_from_common() to get simple configuration


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.

    Args:
        model_name: Model name (e.g., "gpt-5", "gpt-4-turbo")

    Returns:
        Model configuration dictionary (simple configuration)
    """
    return get_model_config_from_common(model_name)


# Format Clarity Judge System Prompt (embedded as per requirements)
JUDGE_SYSTEM_PROMPT = """You are a meticulous AI evaluator specializing in text formatting and stylistic consistency.

**Your Task:**
Your sole task is to evaluate how closely the format of the "Model A Answer" matches the format of the "Model B Answer". The "Model B Answer" is the golden standard, and you are scoring the "Model A Answer" on its ability to mimic that style.

You must provide a score from 1 to 5, using the detailed rubric below. You must return your evaluation as a single, valid JSON object and nothing else.

### **Detailed Scoring Rubric: Format/Clarity (回答の形式)**

*Focus: Compare the "Model A Answer" directly against the "Model B Answer". Evaluate the similarity in markdown (headings, bolding), list structures (bullets/numbers), and overall logical separation of ideas.*

- **5 (Excellent - Near Identical):**
    The Model A response uses virtually identical formatting to the Model B response. It effectively mirrors the use of markdown headings (e.g., `##`), bold text (`**...**`), bullet points (`-`), and logical paragraph breaks. The structure is a clear match.

- **4 (Good - Mostly Similar):**
    The Model A response follows the general structure (e.g., headings, lists) of the Model B response but has minor, non-critical deviations. For example, it might use bullets where Model B used numbers, or miss a single bolded word, but the overall style is clearly aligned.

- **3 (Acceptable - Some Similarities):**
    The Model A response shows some structural similarities but also has significant differences. For example, it might use lists where Model B used paragraphs, or be missing all the headings that Model B used. The style is noticeably different but not completely unrelated.

- **2 (Poor - Mostly Different):**
    The Model A response is mostly different. It might use some basic formatting (like paragraph breaks), but it does not resemble the structure or style (e.g., use of headings and lists) of the Model B response.

- **1 (Very Poor - No Resemblance):**
    The Model A response format is completely different. For example, Model B provided a structured, multi-part answer, and Model A returned a single, dense block of text (or vice-versa).

---
**Required JSON Output Format:**
{
  "format_clarity_evaluation": {
    "score": <score_from_1_to_5>,
    "justification": "<A brief justification for your score, detailing why the 4.5 model's format matches or differs from the 3.5 model's format.>"
  }
}"""


def parse_final_answer(raw_log: str) -> str:
    """
    Parse the raw ReAct log to extract only the Final Answer text.

    This function looks for the "## ✅ Final Answer 回答" heading and extracts
    all text that follows it until it hits "## 🔗 URLs" section or end of string.

    Args:
        raw_log: The complete raw ReAct log string

    Returns:
        The extracted final answer text, or the original log if parsing fails
    """
    if not isinstance(raw_log, str):
        return str(raw_log) if raw_log is not None else ""

    # Pattern to match "## ✅ Final Answer 回答" and capture content until "## 🔗 URLs" or end
    # Using non-greedy match (.+?) and lookahead to stop at URLs section
    # The URLs section may be formatted as "## 🔗 URLs URL" or just "## 🔗 URLs"
    pattern = r"##\s*✅\s*Final\s+Answer\s*回答\s*(.+?)(?=##\s*🔗\s*URLs(?:\s+URL)?|$)"

    # Use DOTALL flag to make . match newlines
    match = re.search(pattern, raw_log, re.DOTALL | re.IGNORECASE)

    if match:
        # Extract the final answer text and strip leading/trailing whitespace
        final_answer = match.group(1).strip()
        return final_answer
    else:
        # If pattern not found, try alternative patterns
        # Sometimes the emoji might be missing or the format slightly different
        alt_patterns = [
            r"##\s*Final\s+Answer\s*回答\s*(.+?)(?=##\s*🔗\s*URLs(?:\s+URL)?|$)",
            r"##\s*✅\s*Final\s+Answer\s*(.+?)(?=##\s*🔗\s*URLs(?:\s+URL)?|$)",
            r"Final\s+Answer\s*回答\s*[:\n]\s*(.+?)(?=##\s*🔗\s*URLs(?:\s+URL)?|$)",
        ]

        for alt_pattern in alt_patterns:
            match = re.search(alt_pattern, raw_log, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If still no match, return a warning message or the original log
        log_warning("Could not parse final answer from log. Using full log.")
        return raw_log


def create_user_prompt(question: str, model_a_answer: str, model_b_answer: str) -> str:
    """
    Create the user prompt that will be sent to the judge model.

    Args:
        question: The original user question
        model_a_answer: Parsed final answer from Model A (to be evaluated)
        model_b_answer: Parsed final answer from Model B (golden standard/baseline)

    Returns:
        Formatted prompt string
    """
    return f"""Please evaluate how closely the formatting and style of the Model A Answer matches the Model B Answer.

**Question:**
{question}

**Model B Answer (Golden Standard):**
{model_b_answer}

**Model A Answer (To Be Evaluated):**
{model_a_answer}

Provide your evaluation as a JSON object following the specified format."""


def call_judge_model(
    client: Union[OpenAI, AzureOpenAI],
    question: str,
    model_a_answer: str,
    model_b_answer: str,
    model_name: str = "gpt-4-turbo",
    is_azure: bool = False,
    max_retries: Optional[int] = None,
    retry_delay: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Call the OpenAI API to evaluate the format similarity.

    This function is a wrapper around the common call_judge_model_common function,
    providing a convenient interface for format_clarity_evaluator.py.

    Args:
        client: OpenAI or AzureOpenAI client instance
        question: The original user question
        model_a_answer: Parsed final answer from Model A (to be evaluated)
        model_b_answer: Parsed final answer from Model B (golden standard/baseline)
        model_name: Model name or Azure deployment name
        is_azure: Whether using Azure OpenAI (affects parameter naming)
        max_retries: Maximum number of retry attempts on failure (defaults to config value)
        retry_delay: Delay in seconds between retries (defaults to config value)

    Returns:
        Parsed JSON response from the judge model, or None if all retries fail
    """
    # Create user prompt
    user_prompt = create_user_prompt(question, model_a_answer, model_b_answer)

    # Get model configuration
    model_config = get_model_config(model_name)

    # Call common function
    return call_judge_model_common(
        client=client,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model_name=model_name,
        model_config=model_config,
        response_validation_keys=["format_clarity_evaluation"],
        enable_token_estimation=False,
        max_retries=max_retries,
        retry_delay=retry_delay,
        timeout=None,  # format_clarity_evaluator doesn't use timeout
    )


def extract_scores_from_evaluation(
    evaluation: Dict[str, Any],
) -> Tuple[Optional[int], str]:
    """
    Extract score and justification from the evaluation JSON.

    Args:
        evaluation: The full evaluation JSON object

    Returns:
        Tuple of (score, justification)
    """
    format_eval = evaluation.get("format_clarity_evaluation", {})
    score = format_eval.get("score", None)
    justification = format_eval.get("justification", "")

    return score, justification


# Output columns definition for format_clarity_evaluator
FORMAT_CLARITY_OUTPUT_COLUMNS = [
    "Question",
    "Model_A_Final_Answer",
    "Model_B_Final_Answer",
    "Format_Clarity_Score",
    "Format_Clarity_Justification",
    "Evaluation_Error",
]


def initialize_openai_client_format_clarity(
    model_name: str,
) -> tuple[Union[OpenAI, AzureOpenAI], bool]:
    """
    Initialize OpenAI client (Azure or Standard) for format_clarity_evaluator.

    Args:
        model_name: Model name for evaluation

    Returns:
        Tuple of (client, is_azure) where is_azure indicates if Azure OpenAI is used

    Raises:
        SystemExit: If no valid credentials are found
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    is_azure = bool(azure_endpoint and azure_api_key)

    if is_azure:
        # Initialize Azure OpenAI client
        if azure_endpoint is None:
            raise ValueError("azure_endpoint is required for Azure OpenAI")
        log_info("Using Azure OpenAI")
        log_info(f"Endpoint: {azure_endpoint}")
        log_info(f"Model/Deployment: {model_name}")
        log_info(f"API Version: {azure_api_version}")
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
        return client, True
    else:
        # Initialize standard OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log_error("Neither Azure OpenAI nor standard OpenAI credentials found.")
            log_error("\nFor Azure OpenAI, set:")
            log_error(
                "  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'"
            )
            log_error("  export AZURE_OPENAI_API_KEY='your-api-key'")
            log_error("  export MODEL_NAME='gpt-5'  # or your deployment name")
            log_error("\nFor standard OpenAI, set:")
            log_error("  export OPENAI_API_KEY='your-api-key-here'")
            sys.exit(1)
        log_info("Using standard OpenAI")
        log_info(f"Model: {model_name}")
        client = OpenAI(api_key=api_key)
        return client, False


def read_and_validate_csv_format_clarity(input_file: str) -> pd.DataFrame:
    """
    Read and validate input CSV file for format_clarity_evaluator.

    Supports Claude format column names (Claude_35_Raw_Log, Claude_45_Raw_Log).

    Args:
        input_file: Path to the input CSV file

    Returns:
        DataFrame with standardized column names

    Raises:
        SystemExit: If file cannot be read or validated
    """
    log_info(f"Reading input file: {input_file}")
    try:
        # Check if CSV has header row
        with open(input_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().lower()

        # If first line looks like headers, use it as header
        if any(
            keyword in first_line
            for keyword in ["question", "model", "answer", "response", "claude"]
        ):
            log_warning("Detected header row in input CSV. Skipping first row.")
            df = pd.read_csv(input_file)
            # Normalize column names to standard format
            # Support both Model_A_Response/Model_B_Response and Claude_35_Raw_Log/Claude_45_Raw_Log
            column_mapping = {}
            for col in df.columns:
                col_lower = (
                    col.lower().replace("_", "").replace("-", "").replace(" ", "")
                )
                if "question" in col_lower or col_lower == "q":
                    column_mapping[col] = "Question"
                elif (
                    ("model" in col_lower and "a" in col_lower)
                    or "claude35" in col_lower
                    or "claude3.5" in col_lower
                ):
                    column_mapping[col] = "Model_A_Response"
                elif (
                    ("model" in col_lower and "b" in col_lower)
                    or "claude45" in col_lower
                    or "claude4.5" in col_lower
                ):
                    column_mapping[col] = "Model_B_Response"

            if column_mapping:
                df = df.rename(columns=column_mapping)
            else:
                # If no mapping found, assume standard column order
                if len(df.columns) >= 3:
                    df.columns = [
                        "Question",
                        "Model_A_Response",
                        "Model_B_Response",
                    ] + list(df.columns[3:])
        else:
            log_warning("No header row detected. Treating first row as data.")
            df = pd.read_csv(
                input_file,
                header=None,
                names=["Question", "Model_A_Response", "Model_B_Response"],
            )

        # Validate columns
        expected_columns = ["Question", "Model_A_Response", "Model_B_Response"]
        if not all(col in df.columns for col in expected_columns):
            log_error(f"CSV must have columns: {expected_columns}")
            log_error(f"Found columns: {list(df.columns)}")
            sys.exit(1)
    except FileNotFoundError:
        log_error(f"Input file '{input_file}' not found.")
        sys.exit(1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        log_error(f"Failed to read input file: {e}")
        log_error(
            "Please check that the file format is correct and the encoding is valid."
        )
        sys.exit(1)
    except PermissionError as e:
        log_error(f"Permission denied accessing file: {e}")
        sys.exit(1)
    except Exception as e:
        # 予期しないエラーの場合は詳細な情報を記録
        log_error(f"Unexpected error occurred: {type(e).__name__}: {e}")
        sys.exit(1)

    log_info(f"Loaded {len(df)} rows from input file.")
    return df


def apply_row_limit_and_confirm_format_clarity(
    df: pd.DataFrame, limit_rows: Optional[int], model_name: str, non_interactive: bool
) -> pd.DataFrame:
    """
    Apply row limit and prompt for confirmation if needed (format_clarity_evaluator version).

    Args:
        df: Input DataFrame
        limit_rows: Optional limit on number of rows to process
        model_name: Model name for logging
        non_interactive: If True, skips confirmation prompt

    Returns:
        DataFrame with row limit applied

    Raises:
        SystemExit: If user cancels the operation
    """
    if limit_rows is not None and limit_rows < len(df):
        df = df.head(limit_rows)
        log_warning(
            f"LIMITING to first {limit_rows} rows for testing (use -n flag to change)"
        )
        log_warning(f"This will make {limit_rows} API calls")
    else:
        log_warning(f"WARNING: This will make {len(df)} API calls to {model_name}")
        log_warning(
            "API costs will be incurred. It is recommended to test with a small number of rows first using the -n flag."
        )

        # Prompt for confirmation if processing many rows (unless non-interactive mode)
        if len(df) > 10 and not non_interactive:
            try:
                response = (
                    input(f"\n🤔 Proceed with {len(df)} API calls? [y/N]: ")
                    .strip()
                    .lower()
                )
                if response != "y" and response != "yes":
                    log_info(
                        "Cancelled. Use -n flag to test with fewer rows: python scripts/format_clarity_evaluator.py input.csv -n 5"
                    )
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                log_info("\nCancelled.")
                sys.exit(0)

    return df


def process_single_row_format_clarity(
    row: pd.Series,
    client: Union[OpenAI, AzureOpenAI],
    model_name: str,
    is_azure: bool,
    output_columns: list[str],
) -> Dict[str, Any]:
    """
    Process a single row from the input DataFrame (format_clarity_evaluator version).

    Args:
        row: Single row from DataFrame
        client: OpenAI or AzureOpenAI client
        model_name: Model name for evaluation
        is_azure: Whether using Azure OpenAI
        output_columns: List of output column names

    Returns:
        Dictionary containing evaluation results for the row
    """
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
    model_a_raw_log = (
        str(model_a_val)
        if not (isinstance(model_a_val, float) and pd.isna(model_a_val))
        else ""
    )
    model_b_raw_log = (
        str(model_b_val)
        if not (isinstance(model_b_val, float) and pd.isna(model_b_val))
        else ""
    )

    # Parse the final answers from the raw logs
    model_a_final_answer = parse_final_answer(model_a_raw_log)
    model_b_final_answer = parse_final_answer(model_b_raw_log)

    # Initialize result row with parsed data
    result_row = {
        "Question": question,
        "Model_A_Final_Answer": model_a_final_answer,
        "Model_B_Final_Answer": model_b_final_answer,
    }

    # Call judge model
    evaluation = call_judge_model(
        client,
        question,
        model_a_final_answer,
        model_b_final_answer,
        model_name=model_name,
        is_azure=is_azure,
    )

    if evaluation is None:
        # If evaluation failed, record error and set score to None
        result_row["Format_Clarity_Score"] = ""  # Use empty string instead of None
        result_row["Format_Clarity_Justification"] = ""
        result_row["Evaluation_Error"] = (
            "Failed to get valid evaluation from judge model"
        )
    else:
        # Extract score and justification
        score, justification = extract_scores_from_evaluation(evaluation)
        result_row["Format_Clarity_Score"] = str(score) if score is not None else ""
        result_row["Format_Clarity_Justification"] = justification
        result_row["Evaluation_Error"] = ""

    return result_row


def write_results_to_csv_format_clarity(
    results: list[Dict[str, Any]], output_file: str, output_columns: list[str]
) -> pd.DataFrame:
    """
    Write evaluation results to CSV file (format_clarity_evaluator version).

    Includes average score calculation and distribution.

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
        output_columns: List of output column names

    Returns:
        Output DataFrame
    """
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

    log_success("Evaluation complete!")
    log_success(f"Results written to: {output_file}")
    log_success(f"Processed {len(results)} rows")

    # Print summary statistics
    errors = output_df[output_df["Evaluation_Error"] != ""].shape[0]
    if errors > 0:
        log_warning(f"Warning: {errors} rows had evaluation errors")

    # Calculate average score (excluding None/empty values)
    if len(output_df) > 0 and "Format_Clarity_Score" in output_df.columns:
        score_col = output_df["Format_Clarity_Score"]
        if isinstance(score_col, pd.Series):
            score_series = pd.to_numeric(score_col, errors="coerce")
            if isinstance(score_series, pd.Series):
                valid_scores = score_series.dropna()
                if len(valid_scores) > 0:
                    avg_score = float(valid_scores.mean())
                    log_info(f"\n📊 Average Format Clarity Score: {avg_score:.2f}/5.0")
                    log_info("📊 Score Distribution:")
                    value_counts = score_series.value_counts()
                    if isinstance(value_counts, pd.Series):
                        log_info(str(value_counts.sort_index()))

    return output_df


def process_csv(
    input_file: str,
    output_file: str = "format_clarity_output.csv",
    limit_rows: Optional[int] = None,
    model_name: Optional[str] = None,
    non_interactive: bool = False,
    max_workers: Optional[int] = None,
) -> None:
    """
    Main processing function that reads the input CSV, parses logs, evaluates format similarity,
    and writes the results to the output CSV.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (default: format_clarity_output.csv)
        limit_rows: Optional limit on number of rows to process (for cost control)
        model_name: Model name for evaluation. If None, uses MODEL_NAME environment variable or default model.
        non_interactive: If True, skips confirmation prompt even for >10 rows. Default is False.
        max_workers: Maximum number of parallel workers. If None, uses config value or sequential processing.
    """
    # Model name can be set via parameter or environment variable
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)

    # Initialize OpenAI client
    client, is_azure = initialize_openai_client_format_clarity(model_name)

    # Read and validate CSV
    df = read_and_validate_csv_format_clarity(input_file)

    # Apply row limit and confirm if needed
    df = apply_row_limit_and_confirm_format_clarity(
        df, limit_rows, model_name, non_interactive
    )

    # Determine max_workers for parallel processing
    if max_workers is None:
        max_workers = get_max_workers()

    # Process each row with progress bar
    log_info("\nParsing logs and evaluating format similarity...")
    results = []

    if max_workers is None or max_workers == 1:
        # Sequential processing (default behavior)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            result_row = process_single_row_format_clarity(
                row, client, model_name, is_azure, FORMAT_CLARITY_OUTPUT_COLUMNS
            )
            results.append(result_row)
    else:
        # Parallel processing with ThreadPoolExecutor
        log_info(f"並列処理モード: {max_workers}ワーカー", indent=1)
        # Create list of (index, row) tuples to maintain order
        rows_with_index = [(idx, row) for idx, row in df.iterrows()]

        def process_row_with_index(idx_row_tuple):
            """Wrapper function to process a single row with its index"""
            idx, row = idx_row_tuple
            return idx, process_single_row_format_clarity(
                row, client, model_name, is_azure, FORMAT_CLARITY_OUTPUT_COLUMNS
            )

        # Process rows in parallel while maintaining order
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_row_with_index, idx_row): idx_row[0]
                for idx_row in rows_with_index
            }

            # Collect results in order
            results_dict = {}
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(rows_with_index),
                desc="Processing rows",
            ):
                idx = future_to_idx[future]
                try:
                    result_idx, result_row = future.result()
                    results_dict[result_idx] = result_row
                except Exception as e:
                    log_error(f"行 {idx} の処理中にエラーが発生しました: {e}", indent=1)
                    # Create error result row
                    result_row = {
                        "Question": df.loc[idx, "Question"] if idx in df.index else "",
                        "Model_A_Final_Answer": (
                            df.loc[idx, "Model_A_Response"] if idx in df.index else ""
                        ),
                        "Model_B_Final_Answer": (
                            df.loc[idx, "Model_B_Response"] if idx in df.index else ""
                        ),
                        "Evaluation_Error": f"Parallel processing error: {e}",
                    }
                    # Fill in empty values for all score columns
                    for col in FORMAT_CLARITY_OUTPUT_COLUMNS:
                        if col not in result_row:
                            result_row[col] = ""
                    results_dict[idx] = result_row

        # Convert results_dict to ordered list
        results = [results_dict[idx] for idx, _ in rows_with_index]

    # Write results to CSV
    write_results_to_csv_format_clarity(
        results, output_file, FORMAT_CLARITY_OUTPUT_COLUMNS
    )


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Format Clarity Evaluator - Compare formatting similarity between two models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/format_clarity_evaluator.py examples/sample_input_format_clarity.csv
    python scripts/format_clarity_evaluator.py input.csv -o output/format_clarity_output.csv
    python scripts/format_clarity_evaluator.py input.csv -n 5 -o output/test_results.csv  # Test with first 5 rows

Input CSV Format:
    - Header row optional
    - Column A: Question
    - Column B: Model A answer (Full raw ReAct log)
    - Column C: Model B answer (Full raw ReAct log)

Setup for Azure OpenAI:
    1. Install dependencies: pip install -r requirements.txt
    2. Set environment variables:
       export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'
       export AZURE_OPENAI_API_KEY='your-api-key'
       export MODEL_NAME='gpt-4-turbo'  # or 'gpt-5', 'gpt-4.1'
       export AZURE_OPENAI_API_VERSION='2024-08-01-preview'  # optional, defaults to this
    3. Run the script with your input CSV file:
       python scripts/format_clarity_evaluator.py input.csv -m gpt-5  # Use -m to specify model

Setup for Standard OpenAI:
    1. Install dependencies: pip install -r requirements.txt
    2. Set API key: export OPENAI_API_KEY='your-api-key-here'
    3. Optionally set model: export MODEL_NAME='gpt-4-turbo'
    4. Run the script with your input CSV file:
       python scripts/format_clarity_evaluator.py input.csv -m gpt-4-turbo  # Use -m to specify model

Supported Models:
    - gpt-5: GPT-5 (uses max_completion_tokens, temperature=1.0)
    - gpt-4.1: GPT-4.1 (uses max_tokens, temperature=0.7)
    - gpt-4-turbo: GPT-4 Turbo (uses max_tokens, temperature=0.7) [default]

You can specify the model via:
    - Command line: -m gpt-5 or --model gpt-4-turbo
    - Environment variable: export MODEL_NAME='gpt-5'
    - Default: gpt-4-turbo

How It Works:
    1. Reads the input CSV (no header row)
    2. Parses each raw ReAct log to extract only the "Final Answer" text
    3. Calls GPT-5 (or GPT-4-turbo) to score format similarity (1-5 scale)
    4. Writes results to output CSV with parsed answers and scores
        """,
    )

    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file (header row optional, columns: Question, Model_A_Response, Model_B_Response)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="format_clarity_output.csv",
        help="Path to the output CSV file (default: format_clarity_output.csv). Note: It's recommended to use output/ directory (e.g., output/format_clarity_output.csv)",
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
        model_name_normalized = args.model.lower().strip()
        for supported_model in SUPPORTED_MODELS:
            if (
                supported_model.lower() == model_name_normalized
                or supported_model.replace("-", "").lower()
                == model_name_normalized.replace("-", "")
            ):
                args.model = supported_model
                break

    log_section("Format Clarity Evaluator - LLM-as-a-Judge")
    log_info("Comparing Model A formatting against Model B (golden standard)")

    # Determine model name
    model_name = args.model or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    if model_name:
        log_info(f"Using model: {model_name}")

    process_csv(
        args.input_csv,
        args.output,
        limit_rows=args.limit,
        model_name=model_name,
        non_interactive=args.yes,
    )


if __name__ == "__main__":
    main()
