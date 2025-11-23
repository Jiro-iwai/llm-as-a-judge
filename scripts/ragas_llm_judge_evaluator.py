#!/usr/bin/env python3
"""
Ragas-Based Evaluation Pipeline for ReAct Chatbot Responses

This script uses the Ragas framework to evaluate two models' responses by automatically
parsing ReAct logs to extract Final Answers and Contexts, then running multiple Ragas
metrics (faithfulness, answer_relevance, context_precision, context_recall) to measure
how grounded and relevant the answers are with respect to the retrieved context and
original question.

This metric doesn't require ground truth, making it suitable for evaluation when
reference answers are not available.

Usage:
    python scripts/ragas_llm_judge_evaluator.py <input_csv_file>

Requirements:
    - Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and MODEL_NAME environment variables
    - Input CSV must have 3 columns with header: Question, Model_A_Full_Log, Model_B_Full_Log
    - Install dependencies: pip install -r requirements.txt (ensure ragas and datasets are included)

Output:
    - ragas_evaluation_output.csv containing all parsed data and Ragas scores
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402
from datasets import Dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import AzureOpenAI  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # type: ignore[attr-defined] # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Supported Ragas metrics (extendable)
AVAILABLE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevance": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

BASIC_METRICS = ["faithfulness", "answer_relevance"]
# Note: context_precision and context_recall require 'reference' column (ground truth)
# These metrics are only available when reference answers are provided
METRICS_WITH_REFERENCE = [
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
]
METRIC_PRESETS = {
    "basic": BASIC_METRICS,
    "with_reference": METRICS_WITH_REFERENCE,  # Requires reference column (ground truth)
}

# Default to basic metrics since we don't have ground truth
DEFAULT_METRICS = tuple(BASIC_METRICS)


def resolve_metrics(metrics: Optional[List[str]], preset: Optional[str]) -> List[str]:
    """
    Determine which metrics to run based on explicit metrics or preset.
    """
    if metrics:
        return list(metrics)
    if preset:
        return list(METRIC_PRESETS[preset])
    return list(DEFAULT_METRICS)


from tqdm import tqdm  # noqa: E402

if TYPE_CHECKING:
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  # noqa: E402

from src.config.model_configs import (  # noqa: E402
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    get_ragas_config as get_model_config_from_common,
)
from src.utils.logging_config import (  # noqa: E402
    log_info,
    log_error,
    log_warning,
    log_success,
    log_section,
    setup_logging,
)

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging system
setup_logging()


def parse_react_log(log_text: str) -> Tuple[str, List[str]]:
    """
    Parse a ReAct log string to extract the Final Answer and Contexts.

    The ReAct log format includes multiple sections, all of which provide context:
    - ## 📝 Task タスク - Task classification
    - ## 💬 Reaction 反応 - Reaction type
    - ## 📂 Classification 分類 - Classification
    - ## 📊 Status 状態 - Status
    - ## 🤖 LLM Thought Process 思考 - The LLM's reasoning
    - ## ⚡ Action 行動 - Action taken
    - ## ⌨️ Action Input 行動入力 - Action input
    - ## 📚 Raw Search Results (Cleaned) 観察 - Retrieved search results
    - ## ✅ Final Answer 回答 - The final answer (not included in contexts)

    Args:
        log_text: The full ReAct log string

    Returns:
        Tuple of (final_answer, contexts_list)
        - final_answer: The extracted final answer text
        - contexts_list: List of context strings (includes thought process + search results)
    """
    # Extract Final Answer
    final_answer = ""
    final_answer_pattern = r"## ✅ Final Answer 回答\s*---\s*(.*?)(?=\n## |$)"
    final_answer_match = re.search(final_answer_pattern, log_text, re.DOTALL)

    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
    else:
        # Fallback: try without the dashes
        final_answer_pattern_alt = r"## ✅ Final Answer 回答\s*(.*?)(?=\n## |$)"
        final_answer_match_alt = re.search(
            final_answer_pattern_alt, log_text, re.DOTALL
        )
        if final_answer_match_alt:
            final_answer = final_answer_match_alt.group(1).strip()

    # Extract all ReAct components as contexts
    contexts = []

    # Helper function to extract a section
    def extract_section(pattern_name: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, log_text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content and content.lower() not in ["none", "n/a", ""]:
                return f"[{pattern_name}]\n{content}"
        return None

    # 1. Extract Task
    task = extract_section("Task", r"## 📝 Task タスク\s*---\s*(.*?)(?=\n## |$)")
    if task:
        contexts.append(task)

    # 2. Extract Classification
    classification = extract_section(
        "Classification", r"## 📂 Classification 分類\s*---\s*(.*?)(?=\n## |$)"
    )
    if classification:
        contexts.append(classification)

    # 3. Extract Status
    status = extract_section("Status", r"## 📊 Status 状態\s*---\s*(.*?)(?=\n## |$)")
    if status:
        contexts.append(status)

    # 4. Extract LLM Thought Process (MOST IMPORTANT)
    thought_process = extract_section(
        "LLM Thought Process",
        r"## 🤖 LLM Thought Process 思考\s*---\s*(.*?)(?=\n## |$)",
    )
    if thought_process:
        contexts.append(thought_process)

    # 5. Extract Action
    action = extract_section("Action", r"## ⚡ Action 行動\s*---\s*(.*?)(?=\n## |$)")
    if action:
        contexts.append(action)

    # 6. Extract Action Input
    action_input = extract_section(
        "Action Input", r"## ⌨️ Action Input 行動入力\s*---\s*(.*?)(?=\n## |$)"
    )
    if action_input:
        contexts.append(action_input)

    # 7. Extract Raw Search Results (can have multiple results)
    search_results_pattern = r"## 📚 Raw Search Results \(Cleaned\) 観察\s*---\s*(.*?)(?=\n## ✅ Final Answer|$)"
    search_results_match = re.search(search_results_pattern, log_text, re.DOTALL)

    if search_results_match:
        search_results_text = search_results_match.group(1)

        # Split by the separator (################################################)
        result_separator = r"#{40,}"
        individual_results = re.split(result_separator, search_results_text)

        # Clean and collect non-empty results
        for idx, result in enumerate(individual_results, 1):
            result_clean = result.strip()
            if (
                result_clean and len(result_clean) > 20
            ):  # Filter out very short/empty results
                # Remove the URLs section from each result if present
                result_without_urls = re.sub(
                    r"## 🔗 URLs URL.*$", "", result_clean, flags=re.DOTALL
                )
                result_without_urls = result_without_urls.strip()
                if result_without_urls:
                    contexts.append(f"[Search Result {idx}]\n{result_without_urls}")

    # If no contexts found, return a placeholder to avoid Ragas errors
    if not contexts:
        contexts = ["No context retrieved"]

    # If no final answer found, use a placeholder
    if not final_answer:
        final_answer = "No answer provided"

    return final_answer, contexts


# Model configuration is now imported from config.model_configs (imported at top)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model (Ragas-compatible)."""
    return get_model_config_from_common(model_name)


def initialize_azure_openai_for_ragas(
    model_name: Optional[str] = None,
) -> Tuple["AzureChatOpenAI", AzureOpenAI, "AzureOpenAIEmbeddings"]:
    """
    Initialize Azure OpenAI client and wrap it for Ragas.

    Args:
        model_name: Optional model name. If None, uses environment variable or default.

    Returns:
        Tuple of (ragas_llm, client, embeddings) where:
        - ragas_llm is the LangChain LLM for Ragas
        - client is the Azure OpenAI client
        - embeddings is the Azure OpenAI Embeddings for Ragas
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    if model_name is None:
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)

    # For embeddings, deployment name is required (chat models cannot be used for embeddings)
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    if not embedding_deployment:
        log_error("Azure OpenAI Embeddings deployment name not found.")
        log_error("\nRagas evaluation requires an embeddings deployment.")
        log_error(
            "Chat models (like gpt-4.1) cannot be used for embeddings operations."
        )
        log_error("\nPlease set the following environment variable in your .env file:")
        log_error("  AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002")
        log_error("  # Replace with your actual embedding deployment name")
        log_error("\nOptional:")
        log_error("  AZURE_OPENAI_EMBEDDING_MODEL_NAME=text-embedding-ada-002")
        sys.exit(1)

    # Embedding model name (optional, defaults to deployment name if not set)
    embedding_model_name = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
    if not embedding_model_name:
        # If not set, use deployment name as model name (common case)
        embedding_model_name = embedding_deployment

    if not azure_endpoint or not azure_api_key:
        log_error("Azure OpenAI credentials not found.")
        log_error("\nPlease set the following environment variables:")
        log_error(
            "  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'"
        )
        log_error("  export AZURE_OPENAI_API_KEY='your-api-key'")
        log_error(
            "  export MODEL_NAME='gpt-5'  # or your deployment name (e.g., 'gpt-4.1')"
        )
        log_error("\nOptional (for Ragas embeddings):")
        log_error(
            "  export AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME='text-embedding-ada-002'"
        )
        log_error("  export AZURE_OPENAI_EMBEDDING_MODEL_NAME='text-embedding-ada-002'")
        sys.exit(1)

    log_info("Initializing Azure OpenAI for Ragas evaluation")
    log_info(f"Endpoint: {azure_endpoint}")
    log_info(f"Model/Deployment: {model_name}")
    log_info(f"API Version: {azure_api_version}")

    # Create Azure OpenAI client for direct API access (if needed)
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version,
    )

    # Create LangChain-compatible LLM for Ragas
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

    # Configure parameters based on model type

    llm_params: Dict[str, Any] = {
        "azure_endpoint": azure_endpoint,
        "api_key": azure_api_key,
        "api_version": azure_api_version,
        "deployment_name": model_name,
        "model_name": model_name,
    }

    # Get model configuration
    model_config = get_model_config(model_name)

    # Set parameters based on model configuration
    if model_config["use_max_completion_tokens"]:
        llm_params["max_completion_tokens"] = model_config["max_completion_tokens"]
    else:
        llm_params["max_tokens"] = model_config["max_tokens"]

    llm_params["temperature"] = model_config["temperature"]

    token_param = (
        "max_completion_tokens"
        if model_config["use_max_completion_tokens"]
        else "max_tokens"
    )
    token_value = model_config.get("max_completion_tokens") or model_config.get(
        "max_tokens"
    )
    log_info(
        f"LLM Configuration: temperature={llm_params['temperature']}, {token_param}={token_value}"
    )

    langchain_llm = AzureChatOpenAI(**llm_params)

    # Create Azure OpenAI Embeddings for Ragas
    # Note: Chat models (like gpt-4.1) cannot be used for embeddings operations
    # embedding_deployment is already validated above (must be set via environment variable)
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,  # type: ignore[arg-type]
        api_version=azure_api_version,
        azure_deployment=embedding_deployment,
        model=embedding_model_name,
    )
    log_info(
        f"Embeddings Model: {embedding_model_name} (Deployment: {embedding_deployment})"
    )

    return langchain_llm, client, embeddings


def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    llm: "AzureChatOpenAI",
    model_name: str = "Model",
    metric_names: Optional[Sequence[str]] = None,
    references: Optional[List[str]] = None,
    embeddings: Optional["AzureOpenAIEmbeddings"] = None,
) -> pd.DataFrame:
    """
    Evaluate responses using Ragas metrics.

    Args:
        questions: List of questions
        answers: List of answers
        contexts_list: List of context lists (each answer has multiple contexts)
        llm: The LLM to use for evaluation (LangChain-compatible)
        model_name: Name for labeling (e.g., "Model_A" or "Model_B")
        metric_names: List of metrics to evaluate
        references: Optional list of reference answers (ground truth) for metrics that require it
        embeddings: Optional Azure OpenAI Embeddings instance (required for Azure OpenAI)

    Returns:
        DataFrame with Ragas scores for the requested metrics.
    """
    metrics_to_run = list(metric_names) if metric_names else list(DEFAULT_METRICS)
    invalid_metrics = [
        metric for metric in metrics_to_run if metric not in AVAILABLE_METRICS
    ]
    if invalid_metrics:
        raise ValueError(f"Unsupported metrics requested: {', '.join(invalid_metrics)}")

    # Check for metrics that require reference column
    metrics_requiring_reference = ["context_precision", "context_recall"]
    requested_metrics_requiring_reference = [
        m for m in metrics_to_run if m in metrics_requiring_reference
    ]

    if requested_metrics_requiring_reference:
        if references is None:
            log_warning(
                f"The following metrics require 'reference' column (ground truth): {', '.join(requested_metrics_requiring_reference)}"
            )
            log_warning(
                "Skipping these metrics. Use --metrics-preset basic or specify metrics that don't require reference."
            )
            # Remove metrics that require reference
            metrics_to_run = [
                m for m in metrics_to_run if m not in metrics_requiring_reference
            ]
            if not metrics_to_run:
                log_error(
                    "No valid metrics remaining after removing metrics requiring reference."
                )
                # Return empty DataFrame with expected structure
                empty_data: Dict[str, List[Any]] = {"question": questions}
                return pd.DataFrame(empty_data)

    log_info(
        f"\nEvaluating {model_name} with Ragas metrics: {', '.join(metrics_to_run)}"
    )

    # Prepare dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }

    # Add reference column if provided and needed
    if references is not None:
        data["reference"] = references

    dataset = Dataset.from_dict(data)

    # Define metrics to evaluate (only those that don't require reference or have reference)
    metrics_to_use = [AVAILABLE_METRICS[metric] for metric in metrics_to_run]

    # Run evaluation
    try:
        # Pass embeddings explicitly if provided (required for Azure OpenAI)
        evaluate_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "metrics": metrics_to_use,
            "llm": llm,
        }
        if embeddings is not None:
            evaluate_kwargs["embeddings"] = embeddings

        results = evaluate(**evaluate_kwargs)

        # Convert results to DataFrame
        # Type checker may not recognize to_pandas() method, but it exists
        results_df: pd.DataFrame = results.to_pandas()  # type: ignore[attr-defined]

        # Ragas returns column names based on metric object names (e.g., "answer_relevancy")
        # but we use normalized names (e.g., "answer_relevance") in our code
        # Map Ragas column names to our normalized names
        ragas_to_normalized = {
            "answer_relevancy": "answer_relevance",
            "faithfulness": "faithfulness",
            "context_precision": "context_precision",
            "context_recall": "context_recall",
        }

        # First, rename Ragas column names to normalized names
        column_mapping = {}
        for col in results_df.columns:
            if col in ragas_to_normalized:
                column_mapping[col] = ragas_to_normalized[col]
        if column_mapping:
            results_df = results_df.rename(columns=column_mapping)

        # Then rename columns with model prefix
        score_columns = {
            metric: f"{model_name}_{metric}_score" for metric in metrics_to_run
        }

        results_df = results_df.rename(columns=score_columns)

        return results_df

    except (ValueError, KeyError, AttributeError, Exception) as e:
        # エラー内容を詳細に記録し、期待される列構造を維持したDataFrameを返す
        log_error(
            f"ERROR during Ragas evaluation for {model_name}: {type(e).__name__}: {e}"
        )
        log_error("Traceback:")
        import traceback

        traceback.print_exc(file=sys.stderr)
        error_data: Dict[str, List[Any]] = {"question": questions}
        for metric in metrics_to_run:
            error_data[f"{model_name}_{metric}_score"] = [None] * len(questions)
        return pd.DataFrame(error_data)


def read_and_validate_csv_ragas(input_file: str) -> pd.DataFrame:
    """
    Read and validate input CSV file for ragas_llm_judge_evaluator.

    Supports Model_A_Full_Log/Model_B_Full_Log column names.

    Args:
        input_file: Path to the input CSV file

    Returns:
        DataFrame with standardized column names

    Raises:
        SystemExit: If file cannot be read or validated
    """
    log_info(f"\nReading input file: {input_file}")
    try:
        # Try to detect if there's a header row by reading first line
        with open(input_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().lower()

        # If first line looks like headers, use it as header
        if any(
            keyword in first_line
            for keyword in ["question", "model", "answer", "response"]
        ):
            log_warning("Detected header row in input CSV.")
            df = pd.read_csv(input_file)
            # Allow both Response and Full_Log naming
            if "Model_A_Full_Log" in df.columns:
                df = df.rename(
                    columns={
                        "Model_A_Full_Log": "Model_A_Response",
                        "Model_B_Full_Log": "Model_B_Response",
                    }
                )
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


def apply_row_limit_ragas(df: pd.DataFrame, limit_rows: Optional[int]) -> pd.DataFrame:
    """
    Apply row limit if specified (ragas_llm_judge_evaluator version).

    Args:
        df: Input DataFrame
        limit_rows: Optional limit on number of rows to process

    Returns:
        DataFrame with row limit applied
    """
    if limit_rows is not None and limit_rows < len(df):
        df = df.head(limit_rows)
        log_warning(f"LIMITING to first {limit_rows} rows for testing")

    return df


def parse_react_logs_for_both_models(
    df: pd.DataFrame,
) -> tuple[List[str], List[List[str]], List[str], List[List[str]]]:
    """
    Parse ReAct logs for both Model A and Model B.

    Args:
        df: DataFrame with Model_A_Response and Model_B_Response columns

    Returns:
        Tuple of (model_a_answers, model_a_contexts, model_b_answers, model_b_contexts)
    """
    log_info("\nParsing ReAct logs...")
    model_a_answers = []
    model_a_contexts = []
    model_b_answers = []
    model_b_contexts = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing logs"):
        # Convert pandas Series to str if needed
        model_a_val = row["Model_A_Response"]
        model_b_val = row["Model_B_Response"]

        # Use isinstance check to avoid Series condition operator error
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

        # Parse Model A log
        a_answer, a_contexts = parse_react_log(model_a_response)
        model_a_answers.append(a_answer)
        model_a_contexts.append(a_contexts)

        # Parse Model B log
        b_answer, b_contexts = parse_react_log(model_b_response)
        model_b_answers.append(b_answer)
        model_b_contexts.append(b_contexts)

    log_success(f"Parsed {len(df)} ReAct logs for both models")
    return model_a_answers, model_a_contexts, model_b_answers, model_b_contexts


def evaluate_both_models_with_ragas(
    df: pd.DataFrame,
    model_a_answers: List[str],
    model_a_contexts: List[List[str]],
    model_b_answers: List[str],
    model_b_contexts: List[List[str]],
    llm: "AzureChatOpenAI",
    metrics_to_run: List[str],
    embeddings: Optional["AzureOpenAIEmbeddings"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate both Model A and Model B with Ragas.

    Args:
        df: DataFrame with questions
        model_a_answers: List of answers for Model A
        model_a_contexts: List of contexts for Model A
        model_b_answers: List of answers for Model B
        model_b_contexts: List of contexts for Model B
        llm: AzureChatOpenAI instance
        metrics_to_run: List of metric names to compute
        embeddings: AzureOpenAIEmbeddings instance

    Returns:
        Tuple of (model_a_results, model_b_results) DataFrames
    """
    # Evaluate Model A with Ragas
    log_section("EVALUATING MODEL A")
    model_a_results = evaluate_with_ragas(
        questions=df["Question"].tolist(),
        answers=model_a_answers,
        contexts_list=model_a_contexts,
        llm=llm,
        model_name="Model_A",
        metric_names=metrics_to_run,
        embeddings=embeddings,
    )

    # Evaluate Model B with Ragas
    log_section("EVALUATING MODEL B")
    model_b_results = evaluate_with_ragas(
        questions=df["Question"].tolist(),
        answers=model_b_answers,
        contexts_list=model_b_contexts,
        llm=llm,
        model_name="Model_B",
        metric_names=metrics_to_run,
        embeddings=embeddings,
    )

    # model_a_results and model_b_results are guaranteed to be DataFrames (not None)
    assert model_a_results is not None, "Model A evaluation failed"
    assert model_b_results is not None, "Model B evaluation failed"

    return model_a_results, model_b_results


def merge_ragas_results_and_write(
    df: pd.DataFrame,
    model_a_results: pd.DataFrame,
    model_b_results: pd.DataFrame,
    model_a_answers: List[str],
    model_a_contexts: List[List[str]],
    model_b_answers: List[str],
    model_b_contexts: List[List[str]],
    metrics_to_run: List[str],
    output_file: str,
) -> pd.DataFrame:
    """
    Merge Ragas evaluation results and write to CSV file.

    Args:
        df: Original DataFrame
        model_a_results: DataFrame with Model A evaluation results
        model_b_results: DataFrame with Model B evaluation results
        model_a_answers: List of answers for Model A
        model_a_contexts: List of contexts for Model A
        model_b_answers: List of answers for Model B
        model_b_contexts: List of contexts for Model B
        metrics_to_run: List of metric names
        output_file: Path to output CSV file

    Returns:
        Output DataFrame
    """
    # Add parsed data to DataFrame
    df["model_A_answer"] = model_a_answers
    df["model_A_contexts"] = model_a_contexts
    df["model_B_answer"] = model_b_answers
    df["model_B_contexts"] = model_b_contexts

    # Merge results back into main DataFrame
    # Only merge the score columns (not the duplicated question/answer/contexts columns)
    score_columns_a = [
        col for col in model_a_results.columns if col.startswith("Model_A_")
    ]
    score_columns_b = [
        col for col in model_b_results.columns if col.startswith("Model_B_")
    ]

    for col in score_columns_a:
        df[col] = model_a_results[col].values

    for col in score_columns_b:
        df[col] = model_b_results[col].values

    # Prepare final output DataFrame with all columns
    output_columns = (
        [
            "Question",
            "Model_A_Response",
            "Model_B_Response",
            "model_A_answer",
            "model_A_contexts",
            "model_B_answer",
            "model_B_contexts",
        ]
        + score_columns_a
        + score_columns_b
    )

    output_df = pd.DataFrame(df[output_columns])

    # Save to CSV
    output_path = Path(output_file)
    if output_path.parent and output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    log_section("✓ EVALUATION COMPLETE!")
    log_success(f"Results written to: {output_path}")
    log_success(f"Processed {len(output_df)} rows")

    # Print summary statistics
    log_section("SUMMARY STATISTICS")

    for display_name, prefix in [("Model A", "Model_A_"), ("Model B", "Model_B_")]:
        log_info(f"\n{display_name}:")
        for metric in metrics_to_run:
            col_name = f"{prefix}{metric}_score"
            if col_name in output_df.columns:
                mean_score = output_df[col_name].mean()
                log_info(f"  {metric:<18}: {mean_score:.4f}")

    return output_df


def process_csv(
    input_file: str,
    output_file: str = "output/ragas_evaluation_output.csv",
    limit_rows: Optional[int] = None,
    model_name: Optional[str] = None,
    metric_names: Optional[List[str]] = None,
) -> None:
    """
    Main processing function that reads the input CSV, parses ReAct logs,
    evaluates with Ragas, and writes results to output CSV.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        limit_rows: Optional limit on number of rows to process
        model_name: Optional model name. If None, uses environment variable or default.
        metric_names: List of Ragas metrics to compute (defaults to preset/CLI selection)
    """
    # Initialize Azure OpenAI for Ragas
    llm, client, embeddings = initialize_azure_openai_for_ragas(model_name=model_name)
    metrics_to_run: List[str] = (
        list(metric_names) if metric_names else list(DEFAULT_METRICS)
    )

    # Read and validate CSV
    df = read_and_validate_csv_ragas(input_file)

    # Apply row limit if specified
    df = apply_row_limit_ragas(df, limit_rows)

    # Parse ReAct logs for both models
    model_a_answers, model_a_contexts, model_b_answers, model_b_contexts = (
        parse_react_logs_for_both_models(df)
    )

    # Evaluate both models with Ragas
    model_a_results, model_b_results = evaluate_both_models_with_ragas(
        df,
        model_a_answers,
        model_a_contexts,
        model_b_answers,
        model_b_contexts,
        llm,
        metrics_to_run,
        embeddings,
    )

    # Merge results and write to CSV
    merge_ragas_results_and_write(
        df,
        model_a_results,
        model_b_results,
        model_a_answers,
        model_a_contexts,
        model_b_answers,
        model_b_contexts,
        metrics_to_run,
        output_file,
    )


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Ragas-Based Evaluation Pipeline for ReAct Chatbot Responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/ragas_llm_judge_evaluator.py examples/sample_input_ragas.csv
    python scripts/ragas_llm_judge_evaluator.py input.csv -o output/ragas_evaluation_output.csv
    python scripts/ragas_llm_judge_evaluator.py input.csv -n 3 -o output/test_results.csv  # Test with 3 rows only

Setup:
    1. Install dependencies:
       pip install -r requirements.txt
       
    2. Ensure requirements.txt includes:
       - ragas
       - datasets
       - langchain-openai
       - openai
       - pandas
       - python-dotenv
       - tqdm
       
    3. Set environment variables:
       export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'
       export AZURE_OPENAI_API_KEY='your-api-key'
       export MODEL_NAME='gpt-4.1'  # or 'gpt-5', 'gpt-4-turbo' (recommended: gpt-4.1 for Ragas)
       export AZURE_OPENAI_API_VERSION='2024-08-01-preview'  # optional
       
       # REQUIRED for Ragas evaluation (chat models cannot be used for embeddings):
       export AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME='text-embedding-ada-002'
       # Optional (defaults to deployment name if not set):
       export AZURE_OPENAI_EMBEDDING_MODEL_NAME='text-embedding-ada-002'
       
       # Or add to .env file:
       # AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
       # AZURE_OPENAI_EMBEDDING_MODEL_NAME=text-embedding-ada-002  # Optional
       
    4. Run the script:
       python scripts/ragas_llm_judge_evaluator.py input.csv -m gpt-4.1  # Use -m to specify model
       
    5. Prepare input CSV with columns:
       - Question
       - Model_A_Response (or Model_A_Full_Log)
       - Model_B_Response (or Model_B_Full_Log)
       
Input Format:
    The script expects ReAct logs with the following structure:
    - ## ✅ Final Answer 回答 section containing the final answer
    - ## 📚 Raw Search Results (Cleaned) 観察 section with search results
    - Results separated by ################################################
    
Output:
    A CSV file (output/ragas_evaluation_output.csv by default) containing:
    - Original columns (Question, Model_A_Response, Model_B_Response)
    - Parsed columns (model_A_answer, model_A_contexts, etc.)
    - Ragas scores for both models (configurable via --metrics / --metrics-preset):
      * Model_A_faithfulness_score / Model_B_faithfulness_score
      * Model_A_answer_relevance_score / Model_B_answer_relevance_score
      * Model_A_context_precision_score / Model_B_context_precision_score
      * Model_A_context_recall_score / Model_B_context_recall_score
        """,
    )

    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file (header row optional, columns: Question, Model_A_Response, Model_B_Response)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output/ragas_evaluation_output.csv",
        help="Path to the output CSV file (default: output/ragas_evaluation_output.csv)",
    )

    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N rows (useful for testing)",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=f"Model to use for evaluation (default: {DEFAULT_MODEL}). Supported models: {', '.join(SUPPORTED_MODELS)}",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(AVAILABLE_METRICS.keys()),
        default=None,
        help=(
            "List of Ragas metrics to compute. "
            f"Choices: {', '.join(sorted(AVAILABLE_METRICS.keys()))}. "
            f"Default: {', '.join(DEFAULT_METRICS)}"
        ),
    )

    parser.add_argument(
        "--metrics-preset",
        choices=sorted(METRIC_PRESETS.keys()),
        default=None,
        help=(
            "Preset of Ragas metrics to compute. "
            "basic: faithfulness + answer_relevance (default, no ground truth required). "
            "with_reference: full metric set including context_precision and context_recall (requires reference column/ground truth). "
            "Ignored when --metrics is provided."
        ),
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and run non-interactively (useful for CI/batch execution). Note: This script does not show confirmation prompts, but this flag is accepted for consistency with other evaluators.",
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

    log_section("Ragas-Based Evaluation Pipeline for ReAct Chatbot Responses")

    # Determine model name
    model_name = args.model or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    if model_name:
        log_info(f"Using model: {model_name}")

    if args.metrics and args.metrics_preset:
        log_warning(
            "--metrics overrides --metrics-preset; ignoring the preset selection."
        )
    selected_metrics = resolve_metrics(args.metrics, args.metrics_preset)
    log_info(f"Using Ragas metrics: {', '.join(selected_metrics)}")

    process_csv(
        args.input_csv,
        args.output,
        limit_rows=args.limit,
        model_name=model_name,
        metric_names=selected_metrics,
    )


if __name__ == "__main__":
    main()
