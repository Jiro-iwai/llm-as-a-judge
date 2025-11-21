#!/usr/bin/env python3
"""
Full Pipeline Script

This script runs the complete pipeline:
1. collect_responses.py - Collect responses from API
2. Evaluation script (llm_judge_evaluator.py, ragas_llm_judge_evaluator.py, or format_clarity_evaluator.py)
3. visualize_results.py - Visualize evaluation results

Usage:
    python run_full_pipeline.py questions.txt
    python run_full_pipeline.py questions.txt --evaluator llm-judge
    python run_full_pipeline.py questions.txt --evaluator ragas --skip-collect
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from utils.logging_config import (
    log_error,
    log_info,
    log_section,
    log_success,
    log_warning,
    setup_logging,
)

# Set up logging system
setup_logging()

# Default file names
DEFAULT_COLLECT_OUTPUT = "collected_responses.csv"
DEFAULT_LLM_JUDGE_OUTPUT = "evaluation_output.csv"
DEFAULT_RAGAS_OUTPUT = "ragas_evaluation_output.csv"
DEFAULT_FORMAT_CLARITY_OUTPUT = "format_clarity_output.csv"


def run_collect_step(
    questions_file: str,
    output_file: str,
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Run collect_responses.py to collect responses from API.

    Args:
        questions_file: Path to questions file
        output_file: Output CSV file path
        model_a: Model A name (optional, defaults to claude3.5-sonnet)
        model_b: Model B name (optional, defaults to claude4.5-haiku)
        api_url: API URL (optional)
        **kwargs: Additional arguments to pass to collect_responses.py

    Returns:
        Tuple of (success: bool, actual_model_a: str, actual_model_b: str)
        Returns the actual model names used (defaults if not specified)
    """
    # Default model names (matching collect_responses.py defaults)
    DEFAULT_MODEL_A = "claude3.5-sonnet"
    DEFAULT_MODEL_B = "claude4.5-haiku"

    actual_model_a = model_a or DEFAULT_MODEL_A
    actual_model_b = model_b or DEFAULT_MODEL_B

    log_section("Step 1: Collecting Responses")
    log_info(f"Input file: {questions_file}")
    log_info(f"Output file: {output_file}")
    log_info(f"Model A: {actual_model_a}")
    log_info(f"Model B: {actual_model_b}")
    log_info("")  # Empty line for readability

    cmd = [sys.executable, "collect_responses.py", questions_file, "-o", output_file]

    # Always pass model names (use defaults if not specified)
    cmd.extend(["--model-a", actual_model_a])
    cmd.extend(["--model-b", actual_model_b])

    if api_url:
        cmd.extend(["--api-url", api_url])

    # Add any additional kwargs as command-line arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    try:
        # Run without capturing output so progress is visible in real-time
        subprocess.run(cmd, check=True)
        log_success("Response collection completed successfully")
        return True, actual_model_a, actual_model_b
    except subprocess.CalledProcessError as e:
        log_error(f"Response collection failed with exit code {e.returncode}")
        return False, actual_model_a, actual_model_b
    except FileNotFoundError:
        log_error("collect_responses.py not found")
        return False, actual_model_a, actual_model_b


def run_evaluation_step(
    evaluator: str,
    input_file: str,
    limit: Optional[int] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> Tuple[bool, str]:
    """
    Run evaluation script.

    Args:
        evaluator: Evaluator type ('llm-judge', 'ragas', 'format-clarity', or 'all')
        input_file: Input CSV file path
        limit: Limit number of rows to process (optional)
        model_name: Model name for evaluation (optional)
        **kwargs: Additional arguments to pass to evaluation script

    Returns:
        Tuple of (success: bool, output_file: str)
    """
    evaluators = {
        "llm-judge": {
            "script": "llm_judge_evaluator.py",
            "output": DEFAULT_LLM_JUDGE_OUTPUT,
        },
        "ragas": {
            "script": "ragas_llm_judge_evaluator.py",
            "output": DEFAULT_RAGAS_OUTPUT,
        },
        "format-clarity": {
            "script": "format_clarity_evaluator.py",
            "output": DEFAULT_FORMAT_CLARITY_OUTPUT,
        },
    }

    if evaluator == "all":
        # Run all evaluators
        results = []
        for eval_name, eval_config in evaluators.items():
            log_section(f"Step 2: Running {eval_name} Evaluation")
            success, output_file = run_single_evaluation(
                eval_name,
                eval_config["script"],
                eval_config["output"],
                input_file,
                limit,
                model_name,
                **kwargs,
            )
            results.append((success, output_file))
            if not success:
                return False, ""
        # Return success if all passed, output file from llm-judge for visualization
        return all(success for success, _ in results), DEFAULT_LLM_JUDGE_OUTPUT
    else:
        if evaluator not in evaluators:
            log_error(f"Invalid evaluator: {evaluator}")
            log_info(f"Valid options: {', '.join(evaluators.keys())}, all")
            return False, ""

        eval_config = evaluators[evaluator]
        log_section(f"Step 2: Running {evaluator} Evaluation")
        return run_single_evaluation(
            evaluator,
            eval_config["script"],
            eval_config["output"],
            input_file,
            limit,
            model_name,
            **kwargs,
        )


def run_single_evaluation(
    evaluator_name: str,
    script_name: str,
    output_file: str,
    input_file: str,
    limit: Optional[int] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> Tuple[bool, str]:
    """
    Run a single evaluation script.

    Args:
        evaluator_name: Name of the evaluator (for logging)
        script_name: Name of the evaluation script
        output_file: Output CSV file path
        input_file: Input CSV file path
        limit: Limit number of rows to process (optional)
        model_name: Model name for evaluation (optional)
        **kwargs: Additional arguments

    Returns:
        Tuple of (success: bool, output_file: str)
    """
    log_info(f"Evaluator: {evaluator_name}")
    log_info(f"Script: {script_name}")
    log_info(f"Input file: {input_file}")
    log_info(f"Output file: {output_file}")
    log_info("")  # Empty line for readability

    cmd = [sys.executable, script_name, input_file, "-o", output_file]

    if limit:
        cmd.extend(["-n", str(limit)])
    if model_name:
        cmd.extend(["-m", model_name])

    # Always add --yes flag for non-interactive execution from pipeline
    cmd.append("--yes")

    # Add any additional kwargs as command-line arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    try:
        # Run without capturing output so progress is visible in real-time
        subprocess.run(cmd, check=True)
        log_success(f"{evaluator_name} evaluation completed successfully")
        return True, output_file
    except subprocess.CalledProcessError as e:
        log_error(f"{evaluator_name} evaluation failed with exit code {e.returncode}")
        return False, ""
    except FileNotFoundError:
        log_error(f"{script_name} not found")
        return False, ""


def run_visualize_step(
    input_file: str,
    model_a_name: Optional[str] = None,
    model_b_name: Optional[str] = None,
) -> bool:
    """
    Run visualize_results.py to visualize evaluation results.

    Args:
        input_file: Input CSV file path (evaluation output)
        model_a_name: Model A name for visualization (optional)
        model_b_name: Model B name for visualization (optional)

    Returns:
        True if successful, False otherwise
    """
    log_section("Step 3: Visualizing Results")
    log_info(f"Input file: {input_file}")
    log_info("")  # Empty line for readability

    cmd = [sys.executable, "visualize_results.py", input_file]

    if model_a_name:
        cmd.extend(["--model-a", model_a_name])
    if model_b_name:
        cmd.extend(["--model-b", model_b_name])

    try:
        # Run without capturing output so progress is visible in real-time
        subprocess.run(cmd, check=True)
        log_success("Visualization completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        log_warning(f"Visualization failed with exit code {e.returncode}")
        # Visualization failure is not critical, so we continue
        return False
    except FileNotFoundError:
        log_warning("visualize_results.py not found")
        return False


def main():
    """Main entry point for the pipeline script"""
    parser = argparse.ArgumentParser(
        description="Run full pipeline: collect responses, evaluate, and visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run pipeline with default llm-judge evaluator
    python run_full_pipeline.py questions.txt

    # Run pipeline with ragas evaluator
    python run_full_pipeline.py questions.txt --evaluator ragas

    # Run pipeline with all evaluators
    python run_full_pipeline.py questions.txt --evaluator all

    # Skip collection step (use existing CSV)
    python run_full_pipeline.py questions.txt --skip-collect

    # Skip visualization step
    python run_full_pipeline.py questions.txt --skip-visualize

    # Custom models and API URL
    python run_full_pipeline.py questions.txt --model-a claude4.5-sonnet --model-b claude4.5-haiku --api-url http://localhost:8080/api/v2/questions

    # Custom judge model for evaluation
    python run_full_pipeline.py questions.txt --judge-model gpt-5
        """,
    )

    parser.add_argument(
        "questions_file",
        help="Path to questions file (.txt or .csv)",
    )

    parser.add_argument(
        "--evaluator",
        choices=["llm-judge", "ragas", "format-clarity", "all"],
        default="llm-judge",
        help="Evaluator to use (default: llm-judge)",
    )

    parser.add_argument(
        "--model-a",
        type=str,
        default=None,
        help="Model A name (passed to collect_responses.py)",
    )

    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Model B name (passed to collect_responses.py)",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API URL (passed to collect_responses.py)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process (passed to evaluation script)",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for evaluation (passed to evaluation script as --model)",
    )

    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip collection step (use existing collected_responses.csv)",
    )

    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip visualization step",
    )

    parser.add_argument(
        "--collect-output",
        type=str,
        default=DEFAULT_COLLECT_OUTPUT,
        help=f"Output file for collection step (default: {DEFAULT_COLLECT_OUTPUT})",
    )

    args = parser.parse_args()

    log_section("Full Pipeline Execution")
    log_info(f"Questions file: {args.questions_file}")
    log_info(f"Evaluator: {args.evaluator}")

    # Step 1: Collect responses (unless skipped)
    # Track actual model names used (for visualization)
    actual_model_a = args.model_a
    actual_model_b = args.model_b

    if not args.skip_collect:
        if not Path(args.questions_file).exists():
            log_error(f"Questions file not found: {args.questions_file}")
            sys.exit(1)

        success, collected_model_a, collected_model_b = run_collect_step(
            args.questions_file,
            args.collect_output,
            model_a=args.model_a,
            model_b=args.model_b,
            api_url=args.api_url,
        )
        if not success:
            log_error("Pipeline failed at collection step")
            sys.exit(1)
        # Update actual model names from collection step
        actual_model_a = collected_model_a
        actual_model_b = collected_model_b
    else:
        log_info("Skipping collection step (using existing file)")
        if not Path(args.collect_output).exists():
            log_error(f"Collection output file not found: {args.collect_output}")
            sys.exit(1)
        # If skip_collect and model names not specified, use defaults
        if not actual_model_a:
            actual_model_a = "claude3.5-sonnet"
        if not actual_model_b:
            actual_model_b = "claude4.5-haiku"

    # Step 2: Run evaluation
    success, eval_output_file = run_evaluation_step(
        args.evaluator,
        args.collect_output,
        limit=args.limit,
        model_name=args.judge_model,
    )
    if not success:
        log_error("Pipeline failed at evaluation step")
        sys.exit(1)

    # Step 3: Visualize results (unless skipped)
    if not args.skip_visualize:
        # Only visualize llm-judge output for now
        if args.evaluator == "llm-judge" or (
            args.evaluator == "all" and eval_output_file == DEFAULT_LLM_JUDGE_OUTPUT
        ):
            visualize_success = run_visualize_step(
                eval_output_file,
                model_a_name=actual_model_a,
                model_b_name=actual_model_b,
            )
            if not visualize_success:
                log_warning("Visualization failed, but pipeline completed")
        elif args.evaluator == "all":
            # For 'all', visualize llm-judge output
            visualize_success = run_visualize_step(
                DEFAULT_LLM_JUDGE_OUTPUT,
                model_a_name=actual_model_a,
                model_b_name=actual_model_b,
            )
            if not visualize_success:
                log_warning("Visualization failed, but pipeline completed")
        else:
            log_info(
                f"Visualization skipped for {args.evaluator} evaluator (only supported for llm-judge)"
            )
    else:
        log_info("Skipping visualization step")

    log_section("Pipeline Completed Successfully")
    log_success(f"Collection output: {args.collect_output}")
    log_success(f"Evaluation output: {eval_output_file}")


if __name__ == "__main__":
    main()
