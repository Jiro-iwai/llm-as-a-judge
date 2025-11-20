"""
Unified model configuration for LLM-as-a-Judge evaluation scripts.

This module provides a single source of truth for model configurations,
eliminating duplication across multiple scripts.
"""

from typing import Dict, Any

# Complete model configurations with all fields
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt-5": {
        # Full configuration (for llm_judge_evaluator.py)
        "max_total_tokens": 128000,  # 128K
        "min_output_tokens": 800,
        "max_output_tokens": 4000,
        "safety_margin": 2000,
        "temperature": 1.0,
        "use_max_completion_tokens": True,  # GPT-5 uses max_completion_tokens
        "timeout": 120,
        # Simple configuration (for format_clarity_evaluator.py)
        "max_tokens": 2000,
        "max_completion_tokens": 2000,
    },
    "gpt-4.1": {
        # Full configuration (for llm_judge_evaluator.py)
        "max_total_tokens": 128000,  # 128K
        "min_output_tokens": 800,
        "max_output_tokens": 4000,
        "safety_margin": 2000,
        "temperature": 0.7,
        "use_max_completion_tokens": False,  # GPT-4.1 uses max_tokens
        "timeout": 120,
        # Simple configuration (for format_clarity_evaluator.py and ragas_llm_judge_evaluator.py)
        "max_tokens": 2000,
    },
    "gpt-4-turbo": {
        # Full configuration (for llm_judge_evaluator.py)
        "max_total_tokens": 128000,  # 128K
        "min_output_tokens": 800,
        "max_output_tokens": 4000,
        "safety_margin": 2000,
        "temperature": 0.7,
        "use_max_completion_tokens": False,
        "timeout": 120,
        # Simple configuration (for format_clarity_evaluator.py and ragas_llm_judge_evaluator.py)
        "max_tokens": 2000,
    },
}

# Default model (using gpt-4.1 as it's the most commonly used)
DEFAULT_MODEL = "gpt-4.1"

# Supported models list
SUPPORTED_MODELS = list(MODEL_CONFIGS.keys())


def _normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for matching (case-insensitive, handle variations).

    Args:
        model_name: Model name to normalize

    Returns:
        Normalized model name
    """
    return model_name.lower().strip()


def _find_model_key(model_name: str) -> str | None:
    """
    Find the model key in MODEL_CONFIGS, handling variations.

    Args:
        model_name: Model name to find

    Returns:
        Model key if found, None otherwise
    """
    normalized = _normalize_model_name(model_name)

    # Try exact match first
    if normalized in MODEL_CONFIGS:
        return normalized

    # Try partial match (e.g., "gpt5" -> "gpt-5")
    for key in MODEL_CONFIGS.keys():
        if key.replace("-", "").lower() == normalized.replace("-", ""):
            return key

    return None


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get complete configuration for a specific model.

    This function returns the full configuration dictionary.
    For script-specific configurations, use get_full_config(), get_simple_config(), or get_ragas_config().

    Args:
        model_name: Model name (e.g., "gpt-5", "gpt-4.1")

    Returns:
        Complete model configuration dictionary
    """
    model_key = _find_model_key(model_name)
    if model_key:
        return MODEL_CONFIGS[model_key].copy()

    # Return default config if not found
    return MODEL_CONFIGS[DEFAULT_MODEL].copy()


def get_full_config(model_name: str) -> Dict[str, Any]:
    """
    Get full configuration for llm_judge_evaluator.py.

    This includes all fields: max_total_tokens, min_output_tokens, max_output_tokens,
    safety_margin, temperature, use_max_completion_tokens, timeout.

    Args:
        model_name: Model name (e.g., "gpt-5", "gpt-4.1")

    Returns:
        Full model configuration dictionary
    """
    config = get_model_config(model_name)
    return {
        "max_total_tokens": config["max_total_tokens"],
        "min_output_tokens": config["min_output_tokens"],
        "max_output_tokens": config["max_output_tokens"],
        "safety_margin": config["safety_margin"],
        "temperature": config["temperature"],
        "use_max_completion_tokens": config["use_max_completion_tokens"],
        "timeout": config["timeout"],
    }


def get_simple_config(model_name: str) -> Dict[str, Any]:
    """
    Get simple configuration for format_clarity_evaluator.py.

    This includes: max_tokens (or max_completion_tokens), temperature, use_max_completion_tokens.

    Args:
        model_name: Model name (e.g., "gpt-5", "gpt-4.1")

    Returns:
        Simple model configuration dictionary
    """
    config = get_model_config(model_name)
    result: Dict[str, Any] = {
        "temperature": config["temperature"],
        "use_max_completion_tokens": config["use_max_completion_tokens"],
    }

    if config["use_max_completion_tokens"]:
        result["max_completion_tokens"] = config["max_completion_tokens"]
        result["max_tokens"] = config.get("max_tokens", config["max_completion_tokens"])
    else:
        result["max_tokens"] = config["max_tokens"]

    return result


def get_ragas_config(model_name: str) -> Dict[str, Any]:
    """
    Get Ragas-compatible configuration for ragas_llm_judge_evaluator.py.

    This includes: max_tokens (or max_completion_tokens), temperature, use_max_completion_tokens.

    Args:
        model_name: Model name (e.g., "gpt-5", "gpt-4.1")

    Returns:
        Ragas-compatible model configuration dictionary
    """
    # Ragas config is the same as simple config
    return get_simple_config(model_name)

