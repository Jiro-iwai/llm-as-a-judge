"""Configuration module for LLM-as-a-Judge evaluation scripts."""

from .model_configs import (
    MODEL_CONFIGS,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    get_model_config,
    get_full_config,
    get_simple_config,
    get_ragas_config,
)

__all__ = [
    "MODEL_CONFIGS",
    "DEFAULT_MODEL",
    "SUPPORTED_MODELS",
    "get_model_config",
    "get_full_config",
    "get_simple_config",
    "get_ragas_config",
]

