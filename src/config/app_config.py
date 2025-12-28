"""
Application configuration module for externalizing hardcoded values.

This module provides a centralized configuration system that:
- Loads default values from code
- Optionally loads from YAML config file (specified via APP_CONFIG_FILE env var)
- Allows environment variables to override config file values
- Provides convenient getter functions for common configuration values
- Validates configuration values for correctness

Configuration Priority (highest to lowest):
1. Environment variables (for overrides)
2. YAML config file (recommended for most settings)
3. Default values (code defaults)

Note: While environment variables can override YAML settings, YAML is the
recommended approach for managing most configuration values. Environment
variables should primarily be used for:
- Required credentials (AZURE_OPENAI_*, OPENAI_API_KEY)
- Runtime overrides (APP_MAX_WORKERS, APP_TIMEOUT)
- CI/CD or deployment-specific settings

Supported environment variables:
- APP_CONFIG_FILE: Path to YAML config file
- APP_TIMEOUT: Request timeout in seconds
- APP_MAX_RETRIES: Maximum number of retry attempts
- APP_RETRY_DELAY: Delay between retries in seconds
- APP_API_DELAY: Delay between API calls in seconds
- APP_DEFAULT_IDENTITY: Default identity for API calls
- APP_MAX_WORKERS: Maximum number of parallel workers (None for sequential processing, int for parallel)
- APP_OUTPUT_FILE_*: Override specific output file paths (deprecated, use YAML instead)
- APP_REGEX_MODEL_A_PATTERN: Override model A regex pattern (deprecated, use YAML instead)
- APP_REGEX_MODEL_B_PATTERN: Override model B regex pattern (deprecated, use YAML instead)
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

# Global config instance
_config: Optional[Dict[str, Any]] = None

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "timeout": 120,
    "max_retries": 3,
    "retry_delay": 2,
    "api_delay": 1.0,
    "default_identity": "USER",
    "max_workers": None,  # None means sequential processing, int means parallel with that many workers
    "output_files": {
        "evaluation_comparison": "output/evaluation_comparison.png",
        "evaluation_distribution": "output/evaluation_distribution.png",
        "evaluation_boxplot": "output/evaluation_boxplot.png",
        "evaluation_summary": "output/evaluation_summary.txt",
        "ragas_evaluation_comparison": "output/ragas_evaluation_comparison.png",
        "ragas_evaluation_distribution": "output/ragas_evaluation_distribution.png",
        "ragas_evaluation_boxplot": "output/ragas_evaluation_boxplot.png",
        "ragas_evaluation_summary": "output/ragas_evaluation_summary.txt",
        "processing_time_comparison": "output/processing_time_comparison.png",
        "processing_time_statistics": "output/processing_time_statistics.png",
        "processing_time_summary": "output/processing_time_summary.txt",
        "processing_time_log": "output/processing_time_log.txt",
    },
    "regex_patterns": {
        "model_a_pattern": r"📥 \[claude4\.5-sonnet\].*?経過時間: ([\d.]+)秒",
        "model_b_pattern": r"📥 \[claude4\.5-haiku\].*?経過時間: ([\d.]+)秒",
    },
}


def reset_config() -> None:
    """Reset the global config (mainly for testing purposes)."""
    global _config
    _config = None


def validate_app_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration values and return a list of issues found.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        List of validation issue messages (empty if no issues found).
    """
    issues: List[str] = []

    # Validate max_workers
    max_workers = config.get("max_workers")
    if max_workers is not None:
        if not isinstance(max_workers, int):
            issues.append(f"max_workers must be int or None, got {type(max_workers).__name__}")
        elif max_workers < 1:
            issues.append(f"max_workers must be >= 1 or None, got {max_workers}")

    # Validate timeout
    timeout = config.get("timeout")
    if timeout is not None:
        if not isinstance(timeout, int):
            issues.append(f"timeout must be int, got {type(timeout).__name__}")
        elif timeout < 1:
            issues.append(f"timeout must be >= 1, got {timeout}")

    # Validate max_retries
    max_retries = config.get("max_retries")
    if max_retries is not None:
        if not isinstance(max_retries, int):
            issues.append(f"max_retries must be int, got {type(max_retries).__name__}")
        elif max_retries < 0:
            issues.append(f"max_retries must be >= 0, got {max_retries}")

    # Validate retry_delay
    retry_delay = config.get("retry_delay")
    if retry_delay is not None:
        if not isinstance(retry_delay, int):
            issues.append(f"retry_delay must be int, got {type(retry_delay).__name__}")
        elif retry_delay < 0:
            issues.append(f"retry_delay must be >= 0, got {retry_delay}")

    # Validate api_delay
    api_delay = config.get("api_delay")
    if api_delay is not None:
        if not isinstance(api_delay, (int, float)):
            issues.append(f"api_delay must be int or float, got {type(api_delay).__name__}")
        elif api_delay < 0:
            issues.append(f"api_delay must be >= 0, got {api_delay}")

    # Validate output_files structure
    output_files = config.get("output_files")
    if output_files is not None:
        if not isinstance(output_files, dict):
            issues.append(f"output_files must be dict, got {type(output_files).__name__}")

    # Validate regex_patterns structure
    regex_patterns = config.get("regex_patterns")
    if regex_patterns is not None:
        if not isinstance(regex_patterns, dict):
            issues.append(f"regex_patterns must be dict, got {type(regex_patterns).__name__}")

    return issues


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file and environment variables.

    Configuration priority (highest to lowest):
    1. Environment variables (for overrides)
    2. YAML config file (recommended for most settings)
    3. Default values (code defaults)

    Args:
        config_file: Optional path to config file. If None, reads from APP_CONFIG_FILE env var.

    Returns:
        Configuration dictionary with all settings.
    """
    global _config

    # Start with default values (deep copy to avoid mutating DEFAULT_CONFIG)
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Load from YAML file if specified
    config_file_path = config_file or os.getenv("APP_CONFIG_FILE")
    # If no config file is specified, try default config.yaml
    if not config_file_path:
        default_config_path = Path("config.yaml")
        if default_config_path.exists():
            config_file_path = str(default_config_path)
    
    if config_file_path and yaml is not None:
        config_path = Path(config_file_path)
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}
                    # Merge file config into default config
                    if isinstance(file_config, dict):
                        config.update(file_config)
            except (OSError, PermissionError) as e:
                # ファイル読み込みエラー、権限エラーを具体的に処理
                import sys

                print(
                    f"Warning: Failed to load config file '{config_file_path}': {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            except Exception as e:
                # YAML解析エラーやその他の予期しないエラー
                import sys

                # yamlが利用可能な場合はYAMLErrorをチェック
                if yaml is not None and isinstance(e, yaml.YAMLError):
                    print(
                        f"Warning: YAML parsing error in config file '{config_file_path}': {e}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Warning: Unexpected error loading config file '{config_file_path}': {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

    # Override with environment variables
    if "APP_TIMEOUT" in os.environ:
        try:
            config["timeout"] = int(os.environ["APP_TIMEOUT"])
        except ValueError:
            pass

    if "APP_MAX_RETRIES" in os.environ:
        try:
            config["max_retries"] = int(os.environ["APP_MAX_RETRIES"])
        except ValueError:
            pass

    if "APP_RETRY_DELAY" in os.environ:
        try:
            config["retry_delay"] = int(os.environ["APP_RETRY_DELAY"])
        except ValueError:
            pass

    if "APP_API_DELAY" in os.environ:
        try:
            config["api_delay"] = float(os.environ["APP_API_DELAY"])
        except ValueError:
            pass

    if "APP_DEFAULT_IDENTITY" in os.environ:
        config["default_identity"] = os.environ["APP_DEFAULT_IDENTITY"]

    if "APP_MAX_WORKERS" in os.environ:
        try:
            max_workers_str = os.environ["APP_MAX_WORKERS"]
            if max_workers_str.lower() in ("none", "null", ""):
                config["max_workers"] = None
            else:
                config["max_workers"] = int(max_workers_str)
        except ValueError:
            pass

    # Output files from env vars (deprecated: use YAML instead)
    output_files = config.get("output_files", {})
    for key in output_files.keys():
        env_key = f"APP_OUTPUT_FILE_{key.upper()}"
        if env_key in os.environ:
            output_files[key] = os.environ[env_key]
            logger.warning(
                f"Using deprecated environment variable {env_key}. "
                "Consider using YAML config file instead."
            )
    config["output_files"] = output_files

    # Regex patterns from env vars (deprecated: use YAML instead)
    regex_patterns = config.get("regex_patterns", {})
    if "APP_REGEX_MODEL_A_PATTERN" in os.environ:
        regex_patterns["model_a_pattern"] = os.environ["APP_REGEX_MODEL_A_PATTERN"]
        logger.warning(
            "Using deprecated environment variable APP_REGEX_MODEL_A_PATTERN. "
            "Consider using YAML config file instead."
        )
    if "APP_REGEX_MODEL_B_PATTERN" in os.environ:
        regex_patterns["model_b_pattern"] = os.environ["APP_REGEX_MODEL_B_PATTERN"]
        logger.warning(
            "Using deprecated environment variable APP_REGEX_MODEL_B_PATTERN. "
            "Consider using YAML config file instead."
        )
    config["regex_patterns"] = regex_patterns

    # Validate configuration
    validation_issues = validate_app_config(config)
    if validation_issues:
        for issue in validation_issues:
            logger.warning(f"Configuration validation issue: {issue}")

    _config = config
    return config


def get_app_config() -> Dict[str, Any]:
    """
    Get the global configuration, loading it if necessary.

    Configuration is validated automatically when loaded.

    Returns:
        Configuration dictionary.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_timeout(override: Optional[int] = None) -> int:
    """
    Get timeout value in seconds.

    Args:
        override: Optional override value. If provided, returns this value instead.

    Returns:
        Timeout in seconds.
    """
    if override is not None:
        return override
    config = get_app_config()
    return config.get("timeout", DEFAULT_CONFIG["timeout"])


def get_max_retries(override: Optional[int] = None) -> int:
    """
    Get maximum number of retry attempts.

    Args:
        override: Optional override value. If provided, returns this value instead.

    Returns:
        Maximum number of retries.
    """
    if override is not None:
        return override
    config = get_app_config()
    return config.get("max_retries", DEFAULT_CONFIG["max_retries"])


def get_retry_delay(override: Optional[int] = None) -> int:
    """
    Get delay between retries in seconds.

    Args:
        override: Optional override value. If provided, returns this value instead.

    Returns:
        Retry delay in seconds.
    """
    if override is not None:
        return override
    config = get_app_config()
    return config.get("retry_delay", DEFAULT_CONFIG["retry_delay"])


def get_api_delay(override: Optional[float] = None) -> float:
    """
    Get delay between API calls in seconds.

    Args:
        override: Optional override value. If provided, returns this value instead.

    Returns:
        API delay in seconds.
    """
    if override is not None:
        return override
    config = get_app_config()
    return config.get("api_delay", DEFAULT_CONFIG["api_delay"])


def get_default_identity(override: Optional[str] = None) -> str:
    """
    Get default identity for API calls.

    Args:
        override: Optional override value. If provided, returns this value instead.

    Returns:
        Default identity string.
    """
    if override is not None:
        return override
    config = get_app_config()
    return config.get("default_identity", DEFAULT_CONFIG["default_identity"])


def get_max_workers(override: Optional[int] = None) -> Optional[int]:
    """
    Get maximum number of workers for parallel processing.

    Args:
        override: Optional override value. If provided, returns this value instead.
                 None means sequential processing, int means parallel with that many workers.

    Returns:
        Maximum number of workers (None for sequential, int for parallel).
        Invalid values (< 1) are automatically converted to None (sequential processing).
    """
    if override is not None:
        # Validate override value
        if override < 1:
            logger.warning(
                f"Invalid max_workers override value: {override}. "
                "Using None (sequential processing) instead."
            )
            return None
        return override
    config = get_app_config()
    max_workers = config.get("max_workers", DEFAULT_CONFIG["max_workers"])
    # Validate and fix invalid values
    if max_workers is not None and max_workers < 1:
        logger.warning(
            f"Invalid max_workers value in config: {max_workers}. "
            "Using None (sequential processing) instead."
        )
        return None
    return max_workers


def get_output_file_names() -> Dict[str, str]:
    """
    Get output file names configuration.

    Returns:
        Dictionary mapping file type keys to file names.
    """
    config = get_app_config()
    return config.get("output_files", DEFAULT_CONFIG["output_files"]).copy()


def get_regex_patterns() -> Dict[str, str]:
    """
    Get regex patterns for processing time extraction.

    Returns:
        Dictionary mapping pattern keys to regex patterns.
    """
    config = get_app_config()
    return config.get("regex_patterns", DEFAULT_CONFIG["regex_patterns"]).copy()

