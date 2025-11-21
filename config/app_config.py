"""
Application configuration module for externalizing hardcoded values.

This module provides a centralized configuration system that:
- Loads default values from code
- Optionally loads from YAML config file (specified via APP_CONFIG_FILE env var)
- Allows environment variables to override config file values
- Provides convenient getter functions for common configuration values

Supported environment variables:
- APP_CONFIG_FILE: Path to YAML config file
- APP_TIMEOUT: Request timeout in seconds
- APP_MAX_RETRIES: Maximum number of retry attempts
- APP_RETRY_DELAY: Delay between retries in seconds
- APP_API_DELAY: Delay between API calls in seconds
- APP_DEFAULT_IDENTITY: Default identity for API calls
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# Global config instance
_config: Optional[Dict[str, Any]] = None

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "timeout": 120,
    "max_retries": 3,
    "retry_delay": 2,
    "api_delay": 1.0,
    "default_identity": "A14804",
    "output_files": {
        "evaluation_comparison": "evaluation_comparison.png",
        "evaluation_distribution": "evaluation_distribution.png",
        "evaluation_boxplot": "evaluation_boxplot.png",
        "evaluation_summary": "evaluation_summary.txt",
        "processing_time_comparison": "processing_time_comparison.png",
        "processing_time_statistics": "processing_time_statistics.png",
        "processing_time_summary": "processing_time_summary.txt",
        "processing_time_log": "processing_time_log.txt",
    },
    "regex_patterns": {
        "model_a_pattern": r"📥 \[claude3\.5-sonnet\].*?経過時間: ([\d.]+)秒",
        "model_b_pattern": r"📥 \[claude4\.5-haiku\].*?経過時間: ([\d.]+)秒",
    },
}


def reset_config() -> None:
    """Reset the global config (mainly for testing purposes)."""
    global _config
    _config = None


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file and environment variables.

    Args:
        config_file: Optional path to config file. If None, reads from APP_CONFIG_FILE env var.

    Returns:
        Configuration dictionary with all settings.
    """
    global _config

    # Start with default values
    config = DEFAULT_CONFIG.copy()

    # Load from YAML file if specified
    config_file_path = config_file or os.getenv("APP_CONFIG_FILE")
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

    # Output files from env vars
    output_files = config.get("output_files", {})
    for key in output_files.keys():
        env_key = f"APP_OUTPUT_FILE_{key.upper()}"
        if env_key in os.environ:
            output_files[key] = os.environ[env_key]
    config["output_files"] = output_files

    # Regex patterns from env vars
    regex_patterns = config.get("regex_patterns", {})
    if "APP_REGEX_MODEL_A_PATTERN" in os.environ:
        regex_patterns["model_a_pattern"] = os.environ["APP_REGEX_MODEL_A_PATTERN"]
    if "APP_REGEX_MODEL_B_PATTERN" in os.environ:
        regex_patterns["model_b_pattern"] = os.environ["APP_REGEX_MODEL_B_PATTERN"]
    config["regex_patterns"] = regex_patterns

    _config = config
    return config


def get_app_config() -> Dict[str, Any]:
    """
    Get the global configuration, loading it if necessary.

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

