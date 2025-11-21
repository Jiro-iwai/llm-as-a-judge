"""
Tests for project structure validation.

These tests verify that the project structure is correct and all imports work
after restructuring.
"""

import sys
from pathlib import Path
from typing import List

import pytest


class TestProjectStructure:
    """Test project directory structure."""

    def test_scripts_directory_exists(self):
        """Verify scripts/ directory exists."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        assert scripts_dir.exists(), "scripts/ directory should exist"
        assert scripts_dir.is_dir(), "scripts/ should be a directory"

    def test_src_directory_exists(self):
        """Verify src/ directory exists."""
        src_dir = Path(__file__).parent.parent / "src"
        assert src_dir.exists(), "src/ directory should exist"
        assert src_dir.is_dir(), "src/ should be a directory"

    def test_src_config_directory_exists(self):
        """Verify src/config/ directory exists."""
        config_dir = Path(__file__).parent.parent / "src" / "config"
        assert config_dir.exists(), "src/config/ directory should exist"
        assert config_dir.is_dir(), "src/config/ should be a directory"

    def test_src_utils_directory_exists(self):
        """Verify src/utils/ directory exists."""
        utils_dir = Path(__file__).parent.parent / "src" / "utils"
        assert utils_dir.exists(), "src/utils/ directory should exist"
        assert utils_dir.is_dir(), "src/utils/ should be a directory"

    def test_examples_directory_exists(self):
        """Verify examples/ directory exists."""
        examples_dir = Path(__file__).parent.parent / "examples"
        assert examples_dir.exists(), "examples/ directory should exist"
        assert examples_dir.is_dir(), "examples/ should be a directory"

    def test_output_directory_exists(self):
        """Verify output/ directory exists."""
        output_dir = Path(__file__).parent.parent / "output"
        assert output_dir.exists(), "output/ directory should exist"
        assert output_dir.is_dir(), "output/ should be a directory"


class TestScriptFiles:
    """Test that script files are in the correct location."""

    @pytest.fixture
    def scripts_dir(self) -> Path:
        """Return scripts directory path."""
        return Path(__file__).parent.parent / "scripts"

    @pytest.fixture
    def expected_scripts(self) -> List[str]:
        """Return list of expected script files."""
        return [
            "llm_judge_evaluator.py",
            "format_clarity_evaluator.py",
            "ragas_llm_judge_evaluator.py",
            "collect_responses.py",
            "compare_processing_time.py",
            "visualize_results.py",
            "run_full_pipeline.py",
        ]

    def test_all_scripts_exist(self, scripts_dir: Path, expected_scripts: List[str]):
        """Verify all expected scripts exist in scripts/ directory."""
        for script in expected_scripts:
            script_path = scripts_dir / script
            assert script_path.exists(), f"{script} should exist in scripts/ directory"
            assert script_path.is_file(), f"{script} should be a file"


class TestModuleImports:
    """Test that modules can be imported from the new structure."""

    def test_src_config_imports(self):
        """Test importing from src.config."""
        try:
            from src.config import app_config  # noqa: F401
            from src.config import model_configs  # noqa: F401
            from src.config.app_config import get_app_config  # noqa: F401
            from src.config.model_configs import MODEL_CONFIGS  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed to import from src.config: {e}")

    def test_src_utils_imports(self):
        """Test importing from src.utils."""
        try:
            from src.utils import logging_config  # noqa: F401
            from src.utils import log_output_simplifier  # noqa: F401
            from src.utils.logging_config import setup_logging  # noqa: F401
            from src.utils.log_output_simplifier import clean_html  # noqa: F401
        except ImportError as e:
            pytest.fail(f"Failed to import from src.utils: {e}")

    def test_config_init_exports(self):
        """Test that config/__init__.py exports work."""
        try:
            from src.config import (  # noqa: F401
                DEFAULT_MODEL,
                MODEL_CONFIGS,
                SUPPORTED_MODELS,
                get_app_config,
                get_model_config,
            )
        except ImportError as e:
            pytest.fail(f"Failed to import from src.config: {e}")


class TestSampleFiles:
    """Test that sample files are in examples/ directory."""

    @pytest.fixture
    def examples_dir(self) -> Path:
        """Return examples directory path."""
        return Path(__file__).parent.parent / "examples"

    @pytest.fixture
    def expected_samples(self) -> List[str]:
        """Return list of expected sample files."""
        return [
            "sample_input_llm_judge.csv",
            "sample_input_ragas.csv",
            "sample_input_format_clarity.csv",
        ]

    def test_all_samples_exist(
        self, examples_dir: Path, expected_samples: List[str]
    ):
        """Verify all expected sample files exist in examples/ directory."""
        for sample in expected_samples:
            sample_path = examples_dir / sample
            assert sample_path.exists(), f"{sample} should exist in examples/ directory"
            assert sample_path.is_file(), f"{sample} should be a file"


class TestScriptImports:
    """Test that scripts can import modules correctly."""

    def test_scripts_can_import_config(self):
        """Test that scripts can import from src.config."""
        # Add scripts directory to path temporarily
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
            try:
                # Try importing a script that uses config
                # This will fail if imports are incorrect
                import importlib.util

                script_path = scripts_dir / "llm_judge_evaluator.py"
                if script_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "test_script", script_path
                    )
                    if spec and spec.loader:
                        # Just check if we can parse the imports, don't execute
                        with open(script_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Check that imports use src.config and src.utils
                            assert (
                                "from src.config" in content
                                or "from config" in content
                            ), "Script should import from src.config or config"
            finally:
                sys.path.remove(str(scripts_dir))

