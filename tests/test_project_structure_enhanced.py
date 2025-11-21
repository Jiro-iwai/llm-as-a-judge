"""
Enhanced tests for project structure validation.

These tests provide comprehensive validation for:
- Import path consistency: All scripts use src.* imports
- File placement rules: Scripts in scripts/, configs in src/config/, etc.
- Cross-reference validation: Files referenced in docs actually exist
- Path reference consistency: No old paths in config files
"""

import ast
import re
from pathlib import Path
from typing import List, Set

import pytest


class TestImportPathConsistency:
    """Test that all scripts use src.* import paths consistently."""

    @pytest.fixture
    def scripts_dir(self) -> Path:
        """Return scripts directory path."""
        return Path(__file__).parent.parent / "scripts"

    @pytest.fixture
    def script_files(self, scripts_dir: Path) -> List[Path]:
        """Return list of Python script files."""
        if not scripts_dir.exists():
            return []
        return [f for f in scripts_dir.iterdir() if f.suffix == ".py" and f.is_file()]

    def test_no_old_config_imports(self, script_files: List[Path]):
        """Test that no scripts import from old config. paths."""
        old_imports = []
        for script_file in script_files:
            try:
                with open(script_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Parse AST to find actual import statements
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ImportFrom):
                                if node.module and node.module.startswith("config."):
                                    if not node.module.startswith("src.config."):
                                        old_imports.append((script_file.name, node.lineno, f"from {node.module}"))
                            elif isinstance(node, ast.Import):
                                for alias in node.names:
                                    if alias.name.startswith("config."):
                                        if not alias.name.startswith("src.config."):
                                            old_imports.append((script_file.name, node.lineno, f"import {alias.name}"))
                    except SyntaxError:
                        # If AST parsing fails, fall back to regex (but skip comments)
                        lines = content.split("\n")
                        in_multiline_comment = False
                        for i, line in enumerate(lines, 1):
                            stripped = line.strip()
                            # Skip single-line comments
                            if stripped.startswith("#"):
                                continue
                            # Skip docstrings
                            if stripped.startswith('"""') or stripped.startswith("'''"):
                                in_multiline_comment = not in_multiline_comment
                                continue
                            if in_multiline_comment:
                                continue
                            # Check for old import patterns (only actual import statements)
                            if re.match(r"^\s*(from|import)\s+config\.", line):
                                if "src.config" not in line:
                                    old_imports.append((script_file.name, i, line.strip()))
            except Exception as e:
                pytest.fail(f"Failed to read {script_file}: {e}")

        assert len(old_imports) == 0, (
            f"Found old config. imports in scripts: {old_imports}. "
            "All imports should use src.config."
        )

    def test_no_old_utils_imports(self, script_files: List[Path]):
        """Test that no scripts import from old utils. paths."""
        old_imports = []
        for script_file in script_files:
            try:
                with open(script_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Check for old import patterns (excluding comments)
                    lines = content.split("\n")
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        # Skip comments and docstrings
                        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                            continue
                        # Check for old import patterns
                        if re.search(r"from\s+utils\.|import\s+utils\.", line):
                            if "src.utils" not in line:
                                old_imports.append((script_file.name, i, line.strip()))
            except Exception as e:
                pytest.fail(f"Failed to read {script_file}: {e}")

        assert len(old_imports) == 0, (
            f"Found old utils. imports in scripts: {old_imports}. "
            "All imports should use src.utils."
        )

    def test_scripts_use_src_imports(self, script_files: List[Path]):
        """Test that scripts use src.config and src.utils imports."""
        scripts_without_src_imports = []
        for script_file in script_files:
            try:
                with open(script_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Check if script uses src.config or src.utils
                    has_src_config = "from src.config" in content or "import src.config" in content
                    has_src_utils = "from src.utils" in content or "import src.utils" in content
                    # Scripts that import config/utils should use src. prefix
                    if ("config" in content or "utils" in content) and not (has_src_config or has_src_utils):
                        # Check if it's actually importing (not just mentioning in comments)
                        tree = ast.parse(content)
                        imports = []
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                imports.extend([alias.name for alias in node.names])
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    imports.append(node.module)
                        
                        if any(imp.startswith("config.") or imp.startswith("utils.") for imp in imports):
                            scripts_without_src_imports.append(script_file.name)
            except SyntaxError:
                # Skip files with syntax errors (they'll be caught by other tests)
                pass
            except Exception as e:
                pytest.fail(f"Failed to parse {script_file}: {e}")

        assert len(scripts_without_src_imports) == 0, (
            f"Scripts using config/utils imports without src. prefix: {scripts_without_src_imports}. "
            "All imports should use src.config or src.utils."
        )


class TestFilePlacementRules:
    """Test that files are placed according to project structure rules."""

    @pytest.fixture
    def scripts_dir(self) -> Path:
        """Return scripts directory path."""
        return Path(__file__).parent.parent / "scripts"

    @pytest.fixture
    def src_config_dir(self) -> Path:
        """Return src/config directory path."""
        return Path(__file__).parent.parent / "src" / "config"

    @pytest.fixture
    def src_utils_dir(self) -> Path:
        """Return src/utils directory path."""
        return Path(__file__).parent.parent / "src" / "utils"

    def test_scripts_dir_only_contains_python_files(self, scripts_dir: Path):
        """Test that scripts/ directory only contains Python files."""
        if not scripts_dir.exists():
            pytest.skip("scripts/ directory does not exist")
        
        non_python_files = []
        for item in scripts_dir.iterdir():
            if item.is_file() and not item.name.endswith(".py"):
                # Allow __pycache__ and other standard Python files
                if item.name not in ["__pycache__", ".gitkeep"]:
                    non_python_files.append(item.name)
        
        assert len(non_python_files) == 0, (
            f"scripts/ directory contains non-Python files: {non_python_files}. "
            "scripts/ should only contain .py files."
        )

    def test_config_files_in_src_config(self, src_config_dir: Path):
        """Test that config files are in src/config/ directory."""
        if not src_config_dir.exists():
            pytest.fail("src/config/ directory does not exist")
        
        expected_config_files = ["app_config.py", "model_configs.py", "__init__.py"]
        missing_files = []
        for expected_file in expected_config_files:
            if not (src_config_dir / expected_file).exists():
                missing_files.append(expected_file)
        
        assert len(missing_files) == 0, (
            f"Missing config files in src/config/: {missing_files}"
        )

    def test_utils_files_in_src_utils(self, src_utils_dir: Path):
        """Test that utils files are in src/utils/ directory."""
        if not src_utils_dir.exists():
            pytest.fail("src/utils/ directory does not exist")
        
        expected_utils_files = ["logging_config.py", "log_output_simplifier.py", "__init__.py"]
        missing_files = []
        for expected_file in expected_utils_files:
            if not (src_utils_dir / expected_file).exists():
                missing_files.append(expected_file)
        
        assert len(missing_files) == 0, (
            f"Missing utils files in src/utils/: {missing_files}"
        )


class TestMakefileConsistency:
    """Test that Makefile references are consistent with project structure."""

    @pytest.fixture
    def makefile_path(self) -> Path:
        """Return path to Makefile."""
        return Path(__file__).parent.parent / "Makefile"

    @pytest.fixture
    def makefile_content(self, makefile_path: Path) -> str:
        """Read Makefile content."""
        return makefile_path.read_text(encoding="utf-8")

    @pytest.fixture
    def scripts_dir(self) -> Path:
        """Return scripts directory path."""
        return Path(__file__).parent.parent / "scripts"

    def test_makefile_script_references_exist(self, makefile_content: str, scripts_dir: Path):
        """Test that all scripts referenced in Makefile exist."""
        # Find all script references in Makefile
        script_pattern = r"scripts/([a-zA-Z_]+\.py)"
        referenced_scripts = set(re.findall(script_pattern, makefile_content))
        
        missing_scripts = []
        for script_name in referenced_scripts:
            script_path = scripts_dir / script_name
            if not script_path.exists():
                missing_scripts.append(script_name)
        
        assert len(missing_scripts) == 0, (
            f"Makefile references scripts that don't exist: {missing_scripts}"
        )

    def test_makefile_uses_scripts_prefix(self, makefile_content: str):
        """Test that Makefile uses scripts/ prefix for script references."""
        # Find script references without scripts/ prefix (excluding help messages)
        # Look for patterns like "python script.py" or "script.py" in execution contexts
        lines = makefile_content.split("\n")
        problematic_lines = []
        
        for i, line in enumerate(lines, 1):
            # Skip help/echo lines that just mention script names
            if line.strip().startswith("@echo") or line.strip().startswith("echo"):
                continue
            
            # Check for script execution patterns
            if re.search(r"\b(llm_judge_evaluator|format_clarity_evaluator|ragas_llm_judge_evaluator|collect_responses|compare_processing_time|visualize_results|run_full_pipeline)\.py\b", line):
                if "scripts/" not in line and "$(PYTHON)" not in line:
                    # This might be in a help message, check context
                    if not any(keyword in line.lower() for keyword in ["help", "usage", "show", "echo"]):
                        problematic_lines.append((i, line.strip()))
        
        # Filter out false positives (help messages)
        actual_problems = [
            (line_num, line) for line_num, line in problematic_lines
            if not any(keyword in line.lower() for keyword in ["help", "usage", "show", "echo", "の使い方"])
        ]
        
        assert len(actual_problems) == 0, (
            f"Makefile contains script references without scripts/ prefix: {actual_problems}. "
            "All script references should use scripts/ prefix."
        )


class TestDocumentationConsistency:
    """Test that files referenced in documentation actually exist."""

    @pytest.fixture
    def readme_path(self) -> Path:
        """Return path to README.md."""
        return Path(__file__).parent.parent / "README.md"

    @pytest.fixture
    def readme_content(self, readme_path: Path) -> str:
        """Read README.md content."""
        return readme_path.read_text(encoding="utf-8")

    def test_readme_script_references_exist(self, readme_content: str):
        """Test that all scripts referenced in README.md exist."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        # Find script references in README
        script_pattern = r"scripts/([a-zA-Z_]+\.py)"
        referenced_scripts = set(re.findall(script_pattern, readme_content))
        
        missing_scripts = []
        for script_name in referenced_scripts:
            script_path = scripts_dir / script_name
            if not script_path.exists():
                missing_scripts.append(script_name)
        
        assert len(missing_scripts) == 0, (
            f"README.md references scripts that don't exist: {missing_scripts}"
        )

    def test_readme_module_references_exist(self, readme_content: str):
        """Test that all modules referenced in README.md exist."""
        project_root = Path(__file__).parent.parent
        
        # Find module references (src/config/*.py, src/utils/*.py)
        module_pattern = r"src/(config|utils)/([a-zA-Z_]+\.py)"
        referenced_modules = set(re.findall(module_pattern, readme_content))
        
        missing_modules = []
        for module_type, module_name in referenced_modules:
            module_path = project_root / "src" / module_type / module_name
            if not module_path.exists():
                missing_modules.append(f"src/{module_type}/{module_name}")
        
        assert len(missing_modules) == 0, (
            f"README.md references modules that don't exist: {missing_modules}"
        )

