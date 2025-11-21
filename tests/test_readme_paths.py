"""
Tests for README.md path references.

This module tests that README.md references use the correct paths
after project structure standardization (PR #41).
"""

import re
from pathlib import Path

import pytest


class TestReadmePathReferences:
    """Test that README.md uses correct path references."""

    @pytest.fixture
    def readme_path(self) -> Path:
        """Return path to README.md."""
        return Path(__file__).parent.parent / "README.md"

    @pytest.fixture
    def readme_content(self, readme_path: Path) -> str:
        """Read README.md content."""
        return readme_path.read_text(encoding="utf-8")

    def test_no_old_config_paths(self, readme_content: str):
        """Test that README.md doesn't reference old config/ paths."""
        # Pattern to match old config/ paths (not src/config/)
        # Look for config/app_config.py that doesn't have src/ before it
        old_config_pattern = r"(?<!src/)config/app_config\.py"
        matches = list(re.finditer(old_config_pattern, readme_content))
        
        problematic_matches = []
        for match in matches:
            # Check context to ensure it's not a false positive
            start = max(0, match.start() - 30)
            end = min(len(readme_content), match.end() + 10)
            context = readme_content[start:end]
            # Skip if it's in a code comment or example that's not about file paths
            if 'src/config' not in context and '`' in context:
                problematic_matches.append(match.group())
        
        assert len(problematic_matches) == 0, (
            f"Found old config/ paths in README.md: {problematic_matches}. "
            "All config paths should use src/config/ prefix."
        )

    def test_no_old_utils_paths(self, readme_content: str):
        """Test that README.md doesn't reference old utils/ paths."""
        # Pattern to match old utils/ paths (not src/utils/)
        # Look for utils/log_output_simplifier.py that doesn't have src/ before it
        old_utils_pattern = r"(?<!src/)utils/log_output_simplifier\.py"
        matches = list(re.finditer(old_utils_pattern, readme_content))
        
        problematic_matches = []
        for match in matches:
            # Check context to ensure it's not a false positive
            start = max(0, match.start() - 30)
            end = min(len(readme_content), match.end() + 10)
            context = readme_content[start:end]
            # Skip if it's in a code comment or example that's not about file paths
            if 'src/utils' not in context and '`' in context:
                problematic_matches.append(match.group())
        
        assert len(problematic_matches) == 0, (
            f"Found old utils/ paths in README.md: {problematic_matches}. "
            "All utils paths should use src/utils/ prefix."
        )

    def test_config_paths_use_src_prefix(self, readme_content: str):
        """Test that config paths in README.md use src/config/ prefix."""
        # Find all config/app_config.py references
        config_refs = re.findall(r"`([^`]*config[^`]*\.py)`", readme_content)
        
        # All config file references should use src/config/ prefix
        old_paths = [ref for ref in config_refs if ref.startswith("config/") and not ref.startswith("src/config/")]
        
        assert len(old_paths) == 0, (
            f"Found config paths without src/ prefix: {old_paths}. "
            "All config paths should use src/config/ prefix."
        )

    def test_utils_paths_use_src_prefix(self, readme_content: str):
        """Test that utils paths in README.md use src/utils/ prefix."""
        # Find all utils/*.py references
        utils_refs = re.findall(r"`([^`]*utils[^`]*\.py)`", readme_content)
        
        # All utils file references should use src/utils/ prefix
        old_paths = [ref for ref in utils_refs if ref.startswith("utils/") and not ref.startswith("src/utils/")]
        
        assert len(old_paths) == 0, (
            f"Found utils paths without src/ prefix: {old_paths}. "
            "All utils paths should use src/utils/ prefix."
        )

