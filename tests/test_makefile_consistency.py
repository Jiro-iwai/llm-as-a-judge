"""
Tests for Makefile consistency with project structure.

These tests verify that Makefile uses consistent script references
with the scripts/ prefix after PR #41.
"""

import re
from pathlib import Path

import pytest


class TestMakefileHelpMessages:
    """Test that Makefile help messages use scripts/ prefix."""

    @pytest.fixture
    def makefile_path(self) -> Path:
        """Return path to Makefile."""
        return Path(__file__).parent.parent / "Makefile"

    @pytest.fixture
    def makefile_content(self, makefile_path: Path) -> str:
        """Read Makefile content."""
        return makefile_path.read_text(encoding="utf-8")

    def test_help_messages_use_scripts_prefix(self, makefile_content: str):
        """Test that help messages reference scripts with scripts/ prefix."""
        # Find help-* target definitions
        help_targets = [
            "help-llm-judge",
            "help-format-clarity",
            "help-ragas",
        ]
        
        problematic_lines = []
        lines = makefile_content.split("\n")
        
        for i, line in enumerate(lines, 1):
            # Check for help target definitions
            for target in help_targets:
                if f"help-{target.replace('help-', '')}" in line or target in line:
                    # Check the next few lines for script references
                    for j in range(i, min(i + 5, len(lines))):
                        check_line = lines[j]
                        # Look for script names without scripts/ prefix
                        if re.search(
                            r"\b(llm_judge_evaluator|format_clarity_evaluator|ragas_llm_judge_evaluator)\.py\b",
                            check_line,
                        ):
                            # Skip if it's in a comment or already has scripts/
                            if "scripts/" not in check_line and not check_line.strip().startswith("#"):
                                # Check if it's an echo statement that should show the full path
                                if "@echo" in check_line or "echo" in check_line:
                                    problematic_lines.append((j + 1, check_line.strip()))
        
        assert len(problematic_lines) == 0, (
            f"Makefile help messages contain script references without scripts/ prefix: {problematic_lines}. "
            "Help messages should reference scripts with scripts/ prefix for consistency."
        )

    def test_help_echo_messages_are_consistent(self, makefile_content: str):
        """Test that echo messages in help targets are consistent."""
        # Find all echo statements that mention script names
        lines = makefile_content.split("\n")
        script_names = [
            "llm_judge_evaluator.py",
            "format_clarity_evaluator.py",
            "ragas_llm_judge_evaluator.py",
        ]
        
        inconsistent_echoes = []
        for i, line in enumerate(lines, 1):
            if "@echo" in line or line.strip().startswith("echo"):
                for script_name in script_names:
                    if script_name in line:
                        # Check if it's showing usage/help message
                        if any(keyword in line.lower() for keyword in ["usage", "使い方", "help"]):
                            # Should show scripts/ prefix for consistency
                            if "scripts/" not in line:
                                inconsistent_echoes.append((i + 1, line.strip()))
        
        assert len(inconsistent_echoes) == 0, (
            f"Makefile help echo messages are inconsistent: {inconsistent_echoes}. "
            "Help messages should show scripts/ prefix for consistency."
        )

