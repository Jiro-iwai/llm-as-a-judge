"""
Unit tests for .gitignore configuration and repository tracking.

This module tests that processing_time_log.txt is properly ignored
and removed from repository tracking.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGitignoreConfiguration:
    """Tests for .gitignore configuration"""

    def test_processing_time_log_in_gitignore(self):
        """Test that processing_time_log.txt is in .gitignore"""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        assert gitignore_path.exists(), ".gitignore file should exist"

        with open(gitignore_path, "r", encoding="utf-8") as f:
            gitignore_content = f.read()

        # Check that processing_time_log.txt is in .gitignore
        assert "processing_time_log.txt" in gitignore_content, (
            "processing_time_log.txt should be in .gitignore"
        )

    def test_processing_time_log_not_tracked(self):
        """Test that processing_time_log.txt is not tracked by git"""
        # Run git ls-files to check if processing_time_log.txt is tracked
        result = subprocess.run(
            ["git", "ls-files", "processing_time_log.txt"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # If file is tracked, git ls-files will return the file path
        # If not tracked, it will return empty output
        assert result.returncode == 0, "git ls-files should succeed"
        assert (
            "processing_time_log.txt" not in result.stdout
        ), "processing_time_log.txt should not be tracked by git"

    def test_processing_time_log_is_ignored_by_git(self, tmp_path):
        """Test that git check-ignore correctly identifies processing_time_log.txt"""
        repo_root = Path(__file__).parent.parent
        
        # Create a test file in the repository root
        test_file = repo_root / "processing_time_log.txt"
        
        # Ensure the file exists for git check-ignore
        if not test_file.exists():
            test_file.write_text("# Test log file\n")
        
        try:
            # Run git check-ignore to verify the file is ignored
            result = subprocess.run(
                ["git", "check-ignore", "-v", "processing_time_log.txt"],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            
            # git check-ignore returns 0 if the file is ignored, 1 if not
            assert result.returncode == 0, (
                "processing_time_log.txt should be ignored by git"
            )
            assert ".gitignore" in result.stdout, (
                "git check-ignore should reference .gitignore"
            )
            assert "processing_time_log.txt" in result.stdout, (
                "git check-ignore output should mention the file"
            )
        finally:
            # Clean up: remove the test file if we created it
            if test_file.exists() and test_file.read_text() == "# Test log file\n":
                test_file.unlink()

