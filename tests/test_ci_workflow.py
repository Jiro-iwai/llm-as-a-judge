"""
Tests for CI workflow configuration.

This module validates that the GitHub Actions CI workflow is correctly configured.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCIWorkflowConfig:
    """Test GitHub Actions CI workflow configuration"""

    def test_ci_workflow_file_exists(self):
        """Test that CI workflow file exists"""
        workflow_path = Path(".github/workflows/ci.yml")
        assert workflow_path.exists(), f"CI workflow file not found: {workflow_path}"

    def test_ci_workflow_yaml_is_valid(self):
        """Test that CI workflow YAML is valid"""
        workflow_path = Path(".github/workflows/ci.yml")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)
        assert isinstance(workflow, dict), "Workflow YAML must be a dictionary"

    def test_ci_workflow_has_required_triggers(self):
        """Test that CI workflow has required triggers"""
        workflow_path = Path(".github/workflows/ci.yml")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        # GitHub Actions uses 'on' as a key, but yaml.safe_load may interpret it differently
        # Check for 'on' key or True key (which is how 'on:' is sometimes parsed)
        triggers = None
        if "on" in workflow:
            triggers = workflow["on"]
        elif True in workflow:  # Sometimes 'on:' is parsed as True key
            triggers = workflow[True]

        assert triggers is not None, "Workflow must have 'on' section"
        assert isinstance(triggers, dict), "Triggers must be a dictionary"

        assert "push" in triggers or "pull_request" in triggers, (
            "Workflow must have 'push' or 'pull_request' trigger"
        )

        # Check for main branch triggers
        if "push" in triggers:
            push_config = triggers["push"]
            if isinstance(push_config, dict) and "branches" in push_config:
                assert "main" in push_config["branches"], (
                    "Push trigger should include 'main' branch"
                )

        if "pull_request" in triggers:
            pr_config = triggers["pull_request"]
            if isinstance(pr_config, dict) and "branches" in pr_config:
                assert "main" in pr_config["branches"], (
                    "Pull request trigger should include 'main' branch"
                )

    def test_ci_workflow_has_test_job(self):
        """Test that CI workflow has a test job"""
        workflow_path = Path(".github/workflows/ci.yml")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        assert "jobs" in workflow, "Workflow must have 'jobs' section"
        jobs = workflow["jobs"]
        assert len(jobs) > 0, "Workflow must have at least one job"

        # Check if there's a job that runs tests (could be named 'ci', 'test', etc.)
        job_names = list(jobs.keys())
        assert any(
            name in ["ci", "test", "tests"] for name in job_names
        ), f"Workflow should have a job named 'ci', 'test', or 'tests'. Found: {job_names}"

    def test_ci_workflow_uses_uv(self):
        """Test that CI workflow uses uv for dependency management"""
        workflow_path = Path(".github/workflows/ci.yml")
        with open(workflow_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that uv is mentioned in the workflow
        assert "uv" in content.lower(), "Workflow should use uv for dependency management"

    def test_ci_workflow_runs_make_test(self):
        """Test that CI workflow runs 'make test'"""
        workflow_path = Path(".github/workflows/ci.yml")
        with open(workflow_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that make test is mentioned
        assert "make test" in content, "Workflow should run 'make test'"

    def test_ci_workflow_has_cache(self):
        """Test that CI workflow has dependency caching"""
        workflow_path = Path(".github/workflows/ci.yml")
        with open(workflow_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that caching is used (actions/cache)
        assert "actions/cache" in content or "cache:" in content.lower(), (
            "Workflow should use caching for dependencies"
        )

    def test_ci_workflow_sets_up_python(self):
        """Test that CI workflow sets up Python"""
        workflow_path = Path(".github/workflows/ci.yml")
        with open(workflow_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that Python is set up
        assert "actions/setup-python" in content or "setup-python" in content, (
            "Workflow should set up Python using actions/setup-python"
        )

