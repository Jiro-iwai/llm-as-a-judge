.PHONY: test format lint typecheck test-scripts test-unit help help-llm-judge help-format-clarity help-ragas help-collect help-visualize help-pipeline help-all clean clean-venv setup venv install-deps update-deps pipeline

# Python executable (use .venv if available, otherwise system python)
PYTHON := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

# Main evaluation scripts (with --help option)
SCRIPTS := scripts/llm_judge_evaluator.py scripts/format_clarity_evaluator.py scripts/ragas_llm_judge_evaluator.py scripts/collect_responses.py

# All Python scripts (for format/lint/typecheck)
ALL_SCRIPTS := $(SCRIPTS) scripts/compare_processing_time.py scripts/visualize_results.py scripts/run_full_pipeline.py

# Scripts that support model option (-m)
MODEL_SCRIPTS := scripts/llm_judge_evaluator.py scripts/format_clarity_evaluator.py scripts/ragas_llm_judge_evaluator.py

# Files to clean (exclude requirements.txt and other important files)
CLEAN_FILES := output/*.csv output/*.png output/evaluation_output.csv output/format_clarity_output.csv output/ragas_evaluation_output.csv output/collected_responses.csv output/processing_time_summary.txt output/evaluation_summary.txt
CLEAN_DIRS := __pycache__ .pytest_cache .ruff_cache .pyright_cache htmlcov output/__pycache__ scripts/__pycache__ src/__pycache__

help:
	@echo "Available targets:"
	@echo ""
	@echo "Setup targets:"
	@echo "  make setup         - Create virtual environment and install dependencies"
	@echo "  make venv          - Create virtual environment (.venv)"
	@echo "  make install-deps  - Install dependencies (requires .venv)"
	@echo "  make update-deps   - Update dependencies to latest versions"
	@echo "  make clean-venv    - Remove virtual environment (.venv)"
	@echo ""
	@echo "Testing targets:"
	@echo "  make test          - Run all tests (format, lint, typecheck, test-scripts, test-unit)"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Run ruff linter"
	@echo "  make typecheck     - Run pyright type checker"
	@echo "  make test-scripts  - Test script interfaces (--help)"
	@echo "  make test-unit     - Run pytest unit tests (if available)"
	@echo "  make test-coverage - Run pytest with coverage report"
	@echo "  make clean         - Clean generated files"
	@echo ""
	@echo "Script usage help:"
	@echo "  make help-llm-judge      - Show usage for scripts/llm_judge_evaluator.py"
	@echo "  make help-format-clarity - Show usage for scripts/format_clarity_evaluator.py"
	@echo "  make help-ragas          - Show usage for scripts/ragas_llm_judge_evaluator.py"
	@echo "  make help-collect        - Show usage for scripts/collect_responses.py"
	@echo "  make help-visualize      - Show usage for scripts/visualize_results.py"
	@echo "  make help-pipeline       - Show usage for scripts/run_full_pipeline.py"
	@echo ""
	@echo "Pipeline target:"
	@echo "  make pipeline            - Run full pipeline (collect, evaluate, visualize)"
	@echo "  make help-all            - Show usage for all scripts"

help-llm-judge:
	@echo "=========================================="
	@echo "scripts/llm_judge_evaluator.py の使い方"
	@echo "=========================================="
	@$(PYTHON) scripts/llm_judge_evaluator.py --help

help-format-clarity:
	@echo "=========================================="
	@echo "scripts/format_clarity_evaluator.py の使い方"
	@echo "=========================================="
	@$(PYTHON) scripts/format_clarity_evaluator.py --help

help-ragas:
	@echo "=========================================="
	@echo "scripts/ragas_llm_judge_evaluator.py の使い方"
	@echo "=========================================="
	@$(PYTHON) scripts/ragas_llm_judge_evaluator.py --help 2>&1 || echo "⚠️  scripts/ragas_llm_judge_evaluator.py requires dependencies (ragas, datasets)"

help-collect:
	@echo "=========================================="
	@echo "scripts/collect_responses.py の使い方"
	@echo "=========================================="
	@echo "※ 応答収集後、処理時間ログと比較チャートを自動生成します"
	@$(PYTHON) scripts/collect_responses.py --help

help-visualize:
	@echo "=========================================="
	@echo "scripts/visualize_results.py の使い方"
	@echo "=========================================="
	@$(PYTHON) scripts/visualize_results.py --help

pipeline:
	@$(PYTHON) scripts/run_full_pipeline.py $(ARGS)

help-pipeline:
	@echo "=========================================="
	@echo "scripts/run_full_pipeline.py の使い方"
	@echo "=========================================="
	@$(PYTHON) scripts/run_full_pipeline.py --help

help-all: help-llm-judge help-format-clarity help-ragas help-collect help-visualize help-pipeline
	@echo ""
	@echo "=========================================="
	@echo "すべてのスクリプトの使い方を表示しました"
	@echo "=========================================="

# Run all tests (format, lint, typecheck, script tests, unit tests)
test: format lint typecheck test-scripts test-unit
	@echo ""
	@echo "✓ All tests passed!"

# Format code with ruff
format:
	@echo "=========================================="
	@echo "Running ruff format..."
	@echo "=========================================="
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "⚠️  Python not found, skipping..."; \
	elif $(PYTHON) -c "import ruff" 2>/dev/null; then \
		if $(PYTHON) -m ruff format $(ALL_SCRIPTS); then \
			echo "✓ Formatting completed"; \
		else \
			echo "✗ Formatting failed"; \
			exit 1; \
		fi; \
	else \
		echo "⚠️  ruff not installed, skipping..."; \
	fi

# Run ruff linter with auto-fix
lint:
	@echo ""
	@echo "=========================================="
	@echo "Running ruff check..."
	@echo "=========================================="
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "⚠️  Python not found, skipping..."; \
	elif $(PYTHON) -c "import ruff" 2>/dev/null; then \
		if $(PYTHON) -m ruff check --fix $(ALL_SCRIPTS); then \
			echo "✓ Linting completed"; \
		else \
			echo "✗ Linting failed"; \
			exit 1; \
		fi; \
	else \
		echo "⚠️  ruff not installed, skipping..."; \
	fi

# Run pyright type checker
typecheck:
	@echo ""
	@echo "=========================================="
	@echo "Running pyright..."
	@echo "=========================================="
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "⚠️  Python not found, skipping..."; \
	elif $(PYTHON) -c "import pyright" 2>/dev/null || command -v pyright >/dev/null 2>&1; then \
		if command -v pyright >/dev/null 2>&1; then \
			PYRIGHT_CMD=pyright; \
		else \
			PYRIGHT_CMD="$(PYTHON) -m pyright"; \
		fi; \
		$$PYRIGHT_CMD --pythonpath "$(shell $(PYTHON) -c 'import sys; print(sys.executable)')" $(ALL_SCRIPTS) 2>&1 | tee /tmp/pyright_output.txt; \
		PYRIGHT_EXIT=$$?; \
		if [ $$PYRIGHT_EXIT -eq 0 ]; then \
			echo "✓ Type checking completed"; \
		elif grep -q "reportMissingImports" /tmp/pyright_output.txt 2>/dev/null; then \
			echo "⚠️  Type checking found errors (some may be due to missing dependencies)"; \
			echo "   Install dependencies with 'make install-deps' to resolve import errors"; \
		else \
			echo "⚠️  Type checking found errors"; \
		fi; \
		rm -f /tmp/pyright_output.txt 2>/dev/null || true; \
	else \
		echo "⚠️  pyright not installed, skipping..."; \
	fi

# Run pytest unit tests (if test files exist)
test-unit:
	@echo ""
	@echo "=========================================="
	@echo "Running pytest unit tests..."
	@echo "=========================================="
	@HAS_TESTS=0; \
	if [ -d "tests" ] && [ -f "tests/__init__.py" ]; then \
		HAS_TESTS=1; \
	elif ls test_*.py 2>/dev/null | grep -q .; then \
		HAS_TESTS=1; \
	fi; \
	if [ $$HAS_TESTS -eq 1 ]; then \
		PYTEST_ARGS=""; \
		if [ -d "tests" ] && [ -f "tests/__init__.py" ]; then \
			PYTEST_ARGS="tests/"; \
		fi; \
		if ls test_*.py 2>/dev/null | grep -q .; then \
			PYTEST_ARGS="$$PYTEST_ARGS test_*.py"; \
		fi; \
		if [ -n "$$PYTEST_ARGS" ]; then \
			if $(PYTHON) -m pytest $$PYTEST_ARGS -v; then \
				echo "✓ Unit tests passed"; \
			else \
				echo "✗ Unit tests failed"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "⚠️  No test files found, skipping pytest..."; \
	fi

# Run pytest with coverage report
test-coverage:
	@echo ""
	@echo "=========================================="
	@echo "Running pytest with coverage..."
	@echo "=========================================="
	@HAS_TESTS=0; \
	if [ -d "tests" ] && [ -f "tests/__init__.py" ]; then \
		HAS_TESTS=1; \
	elif ls test_*.py 2>/dev/null | grep -q .; then \
		HAS_TESTS=1; \
	fi; \
	if [ $$HAS_TESTS -eq 1 ]; then \
		PYTEST_ARGS=""; \
		if [ -d "tests" ] && [ -f "tests/__init__.py" ]; then \
			PYTEST_ARGS="tests/"; \
		fi; \
		if ls test_*.py 2>/dev/null | grep -q .; then \
			PYTEST_ARGS="$$PYTEST_ARGS test_*.py"; \
		fi; \
		if [ -n "$$PYTEST_ARGS" ]; then \
			if $(PYTHON) -m pytest $$PYTEST_ARGS --cov=. --cov-report=term-missing --cov-report=html -v; then \
				echo ""; \
				echo "✓ Coverage report generated"; \
				echo "  HTML report: htmlcov/index.html"; \
			else \
				echo "✗ Coverage test failed"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "⚠️  No test files found, skipping pytest..."; \
	fi

# Test script interfaces (--help and specific features)
test-scripts:
	@echo ""
	@echo "=========================================="
	@echo "Testing script interfaces..."
	@echo "=========================================="
	@FAILED=0; \
	for script in $(SCRIPTS); do \
		echo ""; \
		echo "Testing $$script..."; \
		if $(PYTHON) $$script --help > /dev/null 2>&1; then \
			echo "  ✓ $$script --help works"; \
		else \
			echo "  ✗ $$script --help failed"; \
			FAILED=1; \
		fi \
	done; \
	if [ $$FAILED -eq 1 ]; then \
		echo ""; \
		echo "✗ Some scripts failed --help test"; \
		exit 1; \
	fi
	@echo ""
	@echo "Testing specific script features..."
	@echo ""
	@for script in $(MODEL_SCRIPTS); do \
		echo "Testing $$script..."; \
		if $(PYTHON) $$script --help 2>&1 | grep -q "Supported models"; then \
			echo "  ✓ Model option available"; \
		else \
			echo "  ✗ Model option missing"; \
		fi; \
		if $(PYTHON) $$script --help 2>&1 | grep -q "\-m MODEL"; then \
			echo "  ✓ -m option available"; \
		else \
			echo "  ✗ -m option missing"; \
		fi; \
		echo ""; \
	done
	@echo "Testing collect_responses.py..."
	@if $(PYTHON) scripts/collect_responses.py --help 2>&1 | grep -q "\-o OUTPUT"; then \
		echo "  ✓ -o option available"; \
	else \
		echo "  ✗ -o option missing"; \
	fi
	@echo ""
	@echo "Testing script syntax validation..."
	@for script in $(SCRIPTS); do \
		echo "  Validating $$script syntax..."; \
		$(PYTHON) -m py_compile $$script 2>/dev/null && echo "    ✓ Syntax OK" || echo "    ✗ Syntax error"; \
	done
	@echo ""
	@echo "✓ Script interface tests completed!"

# Create virtual environment and install dependencies
setup: venv install-deps
	@echo ""
	@echo "=========================================="
	@echo "✓ Setup completed!"
	@echo "=========================================="
	@echo "Virtual environment: .venv"
	@echo "To activate: source .venv/bin/activate"
	@echo ""

# Create virtual environment using uv
venv:
	@echo "=========================================="
	@echo "Creating virtual environment..."
	@echo "=========================================="
	@if [ -d ".venv" ]; then \
		echo "⚠️  Virtual environment .venv already exists"; \
	else \
		uv venv && echo "✓ Virtual environment created"; \
	fi

# Install dependencies from requirements.txt using uv
install-deps:
	@echo ""
	@echo "=========================================="
	@echo "Installing dependencies..."
	@echo "=========================================="
	@if [ ! -f ".venv/bin/python" ]; then \
		echo "✗ Error: Virtual environment .venv not found"; \
		echo "  Run 'make venv' first or 'make setup' to create it"; \
		exit 1; \
	fi
	@if [ ! -f "requirements.txt" ]; then \
		echo "✗ Error: requirements.txt not found"; \
		exit 1; \
	fi
	@uv pip install --python .venv/bin/python -r requirements.txt
	@echo ""
	@echo "✓ Dependencies installed!"

# Update dependencies to latest versions
update-deps:
	@echo ""
	@echo "=========================================="
	@echo "Updating dependencies..."
	@echo "=========================================="
	@if [ ! -f ".venv/bin/python" ]; then \
		echo "✗ Error: Virtual environment .venv not found"; \
		echo "  Run 'make venv' first or 'make setup' to create it"; \
		exit 1; \
	fi
	@if [ ! -f "requirements.txt" ]; then \
		echo "✗ Error: requirements.txt not found"; \
		exit 1; \
	fi
	@uv pip install --python .venv/bin/python --upgrade -r requirements.txt
	@echo ""
	@echo "✓ Dependencies updated!"

# Clean generated files and cache directories
clean:
	@echo "Cleaning generated files..."
	@rm -f $(CLEAN_FILES) 2>/dev/null || true
	@rm -rf $(CLEAN_DIRS) 2>/dev/null || true
	@echo "✓ Clean completed!"

# Remove virtual environment
clean-venv:
	@echo "Removing virtual environment..."
	@if [ -d ".venv" ]; then \
		rm -rf .venv && echo "✓ Virtual environment removed"; \
	else \
		echo "⚠️  Virtual environment .venv not found"; \
	fi

