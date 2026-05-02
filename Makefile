.PHONY: test lint format typecheck check install-dev benchmark-smoke

# Install dev dependencies (requires pip)
install-dev:
	pip install -e ".[dev]"

# Run all unit tests with verbose output
test:
	pytest tests/ -v

# Run tests with coverage report
test-cov:
	pytest tests/ -v --tb=short --cov=token_reducer --cov-report=term-missing

# Lint: check for errors and style issues
lint:
	ruff check scripts/ tests/

# Format: auto-fix lint issues and reformat code
format:
	ruff check --fix scripts/ tests/
	ruff format scripts/ tests/

# Type check with mypy (strict mode)
typecheck:
	mypy scripts/token_reducer/ --strict

# Run pyright for additional static analysis
pyright:
	pyright scripts/token_reducer/

# Benchmark harness (requires repo root as CWD; PYTHONPATH=scripts if not pip install -e .)
benchmark-smoke:
	PYTHONPATH=scripts python -m token_reducer.cli proof-harness run --tier smoke --repo-root .

