.PHONY: install test lint format typecheck run-api clean

install:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest -x --no-cov

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

run-api:
	uvicorn src.api.main:app --reload --port 8000

notebook:
	jupyter lab notebooks/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name "*.pyc" -delete

check: lint typecheck test
