# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a multi-project portfolio repository for an AI/ML career transition, built across a ~14-week incremental plan. See `side_project_build_plans.md` for the full schedule and design intent for each project.

## Projects

| Directory | Status | Description |
|-----------|--------|-------------|
| `hotel-booking-ml/` | Active (Week 1 done) | End-to-end ML pipeline predicting hotel booking cancellations |
| *(AI Interview Coach)* | Planned (Week 4) | FastAPI backend + Next.js frontend, LLM-powered interview evaluation |
| *(Code/Test Analyzer)* | Planned (Week 10) | Python AST parser + LLM integration for test coverage analysis |

## hotel-booking-ml

Each sub-project has its own `CLAUDE.md`. For `hotel-booking-ml`, see `hotel-booking-ml/CLAUDE.md` — it covers commands, architecture, and constraints in full.

Quick reference:

```bash
cd hotel-booking-ml
pip install -e ".[dev]"   # one-time setup

make test         # full test suite with coverage
make test-fast    # stop on first failure, no coverage
make lint         # ruff check src tests
make typecheck    # mypy src (strict)
make check        # lint + typecheck + test
```

Run a single test file: `pytest tests/data/test_loader.py -v`

## Cross-project conventions

- Python `src/` layout with `pyproject.toml` for each project
- `ruff` for linting/formatting, `mypy` with `strict = true` for type checking
- `from __future__ import annotations` required at the top of every module
- 80%+ test coverage target; tests live in a sibling `tests/` directory mirroring `src/`
- Design patterns are intentional and named — Template Method, Strategy, Pipeline, Observer, etc. Preserve them when extending code.
