# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"   # install all dependencies

make test         # run full test suite with coverage
make test-fast    # run tests without coverage, stop on first failure (-x)
make lint         # ruff check src tests
make format       # ruff format src tests
make typecheck    # mypy src (strict mode)
make check        # lint + typecheck + test
make run-api      # uvicorn on port 8000 with --reload
make notebook     # jupyter lab notebooks/
```

Run a single test file: `pytest tests/data/test_loader.py -v`

## Architecture

This is a **portfolio ML project** being built incrementally across a ~7-week plan (see `../side_project_build_plans.md`). Each week adds a layer:

- **Week 1 (done):** `DataLoader` Template Method base class + `HotelBookingLoader` + `FrequencyMap` DSA exercise + EDA notebook
- **Week 2 (next):** Feature engineering pipeline (`src/features/`), LRU cache DSA, Pipeline pattern with composable `Transformer` objects
- **Week 3:** `DecisionTreeClassifier` from scratch (`src/models/`), Visitor pattern
- **Weeks 4–7:** sklearn models, MLflow tracking, FastAPI (`src/api/main.py`), monitoring

### Data flow

`HotelBookingLoader.prepare()` → `_load` → `_validate` → `_preprocess` → `_split` → `DataSplit`

The `prepare()` template method must never be overridden. Only `_load`, `_validate`, `_preprocess` are abstract. `_split` has a stratified default that subclasses can override.

### Key constraints

- **Data leakage:** `reservation_status` and `reservation_status_date` directly encode the target and are stripped in `_preprocess`. Never use them as features.
- **`_preprocess` contract:** must return a DataFrame with a column named `target`; `_split` will raise if it's missing.
- **Dataset not committed:** place the Kaggle CSV at `data/raw/hotel_bookings.csv` before running anything that touches real data.
- **mypy strict mode:** all new code must pass `mypy src` with `strict = true`. Use `from __future__ import annotations` at the top of every module.
- **Line length:** 100 (ruff configured). Ruff rule sets: E, F, I, UP, B, SIM.
