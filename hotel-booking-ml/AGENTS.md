# Agent Instructions for hotel-booking-ml

This repository is an end-to-end Python ML pipeline for hotel booking cancellation prediction.

## Recommended workflow

- Install dependencies with `make install`
- Run tests with `make test`
- Run lint with `make lint`
- Run type checking with `make typecheck`
- Use `make check` to run lint, typecheck, and tests together
- Run the API locally with `make run-api`

## Key project structure

- `src/data/` — data loader implementations
- `src/features/` — feature engineering and pipeline code
- `src/models/` — model implementations and training logic
- `src/utils/` — reusable utilities such as `FrequencyMap`
- `tests/` — unit tests for the codebase
- `docs/` — project artifacts and exploratory materials
- `notebooks/` — EDA notebooks

## Important context

- The dataset is not committed. Place `hotel_bookings.csv` at `data/raw/hotel_bookings.csv`.
- Avoid data leakage: `reservation_status` encodes the target and must not be used as input.
- `HotelBookingLoader` already handles this leakage concern.
- The repository uses `ruff` and `mypy` for static checks. Respect the configured settings in `pyproject.toml`.

## Agent behavior guidance

- Prefer concise, precise code changes.
- Prioritize thorough review and verification.
- Always back changes with tests when modifying behavior.
- Preserve the repository's existing architecture and naming conventions.

## References

- [README.md](README.md)
- [pyproject.toml](pyproject.toml)
- [Makefile](Makefile)
