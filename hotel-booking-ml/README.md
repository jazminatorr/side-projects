# Hotel Booking Cancellation Predictor

An end-to-end ML pipeline predicting whether a hotel booking will be cancelled,
built as a portfolio project to demonstrate production-quality ML engineering.

## What it does

Given features available at booking time (lead time, deposit type, market segment, etc.),
the model predicts cancellation probability. The pipeline covers the full ML lifecycle:
EDA → feature engineering → model training → evaluation → REST API deployment.

## Tech Stack

- **Python 3.11** · pandas · scikit-learn · numpy
- **MLflow** for experiment tracking
- **FastAPI** for the prediction API
- **pytest** for testing (80%+ coverage)

## Project Structure

```
hotel-booking-ml/
├── src/
│   ├── data/        # DataLoader (Template Method pattern)
│   ├── features/    # Feature engineering pipeline
│   ├── models/      # Model implementations (decision tree from scratch + sklearn)
│   └── utils/       # FrequencyMap and other utilities
├── notebooks/
│   └── 01_eda.ipynb # Exploratory data analysis
├── tests/           # Full test suite
├── data/
│   ├── raw/         # Original CSV (not committed)
│   └── processed/   # Engineered features (not committed)
└── docs/            # EDA charts and artifacts
```

## Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
make test

# Start EDA
make notebook
```

## Dataset

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — download and save as `data/raw/hotel_bookings.csv`.

⚠️ **Key data leakage warning:** `reservation_status` directly encodes the target and must be dropped before any modeling. This is handled automatically by `HotelBookingLoader`.

## Design Patterns Used

- **Template Method** — `DataLoader` base class defines the loading pipeline skeleton; `HotelBookingLoader` implements domain-specific steps
- **Strategy** — pluggable evaluation metrics
- **Pipeline** — composable feature transformer chain
- **Observer** — training callbacks for MLflow logging and early stopping
