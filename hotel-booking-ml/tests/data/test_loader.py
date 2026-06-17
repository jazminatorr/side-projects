"""
Tests for the DataLoader base class and HotelBookingLoader.

The key things to test:
- Template method enforces the correct step order
- Validation catches bad data before it reaches preprocessing
- Splits are stratified and the sizes are correct
- Leaky columns are dropped
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader, DataSplit, HotelBookingLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_minimal_df(n: int = 200, cancel_rate: float = 0.37) -> pd.DataFrame:
    """
    Create a minimal DataFrame that matches the hotel_bookings schema.
    Used in tests so we don't need the real dataset on disk.
    """
    rng = np.random.default_rng(42)
    n_cancel = int(n * cancel_rate)
    target = np.array([1] * n_cancel + [0] * (n - n_cancel))
    rng.shuffle(target)

    return pd.DataFrame({
        "is_canceled": target,
        "lead_time": rng.integers(0, 365, size=n),
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], size=n),
        "arrival_date_year": rng.choice([2015, 2016, 2017], size=n),
        "arrival_date_month": rng.choice(
            ["January", "February", "March", "April"], size=n
        ),
        "arrival_date_week_number": rng.integers(1, 53, size=n),
        "arrival_date_day_of_month": rng.integers(1, 32, size=n),
        "stays_in_weekend_nights": rng.integers(0, 5, size=n),
        "stays_in_week_nights": rng.integers(0, 8, size=n),
        "adults": rng.integers(1, 4, size=n),
        "children": rng.integers(0, 3, size=n).astype(float),
        "babies": rng.integers(0, 2, size=n),
        "meal": rng.choice(["BB", "HB", "SC"], size=n),
        "country": rng.choice(["PRT", "GBR", "FRA", "ESP"], size=n),
        "market_segment": rng.choice(["Online TA", "Direct", "Corporate"], size=n),
        "distribution_channel": rng.choice(["TA/TO", "Direct"], size=n),
        "is_repeated_guest": rng.integers(0, 2, size=n),
        "previous_cancellations": rng.integers(0, 5, size=n),
        "previous_bookings_not_canceled": rng.integers(0, 5, size=n),
        "reserved_room_type": rng.choice(["A", "B", "C"], size=n),
        "assigned_room_type": rng.choice(["A", "B", "C"], size=n),
        "booking_changes": rng.integers(0, 5, size=n),
        "deposit_type": rng.choice(["No Deposit", "Non Refund"], size=n),
        "agent": rng.choice([9, 14, 240, np.nan], size=n),
        "company": rng.choice([np.nan, 40, 223], size=n),
        "days_in_waiting_list": rng.integers(0, 100, size=n),
        "customer_type": rng.choice(["Transient", "Contract"], size=n),
        "adr": rng.uniform(0, 300, size=n).round(2),
        "required_car_parking_spaces": rng.integers(0, 3, size=n),
        "total_of_special_requests": rng.integers(0, 5, size=n),
        "reservation_status": np.where(target == 1, "Canceled", "Check-Out"),
        "reservation_status_date": pd.date_range("2015-01-01", periods=n).astype(str),
    })


@pytest.fixture
def csv_path(tmp_path) -> Path:
    """Write a minimal hotel bookings CSV to a temp file and return its path."""
    df = _make_minimal_df(n=500)
    path = tmp_path / "hotel_bookings.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# DataSplit tests
# ---------------------------------------------------------------------------

class TestDataSplit:
    def test_sizes_reported_correctly(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series([0] * 100)
        split = DataSplit(
            X_train=X.iloc[:70], X_val=X.iloc[70:85], X_test=X.iloc[85:],
            y_train=y.iloc[:70], y_val=y.iloc[70:85], y_test=y.iloc[85:],
        )
        assert split.train_size == 70
        assert split.val_size == 15
        assert split.test_size == 15

    def test_summary_string(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series([0] * 100)
        split = DataSplit(
            X_train=X.iloc[:70], X_val=X.iloc[70:85], X_test=X.iloc[85:],
            y_train=y.iloc[:70], y_val=y.iloc[70:85], y_test=y.iloc[85:],
        )
        summary = split.summary()
        assert "train=" in summary
        assert "val=" in summary
        assert "test=" in summary


# ---------------------------------------------------------------------------
# HotelBookingLoader tests
# ---------------------------------------------------------------------------

class TestHotelBookingLoader:
    def test_prepare_returns_data_split(self, csv_path):
        loader = HotelBookingLoader(csv_path)
        split = loader.prepare()
        assert isinstance(split, DataSplit)

    def test_split_sizes_sum_to_total(self, csv_path):
        loader = HotelBookingLoader(csv_path)
        split = loader.prepare()
        total = split.train_size + split.val_size + split.test_size
        assert total == 500

    def test_leaky_columns_dropped(self, csv_path):
        loader = HotelBookingLoader(csv_path)
        split = loader.prepare()
        for col in ["reservation_status", "reservation_status_date"]:
            assert col not in split.X_train.columns, f"{col} should be dropped"

    def test_target_not_in_features(self, csv_path):
        loader = HotelBookingLoader(csv_path)
        split = loader.prepare()
        assert "is_canceled" not in split.X_train.columns
        assert "target" not in split.X_train.columns

    def test_target_is_binary(self, csv_path):
        loader = HotelBookingLoader(csv_path)
        split = loader.prepare()
        for y in [split.y_train, split.y_val, split.y_test]:
            assert set(y.unique()).issubset({0, 1})

    def test_stratified_cancel_rate(self, csv_path):
        """Train, val, and test sets should have similar cancellation rates."""
        loader = HotelBookingLoader(csv_path)
        split = loader.prepare()
        overall = pd.concat([split.y_train, split.y_val, split.y_test]).mean()
        for y, name in [
            (split.y_train, "train"),
            (split.y_val, "val"),
            (split.y_test, "test"),
        ]:
            diff = abs(y.mean() - overall)
            assert diff < 0.05, (
                f"{name} cancel rate {y.mean():.3f} deviates too much "
                f"from overall {overall:.3f}"
            )

    def test_file_not_found_raises(self, tmp_path):
        loader = HotelBookingLoader(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.prepare()

    def test_raw_data_accessible_after_prepare(self, csv_path):
        loader = HotelBookingLoader(csv_path)
        loader.prepare()
        assert loader.raw_data is not None
        assert "reservation_status" in loader.raw_data.columns  # raw has leaky cols

    def test_raw_data_raises_before_prepare(self, csv_path):
        loader = HotelBookingLoader(csv_path)
        with pytest.raises(RuntimeError):
            _ = loader.raw_data
