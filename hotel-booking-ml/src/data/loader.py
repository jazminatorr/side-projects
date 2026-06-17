"""
DataLoader — Template Method pattern for data loading pipelines.

The Template Method pattern defines the skeleton of an algorithm in a base class,
deferring specific steps to subclasses. This is exactly how sklearn's BaseEstimator
and many ML frameworks (Keras, PyTorch Lightning) structure their APIs.

The invariant sequence is: load → validate → preprocess → split
Each step is defined as an abstract method that subclasses must implement.
The `prepare()` method is the "template method" — it orchestrates the steps
and should never be overridden.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Holds the result of a train/validation/test split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def val_size(self) -> int:
        return len(self.X_val)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    def summary(self) -> str:
        total = self.train_size + self.val_size + self.test_size
        return (
            f"DataSplit: train={self.train_size} ({self.train_size/total:.0%}), "
            f"val={self.val_size} ({self.val_size/total:.0%}), "
            f"test={self.test_size} ({self.test_size/total:.0%})"
        )


class DataLoader(ABC):
    """
    Abstract base class for dataset loading pipelines.

    Defines the invariant steps of loading, validating, preprocessing,
    and splitting a dataset. Subclasses implement the domain-specific
    logic for each step.

    Usage:
        loader = HotelBookingLoader("data/raw/hotel_bookings.csv")
        split = loader.prepare()
    """

    def __init__(
        self,
        data_path: str | Path,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.data_path = Path(data_path)
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self._raw_data: pd.DataFrame | None = None
        self._processed_data: pd.DataFrame | None = None

    # -------------------------------------------------------------------------
    # Template method — the invariant algorithm. Do not override.
    # -------------------------------------------------------------------------

    def prepare(self) -> DataSplit:
        """
        Execute the full loading pipeline in order.

        This is the template method. It defines *what* happens and *when*;
        subclasses define *how* each step works.
        """
        logger.info("Step 1/4: Loading data from %s", self.data_path)
        raw = self._load()
        self._raw_data = raw

        logger.info("Step 2/4: Validating schema and data quality")
        self._validate(raw)

        logger.info("Step 3/4: Preprocessing features")
        processed = self._preprocess(raw.copy())
        self._processed_data = processed

        logger.info("Step 4/4: Splitting into train / val / test")
        split = self._split(processed)

        logger.info("Done. %s", split.summary())
        return split

    # -------------------------------------------------------------------------
    # Abstract steps — subclasses must implement these.
    # -------------------------------------------------------------------------

    @abstractmethod
    def _load(self) -> pd.DataFrame:
        """Read raw data from disk. Return a DataFrame."""
        ...

    @abstractmethod
    def _validate(self, df: pd.DataFrame) -> None:
        """
        Assert that the loaded data meets minimum quality requirements.
        Raise ValueError with a descriptive message if validation fails.
        """
        ...

    @abstractmethod
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering and cleaning.
        Receives a copy of the raw data. Return the processed DataFrame
        with a 'target' column containing the labels.
        """
        ...

    # -------------------------------------------------------------------------
    # Concrete step with a sensible default — subclasses may override.
    # -------------------------------------------------------------------------

    def _split(self, df: pd.DataFrame) -> DataSplit:
        """
        Stratified train / val / test split.

        Stratification preserves the class balance across splits, which matters
        for imbalanced datasets. The default implementation handles binary targets;
        override for multi-class or regression targets.
        """
        if "target" not in df.columns:
            raise ValueError(
                "_preprocess() must produce a 'target' column. "
                f"Got columns: {list(df.columns)}"
            )

        X = df.drop(columns=["target"])
        y = df["target"]

        # First split: carve out test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        # Second split: carve val from remaining
        relative_val_size = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val_size,
            stratify=y_temp,
            random_state=self.random_state,
        )

        return DataSplit(
            X_train=X_train.reset_index(drop=True),
            X_val=X_val.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_val=y_val.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        )

    # -------------------------------------------------------------------------
    # Convenience properties (available after prepare() is called)
    # -------------------------------------------------------------------------

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data is None:
            raise RuntimeError("Call prepare() first.")
        return self._raw_data

    @property
    def processed_data(self) -> pd.DataFrame:
        if self._processed_data is None:
            raise RuntimeError("Call prepare() first.")
        return self._processed_data


# -----------------------------------------------------------------------------
# Concrete implementation for the Hotel Booking dataset
# You will flesh out _validate and _preprocess in Week 2.
# For Week 1, the stub is enough to run the EDA notebook.
# -----------------------------------------------------------------------------

# Columns that leak the target (reservation_status tells you if it was canceled)
_LEAKY_COLUMNS = ["reservation_status", "reservation_status_date"]

# High-cardinality identifier columns — drop before modeling
_ID_COLUMNS = ["agent", "company"]

# The target column in the raw dataset
_TARGET_COLUMN = "is_canceled"


class HotelBookingLoader(DataLoader):
    """
    DataLoader for the Hotel Booking Demand dataset.

    Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
    Paper: Antonio, Almeida & Nunes (2019)

    Key gotcha: `reservation_status` directly encodes the target. It MUST be
    dropped before any modeling — this is your first lesson in data leakage.
    """

    REQUIRED_COLUMNS = {
        "is_canceled", "lead_time", "hotel", "arrival_date_month",
        "stays_in_weekend_nights", "stays_in_week_nights", "adults",
        "meal", "market_segment", "deposit_type", "customer_type",
        "adr", "total_of_special_requests", "reservation_status",
    }

    def _load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}.\n"
                "Download from: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand\n"
                "Save as: data/raw/hotel_bookings.csv"
            )
        df = pd.read_csv(self.data_path)
        logger.info("Loaded %d rows × %d columns", *df.shape)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {missing}")

        if df["is_canceled"].nunique() > 2:
            raise ValueError("Target column 'is_canceled' should be binary (0/1).")

        null_pct = df.isnull().mean()
        high_null = null_pct[null_pct > 0.5]
        if not high_null.empty:
            logger.warning("Columns with >50%% nulls: %s", high_null.to_dict())

        logger.info("Validation passed. Shape: %s", df.shape)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Week 1 stub — just drops leaky columns and renames target.
        Week 2 will add full feature engineering here.
        """
        # Drop leaky columns (they directly encode the outcome)
        df = df.drop(columns=_LEAKY_COLUMNS, errors="ignore")

        # Drop high-cardinality ID columns
        df = df.drop(columns=_ID_COLUMNS, errors="ignore")

        # Rename target to the standard name the pipeline expects
        df = df.rename(columns={_TARGET_COLUMN: "target"})

        return df
