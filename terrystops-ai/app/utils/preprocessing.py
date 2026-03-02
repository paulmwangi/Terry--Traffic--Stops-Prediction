"""Data preprocessing pipeline for the Terry Stops dataset.

Provides the TerryStopsPreprocessor class that handles loading,
cleaning, feature engineering, encoding, class-imbalance handling,
and train/test splitting.
"""

import logging
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


class TerryStopsPreprocessor:
    """End-to-end preprocessing for the Terry Stops CSV dataset.

    Args:
        data_path: Absolute or relative path to ``Terry_Stops.csv``.

    Attributes:
        df: The working pandas DataFrame.
        encoder: Fitted OrdinalEncoder for categorical columns.
        categorical_cols: List of categorical column names to encode.
    """

    TARGET = "Arrest Flag"
    CATEGORICAL_COLS = [
        "Subject Perceived Race",
        "Subject Perceived Gender",
        "Officer Gender",
        "Precinct",
        "Weapon Type",
        "Call Type",
    ]
    FEATURE_COLS = [
        "Subject Perceived Race",
        "Subject Perceived Gender",
        "Officer Gender",
        "Precinct",
        "Weapon Type",
        "Call Type",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
    ]

    def __init__(self, data_path):
        """Load the raw CSV into a DataFrame.

        Args:
            data_path: Path to the Terry_Stops.csv file.

        Raises:
            FileNotFoundError: If *data_path* does not exist.
        """
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.df = pd.read_csv(data_path)
        self.encoder = None
        self.categorical_cols = list(self.CATEGORICAL_COLS)
        logger.info("Loaded %d rows from %s", len(self.df), data_path)

    def clean_data(self):
        """Handle missing values with median/mode imputation.

        Numeric columns are filled with their median; categorical
        columns are filled with their mode.

        Returns:
            self, for method chaining.
        """
        for col in self.df.select_dtypes(include="number").columns:
            if self.df[col].isna().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                logger.info("Imputed %s with median %.4f", col, median_val)

        for col in self.df.select_dtypes(include="object").columns:
            if self.df[col].isna().any():
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
                logger.info("Imputed %s with mode '%s'", col, mode_val)

        # Replace sentinel strings with NaN then re-impute.
        self.df.replace({"-": np.nan, "": np.nan}, inplace=True)
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

        logger.info("Data cleaning complete. Shape: %s", self.df.shape)
        return self

    def engineer_features(self):
        """Create derived features from raw date/time columns.

        New columns:
            * ``hour_of_day`` – extracted from *Reported Time*.
            * ``day_of_week`` – Monday=0 … Sunday=6 from *Reported Date*.
            * ``is_weekend`` – 1 if Saturday or Sunday, else 0.

        The target column ``Arrest Flag`` is converted to binary
        (``Y`` → 1, ``N`` → 0).

        Returns:
            self, for method chaining.
        """
        if "Reported Time" in self.df.columns:
            self.df["hour_of_day"] = (
                pd.to_datetime(self.df["Reported Time"], format="%H:%M:%S.%f", errors="coerce")
                .dt.hour
                .fillna(0)
                .astype(int)
            )
        else:
            self.df["hour_of_day"] = 0

        if "Reported Date" in self.df.columns:
            dt = pd.to_datetime(self.df["Reported Date"], errors="coerce")
            self.df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
            self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)
        else:
            self.df["day_of_week"] = 0
            self.df["is_weekend"] = 0

        if self.TARGET in self.df.columns:
            self.df[self.TARGET] = self.df[self.TARGET].map({"Y": 1, "N": 0})
            self.df[self.TARGET].fillna(0, inplace=True)
            self.df[self.TARGET] = self.df[self.TARGET].astype(int)

        logger.info("Feature engineering complete.")
        return self

    def encode_features(self):
        """Ordinally encode categorical feature columns.

        Uses ``OrdinalEncoder`` with ``handle_unknown='use_encoded_value'``
        and ``unknown_value=-1`` so unseen categories at inference time
        do not raise errors.

        Returns:
            self, for method chaining.
        """
        cols_to_encode = [c for c in self.categorical_cols if c in self.df.columns]
        if not cols_to_encode:
            logger.warning("No categorical columns found to encode.")
            return self

        self.df[cols_to_encode] = self.df[cols_to_encode].astype(str)

        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.df[cols_to_encode] = self.encoder.fit_transform(self.df[cols_to_encode])
        logger.info("Encoded %d categorical columns.", len(cols_to_encode))
        return self

    def handle_imbalance(self, X, y):
        """Apply SMOTE to balance the training data.

        Args:
            X: Feature matrix (array-like).
            y: Target vector (array-like).

        Returns:
            Tuple ``(X_resampled, y_resampled)`` after SMOTE.
        """
        logger.info("Class distribution before SMOTE: %s", dict(pd.Series(y).value_counts()))
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info("Class distribution after  SMOTE: %s", dict(pd.Series(y_res).value_counts()))
        return X_res, y_res

    def get_train_test_split(self, test_size=0.2, apply_smote=True):
        """Run the full pipeline and return train/test splits.

        Calls :meth:`clean_data`, :meth:`engineer_features`, and
        :meth:`encode_features` in sequence, then splits the data.

        Args:
            test_size: Fraction of data reserved for testing.
            apply_smote: Whether to apply SMOTE to the training set.

        Returns:
            Tuple ``(X_train, X_test, y_train, y_test)`` as numpy
            arrays.
        """
        self.clean_data().engineer_features().encode_features()

        available = [c for c in self.FEATURE_COLS if c in self.df.columns]
        X = self.df[available].values
        y = self.df[self.TARGET].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        if apply_smote:
            X_train, y_train = self.handle_imbalance(X_train, y_train)

        logger.info(
            "Split complete. Train: %d, Test: %d", len(X_train), len(X_test)
        )
        return X_train, X_test, y_train, y_test

    def save_processed(self, output_dir):
        """Save the processed DataFrame as a Parquet file.

        Args:
            output_dir: Directory in which to write
                ``processed_terry_stops.parquet``.
        """
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "processed_terry_stops.parquet")
        self.df.to_parquet(out_path, index=False)
        logger.info("Saved processed data to %s", out_path)
