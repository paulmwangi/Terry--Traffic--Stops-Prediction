"""Unit tests for TerryStopsPreprocessor."""
import sys
import os
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.utils.preprocessing import TerryStopsPreprocessor  # noqa: E402


@pytest.fixture
def synthetic_csv(sample_data):
    """Write sample_data to a temporary CSV and return its path."""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False
    ) as f:
        sample_data.to_csv(f, index=False)
        path = f.name
    yield path
    os.unlink(path)


def test_preprocessor_creates_features(synthetic_csv):
    """Verify that feature engineering produces expected columns."""
    pp = TerryStopsPreprocessor(synthetic_csv)
    pp.clean_data().engineer_features()
    assert 'hour_of_day' in pp.df.columns
    assert 'day_of_week' in pp.df.columns
    assert 'is_weekend' in pp.df.columns


def test_preprocessor_handles_missing(sample_data):
    """Verify that missing values are imputed."""
    sample_data.loc[0, 'Weapon Type'] = None
    sample_data.loc[1, 'Precinct'] = None
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False
    ) as f:
        sample_data.to_csv(f, index=False)
        path = f.name
    try:
        pp = TerryStopsPreprocessor(path)
        pp.clean_data()
        assert pp.df['Weapon Type'].isna().sum() == 0
        assert pp.df['Precinct'].isna().sum() == 0
    finally:
        os.unlink(path)


def test_train_test_split_shapes(synthetic_csv):
    """Verify that the split produces correct shapes."""
    pp = TerryStopsPreprocessor(synthetic_csv)
    X_train, X_test, y_train, y_test = pp.get_train_test_split(
        test_size=0.4, apply_smote=False
    )
    total = len(X_train) + len(X_test)
    assert total == 5
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
