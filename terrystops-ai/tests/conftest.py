"""Shared pytest fixtures for TerryStops-AI tests."""
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app, login_manager  # noqa: E402


@login_manager.user_loader
def _load_user(user_id):
    """Dummy user loader for testing."""
    return None


@pytest.fixture
def app():
    """Create a Flask application configured for testing."""
    application = create_app('testing')
    yield application


@pytest.fixture
def client(app):
    """Create a Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_data():
    """Return a small synthetic DataFrame mimicking Terry Stops data."""
    return pd.DataFrame({
        'Subject Perceived Race': ['White', 'Black or African American',
                                   'Hispanic or Latino', 'Asian', 'White'],
        'Subject Perceived Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Officer Gender': ['M', 'F', 'M', 'M', 'F'],
        'Precinct': ['North', 'South', 'East', 'West', 'North'],
        'Weapon Type': ['None', 'Firearm', 'None', 'Knife', 'None'],
        'Call Type': ['911', 'ONVIEW', '911', 'ONVIEW', '911'],
        'Reported Date': ['2023-01-01', '2023-01-02', '2023-01-03',
                          '2023-01-07', '2023-01-05'],
        'Reported Time': ['08:30:00.000', '14:15:00.000', '22:00:00.000',
                          '03:45:00.000', '18:00:00.000'],
        'Arrest Flag': ['Y', 'N', 'N', 'Y', 'N'],
    })
