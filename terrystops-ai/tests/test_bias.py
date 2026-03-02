"""Bias audit unit tests for BiasAuditor."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.utils.bias_audit import BiasAuditor  # noqa: E402


@pytest.fixture
def auditor():
    """Create a BiasAuditor with known values."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0, 1, 1])
    groups = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    return BiasAuditor(y_true, y_pred, groups)


def test_demographic_parity(auditor):
    """Verify demographic parity computation with known inputs."""
    result = auditor.demographic_parity_difference()
    assert 'selection_rates' in result
    assert 'demographic_parity_diff' in result
    assert 'A' in result['selection_rates']
    assert 'B' in result['selection_rates']
    # Group A: preds [1,0,0,1] => rate 0.5; Group B: preds [1,0,1,1] => rate 0.75
    assert result['selection_rates']['A'] == pytest.approx(0.5)
    assert result['selection_rates']['B'] == pytest.approx(0.75)
    assert result['demographic_parity_diff'] == pytest.approx(0.25)


def test_fpr_by_group(auditor):
    """Verify false positive rates are computed correctly."""
    fpr = auditor.false_positive_rate_by_group()
    assert 'A' in fpr
    assert 'B' in fpr
    # Group A: true=[1,0,1,0], pred=[1,0,0,1] => neg idx 1,3 => FP=1, TN=1 => FPR=0.5
    assert fpr['A'] == pytest.approx(0.5)
    # Group B: true=[1,0,1,0], pred=[1,0,1,1] => neg idx 1,3 => FP=1, TN=1 => FPR=0.5
    assert fpr['B'] == pytest.approx(0.5)


def test_fnr_by_group(auditor):
    """Verify false negative rates are computed correctly."""
    fnr = auditor.false_negative_rate_by_group()
    assert 'A' in fnr
    assert 'B' in fnr
    # Group A: true=[1,0,1,0], pred=[1,0,0,1] => pos idx 0,2 => FN=1, TP=1 => FNR=0.5
    assert fnr['A'] == pytest.approx(0.5)
    # Group B: true=[1,0,1,0], pred=[1,0,1,1] => pos idx 0,2 => FN=0, TP=2 => FNR=0.0
    assert fnr['B'] == pytest.approx(0.0)


def test_report_generation(auditor):
    """Verify report has expected structure."""
    report = auditor.generate_report()
    assert 'metrics' in report
    assert 'status' in report
    assert 'demographic_parity' in report['metrics']
    assert 'fpr_by_group' in report['metrics']
    assert 'fnr_by_group' in report['metrics']
    assert 'demographic_parity' in report['status']
    assert 'fpr_disparity' in report['status']
    assert 'fnr_disparity' in report['status']
    assert report['status']['demographic_parity'] in ('green', 'yellow', 'red')
