"""Training script for the Terry Stops arrest-prediction models.

Trains five classifiers, cross-validates each, builds a VotingClassifier
from the top three, selects the best model by F1 score, and persists all
artifacts to ``ml/saved_models/``.

Usage::

    python -m ml.train
"""

import json
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Append project root so sibling packages are importable.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.utils.preprocessing import TerryStopsPreprocessor  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Paths ----------------------------------------------------------------
_BASE_DIR = os.path.dirname(__file__)
_DATA_PATH = os.path.join(_BASE_DIR, "..", "data", "raw", "Terry_Stops.csv")
_SAVE_DIR = os.path.join(_BASE_DIR, "saved_models")


def _build_candidates():
    """Return a list of ``(name, estimator)`` tuples to evaluate.

    Returns:
        List of 2-tuples with model name and unfitted estimator.
    """
    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ]

    try:
        from xgboost import XGBClassifier

        candidates.append(
            ("XGBoost", XGBClassifier(n_estimators=200, use_label_encoder=False,
                                      eval_metric="logloss", random_state=42))
        )
    except ImportError:
        logger.warning("xgboost not installed – skipping XGBClassifier.")

    try:
        from lightgbm import LGBMClassifier

        candidates.append(
            ("LightGBM", LGBMClassifier(n_estimators=200, random_state=42, verbose=-1))
        )
    except ImportError:
        logger.warning("lightgbm not installed – skipping LGBMClassifier.")

    return candidates


def train():
    """Execute the full training pipeline.

    Steps:
        1. Load and preprocess data via ``TerryStopsPreprocessor``.
        2. Train each candidate model with 5-fold cross-validation.
        3. Build a ``VotingClassifier`` from the top-3 models.
        4. Save every model and its metadata to disk.
        5. Mark the best model (by mean CV F1) as *active*.

    Returns:
        Dict mapping model name to its cross-validation metrics.
    """
    # 1. Data --------------------------------------------------------------
    preprocessor = TerryStopsPreprocessor(_DATA_PATH)
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_split()
    feature_names = [
        c for c in preprocessor.FEATURE_COLS if c in preprocessor.df.columns
    ]

    os.makedirs(_SAVE_DIR, exist_ok=True)

    # 2. Train & cross-validate --------------------------------------------
    candidates = _build_candidates()
    results = {}

    for name, model in candidates:
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)

        cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        metrics = {
            "model_name": name,
            "cv_f1_mean": round(float(np.mean(cv_f1)), 4),
            "cv_f1_std": round(float(np.std(cv_f1)), 4),
            "cv_accuracy_mean": round(float(np.mean(cv_acc)), 4),
            "feature_names": feature_names,
            "is_active": False,
        }
        results[name] = metrics

        # Persist model + metadata.
        model_path = os.path.join(_SAVE_DIR, f"{name}.pkl")
        meta_path = os.path.join(_SAVE_DIR, f"{name}_metadata.json")
        joblib.dump(model, model_path)
        with open(meta_path, "w") as fh:
            json.dump(metrics, fh, indent=2)

        print(f"{name:25s}  F1={metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}  "
              f"Acc={metrics['cv_accuracy_mean']:.4f}")

    # 3. Voting ensemble from top-3 ----------------------------------------
    sorted_models = sorted(results.items(), key=lambda x: x[1]["cv_f1_mean"], reverse=True)
    top3 = sorted_models[:3]
    top3_estimators = [(n, dict(candidates)[n]) for n, _ in top3]

    # Re-fit top-3 estimators so the VotingClassifier has fitted sub-models.
    for _, est in top3_estimators:
        est.fit(X_train, y_train)

    voting = VotingClassifier(estimators=top3_estimators, voting="soft")
    voting.fit(X_train, y_train)
    cv_f1_voting = cross_val_score(voting, X_train, y_train, cv=5, scoring="f1")

    voting_metrics = {
        "model_name": "VotingEnsemble",
        "cv_f1_mean": round(float(np.mean(cv_f1_voting)), 4),
        "cv_f1_std": round(float(np.std(cv_f1_voting)), 4),
        "component_models": [n for n, _ in top3],
        "feature_names": feature_names,
        "is_active": False,
    }
    results["VotingEnsemble"] = voting_metrics

    voting_path = os.path.join(_SAVE_DIR, "VotingEnsemble.pkl")
    voting_meta = os.path.join(_SAVE_DIR, "VotingEnsemble_metadata.json")
    joblib.dump(voting, voting_path)
    with open(voting_meta, "w") as fh:
        json.dump(voting_metrics, fh, indent=2)

    print(f"{'VotingEnsemble':25s}  F1={voting_metrics['cv_f1_mean']:.4f} "
          f"± {voting_metrics['cv_f1_std']:.4f}")

    # 4. Select best by F1 and mark active ---------------------------------
    all_sorted = sorted(results.items(), key=lambda x: x[1]["cv_f1_mean"], reverse=True)
    best_name = all_sorted[0][0]
    results[best_name]["is_active"] = True

    best_meta_path = os.path.join(_SAVE_DIR, f"{best_name}_metadata.json")
    with open(best_meta_path, "w") as fh:
        json.dump(results[best_name], fh, indent=2)

    print(f"\n✓ Best model: {best_name} (F1={results[best_name]['cv_f1_mean']:.4f})")
    return results


if __name__ == "__main__":
    train()
