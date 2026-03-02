"""Model evaluation utilities.

Provides functions for computing classification metrics, finding an
optimal decision threshold, and generating diagnostic plots.
"""

import logging
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name="model"):
    """Compute standard classification metrics on a test set.

    Args:
        model: A fitted estimator with ``predict`` and, optionally,
            ``predict_proba`` methods.
        X_test: Test feature matrix.
        y_test: True labels for the test set.
        model_name: Human-readable model name for logging.

    Returns:
        Dict with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, and ``roc_auc``.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
    else:
        metrics["roc_auc"] = None

    logger.info("%s metrics: %s", model_name, metrics)
    return metrics


def find_optimal_threshold(model, X_test, y_test):
    """Find the decision threshold that maximises F1 score.

    Iterates over thresholds from 0.01 to 0.99 in steps of 0.01.

    Args:
        model: Fitted estimator with ``predict_proba``.
        X_test: Test feature matrix.
        y_test: True labels.

    Returns:
        Dict with ``best_threshold`` and ``best_f1``.

    Raises:
        AttributeError: If *model* does not support ``predict_proba``.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_test, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    logger.info("Optimal threshold: %.2f  (F1=%.4f)", best_threshold, best_f1)
    return {
        "best_threshold": round(float(best_threshold), 2),
        "best_f1": round(float(best_f1), 4),
    }


def generate_evaluation_plots(model, X_test, y_test, output_dir):
    """Generate and save confusion-matrix and ROC-curve plots.

    Args:
        model: Fitted estimator with ``predict`` and ``predict_proba``.
        X_test: Test feature matrix.
        y_test: True labels.
        output_dir: Directory where PNG files are written.

    Returns:
        Dict with paths to the saved plots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    y_pred = model.predict(X_test)

    # --- Confusion matrix -------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    paths["confusion_matrix"] = cm_path

    # --- ROC curve --------------------------------------------------------
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        roc_path = os.path.join(output_dir, "roc_curve.png")
        fig.savefig(roc_path, bbox_inches="tight")
        plt.close(fig)
        paths["roc_curve"] = roc_path

    logger.info("Evaluation plots saved to %s", output_dir)
    return paths
