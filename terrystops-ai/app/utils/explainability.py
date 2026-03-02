"""Model explainability utilities powered by SHAP.

Provides the ModelExplainer class for computing global and local
feature-importance explanations.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ModelExplainer:
    """SHAP-based explainer for tree and linear models.

    Args:
        model: A fitted scikit-learn-compatible estimator.
        feature_names: Ordered list of feature names.
        X_background: Background dataset (numpy array or DataFrame)
            used by the SHAP explainer for expected-value estimation.

    Attributes:
        model: The wrapped estimator.
        feature_names: List of feature names.
        explainer: A SHAP ``TreeExplainer`` (or ``None`` on failure).
        shap_values: Cached SHAP values for *X_background*.
    """

    def __init__(self, model, feature_names, X_background):
        """Initialise the SHAP explainer.

        Args:
            model: Fitted estimator.
            feature_names: Feature name list matching column order.
            X_background: Background sample array for SHAP.
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.X_background = np.asarray(X_background)
        self.explainer = None
        self.shap_values = None

        try:
            import shap

            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialised successfully.")
        except Exception:
            logger.warning(
                "Could not create TreeExplainer; SHAP features disabled.",
                exc_info=True,
            )

    def _ensure_shap_values(self):
        """Compute SHAP values for the background set if not cached."""
        if self.shap_values is not None:
            return
        if self.explainer is None:
            return
        try:
            sv = self.explainer.shap_values(self.X_background)
            # For binary classifiers shap_values returns a list of two arrays.
            self.shap_values = sv[1] if isinstance(sv, list) else sv
        except Exception:
            logger.warning("Failed to compute SHAP values.", exc_info=True)

    def global_feature_importance(self):
        """Compute mean absolute SHAP value per feature.

        Returns:
            Dict mapping feature name to its mean |SHAP| value, sorted
            descending.  Returns an empty dict if SHAP is unavailable.
        """
        self._ensure_shap_values()
        if self.shap_values is None:
            return {}

        mean_abs = np.abs(self.shap_values).mean(axis=0)
        importance = {
            name: round(float(val), 6)
            for name, val in zip(self.feature_names, mean_abs)
        }
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def explain_single(self, instance):
        """Explain a single prediction.

        Args:
            instance: 1-D array-like of feature values (same order as
                ``feature_names``).

        Returns:
            Dict with ``feature_contributions`` mapping feature name
            to its SHAP value, and ``base_value`` (the expected model
            output).  Returns an empty dict on failure.
        """
        if self.explainer is None:
            return {}

        try:
            row = np.asarray(instance).reshape(1, -1)
            sv = self.explainer.shap_values(row)
            if isinstance(sv, list):
                sv = sv[1]

            contributions = {
                name: round(float(val), 6)
                for name, val in zip(self.feature_names, sv[0])
            }

            base = self.explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = base[1] if len(base) > 1 else base[0]

            return {
                "feature_contributions": contributions,
                "base_value": round(float(base), 6),
            }
        except Exception:
            logger.warning("Single-instance SHAP explanation failed.", exc_info=True)
            return {}

    def get_summary_data(self):
        """Return data suitable for rendering a SHAP summary plot.

        Returns:
            Dict with ``shap_values`` (2-D list), ``feature_names``,
            and ``X_background`` (2-D list).  Returns an empty dict if
            SHAP values are unavailable.
        """
        self._ensure_shap_values()
        if self.shap_values is None:
            return {}

        return {
            "shap_values": self.shap_values.tolist(),
            "feature_names": self.feature_names,
            "X_background": self.X_background.tolist(),
        }
