"""Fairness and bias auditing utilities.

Provides the BiasAuditor class for computing group-level fairness
metrics such as demographic parity, false-positive rates, and
false-negative rates across sensitive attribute groups.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BiasAuditor:
    """Audits a classifier for bias across sensitive-feature groups.

    Args:
        y_true: Ground-truth binary labels (array-like of 0/1).
        y_pred: Predicted binary labels (array-like of 0/1).
        sensitive_features: Array-like of group labels (e.g. race or
            gender) with the same length as *y_true*.

    Attributes:
        y_true: Ground-truth labels as a numpy array.
        y_pred: Predicted labels as a numpy array.
        groups: Sensitive-feature array as a numpy array.
    """

    # Thresholds for the traffic-light status indicator.
    _YELLOW_THRESHOLD = 0.10
    _RED_THRESHOLD = 0.20

    def __init__(self, y_true, y_pred, sensitive_features):
        """Store arrays and validate lengths.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            sensitive_features: Group membership labels.

        Raises:
            ValueError: If input arrays have mismatched lengths.
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.groups = np.asarray(sensitive_features)

        if not (len(self.y_true) == len(self.y_pred) == len(self.groups)):
            raise ValueError("y_true, y_pred, and sensitive_features must have equal length.")

    def demographic_parity_difference(self):
        """Compute the demographic parity difference.

        The selection rate for a group is the proportion of positive
        predictions within that group.  The demographic parity
        difference is ``max(rate) - min(rate)`` across all groups.

        Returns:
            A dict with ``selection_rates`` per group and the overall
            ``demographic_parity_diff``.
        """
        unique_groups = np.unique(self.groups)
        rates = {}
        for g in unique_groups:
            mask = self.groups == g
            rates[str(g)] = float(self.y_pred[mask].mean()) if mask.sum() > 0 else 0.0

        dp_diff = max(rates.values()) - min(rates.values()) if rates else 0.0
        return {"selection_rates": rates, "demographic_parity_diff": round(dp_diff, 4)}

    def false_positive_rate_by_group(self):
        """Compute the false-positive rate for each group.

        FPR = FP / (FP + TN) within each group.

        Returns:
            Dict mapping group label to its FPR.
        """
        unique_groups = np.unique(self.groups)
        fpr = {}
        for g in unique_groups:
            mask = self.groups == g
            yt = self.y_true[mask]
            yp = self.y_pred[mask]
            neg = (yt == 0)
            fp = ((yp == 1) & neg).sum()
            tn = ((yp == 0) & neg).sum()
            fpr[str(g)] = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
        return fpr

    def false_negative_rate_by_group(self):
        """Compute the false-negative rate for each group.

        FNR = FN / (FN + TP) within each group.

        Returns:
            Dict mapping group label to its FNR.
        """
        unique_groups = np.unique(self.groups)
        fnr = {}
        for g in unique_groups:
            mask = self.groups == g
            yt = self.y_true[mask]
            yp = self.y_pred[mask]
            pos = (yt == 1)
            fn = ((yp == 0) & pos).sum()
            tp = ((yp == 1) & pos).sum()
            fnr[str(g)] = round(float(fn / (fn + tp)), 4) if (fn + tp) > 0 else 0.0
        return fnr

    def compute_all_metrics(self):
        """Return all fairness metrics in a single dict.

        Returns:
            Dict with keys ``demographic_parity``, ``fpr_by_group``,
            and ``fnr_by_group``.
        """
        return {
            "demographic_parity": self.demographic_parity_difference(),
            "fpr_by_group": self.false_positive_rate_by_group(),
            "fnr_by_group": self.false_negative_rate_by_group(),
        }

    def generate_report(self):
        """Generate a summary report with traffic-light indicators.

        Status levels:
            * **green** – disparity < 0.10
            * **yellow** – 0.10 ≤ disparity < 0.20
            * **red** – disparity ≥ 0.20

        Returns:
            Dict containing ``metrics`` (all computed fairness metrics)
            and ``status`` (traffic-light indicator per metric category).
        """
        metrics = self.compute_all_metrics()
        dp_diff = metrics["demographic_parity"]["demographic_parity_diff"]

        fpr_vals = list(metrics["fpr_by_group"].values())
        fpr_range = max(fpr_vals) - min(fpr_vals) if fpr_vals else 0.0

        fnr_vals = list(metrics["fnr_by_group"].values())
        fnr_range = max(fnr_vals) - min(fnr_vals) if fnr_vals else 0.0

        def _status(value):
            if value >= self._RED_THRESHOLD:
                return "red"
            if value >= self._YELLOW_THRESHOLD:
                return "yellow"
            return "green"

        return {
            "metrics": metrics,
            "status": {
                "demographic_parity": _status(dp_diff),
                "fpr_disparity": _status(fpr_range),
                "fnr_disparity": _status(fnr_range),
            },
        }
