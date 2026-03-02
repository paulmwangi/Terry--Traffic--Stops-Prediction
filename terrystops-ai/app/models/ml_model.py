"""ML model loading and prediction interface.

Provides the ModelPredictor class that wraps a trained scikit-learn
model for inference, including probability output and optional SHAP
explanations.
"""

import glob
import json
import logging
import os

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Loads a trained model from disk and exposes a prediction API.

    Args:
        model_dir: Path to the directory containing saved model files.
            Expects ``*.pkl`` or ``*.joblib`` files and an optional
            ``*_metadata.json`` sidecar with feature names and metrics.

    Attributes:
        model: The loaded scikit-learn estimator (or ``None``).
        feature_names: List of feature names the model was trained on.
        model_path: Filesystem path of the loaded model file.
        metadata: Dict of model metadata loaded from the JSON sidecar.
    """

    LABEL_MAP = {1: "Arrest", 0: "No Arrest"}

    def __init__(self, model_dir):
        """Initialise the predictor by scanning *model_dir* for models.

        The loader first looks for a file whose companion JSON metadata
        has ``"is_active": true``.  If none is found it falls back to the
        first ``.pkl`` / ``.joblib`` file discovered.

        Args:
            model_dir: Directory that contains serialized model files.
        """
        self.model = None
        self.feature_names = []
        self.model_path = None
        self.metadata = {}
        self._load_model(model_dir)

    def _load_model(self, model_dir):
        """Scan *model_dir* and load the best available model.

        Args:
            model_dir: Directory containing model artifacts.
        """
        if not os.path.isdir(model_dir):
            logger.warning("Model directory does not exist: %s", model_dir)
            return

        model_files = sorted(
            glob.glob(os.path.join(model_dir, "*.pkl"))
            + glob.glob(os.path.join(model_dir, "*.joblib"))
        )

        if not model_files:
            logger.warning("No model files found in %s", model_dir)
            return

        # Try to find the active model via metadata sidecars.
        active_path = None
        for mf in model_files:
            meta_path = os.path.splitext(mf)[0] + "_metadata.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as fh:
                    meta = json.load(fh)
                if meta.get("is_active"):
                    active_path = mf
                    self.metadata = meta
                    break

        chosen = active_path or model_files[0]
        try:
            self.model = joblib.load(chosen)
            self.model_path = chosen
            logger.info("Loaded model from %s", chosen)
        except Exception:
            logger.exception("Failed to load model from %s", chosen)
            return

        # Load metadata if not already loaded.
        if not self.metadata:
            meta_path = os.path.splitext(chosen)[0] + "_metadata.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as fh:
                    self.metadata = json.load(fh)

        self.feature_names = self.metadata.get("feature_names", [])

    def predict(self, features_dict):
        """Run a single prediction from a raw feature dictionary.

        Args:
            features_dict: Mapping of feature-name → value.  Keys that
                are not in ``self.feature_names`` are silently ignored;
                missing features are filled with ``0``.

        Returns:
            A dict with keys ``prediction`` (int 0/1), ``probability``
            (float), ``label`` (str), and ``shap_values`` (dict).
            Returns ``None`` if no model is loaded.
        """
        if self.model is None:
            logger.error("No model loaded – cannot predict.")
            return None

        feature_names = self.feature_names
        if not feature_names:
            # Fall back to dict keys in deterministic order.
            feature_names = sorted(features_dict.keys())

        row = np.array(
            [features_dict.get(f, 0) for f in feature_names]
        ).reshape(1, -1)

        prediction = int(self.model.predict(row)[0])

        probability = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(row)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])

        shap_values = self._compute_shap(row, feature_names)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "label": self.LABEL_MAP.get(prediction, "Unknown"),
            "shap_values": shap_values,
        }

    def _compute_shap(self, row, feature_names):
        """Attempt to compute SHAP values for a single instance.

        Args:
            row: 2-D numpy array of shape ``(1, n_features)``.
            feature_names: Ordered list of feature names.

        Returns:
            Dict mapping feature name to its SHAP contribution, or an
            empty dict on failure.
        """
        try:
            import shap

            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(row)
            if isinstance(sv, list):
                sv = sv[1]
            return dict(zip(feature_names, sv[0].tolist()))
        except Exception:
            logger.debug("SHAP computation skipped or failed.", exc_info=True)
            return {}

    def get_model_info(self):
        """Return metadata about the currently loaded model.

        Returns:
            A dict with ``model_path``, ``model_type``,
            ``feature_names``, ``is_loaded``, and any extra fields from
            the metadata JSON.
        """
        info = {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__ if self.model else None,
            "feature_names": self.feature_names,
            "is_loaded": self.model is not None,
        }
        info.update(self.metadata)
        return info
