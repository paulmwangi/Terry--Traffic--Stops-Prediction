"""SQLAlchemy database models for the TerryStops-AI application.

Defines the ORM models for prediction logging, model registry,
daily statistics, and admin audit logging.
"""

from datetime import datetime, date

from app import db


class PredictionLog(db.Model):
    """Stores individual prediction requests and their results.

    Attributes:
        id: Primary key.
        timestamp: UTC time of the prediction request.
        input_features: JSON string of the input feature dictionary.
        prediction: Binary prediction result (0 or 1).
        probability: Model confidence probability.
        model_version: Version string of the model used.
        ip_hash: SHA-256 hash of the requester's IP address.
    """

    __tablename__ = "prediction_log"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    input_features = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(50), nullable=True)
    ip_hash = db.Column(db.String(64), nullable=True)

    def __repr__(self):
        return f"<PredictionLog {self.id} pred={self.prediction}>"


class ModelRegistry(db.Model):
    """Registry of trained ML models and their evaluation metrics.

    Attributes:
        id: Primary key.
        name: Human-readable model name.
        version: Semantic version string.
        file_path: Path to the serialized model file.
        accuracy: Accuracy score on the test set.
        f1_score: F1 score on the test set.
        precision_score: Precision score on the test set.
        recall_score: Recall score on the test set.
        roc_auc: ROC-AUC score on the test set.
        metrics_json: Full metrics dictionary as a JSON string.
        is_active: Whether this model is the currently active model.
        trained_at: UTC time the model was trained.
    """

    __tablename__ = "model_registry"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    version = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    precision_score = db.Column(db.Float, nullable=True)
    recall_score = db.Column(db.Float, nullable=True)
    roc_auc = db.Column(db.Float, nullable=True)
    metrics_json = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=False)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelRegistry {self.name} v{self.version} active={self.is_active}>"


class DailyStats(db.Model):
    """Aggregated daily prediction statistics.

    Attributes:
        id: Primary key.
        date: Calendar date (unique).
        total_predictions: Number of predictions made on this date.
        arrest_predictions: Number of positive (arrest) predictions.
        avg_probability: Mean prediction probability for the day.
    """

    __tablename__ = "daily_stats"

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, unique=True, nullable=False, default=date.today)
    total_predictions = db.Column(db.Integer, default=0)
    arrest_predictions = db.Column(db.Integer, default=0)
    avg_probability = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f"<DailyStats {self.date} total={self.total_predictions}>"


class AdminLog(db.Model):
    """Audit log for administrative actions.

    Attributes:
        id: Primary key.
        action: Short description of the action performed.
        timestamp: UTC time of the action.
        details: Optional extended details or context.
    """

    __tablename__ = "admin_log"

    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<AdminLog {self.id} action={self.action!r}>"
