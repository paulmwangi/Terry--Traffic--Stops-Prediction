"""Main routes blueprint."""
import glob
import json
import os

from flask import Blueprint, render_template, jsonify, current_app

main_bp = Blueprint('main', __name__)


def _load_stats():
    """Load basic statistics from CSV data and model registry."""
    stats = {'total_stops': 0, 'arrest_rate': 0.0, 'model_accuracy': 0.0}
    try:
        import pandas as pd
        csv_path = os.path.join(current_app.config.get('DATA_DIR', ''), 'raw', 'Terry_Stops.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            stats['total_stops'] = len(df)
            if 'Arrest Flag' in df.columns:
                arrest = df['Arrest Flag'].map({'Y': 1, 'N': 0}).fillna(0)
                stats['arrest_rate'] = round(arrest.mean() * 100, 2)
    except Exception:
        pass

    try:
        from app.models.db_models import ModelRegistry
        active_model = ModelRegistry.query.filter_by(is_active=True).first()
        if active_model and hasattr(active_model, 'accuracy'):
            stats['model_accuracy'] = round(active_model.accuracy * 100, 2)
    except Exception:
        pass

    return stats


@main_bp.route('/')
def index():
    """Render the home page with summary statistics."""
    stats = _load_stats()
    return render_template('index.html', stats=stats)


@main_bp.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html')


def _load_metrics_from_files():
    """Fall back to reading model metadata from JSON sidecar files."""
    model_dir = current_app.config.get('MODEL_DIR', '')
    if not os.path.isdir(model_dir):
        return {}
    meta_files = sorted(glob.glob(os.path.join(model_dir, '*_metadata.json')))
    for mf in meta_files:
        try:
            with open(mf, 'r') as fh:
                meta = json.load(fh)
            if meta.get('is_active'):
                raw_id = os.path.splitext(os.path.basename(mf))[0]
                model_id = raw_id.replace('_metadata', '')
                return {
                    'id': model_id,
                    'name': meta.get('model_name', 'N/A'),
                    'accuracy': meta.get('cv_accuracy_mean', 0),
                    'precision': meta.get('precision', 0),
                    'recall': meta.get('recall', 0),
                    'f1_score': meta.get('cv_f1_mean', 0),
                    'roc_auc': meta.get('roc_auc', 0),
                    'created_at': meta.get('created_at', None),
                }
        except (OSError, json.JSONDecodeError):
            continue
    return {}


@main_bp.route('/model-info')
def model_info():
    """Render model information page with active model metrics."""
    metrics = {}
    try:
        from app.models.db_models import ModelRegistry
        active_model = ModelRegistry.query.filter_by(is_active=True).first()
        if active_model:
            metrics = {
                'id': active_model.id,
                'name': getattr(active_model, 'name', 'N/A'),
                'accuracy': getattr(active_model, 'accuracy', 0),
                'precision': getattr(active_model, 'precision_score', 0),
                'recall': getattr(active_model, 'recall_score', 0),
                'f1_score': getattr(active_model, 'f1_score', 0),
                'roc_auc': getattr(active_model, 'roc_auc', 0),
                'created_at': getattr(active_model, 'trained_at', None),
            }
    except Exception:
        pass

    if not metrics:
        metrics = _load_metrics_from_files()

    return render_template('model_info.html', metrics=metrics)


@main_bp.route('/health')
def health():
    """Return JSON health status."""
    return jsonify({'status': 'healthy', 'app': 'TerryStops-AI'})
