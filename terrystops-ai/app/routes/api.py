"""API v1 routes blueprint."""
import glob
import json
import os

from flask import Blueprint, request, jsonify, current_app

from app import csrf

api_bp = Blueprint('api', __name__)
csrf.exempt(api_bp)


@api_bp.route('/predict', methods=['POST'])
def api_predict():
    """Accept JSON input and return prediction with probability."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON payload'}), 400

    required_fields = [
        'subject_race', 'subject_gender', 'officer_gender',
        'precinct', 'time_of_day', 'call_type', 'weapon_type',
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

    try:
        from app.models.ml_model import ModelPredictor
        from flask import current_app
        predictor = ModelPredictor(current_app.config['MODEL_DIR'])
        result = predictor.predict(data)
        if result is None:
            return jsonify({'error': 'No model loaded. Please train a model first.'}), 503
    except Exception:
        return jsonify({'error': 'Prediction failed. Please try again later.'}), 500

    try:
        from app.models.db_models import db, PredictionLog
        log = PredictionLog(
            input_features=str(data),
            prediction=result.get('prediction', 0),
            probability=result.get('probability', 0.0),
        )
        db.session.add(log)
        db.session.commit()
    except Exception:
        pass

    return jsonify({
        'prediction': result.get('prediction'),
        'probability': result.get('probability'),
        'shap_values': result.get('shap_values'),
    })


@api_bp.route('/stats', methods=['GET'])
def api_stats():
    """Return aggregate statistics as JSON."""
    stats = {'total_predictions': 0, 'arrest_predictions': 0}
    try:
        from app.models.db_models import PredictionLog
        stats['total_predictions'] = PredictionLog.query.count()
        stats['arrest_predictions'] = PredictionLog.query.filter_by(prediction=1).count()
    except Exception:
        pass
    return jsonify(stats)


def _load_all_models_from_files():
    """Load model metadata from JSON sidecar files on disk."""
    model_dir = current_app.config.get('MODEL_DIR', '')
    if not os.path.isdir(model_dir):
        return []
    models = []
    meta_files = sorted(glob.glob(os.path.join(model_dir, '*_metadata.json')))
    for mf in meta_files:
        try:
            with open(mf, 'r') as fh:
                meta = json.load(fh)
            raw_id = os.path.splitext(os.path.basename(mf))[0]
            model_id = raw_id.replace('_metadata', '')
            models.append({
                'id': model_id,
                'name': meta.get('model_name', 'N/A'),
                'accuracy': meta.get('cv_accuracy_mean', 0),
                'precision': meta.get('precision', 0),
                'recall': meta.get('recall', 0),
                'f1_score': meta.get('cv_f1_mean', 0),
                'roc_auc': meta.get('roc_auc', 0),
                'is_active': meta.get('is_active', False),
                'created_at': meta.get('created_at', ''),
            })
        except (OSError, json.JSONDecodeError):
            continue
    return models


@api_bp.route('/model', methods=['GET'])
def api_model():
    """Return active model metadata and all models as JSON."""
    model_info = {'active_model': None, 'all_models': []}
    try:
        from app.models.db_models import ModelRegistry
        active = ModelRegistry.query.filter_by(is_active=True).first()
        if active:
            model_info['active_model'] = {
                'id': active.id,
                'name': getattr(active, 'name', 'N/A'),
                'accuracy': getattr(active, 'accuracy', 0),
                'precision': getattr(active, 'precision_score', 0),
                'recall': getattr(active, 'recall_score', 0),
                'f1_score': getattr(active, 'f1_score', 0),
                'roc_auc': getattr(active, 'roc_auc', 0),
                'is_active': active.is_active,
                'created_at': str(getattr(active, 'trained_at', '')),
            }
        all_db_models = ModelRegistry.query.all()
        if all_db_models:
            model_info['all_models'] = [{
                'id': m.id,
                'name': getattr(m, 'name', 'N/A'),
                'accuracy': getattr(m, 'accuracy', 0),
                'precision': getattr(m, 'precision_score', 0),
                'recall': getattr(m, 'recall_score', 0),
                'f1_score': getattr(m, 'f1_score', 0),
                'roc_auc': getattr(m, 'roc_auc', 0),
                'is_active': m.is_active,
                'created_at': str(getattr(m, 'trained_at', '')),
            } for m in all_db_models]
    except Exception:
        pass

    # Fall back to filesystem metadata when DB has no models
    if not model_info['all_models']:
        file_models = _load_all_models_from_files()
        if file_models:
            model_info['all_models'] = file_models
            active_from_files = [m for m in file_models if m.get('is_active')]
            if active_from_files and not model_info['active_model']:
                model_info['active_model'] = active_from_files[0]

    return jsonify(model_info)
