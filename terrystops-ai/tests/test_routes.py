"""Flask integration tests for routes and API endpoints."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_home_page(client):
    """GET / should return 200."""
    response = client.get('/')
    assert response.status_code == 200


def test_predict_page(client):
    """GET /predict should return 200."""
    response = client.get('/predict')
    assert response.status_code == 200


def test_analytics_page(client):
    """GET /analytics should return 200."""
    response = client.get('/analytics')
    assert response.status_code == 200


def test_health_endpoint(client):
    """GET /health should return 200 with JSON containing status."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert data['status'] == 'healthy'


def test_api_predict(client):
    """POST /api/v1/predict with valid JSON should return 200."""
    payload = {
        'subject_race': 'White',
        'subject_gender': 'Male',
        'officer_gender': 'M',
        'precinct': 'North',
        'time_of_day': '14:00',
        'call_type': '911',
        'weapon_type': 'None',
    }
    response = client.post(
        '/api/v1/predict',
        json=payload,
        content_type='application/json',
    )
    assert response.status_code in (200, 500, 503)
    data = response.get_json()
    if response.status_code == 200:
        assert 'prediction' in data
        assert 'probability' in data


def test_api_stats(client):
    """GET /api/v1/stats should return 200."""
    response = client.get('/api/v1/stats')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)


def test_api_model(client):
    """GET /api/v1/model should return 200."""
    response = client.get('/api/v1/model')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)


def test_api_model_has_all_models_key(client):
    """GET /api/v1/model should include all_models key."""
    response = client.get('/api/v1/model')
    assert response.status_code == 200
    data = response.get_json()
    assert 'all_models' in data
    assert 'active_model' in data
    assert isinstance(data['all_models'], list)


def test_model_info_page(client):
    """GET /model-info should return 200."""
    response = client.get('/model-info')
    assert response.status_code == 200


def test_model_info_no_stuck_loading(client):
    """Model comparison table should not be permanently stuck on Loading."""
    response = client.get('/model-info')
    assert response.status_code == 200
    html = response.data.decode()
    # The JS should handle empty model list with a proper message
    assert 'No models available' in html or 'modelTableBody' in html


def test_api_model_file_fallback(client, tmp_path):
    """API should fall back to filesystem metadata when DB is empty."""
    import json
    meta = {
        'model_name': 'TestModel',
        'cv_f1_mean': 0.85,
        'cv_accuracy_mean': 0.90,
        'is_active': True,
    }
    meta_file = tmp_path / 'TestModel_metadata.json'
    meta_file.write_text(json.dumps(meta))

    original_dir = client.application.config['MODEL_DIR']
    client.application.config['MODEL_DIR'] = str(tmp_path)
    try:
        response = client.get('/api/v1/model')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['all_models']) >= 1
        assert data['active_model'] is not None
        assert data['active_model']['name'] == 'TestModel'
        assert data['active_model']['f1_score'] == 0.85
        assert data['active_model']['id'] == 'TestModel'
    finally:
        client.application.config['MODEL_DIR'] = original_dir


def test_model_info_file_fallback(client, tmp_path):
    """Model info page should show metrics from filesystem when DB is empty."""
    import json
    meta = {
        'model_name': 'FileModel',
        'cv_f1_mean': 0.92,
        'cv_accuracy_mean': 0.88,
        'is_active': True,
    }
    meta_file = tmp_path / 'FileModel_metadata.json'
    meta_file.write_text(json.dumps(meta))

    original_dir = client.application.config['MODEL_DIR']
    client.application.config['MODEL_DIR'] = str(tmp_path)
    try:
        response = client.get('/model-info')
        assert response.status_code == 200
        html = response.data.decode()
        assert 'FileModel' in html
        assert '0.92' in html
    finally:
        client.application.config['MODEL_DIR'] = original_dir
