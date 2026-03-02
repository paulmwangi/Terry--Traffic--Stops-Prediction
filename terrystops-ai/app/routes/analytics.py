"""Analytics routes blueprint."""
import os
from flask import Blueprint, render_template, jsonify, current_app

analytics_bp = Blueprint('analytics', __name__)


def _load_analytics_data():
    """Load and aggregate data for analytics charts."""
    data = {'by_hour': [], 'by_race': [], 'by_precinct': []}
    try:
        import pandas as pd
        csv_path = os.path.join(current_app.config.get('DATA_DIR', ''), 'raw', 'Terry_Stops.csv')
        if not os.path.exists(csv_path):
            return data
        df = pd.read_csv(csv_path)

        # Convert Arrest Flag from Y/N to numeric
        if 'Arrest Flag' in df.columns:
            df['arrest_numeric'] = df['Arrest Flag'].map({'Y': 1, 'N': 0}).fillna(0)

        # Arrests by hour
        if 'Reported Time' in df.columns and 'arrest_numeric' in df.columns:
            try:
                df['hour'] = pd.to_datetime(df['Reported Time'], format='%H:%M:%S.%f', errors='coerce').dt.hour
                hourly = df.groupby('hour')['arrest_numeric'].mean().reset_index()
                hourly.columns = ['hour', 'arrest_rate']
                data['by_hour'] = hourly.dropna().to_dict('records')
            except Exception:
                pass

        # Arrests by race
        if 'Subject Perceived Race' in df.columns and 'arrest_numeric' in df.columns:
            race_stats = df.groupby('Subject Perceived Race')['arrest_numeric'].agg(
                ['mean', 'count']
            ).reset_index()
            race_stats.columns = ['race', 'arrest_rate', 'count']
            data['by_race'] = race_stats.to_dict('records')

        # Arrests by precinct
        if 'Precinct' in df.columns and 'arrest_numeric' in df.columns:
            precinct_stats = df.groupby('Precinct')['arrest_numeric'].agg(
                ['mean', 'count']
            ).reset_index()
            precinct_stats.columns = ['precinct', 'arrest_rate', 'count']
            data['by_precinct'] = precinct_stats.to_dict('records')

    except Exception:
        pass
    return data


@analytics_bp.route('/analytics')
def analytics():
    """Render the analytics page."""
    return render_template('analytics.html')


@analytics_bp.route('/analytics/data')
def analytics_data():
    """Return aggregated analytics data as JSON for charts."""
    data = _load_analytics_data()
    return jsonify(data)


@analytics_bp.route('/analytics/feature-importance')
def feature_importance():
    """Render the feature importance page."""
    importance = {}
    try:
        from app.models.ml_model import ModelPredictor
        predictor = ModelPredictor(current_app.config['MODEL_DIR'])
        if hasattr(predictor, 'get_feature_importance'):
            importance = predictor.get_feature_importance()
    except Exception:
        pass
    return render_template('analytics.html', importance=importance)


@analytics_bp.route('/bias-report')
def bias_report():
    """Render the bias report page."""
    return render_template('bias_report.html')
