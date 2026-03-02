"""Admin routes blueprint."""
from flask import Blueprint, render_template, redirect, url_for, flash

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/admin/models')
def list_models():
    """List all registered models."""
    models = []
    try:
        from app.models.db_models import ModelRegistry
        models = ModelRegistry.query.order_by(ModelRegistry.id.desc()).all()
    except Exception:
        flash('Unable to load model registry.', 'warning')
    return render_template('admin_models.html', models=models)


@admin_bp.route('/admin/models/promote/<int:model_id>', methods=['POST'])
def promote_model(model_id):
    """Promote a model to active status."""
    try:
        from app.models.db_models import db, ModelRegistry
        # Deactivate all current models
        ModelRegistry.query.update({'is_active': False})
        # Activate the selected model
        model = ModelRegistry.query.get_or_404(model_id)
        model.is_active = True
        db.session.commit()
        flash(f'Model {model_id} promoted to active.', 'success')
    except Exception:
        flash('Unable to promote model. Please try again.', 'danger')
    return redirect(url_for('admin.list_models'))
