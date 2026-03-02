"""Prediction routes blueprint."""
from flask import Blueprint, render_template, flash, current_app
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField
from wtforms.validators import DataRequired

predict_bp = Blueprint('predict', __name__)


class StopPredictionForm(FlaskForm):
    """Form for Terry Stop prediction input."""
    subject_race = SelectField(
        'Subject Race',
        choices=[
            ('White', 'White'),
            ('Black or African American', 'Black or African American'),
            ('Hispanic or Latino', 'Hispanic or Latino'),
            ('Asian', 'Asian'),
            ('American Indian/Alaska Native', 'American Indian/Alaska Native'),
            ('Native Hawaiian/Other Pacific Islander', 'Native Hawaiian/Other Pacific Islander'),
            ('Other', 'Other'),
        ],
        validators=[DataRequired()],
    )
    subject_gender = SelectField(
        'Subject Gender',
        choices=[
            ('Male', 'Male'),
            ('Female', 'Female'),
            ('Non-Binary', 'Non-Binary'),
        ],
        validators=[DataRequired()],
    )
    officer_gender = SelectField(
        'Officer Gender',
        choices=[('M', 'Male'), ('F', 'Female')],
        validators=[DataRequired()],
    )
    precinct = SelectField(
        'Precinct',
        choices=[
            ('North', 'North'),
            ('South', 'South'),
            ('East', 'East'),
            ('West', 'West'),
            ('Southwest', 'Southwest'),
        ],
        validators=[DataRequired()],
    )
    time_of_day = StringField('Time of Day', validators=[DataRequired()])
    call_type = SelectField(
        'Call Type',
        choices=[
            ('911', '911'),
            ('ONVIEW', 'On View'),
            ('TELEPHONE OTHER', 'Telephone Other'),
        ],
        validators=[DataRequired()],
    )
    weapon_type = SelectField(
        'Weapon Type',
        choices=[
            ('None', 'None'),
            ('Firearm', 'Firearm'),
            ('Knife', 'Knife'),
            ('Other', 'Other'),
        ],
        validators=[DataRequired()],
    )


@predict_bp.route('/predict', methods=['GET'])
def predict_form():
    """Render the prediction form."""
    form = StopPredictionForm()
    return render_template('predict.html', form=form)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Process prediction form and return results."""
    form = StopPredictionForm()
    if not form.validate_on_submit():
        flash('Please correct the errors in the form.', 'danger')
        return render_template('predict.html', form=form)

    features = {
        'subject_race': form.subject_race.data,
        'subject_gender': form.subject_gender.data,
        'officer_gender': form.officer_gender.data,
        'precinct': form.precinct.data,
        'time_of_day': form.time_of_day.data,
        'call_type': form.call_type.data,
        'weapon_type': form.weapon_type.data,
    }

    try:
        from app.models.ml_model import ModelPredictor
        predictor = ModelPredictor(current_app.config['MODEL_DIR'])
        result = predictor.predict(features)
        if result is None:
            flash('No model is currently loaded. Please train a model first.', 'warning')
            return render_template('predict.html', form=form)
    except Exception:
        flash('Unable to process prediction. Please try again later.', 'danger')
        return render_template('predict.html', form=form)

    try:
        from app.models.db_models import db, PredictionLog
        log = PredictionLog(
            input_features=str(features),
            prediction=result.get('prediction', 0),
            probability=result.get('probability', 0.0),
        )
        db.session.add(log)
        db.session.commit()
    except Exception:
        pass

    return render_template('predict.html', form=form, result=result, features=features)
