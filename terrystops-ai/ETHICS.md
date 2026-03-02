# ⚖️ Ethics & Responsible Use Policy

**This tool is intended strictly for research and educational purposes.**

TerryStops-AI analyzes historical Terry Stop data to surface patterns, measure bias, and make model decisions transparent. It must **never** be used to make or influence real-world policing decisions.

---

## Intended Use

This project is designed for:

- **Academic Research** — studying predictive policing models, their accuracy, and their limitations.
- **Bias Detection** — identifying and quantifying demographic disparities embedded in historical stop data.
- **Policy Analysis** — informing evidence-based discussions about policing practices and reform.
- **Education** — teaching responsible machine learning, fairness-aware modeling, and model explainability.

---

## Prohibited Use

TerryStops-AI **must NOT** be used for:

- **Actual policing decisions** — including, but not limited to, deciding whom to stop, search, or arrest.
- **Profiling** — targeting individuals or communities based on race, ethnicity, gender, age, or any other protected characteristic.
- **Discrimination** — any application that reinforces or amplifies discriminatory practices.
- **Surveillance** — monitoring or tracking individuals or groups.

Any use that violates these prohibitions is a misuse of this software and is contrary to the values of this project.

---

## Bias Awareness

The data used by this application reflects **historical policing practices** that may contain systemic bias. Patterns learned by the model are a reflection of *past decisions made by officers*, not objective ground truth. Users must interpret all outputs with this context in mind.

Key considerations:

- Arrest rates may differ across demographic groups due to policing practices, not underlying behavior.
- Feature importance rankings can reflect biased data collection, not causal relationships.
- High model accuracy does **not** imply fairness or correctness.

---

## Fairness Metrics

The built-in **bias audit module** (`app/utils/bias_audit.py`) computes the following fairness metrics across demographic groups:

| Metric | Description |
|---|---|
| **Demographic Parity** | Whether positive prediction rates are equal across groups |
| **Equalized Odds** | Whether true-positive and false-positive rates are equal across groups |
| **Disparate Impact Ratio** | Ratio of positive prediction rates between groups |
| **Calibration** | Whether predicted probabilities align with actual outcomes per group |

These metrics are surfaced in the bias audit dashboard and API responses to promote accountability and transparency.

---

## Data Privacy

- **No personally identifiable information (PII) is exposed** by this application.
- All subject and officer IDs are hashed or anonymized before processing.
- The dataset is sourced from publicly available Seattle Police Department open data.
- No additional data enrichment or re-identification is performed.

---

## Transparency

All model predictions are accompanied by explanations:

- **SHAP values** provide global and local feature-importance breakdowns.
- **LIME explanations** offer human-readable reasoning for individual predictions.
- Model metadata (algorithm, hyperparameters, training date, performance metrics) is available via the `/api/v1/model` endpoint.

Users can inspect *why* a prediction was made, not just *what* the prediction is.

---

## Limitations

- The model reflects **historical patterns** which may embed systemic bias in policing.
- Predictions are probabilistic estimates, not determinations of guilt or innocence.
- The model has been trained on Seattle-specific data and may not generalize to other jurisdictions.
- Changes in policing policy, demographics, or data collection practices may reduce model relevance over time.
- This tool does **not** account for unreported stops or encounters absent from the dataset.

---

## Contact

If you have concerns about the ethical implications of this project or believe it is being misused, please open an issue in the repository or contact the maintainers directly.
