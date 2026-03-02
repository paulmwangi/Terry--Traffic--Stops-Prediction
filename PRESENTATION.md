# Terry Traffic Stops: Predicting Arrest Outcomes

**Author:** [Paul Mwangi](https://github.com/paulmwangi)

---

## Slide 1 — Title

### Terry Traffic Stops: Predicting Arrest Outcomes

A machine-learning approach to understanding and predicting arrest decisions during Terry Stops, with built-in explainability and fairness auditing.

**Dataset:** [Seattle Police Department — Terry Stops](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8/about_data)

---

## Slide 2 — Background

### What Are Terry Stops?

- In **Terry v. Ohio (1968)**, the U.S. Supreme Court ruled that police may briefly detain a person based on **reasonable suspicion** of criminal activity.
- This standard is lower than probable cause required for a formal arrest.
- **Terry Stops** — also called "stop-and-frisk" — are one of the most common forms of police–citizen encounters.
- Understanding the factors that lead to arrest during these stops is critical for transparency and accountability.

---

## Slide 3 — Problem Statement

### Why Predict Arrest Outcomes?

| Stakeholder | Need |
|---|---|
| **Law enforcement agencies** | Data-driven insights into which factors most influence arrest decisions |
| **Policymakers** | Evidence for evaluating whether stop practices are equitable |
| **Researchers** | A transparent, reproducible analytical framework for studying policing patterns |
| **Community** | Accountability and visibility into how enforcement decisions are made |

### Research Questions

1. What factors are most influential in determining whether an arrest occurs?
2. How does the presence of weapons impact arrest outcomes?
3. Is there evidence of demographic bias in arrest predictions?

---

## Slide 4 — Dataset Overview

### Seattle Police Department Terry Stops Data

| Characteristic | Detail |
|---|---|
| **Source** | Seattle Open Data Portal |
| **Records** | ~50,000+ stop encounters |
| **Time span** | Multiple years of data |
| **Target variable** | `Arrest Flag` (Y/N) |

### Key Features

- **Subject demographics:** Perceived race, perceived gender
- **Officer information:** Gender, ID
- **Stop context:** Precinct, call type (911, On View, Telephone)
- **Situational:** Weapon type, time of day
- **Temporal:** Date, time, day of week, weekend flag

### Class Distribution

The dataset is **imbalanced** — arrests occur in a minority of stops, making naive accuracy metrics misleading.

---

## Slide 5 — Data Preprocessing Pipeline

### Step-by-Step

```
Raw CSV → Missing Value Imputation → Sentinel Replacement
       → Feature Engineering → Categorical Encoding
       → SMOTE Resampling → Train/Test Split
```

| Step | Method | Rationale |
|---|---|---|
| **Missing values (numeric)** | Median imputation | Robust to outliers |
| **Missing values (categorical)** | Mode imputation | Preserves distribution |
| **Sentinel strings** | Replace `"-"`, `""` with NaN, then ffill/bfill | Handles data entry artifacts |
| **Feature engineering** | Extract `hour_of_day`, `day_of_week`, `is_weekend` | Temporal patterns in policing |
| **Categorical encoding** | OrdinalEncoder (unknown → -1) | Tree-model compatible, low dimensionality |
| **Class imbalance** | SMOTE (applied to training set only) | Generates synthetic minority samples |
| **Split** | 80/20 stratified | Preserves class ratio in both sets |

---

## Slide 6 — Feature Engineering

### Derived Features

| Feature | Source | Method |
|---|---|---|
| `hour_of_day` | Reported Time | Parse time, extract hour (0–23) |
| `day_of_week` | Reported Date | Monday=0 … Sunday=6 |
| `is_weekend` | day_of_week | 1 if Saturday/Sunday |
| `officer_stop_count` | Officer ID | Count of stops per officer (experience proxy) |
| `precinct_arrest_rate` | Precinct + Arrest Flag | Historical arrest rate per precinct |
| `is_repeat_subject` | Subject ID | Boolean: subject appears more than once |

### Why These Features?

- **Temporal features** capture known patterns (e.g., higher arrest rates at night).
- **Experience proxy** tests whether officer familiarity with stops correlates with outcomes.
- **Precinct arrest rate** provides a neighbourhood-level risk signal.
- **Repeat subject** flag captures recidivism effects.

---

## Slide 7 — Models Evaluated

### Five Classifiers + Ensemble

| Model | Type | Key Strengths |
|---|---|---|
| **Logistic Regression** | Linear | Interpretable baseline; fast |
| **Random Forest** | Bagging ensemble | Robust to overfitting; feature importance |
| **Gradient Boosting** | Sequential boosting | High accuracy; handles non-linearity |
| **XGBoost** | Optimised boosting | Regularisation; best-in-class tabular performance |
| **LightGBM** | Optimised boosting | Fastest training; leaf-wise growth |
| **Voting Ensemble** | Soft voting | Combines top-3 models; reduces variance |

### Selection Process

1. Train all five models on the SMOTE-resampled training set.
2. Evaluate each with **5-fold stratified cross-validation** on F1 and accuracy.
3. Build a **soft-voting ensemble** from the top 3 by F1.
4. Select the overall best model by mean CV F1 score.

---

## Slide 8 — Evaluation Metrics

### Why Multiple Metrics?

No single metric tells the whole story — especially with imbalanced data.

| Metric | Formula | What It Measures |
|---|---|---|
| **Accuracy** | (TP+TN) / Total | Overall correctness |
| **Precision** | TP / (TP+FP) | How many predicted arrests are real |
| **Recall** | TP / (TP+FN) | How many real arrests are caught |
| **F1 Score** | Harmonic mean of P & R | Balance of precision and recall |
| **ROC-AUC** | Area under ROC curve | Discrimination across all thresholds |

### Primary Selection Metric: **F1 Score**

F1 was chosen as the primary metric because it balances the cost of false positives (unwarranted scrutiny) and false negatives (missed arrests).

### Optimal Threshold Tuning

The default 0.5 probability threshold is not necessarily optimal. We sweep thresholds from 0.01 to 0.99 and select the one that maximises F1 on the test set.

---

## Slide 9 — Model Explainability

### SHAP Values (SHapley Additive exPlanations)

- Based on **Shapley values** from cooperative game theory.
- For each prediction, SHAP quantifies how much each feature pushed the probability toward or away from arrest.
- **TreeExplainer** computes exact SHAP values efficiently for tree models.

### Two Levels of Explanation

1. **Global:** Mean |SHAP| across all predictions → overall feature importance ranking.
2. **Local:** Per-prediction SHAP breakdown → explains *why* a specific stop was flagged.

### Example Interpretation

> "Weapon type = Firearm increased the arrest probability by +0.28, while Precinct = North decreased it by -0.05."

### Why Explainability Matters

- Builds **trust** with stakeholders.
- Enables **accountability** — predictions can be audited.
- Helps identify potential **bias pathways** in the model.

---

## Slide 10 — Bias & Fairness Auditing

### Three Fairness Metrics

| Metric | What It Checks | Acceptable Range |
|---|---|---|
| **Demographic Parity** | Same positive-prediction rate across groups | Disparity < 0.10 |
| **FPR Disparity** | Same false-positive rate across groups | Disparity < 0.10 |
| **FNR Disparity** | Same false-negative rate across groups | Disparity < 0.10 |

### Traffic-Light System

| Status | Disparity | Action |
|---|---|---|
| 🟢 Green | < 0.10 | Acceptable |
| 🟡 Yellow | 0.10 – 0.19 | Monitor and investigate |
| 🔴 Red | ≥ 0.20 | Significant — requires intervention |

### Key Finding

The analysis reveals measurable disparities in arrest predictions across racial groups. These findings underscore the importance of continuous bias monitoring and the use of this tool for **research only, not enforcement**.

---

## Slide 11 — Web Application

### TerryStops-AI Flask Application

The project includes a full-stack web application with:

- **Multi-step prediction form** — enter stop characteristics and receive a probability estimate with SHAP explanations.
- **Interactive analytics dashboard** — Plotly charts showing arrest rates by hour, race, precinct, and weapon type.
- **Bias report page** — fairness metrics with traffic-light indicators and detailed demographic analysis.
- **Model information page** — view active model metrics, feature importance, and confusion matrix.
- **REST API** — JSON endpoints for programmatic access (`/api/v1/predict`, `/api/v1/stats`, `/api/v1/model`).
- **Model registry** — manage trained models and promote the best one to active status.

### Technology Stack

| Layer | Technology |
|---|---|
| Backend | Flask 3.0, SQLAlchemy, Gunicorn |
| Frontend | Bootstrap 5 (dark theme), Plotly.js, Font Awesome |
| ML | scikit-learn, XGBoost, LightGBM, SHAP |
| Deployment | Docker, Docker Compose |

---

## Slide 12 — Key Findings

### What Drives Arrest Outcomes?

Based on model analysis and SHAP values:

1. **Weapon type** is the strongest predictor — stops involving firearms have significantly higher arrest rates.
2. **Call type** matters — 911-initiated stops differ from officer-initiated ("On View") stops.
3. **Precinct** captures neighbourhood-level variation in arrest patterns.
4. **Time of day** shows distinct patterns — late-night and early-morning stops have elevated arrest rates.
5. **Demographic features** (race, gender) contribute to predictions, reflecting patterns in the historical data that warrant careful fairness analysis.

### Model Performance

- All models exceeded the **80% accuracy** target.
- The best model (selected by F1 score) balances precision and recall effectively.
- The voting ensemble provides robust predictions by combining the strengths of multiple algorithms.

---

## Slide 13 — Limitations & Future Work

### Current Limitations

- **Historical bias:** The model learns from past policing data, which may encode systemic biases.
- **Feature proxies:** Variables like precinct or time of day may correlate with protected attributes.
- **Geographic scope:** Trained on Seattle data only — results may not generalise to other cities.
- **Static model:** Does not automatically adapt to changing policing practices over time.

### Future Directions

- **Fairness-constrained training** — incorporate fairness penalties directly into the loss function.
- **Temporal analysis** — study trends over time to detect evolving patterns.
- **Causal inference** — move beyond correlation to identify causal drivers of arrest decisions.
- **Multi-city comparison** — extend the analysis to other jurisdictions for broader insights.
- **Real-time monitoring** — deploy model drift detection for production environments.

---

## Slide 14 — Conclusion

### Summary

1. **Built a robust ML pipeline** that trains, evaluates, and compares multiple classifiers for predicting Terry Stop arrest outcomes.
2. **Achieved strong predictive performance** with an ensemble of the top-performing models.
3. **Integrated explainability** (SHAP) so every prediction can be audited and understood.
4. **Implemented fairness auditing** to detect and monitor demographic disparities.
5. **Delivered a full-stack web application** for interactive exploration and API access.

### Core Principle

> This tool exists to **increase transparency** in policing analytics, not to replace human judgment. See [ETHICS.md](terrystops-ai/ETHICS.md) for responsible-use guidelines.

---

## Slide 15 — Resources

| Resource | Link |
|---|---|
| **Full Analysis Notebook** | [Terry-Stops-Classification.ipynb](Terry-Stops-Classification.ipynb) |
| **Web Application** | [terrystops-ai/](terrystops-ai/) |
| **Methodology Documentation** | [METHODOLOGY.md](METHODOLOGY.md) |
| **Ethics Guidelines** | [terrystops-ai/ETHICS.md](terrystops-ai/ETHICS.md) |
| **Contributing Guide** | [terrystops-ai/CONTRIBUTING.md](terrystops-ai/CONTRIBUTING.md) |
| **Dataset** | [Seattle Open Data Portal](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8/about_data) |
| **Repository** | [github.com/paulmwangi/Terry--Traffic--Stops-Prediction](https://github.com/paulmwangi/Terry--Traffic--Stops-Prediction) |
