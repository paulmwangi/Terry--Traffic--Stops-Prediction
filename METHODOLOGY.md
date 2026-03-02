# Methodology: Terry Stops Arrest Outcome Prediction

This document provides an in-depth explanation of every method, algorithm, and design decision used in the Terry Stops arrest-prediction pipeline. It is intended for researchers, data scientists, and reviewers who want to understand **what** the project does, **how** each component works, and **why** each choice was made.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Data Collection & Description](#2-data-collection--description)
3. [Data Preprocessing](#3-data-preprocessing)
   - 3.1 [Missing-Value Imputation](#31-missing-value-imputation)
   - 3.2 [Sentinel-String Replacement](#32-sentinel-string-replacement)
4. [Feature Engineering](#4-feature-engineering)
   - 4.1 [Temporal Features](#41-temporal-features)
   - 4.2 [Target Encoding](#42-target-encoding)
   - 4.3 [Categorical Encoding — Ordinal Encoder](#43-categorical-encoding--ordinal-encoder)
   - 4.4 [Additional Feature Engineering Helpers](#44-additional-feature-engineering-helpers)
5. [Class Imbalance Handling — SMOTE](#5-class-imbalance-handling--smote)
6. [Train / Test Split](#6-train--test-split)
7. [Model Selection & Training](#7-model-selection--training)
   - 7.1 [Logistic Regression](#71-logistic-regression)
   - 7.2 [Random Forest](#72-random-forest)
   - 7.3 [Gradient Boosting](#73-gradient-boosting)
   - 7.4 [XGBoost](#74-xgboost)
   - 7.5 [LightGBM](#75-lightgbm)
   - 7.6 [Voting Ensemble](#76-voting-ensemble)
8. [Cross-Validation Strategy](#8-cross-validation-strategy)
9. [Evaluation Metrics](#9-evaluation-metrics)
   - 9.1 [Accuracy](#91-accuracy)
   - 9.2 [Precision](#92-precision)
   - 9.3 [Recall](#93-recall)
   - 9.4 [F1 Score](#94-f1-score)
   - 9.5 [ROC-AUC](#95-roc-auc)
   - 9.6 [Confusion Matrix](#96-confusion-matrix)
   - 9.7 [Optimal Threshold Selection](#97-optimal-threshold-selection)
10. [Model Explainability](#10-model-explainability)
    - 10.1 [SHAP (SHapley Additive exPlanations)](#101-shap-shapley-additive-explanations)
    - 10.2 [LIME (Local Interpretable Model-agnostic Explanations)](#102-lime-local-interpretable-model-agnostic-explanations)
11. [Bias Auditing & Fairness](#11-bias-auditing--fairness)
    - 11.1 [Demographic Parity](#111-demographic-parity)
    - 11.2 [False Positive Rate by Group](#112-false-positive-rate-by-group)
    - 11.3 [False Negative Rate by Group](#113-false-negative-rate-by-group)
    - 11.4 [Traffic-Light Status Indicators](#114-traffic-light-status-indicators)
12. [Model Persistence & Registry](#12-model-persistence--registry)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Limitations & Ethical Considerations](#14-limitations--ethical-considerations)
15. [References](#15-references)

---

## 1. Problem Framing

The task is a **binary classification** problem:

> *Given observable characteristics of a Terry Stop (location, time, subject demographics, weapon presence, call type, officer information), predict whether the stop will result in an arrest.*

| Aspect | Detail |
|---|---|
| **Type** | Supervised binary classification |
| **Target variable** | `Arrest Flag` — `Y` (arrest) or `N` (no arrest) |
| **Positive class** | 1 (arrest made) |
| **Negative class** | 0 (no arrest) |

**Why binary classification?**  The arrest outcome is inherently dichotomous. Binary classification algorithms are well-studied, interpretable, and allow the use of calibrated probability outputs — essential when the model serves as a decision-support tool rather than a decision-maker.

---

## 2. Data Collection & Description

The dataset is sourced from the [Seattle Police Department's Terry Stops open data portal](https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8/about_data). Each row represents one Terry Stop encounter.

### Key columns used

| Column | Type | Description |
|---|---|---|
| `Subject Perceived Race` | Categorical | Officer-perceived race of the subject |
| `Subject Perceived Gender` | Categorical | Officer-perceived gender of the subject |
| `Officer Gender` | Categorical | Gender of the officer conducting the stop |
| `Precinct` | Categorical | Seattle police precinct (North, South, East, West, Southwest) |
| `Weapon Type` | Categorical | Type of weapon found, if any |
| `Call Type` | Categorical | How the stop was initiated (911, On View, Telephone) |
| `Reported Date` | DateTime | Date the stop was reported |
| `Reported Time` | DateTime | Time the stop was reported |
| `Arrest Flag` | Binary | **Target** — Y or N |

**Why this dataset?**  It is publicly available, officially maintained, and contains a rich set of features relevant to the research questions. The combination of demographic, temporal, and situational variables enables the study of both predictive accuracy and fairness.

---

## 3. Data Preprocessing

Preprocessing is handled by the `TerryStopsPreprocessor` class in `app/utils/preprocessing.py`.

### 3.1 Missing-Value Imputation

```python
# Numeric columns → median imputation
for col in df.select_dtypes(include="number").columns:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns → mode imputation
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
```

**Why median for numeric columns?**  The median is robust to outliers. Unlike the mean, a few extreme values (e.g., an unusual officer ID number) will not skew the imputed value.

**Why mode for categorical columns?**  The mode replaces missing categories with the most common value, preserving the distribution shape better than introducing an arbitrary "Unknown" category.

### 3.2 Sentinel-String Replacement

```python
df.replace({"-": np.nan, "": np.nan}, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
```

**Why?**  The raw dataset uses `"-"` and empty strings as placeholders for missing data. These are replaced with `NaN` and then filled using forward-fill and back-fill to handle any remaining gaps after the initial imputation pass.

---

## 4. Feature Engineering

### 4.1 Temporal Features

Three features are derived from the raw date/time columns:

| Feature | Source Column | Derivation | Rationale |
|---|---|---|---|
| `hour_of_day` | `Reported Time` | Parsed as `%H:%M:%S.%f`, extract `.dt.hour` | Arrest rates vary significantly by time of day (e.g., late-night stops may have higher arrest rates) |
| `day_of_week` | `Reported Date` | `.dt.dayofweek` (Monday=0, Sunday=6) | Policing patterns and crime patterns vary by day |
| `is_weekend` | `day_of_week` | 1 if Saturday or Sunday, else 0 | Weekend stops may have different characteristics (e.g., more alcohol-related encounters) |

**Why extract these features instead of using raw timestamps?**  Raw timestamps are high-cardinality and non-repeating. Extracting cyclical components (hour, day) reduces dimensionality while preserving the signal. Tree-based models can split on these integer features to capture non-linear temporal patterns.

### 4.2 Target Encoding

```python
df["Arrest Flag"] = df["Arrest Flag"].map({"Y": 1, "N": 0})
```

**Why?**  Machine learning models require numeric targets. A simple binary mapping preserves the semantics: 1 = arrest (positive class), 0 = no arrest (negative class).

### 4.3 Categorical Encoding — Ordinal Encoder

```python
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
```

**Why OrdinalEncoder over OneHotEncoder?**

- **Tree-based models** (Random Forest, Gradient Boosting, XGBoost, LightGBM) can handle ordinal-encoded features natively — they split on thresholds, so the numeric assignment works well.
- **One-Hot Encoding** would dramatically increase dimensionality (e.g., `Subject Perceived Race` has 7+ categories), leading to sparse features and potential overfitting.
- The `handle_unknown="use_encoded_value"` with `unknown_value=-1` ensures that new categories at inference time produce a recognisable sentinel value rather than raising an error.

**Why not Target Encoding?**  Target encoding risks data leakage if not applied with proper cross-validation folds. Ordinal encoding is simpler, stateless, and sufficient for the tree-based ensemble models used here.

### 4.4 Additional Feature Engineering Helpers

The `ml/feature_engineering.py` module provides three supplementary feature functions:

| Function | Feature Created | Method | Rationale |
|---|---|---|---|
| `create_officer_experience_proxy` | `officer_stop_count` | Count of stops per `Officer ID` | Officers with more experience (more stops) may have different arrest patterns |
| `create_neighborhood_risk_score` | `precinct_arrest_rate` | Mean arrest rate per `Precinct` | Provides a continuous, history-based risk signal rather than a categorical label |
| `create_repeat_subject_flag` | `is_repeat_subject` | Boolean flag if `Subject ID` appears more than once | Repeat encounters may correlate with higher arrest likelihood |

---

## 5. Class Imbalance Handling — SMOTE

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**What is SMOTE?**  Synthetic Minority Over-sampling Technique generates new synthetic examples of the minority class by interpolating between existing minority-class instances in feature space. For each minority sample, SMOTE:

1. Finds its *k* nearest neighbours (default *k*=5).
2. Randomly selects one neighbour.
3. Creates a new sample at a random point along the line segment between the original and the neighbour.

**Why SMOTE?**

- The Terry Stops dataset is **imbalanced** — arrests occur in a minority of stops. Without resampling, classifiers tend to favour the majority class (no arrest), yielding high accuracy but poor recall for arrests.
- SMOTE is preferred over **random oversampling** because it creates novel synthetic points rather than duplicating existing ones, which reduces overfitting risk.
- SMOTE is applied **only to the training set** to prevent data leakage. The test set remains untouched to give an honest estimate of real-world performance.

**Why not ADASYN or Borderline-SMOTE?**  SMOTE is the most widely validated synthetic oversampling technique and provides a good baseline. More advanced variants could be explored in future iterations but add complexity without guaranteed improvement.

---

## 6. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `test_size` | 0.2 | 80/20 split is a standard partition that balances having enough training data with a meaningful test set |
| `random_state` | 42 | Ensures reproducibility across runs |
| `stratify=y` | Target column | Preserves the class distribution in both splits, critical for imbalanced datasets |

---

## 7. Model Selection & Training

Five candidate classifiers plus one ensemble are trained and compared. Each was chosen to cover a spectrum from simple linear models to state-of-the-art gradient boosting.

### 7.1 Logistic Regression

```python
LogisticRegression(max_iter=1000, random_state=42)
```

**How it works:**  Logistic Regression models the log-odds of the positive class as a linear combination of features:

$$\log\frac{P(y=1|x)}{1 - P(y=1|x)} = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$$

The sigmoid function converts the linear output to a probability between 0 and 1.

**Why included?**

- Serves as an **interpretable baseline**. If a simple linear model performs adequately, complex models may not be necessary.
- Coefficients are directly interpretable as log-odds multipliers.
- Fast to train and predict.
- `max_iter=1000` ensures convergence on the resampled dataset.

### 7.2 Random Forest

```python
RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
```

**How it works:**  Random Forest builds an ensemble of `n_estimators` decision trees, each trained on a bootstrap sample of the data and a random subset of features. The final prediction is the majority vote (classification) or average (probability) across all trees.

**Why included?**

- **Robust to overfitting** due to bagging and feature subsampling.
- Handles non-linear relationships and feature interactions automatically.
- Provides built-in **feature importance** via mean decrease in impurity.
- `n_estimators=200` provides a good balance between predictive power and training time.
- `n_jobs=-1` enables parallel tree construction for faster training.

### 7.3 Gradient Boosting

```python
GradientBoostingClassifier(n_estimators=200, random_state=42)
```

**How it works:**  Gradient Boosting builds trees sequentially. Each new tree is trained on the residual errors (negative gradient of the loss function) of the previous ensemble. The final prediction is the weighted sum of all trees.

**Why included?**

- Gradient Boosting often achieves **higher accuracy** than Random Forest because it focuses on correcting mistakes of previous iterations.
- Provides a different bias-variance trade-off compared to bagging methods.
- Acts as a benchmark for the more optimised boosting implementations (XGBoost, LightGBM).

### 7.4 XGBoost

```python
XGBClassifier(n_estimators=200, use_label_encoder=False,
              eval_metric="logloss", random_state=42)
```

**How it works:**  XGBoost (eXtreme Gradient Boosting) is an optimised implementation of gradient boosting that adds:

- **Regularisation** (L1 and L2 penalties on leaf weights) to reduce overfitting.
- **Efficient split finding** using histogram-based algorithms.
- **Built-in handling of missing values** via learned default split directions.
- **Parallel tree construction** across features.

**Why included?**

- Consistently ranks among the **top-performing algorithms** on tabular data in academic benchmarks and Kaggle competitions.
- Regularisation helps prevent overfitting on the SMOTE-augmented training set.
- `eval_metric="logloss"` aligns the optimisation objective with probabilistic calibration.

### 7.5 LightGBM

```python
LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
```

**How it works:**  LightGBM uses two key innovations:

- **Gradient-based One-Side Sampling (GOSS):** Keeps instances with large gradients (high error) and randomly samples instances with small gradients, reducing computation without sacrificing accuracy.
- **Exclusive Feature Bundling (EFB):** Bundles mutually exclusive sparse features to reduce dimensionality.
- **Leaf-wise tree growth** (vs. level-wise in XGBoost), which often produces deeper, more accurate trees.

**Why included?**

- Significantly **faster training** than XGBoost on large datasets.
- Often achieves comparable or better accuracy.
- `verbose=-1` suppresses training logs for cleaner output.

### 7.6 Voting Ensemble

```python
VotingClassifier(estimators=top3_estimators, voting="soft")
```

**How it works:**  A soft-voting ensemble averages the predicted probabilities from multiple base models and assigns the class with the highest average probability. Unlike hard voting (majority of predicted classes), soft voting leverages confidence information.

**Why included?**

- Ensemble methods **reduce variance** by combining multiple models, often outperforming any individual member.
- The top-3 models (by cross-validated F1) are selected automatically, ensuring the ensemble is composed of the strongest performers.
- `voting="soft"` is preferred over `"hard"` because it uses probability estimates, which are smoother and better calibrated.

---

## 8. Cross-Validation Strategy

```python
cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
```

**Method:** 5-fold stratified cross-validation.

**Why 5 folds?**

- Standard choice that balances **bias and variance** of the performance estimate.
- 3 folds may underestimate performance; 10 folds increase computation time with marginal improvement in estimate quality.
- Stratification ensures each fold preserves the class distribution.

**Why cross-validate on the training set?**

- The test set is held out entirely for final evaluation. Cross-validation on the training set provides an unbiased estimate of generalisation performance during model selection without touching the test set.

---

## 9. Evaluation Metrics

The project uses multiple complementary metrics because no single metric captures all aspects of model quality, especially with imbalanced data.

### 9.1 Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Why:** Easy to interpret — the fraction of correct predictions. However, it can be misleading on imbalanced datasets (a model that always predicts "no arrest" could achieve >80% accuracy).

### 9.2 Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Why:** Measures how many of the predicted arrests are actually arrests. High precision is important to avoid **false alarms** that could lead to unwarranted scrutiny.

### 9.3 Recall

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Why:** Measures how many actual arrests the model correctly identifies. High recall is important to avoid **missing** genuine arrest situations.

### 9.4 F1 Score

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Why:** The harmonic mean of precision and recall. It penalises models that sacrifice one for the other and is the **primary model selection metric** in this project because it balances both types of errors.

### 9.5 ROC-AUC

The area under the Receiver Operating Characteristic curve. It measures the model's ability to discriminate between classes across all possible decision thresholds.

**Why:** Threshold-invariant evaluation — useful for comparing models before choosing a specific operating point. A value of 0.5 indicates random guessing; 1.0 indicates perfect discrimination.

### 9.6 Confusion Matrix

A 2×2 matrix showing TP, TN, FP, and FN counts. The project generates a confusion matrix heatmap plot for visual inspection.

**Why:** Provides a concrete, interpretable breakdown of errors that summary statistics can obscure.

### 9.7 Optimal Threshold Selection

```python
for threshold in np.arange(0.01, 1.0, 0.01):
    y_pred = (y_proba >= threshold).astype(int)
    score = f1_score(y_test, y_pred, zero_division=0)
```

**Why:** The default 0.5 threshold may not maximise F1 on an imbalanced dataset. Sweeping thresholds from 0.01 to 0.99 finds the operating point that maximises the F1 score, enabling a more nuanced trade-off between precision and recall.

---

## 10. Model Explainability

### 10.1 SHAP (SHapley Additive exPlanations)

**Implementation:** `app/utils/explainability.py` and `app/models/ml_model.py`

**How it works:**  SHAP values are based on Shapley values from cooperative game theory. For each prediction, a SHAP value quantifies the contribution of each feature to the difference between the actual prediction and the average prediction. Formally, for feature *i*:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

The project uses `TreeExplainer`, which computes exact SHAP values in polynomial time for tree-based models.

**Why SHAP?**

- **Consistency:** If a feature's contribution increases, its SHAP value increases.
- **Local accuracy:** SHAP values for a single prediction sum to the difference between the prediction and the expected value.
- **Global importance:** Mean |SHAP| across all samples provides feature importance rankings.
- Tree-based SHAP is computationally efficient for the ensemble models used.

**Usage in the application:**

- **Per-prediction explanations:** Each prediction result includes a SHAP breakdown showing which features pushed the probability toward or away from an arrest prediction.
- **Global feature importance:** The `global_feature_importance()` method computes mean absolute SHAP values across a background dataset.

### 10.2 LIME (Local Interpretable Model-agnostic Explanations)

LIME is included as a dependency and can be used for model-agnostic local explanations. It works by:

1. Perturbing the input instance to create a neighbourhood dataset.
2. Querying the model on these perturbed instances.
3. Fitting a simple interpretable model (e.g., linear regression) on the local neighbourhood.
4. Using the simple model's coefficients as feature importance scores.

**Why LIME in addition to SHAP?**  LIME provides a complementary perspective — it is model-agnostic and can be used when TreeExplainer is not applicable (e.g., neural networks or non-tree models added in the future).

---

## 11. Bias Auditing & Fairness

**Implementation:** `app/utils/bias_audit.py` — the `BiasAuditor` class.

### 11.1 Demographic Parity

$$\text{DP Difference} = \max_g \left( P(\hat{y}=1 | G=g) \right) - \min_g \left( P(\hat{y}=1 | G=g) \right)$$

**What it measures:**  Whether the positive prediction rate is the same across all demographic groups. A difference of 0 indicates perfect parity.

**Why:** Demographic parity is a fundamental fairness criterion. If the model predicts arrest more frequently for one racial group than another (regardless of ground truth), it may reflect or amplify systemic biases.

### 11.2 False Positive Rate by Group

$$\text{FPR}_g = \frac{FP_g}{FP_g + TN_g}$$

**What it measures:** How often the model incorrectly predicts an arrest for each group. Disparate FPR values mean some groups face more false accusations than others.

**Why:** A high FPR for a specific group indicates the model disproportionately flags members of that group, which could lead to unwarranted stops or heightened suspicion.

### 11.3 False Negative Rate by Group

$$\text{FNR}_g = \frac{FN_g}{FN_g + TP_g}$$

**What it measures:** How often the model misses actual arrests for each group. Disparate FNR values mean the model underperforms for certain groups.

**Why:** A high FNR for a group means the model fails to identify genuine arrest situations, potentially leading to under-allocation of resources.

### 11.4 Traffic-Light Status Indicators

| Disparity Range | Status | Interpretation |
|---|---|---|
| < 0.10 | 🟢 Green | Acceptable disparity |
| 0.10 – 0.19 | 🟡 Yellow | Monitor — warrants investigation |
| ≥ 0.20 | 🔴 Red | Significant disparity — action required |

**Why these thresholds?**  They follow commonly used industry guidelines (e.g., the "four-fifths rule" adapted to continuous disparity measures). They provide actionable, at-a-glance guidance for non-technical stakeholders.

---

## 12. Model Persistence & Registry

### Serialisation

```python
joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")
```

**Why joblib?**  `joblib` is optimised for serialising large numpy arrays and scikit-learn pipelines. It is faster and produces smaller files than Python's built-in `pickle` for these use cases.

### Metadata Sidecars

Each model file has a companion `*_metadata.json` containing:

- `model_name`, `feature_names`, `is_active`
- Cross-validation metrics (`cv_f1_mean`, `cv_f1_std`, `cv_accuracy_mean`)

**Why JSON sidecars?**  They allow the application to inspect model metadata without deserialising the model itself — useful for the model registry UI and for selecting the active model.

### Database Registry

The `ModelRegistry` SQLAlchemy model stores model metadata in the application database, enabling:

- Version tracking and comparison.
- One-click model promotion (marking a model as "active").
- Audit logging of model changes.

---

## 13. Deployment Architecture

| Component | Technology | Purpose |
|---|---|---|
| **Web framework** | Flask 3.0 | Routes, templates, API endpoints |
| **Database** | SQLite (dev) / PostgreSQL (prod) | Prediction logging, model registry |
| **ML inference** | scikit-learn + joblib | Load and run trained models |
| **Containerisation** | Docker + Docker Compose | Reproducible deployment |
| **WSGI server** | Gunicorn (4 workers) | Production-grade request handling |
| **Frontend** | Bootstrap 5 + Plotly.js | Responsive dark-theme UI with interactive charts |

**Why Flask?**  Flask is lightweight, well-documented, and provides enough structure (blueprints, extensions) for a medium-sized application without the overhead of a full framework like Django.

**Why Docker?**  Containers ensure that the application runs identically on any machine — local development, CI, and production — eliminating "it works on my machine" problems.

---

## 14. Limitations & Ethical Considerations

1. **Historical bias:** The model is trained on historical policing data, which may reflect systemic biases in how stops are conducted. Predictions should not be treated as ground truth.

2. **Proxy discrimination:** Features like precinct or time of day may serve as proxies for race or socioeconomic status, introducing indirect bias even when protected attributes are removed.

3. **Temporal drift:** Policing practices, demographics, and crime patterns change over time. Models should be retrained periodically on recent data.

4. **Not for enforcement:** This tool is designed for **research and educational purposes only**. It should never be the sole basis for enforcement decisions.

5. **Sample bias:** The dataset represents Seattle-specific policing patterns and may not generalise to other jurisdictions.

See [ETHICS.md](terrystops-ai/ETHICS.md) for the full responsible-use policy.

---

## 15. References

1. **Terry v. Ohio, 392 U.S. 1 (1968)** — Supreme Court decision establishing the legal basis for stop-and-frisk encounters.
2. **Chawla, N. V., et al. (2002).** SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
3. **Lundberg, S. M., & Lee, S.-I. (2017).** A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.
4. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD*.
5. **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32.
6. **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*.
7. **Ke, G., et al. (2017).** LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems (NeurIPS)*.
8. **Friedman, J. H. (2001).** Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.
9. **Seattle Police Department Open Data Portal.** Terry Stops dataset. https://data.seattle.gov/Public-Safety/Terry-Stops/28ny-9ts8
