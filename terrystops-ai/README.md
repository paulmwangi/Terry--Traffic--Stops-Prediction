# 🚔 TerryStops-AI

**Machine-learning powered prediction of arrest outcomes during Terry Stops — with bias auditing and explainability built in.**

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

---

## 📖 Overview

TerryStops-AI is a full-stack analytics application that uses machine learning to predict arrest outcomes from Seattle Police Department Terry Stop data. The platform goes beyond simple prediction by integrating **SHAP/LIME explainability**, a **bias audit dashboard**, and an interactive analytics interface — making model decisions transparent, auditable, and research-ready.

> **⚠️ This tool is for research and educational purposes only.** See [ETHICS.md](ETHICS.md) for responsible use guidelines.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Model ML Pipeline** | Train and compare Logistic Regression, Random Forest, XGBoost, and more |
| **SHAP / LIME Explainability** | Understand *why* the model makes each prediction |
| **Bias Audit Dashboard** | Measure fairness metrics across demographic groups |
| **Real-Time Prediction API** | RESTful JSON API for on-demand predictions |
| **Interactive Analytics** | Web-based dashboards for exploratory data analysis |
| **Docker Deployment** | One-command setup with Docker Compose |

---

## 🏗️ Architecture

<!-- Architecture diagram coming soon -->

---

## 🚀 Quick Start

### Docker (recommended)

```bash
git clone https://github.com/paulmwangi/Terry--Traffic--Stops-Prediction.git
cd Terry--Traffic--Stops-Prediction/terrystops-ai
docker-compose up --build
docker-compose exec web python -m ml.train
```

The app will be available at `http://localhost:5000`.

### Manual Setup

```bash
cd terrystops-ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Submit features and receive an arrest-outcome prediction |
| `GET` | `/api/v1/stats` | Retrieve dataset and model performance statistics |
| `GET` | `/api/v1/model` | Get metadata about the currently loaded model |

---

## 📁 Project Structure

```
terrystops-ai/
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── db_models.py        # Database models
│   │   └── ml_model.py         # ML model wrapper
│   ├── routes/
│   │   ├── admin.py            # Admin endpoints
│   │   ├── analytics.py        # Analytics dashboard routes
│   │   ├── api.py              # REST API endpoints
│   │   ├── main.py             # Main/index routes
│   │   └── predict.py          # Prediction routes
│   ├── static/                 # CSS, JS, images
│   ├── templates/              # Jinja2 HTML templates
│   └── utils/
│       ├── bias_audit.py       # Fairness & bias metrics
│       ├── explainability.py   # SHAP / LIME explanations
│       └── preprocessing.py    # Data preprocessing helpers
├── config.py                   # App configuration
├── data/
│   ├── exports/
│   ├── processed/
│   └── raw/
├── ml/
│   ├── evaluate.py             # Model evaluation scripts
│   ├── feature_engineering.py  # Feature engineering pipeline
│   ├── saved_models/           # Serialized model artifacts
│   └── train.py                # Model training scripts
├── notebooks/                  # Jupyter exploration notebooks
├── tests/
│   ├── conftest.py
│   ├── test_bias.py
│   ├── test_model.py
│   └── test_routes.py
├── Dockerfile
├── docker-compose.yml
├── Procfile
├── requirements.txt
├── run.py
├── CONTRIBUTING.md
├── ETHICS.md
└── README.md
```

---

## 📸 Screenshots

<!-- Screenshots coming soon -->

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Flask |
| ML Libraries | scikit-learn, XGBoost |
| Explainability | SHAP, LIME |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Containerization | Docker, Docker Compose |
| Testing | pytest, pytest-cov |

---

## 🧪 Testing

Run the full test suite with coverage:

```bash
pytest --cov=app tests/
```

All pull requests must pass the existing test suite and maintain **≥ 80 % code coverage**.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

---

## ⚖️ Ethics & Responsible Use

This project handles sensitive policing data. Please review our [ETHICS.md](ETHICS.md) before using or extending the platform.

---

## 📄 License

This project is licensed under the **MIT License**. See the root [LICENSE](../LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Seattle Police Department Open Data** — for providing the Terry Stops dataset.
- **Terry v. Ohio, 392 U.S. 1 (1968)** — the landmark Supreme Court decision that established the legal foundation for stop-and-frisk encounters.
