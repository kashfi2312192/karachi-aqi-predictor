# 🌫️ Pearls AQI Predictor

### AI-Powered 3-Day Air Quality Forecasting System for Karachi, Pakistan

Pearls AQI Predictor is an end-to-end machine learning and MLOps project designed to forecast **Karachi's Air Quality Index (AQI) for the next three days**.

The system combines historical AQI observations, meteorological data, feature engineering, multiple machine learning models, Hopsworks feature and model management, automated pipelines, a Flask REST API, and an interactive Streamlit dashboard.

The final system provides AQI predictions together with air-quality health categories through a web-based dashboard.

🔗 **Live Dashboard:** [karachi-aqi-predictor-2026.streamlit.app](https://karachi-aqi-predictor-2026.streamlit.app/#pearls-aqi-predictor)
🔗 **Repository:** [github.com/kashfi2312192/karachi-aqi-predictor](https://github.com/kashfi2312192/karachi-aqi-predictor)

---

## 📌 Project Overview

Air pollution can vary significantly from day to day due to weather conditions, pollutant levels, and temporal patterns.

Pearls AQI Predictor addresses this problem by forecasting future AQI values for:

- **1 day ahead**
- **2 days ahead**
- **3 days ahead**

Instead of relying on a single model for every forecasting horizon, multiple machine learning approaches are evaluated and specialized models are selected for each prediction horizon.

### Current Forecast Example

| Forecast Horizon | Predicted AQI | Category |
|---|---:|---|
| Day 1 | 67.41 | Moderate |
| Day 2 | 69.81 | Moderate |
| Day 3 | 79.78 | Moderate |

The current predictions are generated from the latest available historical data and exposed through the Flask REST API.

---

## 🎯 Objectives

The main objectives of the project are:

- Predict Karachi's AQI for the next 3 days.
- Use historical pollution and weather information as predictive features.
- Engineer time-series features to improve forecasting performance.
- Compare multiple machine learning algorithms.
- Select specialized models for different forecast horizons.
- Store and manage features using Hopsworks.
- Register and track trained models.
- Perform model evaluation and diagnostic analysis.
- Explain model predictions using SHAP.
- Automate feature generation and model training using scheduled pipelines.
- Expose predictions through a REST API.
- Provide an interactive dashboard for users.

---

## ✨ Key Features

### 🤖 Multi-Horizon AQI Forecasting

The system produces independent predictions for:

- AQI at **t+1**
- AQI at **t+2**
- AQI at **t+3**

Different models are used for different horizons based on model evaluation results.

---

### 🌦️ Weather + Pollution Features

The forecasting pipeline combines historical AQI and environmental information with engineered time-series features.

The feature engineering process includes information such as:

- Historical AQI
- Weather conditions
- Temporal patterns
- Lagged observations
- Recency information
- Other engineered predictors

These features allow the models to learn relationships between environmental conditions and future AQI.

---

### 🧠 Multiple Model Evaluation

The project evaluates multiple machine learning approaches, including:

- Support Vector Regression (SVR)
- CatBoost
- Random Forest
- XGBoost
- Ridge
- Lasso
- ElasticNet
- LSTM
- Prophet
- Multi-Output Random Forest

The final production models are selected independently for each forecasting horizon.

---

## 🏆 Final Production Models

### Day 1 — SVR

**Model:** SVR + StandardScaler + Recency

| Metric | Value |
|---|---:|
| Model Registry | `karachi_aqi_svr_t_1` |
| Version | 1 |
| RMSE | 13.6674 |
| MAE | 9.3747 |
| R² | 0.7137 |

---

### Day 2 — SVR

**Model:** SVR + RobustScaler

| Metric | Value |
|---|---:|
| Model Registry | `karachi_aqi_svr_t_2` |
| Version | 1 |
| RMSE | 18.3090 |
| MAE | 13.5716 |
| R² | 0.4830 |

---

### Day 3 — CatBoost

**Model:** CatBoost Regressor

| Metric | Value |
|---|---:|
| Model Registry | `karachi_aqi_catboost_t_3` |
| Version | 1 |
| RMSE | 19.7632 |
| MAE | 15.4517 |
| R² | 0.4000 |

---

## 📊 Model Performance

| Horizon | Model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| 1 Day | SVR + StandardScaler + Recency | 13.6674 | 9.3747 | **0.7137** |
| 2 Days | SVR + RobustScaler | 18.3090 | 13.5716 | **0.4830** |
| 3 Days | CatBoost Regressor | 19.7632 | 15.4517 | **0.4000** |

### Interpretation

The 1-day forecast provides the strongest predictive performance, with an R² of approximately **0.71**.

Prediction becomes more challenging as the forecasting horizon increases because uncertainty increases further into the future.

The system therefore uses **horizon-specific models instead of forcing a single model to handle all three prediction horizons**.

---

## 🔬 Model Explainability

The project includes SHAP-based explainability analysis.

SHAP is used to analyze how individual features influence model predictions.

The repository contains:

- Feature importance CSV files
- SHAP summary plots
- Top-20 feature visualizations
- Day-3 CatBoost SHAP analysis

These results provide insight into the environmental and temporal variables influencing AQI predictions.

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML/Data:** scikit-learn, XGBoost, CatBoost, pandas, numpy, SHAP
- **MLOps:** Hopsworks (Feature Store + Model Registry)
- **Backend:** Flask (REST API)
- **Frontend:** Streamlit
- **Automation:** GitHub Actions (scheduled workflows)
- **Deployment:** Render (Flask API hosting)

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/kashfi2312192/karachi-aqi-predictor.git
cd karachi-aqi-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-ml.txt   # for training/ML dependencies
```

### Environment Variables

Create a `.env` file in the project root with:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=your_project_name
```

---

## ▶️ Running Locally

**Start the Flask API:**
```bash
python src/api/app.py
# API available at http://localhost:5000
```

**Start the Streamlit dashboard:**
```bash
streamlit run src/dashboard/app.py
# Dashboard available at http://localhost:8501
```

> Update the paths above if your actual entry-point filenames differ.

---

## 📁 Project Structure

```
karachi-aqi-predictor/
├── .github/workflows/      # GitHub Actions cron jobs (hourly feature pipeline, daily training)
├── data/                   # Raw and processed AQI/weather data
├── models/                 # Serialized trained models per horizon
├── results/                # Metrics, SHAP outputs, evaluation reports
├── scripts/                # Standalone/CLI scripts (training, prediction, automation)
├── src/                    # Core application code (pipeline, API, dashboard)
├── requirements.txt        # Core dependencies
├── requirements-ml.txt     # ML/training-specific dependencies
└── README.md
```

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │    Open-Meteo Data  │
                    │ Weather + AQI Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Processing   │
                    │ Cleaning / AQI Prep │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Lag / Time / Weather│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Hopsworks      │
                    │   Feature Storage   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Training &    │
                    │ Model Evaluation    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
              Day 1          Day 2         Day 3
               SVR            SVR          CatBoost
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction Pipeline │
                    └──────────┬──────────┘
                               │
                               ▼
                   latest_predictions.csv
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Flask API      │
                    │   (hosted on Render)│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘
```

---

## 🔌 API Reference

**Base URL (Production):** `https://<your-render-app-name>.onrender.com`

**GET `/predict`** — Returns the latest 3-day AQI forecast

```json
{
  "day_1": { "aqi": 67.41, "category": "Moderate" },
  "day_2": { "aqi": 69.81, "category": "Moderate" },
  "day_3": { "aqi": 79.78, "category": "Moderate" },
  "generated_at": "2026-09-02T06:00:00Z"
}
```

> Update the endpoint path/response shape above to match your actual Flask routes.

---

## 🤖 Automation (GitHub Actions)

The project runs two scheduled workflows so the system stays current without manual intervention:

| Workflow | Schedule | Purpose |
|---|---|---|
| `fetch_features.yml` | Hourly | Pulls latest weather/AQI data, regenerates features, pushes to the Hopsworks feature store |
| `train_models.yml` | Daily | Retrains candidate models, re-evaluates performance per horizon, updates registered production models |

> Rename to match your actual `.yml` filenames in `.github/workflows/`.

---

## 🚀 Deployment

- **Flask REST API** — deployed on [Render](https://render.com)
- **Streamlit Dashboard** — [Live Demo](https://karachi-aqi-predictor-2026.streamlit.app/#pearls-aqi-predictor)
- **Feature Store & Model Registry** — [Hopsworks](https://www.hopsworks.ai/)

---

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## 👤 Author

**Kashfi Karim Ali**
[GitHub](https://github.com/kashfi2312192)
