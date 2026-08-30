\# 🌫️ Pearls AQI Predictor



\### AI-Powered 3-Day Air Quality Forecasting System for Karachi, Pakistan



Pearls AQI Predictor is an end-to-end machine learning and MLOps project designed to forecast \*\*Karachi's Air Quality Index (AQI) for the next three days\*\*.



The system combines historical AQI observations, meteorological data, feature engineering, multiple machine learning models, Hopsworks feature/model management, automated daily processing, a Flask REST API, and an interactive Streamlit dashboard.



The final system provides both \*\*AQI predictions and actionable air-quality health guidance\*\* through a web-based observatory dashboard.



\---



\## 📌 Project Overview



Air pollution can vary significantly from day to day due to weather conditions, pollutant levels, and temporal patterns.



Pearls AQI Predictor addresses this problem by using historical air-quality and weather data to predict future AQI values for:



\- \*\*1 day ahead\*\*

\- \*\*2 days ahead\*\*

\- \*\*3 days ahead\*\*



Instead of relying on a single model for all forecasting horizons, the project evaluates multiple approaches and uses the best-performing model for each prediction horizon.



\### Current Forecast Example



| Forecast Horizon | Predicted AQI | Category |

|---|---:|---|

| Day 1 | 67.4 | Moderate |

| Day 2 | 69.8 | Moderate |

| Day 3 | 79.8 | Moderate |



Current predictions are generated from the latest available historical data and exposed through the Flask API.



\---



\# 🎯 Objectives



The main objectives of the project are:



\- Predict Karachi's AQI for the next 3 days.

\- Use historical pollution and weather information as predictive features.

\- Engineer time-series features to improve forecasting performance.

\- Compare different machine learning algorithms.

\- Select specialized models for different forecast horizons.

\- Store and manage features using Hopsworks.

\- Register and track models.

\- Provide model evaluation and diagnostic analysis.

\- Explain model predictions using SHAP.

\- Automate daily data processing and prediction generation.

\- Expose predictions through a REST API.

\- Provide an interactive dashboard for end users.



\---



\# ✨ Key Features



\## 🤖 Machine Learning Forecasting



The system produces independent predictions for:



\- AQI at t+1

\- AQI at t+2

\- AQI at t+3



Different models are used for different horizons based on validation performance.



\---



\## 🌦️ Weather + Pollution Features



The forecasting pipeline uses historical environmental information together with engineered time-series features.



These features allow the models to learn relationships between:



\- Historical AQI

\- Weather conditions

\- Temporal patterns

\- Recency

\- Lagged observations

\- Other engineered predictors



\---



\## 🧠 Multiple Model Evaluation



The project evaluates multiple machine learning approaches, including:



\- SVR

\- CatBoost

\- Random Forest

\- XGBoost

\- Ridge

\- Lasso

\- ElasticNet

\- LSTM

\- Prophet

\- Multi-output Random Forest



The final production models were selected separately for each forecasting horizon.



\---



\## 🏆 Final Production Models



\### Day 1



\*\*SVR + StandardScaler + Recency\*\*



\- Registry: `karachi\_aqi\_svr\_t\_1`

\- Version: `1`

\- RMSE: `13.6674`

\- MAE: `9.3747`

\- R²: `0.7137`



\### Day 2



\*\*SVR + RobustScaler\*\*



\- Registry: `karachi\_aqi\_svr\_t\_2`

\- Version: `1`

\- RMSE: `18.3090`

\- MAE: `13.5716`

\- R²: `0.4830`



\### Day 3



\*\*CatBoost Regressor\*\*



\- Registry: `karachi\_aqi\_catboost\_t\_3`

\- Version: `1`

\- RMSE: `19.7632`

\- MAE: `15.4517`

\- R²: `0.4000`



\---



\# 📊 Model Performance



| Horizon | Model | RMSE | MAE | R² |

|---|---|---:|---:|---:|

| 1 Day | SVR + StandardScaler + Recency | 13.6674 | 9.3747 | \*\*0.7137\*\* |

| 2 Days | SVR + RobustScaler | 18.3090 | 13.5716 | \*\*0.4830\*\* |

| 3 Days | CatBoost Regressor | 19.7632 | 15.4517 | \*\*0.4000\*\* |



\### Interpretation



The 1-day forecast provides the strongest predictive performance, with an R² of approximately \*\*0.71\*\*.



Performance naturally becomes more challenging as the forecasting horizon increases because uncertainty accumulates further into the future.



The system therefore uses \*\*horizon-specific models rather than forcing one model to handle all three prediction horizons\*\*.



\---



\# 🔬 Model Explainability



The project includes SHAP-based explainability analysis.



SHAP is used to understand how individual features influence model predictions.



The repository contains:



\- Feature importance CSV files

\- SHAP summary plots

\- Top-20 feature visualizations

\- Day-3 CatBoost SHAP analysis



These results help make the forecasting system more interpretable and provide insight into which environmental and temporal variables influence AQI predictions.



\---



\# 🏗️ System Architecture



```text

&#x20;                        ┌─────────────────────┐

&#x20;                        │   Open-Meteo Data    │

&#x20;                        │  Weather + AQI Data  │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │   Data Processing   │

&#x20;                        │ Cleaning / AQI Prep  │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │ Feature Engineering │

&#x20;                        │ Lag / Time / Weather│

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │      Hopsworks      │

&#x20;                        │   Feature Storage   │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │ Model Training \&    │

&#x20;                        │ Model Evaluation    │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                  ┌────────────────┼────────────────┐

&#x20;                  │                │                │

&#x20;                  ▼                ▼                ▼

&#x20;               Day 1             Day 2            Day 3

&#x20;                SVR               SVR             CatBoost

&#x20;                  │                │                │

&#x20;                  └────────────────┼────────────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │ Prediction Pipeline │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        latest\_predictions.csv

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │     Flask API       │

&#x20;                        │      Port 5000      │

&#x20;                        └──────────┬──────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌─────────────────────┐

&#x20;                        │ Streamlit Dashboard │

&#x20;                        │      Port 8501      │

&#x20;                        └─────────────────────┘

