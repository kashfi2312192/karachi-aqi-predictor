import os
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

# ============================================================
# PEARLS AQI PREDICTOR — KARACHI AQI FLASK API
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PREDICTION_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "latest_predictions.csv")
HISTORICAL_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "hopsworks_training_data.csv")
SHAP_DIR = os.path.join(PROJECT_ROOT, "results", "shap")

MODEL_INFO = {
    "1_day": {"registry_name": "karachi_aqi_svr_t_1", "version": 1, "model": "SVR + StandardScaler + Recency", "rmse": 13.6674, "mae": 9.3747, "r2": 0.7137},
    "2_days": {"registry_name": "karachi_aqi_svr_t_2", "version": 1, "model": "SVR + RobustScaler", "rmse": 18.3090, "mae": 13.5716, "r2": 0.4830},
    "3_days": {"registry_name": "karachi_aqi_catboost_t_3", "version": 1, "model": "CatBoost Regressor", "rmse": 19.7632, "mae": 15.4517, "r2": 0.4000},
}

app = Flask(__name__)
CORS(app)


def get_aqi_category(aqi):
    aqi = float(aqi)
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File is empty: {path}")
    return df


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def get_source_date():
    if not os.path.exists(HISTORICAL_FILE):
        return None
    df = pd.read_csv(HISTORICAL_FILE)
    date_col = find_column(df, ["date", "Date", "datetime", "timestamp"])
    if not date_col:
        return None
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    return dates.max().strftime("%Y-%m-%d") if not dates.empty else None


def get_latest_actual_aqi():
    """Return latest observed AQI from historical data, never a forecast."""
    if not os.path.exists(HISTORICAL_FILE):
        return None

    df = pd.read_csv(HISTORICAL_FILE)
    if df.empty:
        return None

    date_col = find_column(df, ["date", "Date", "datetime", "timestamp"])
    aqi_col = find_column(df, [
        "AQI", "aqi", "actual_aqi", "Actual_AQI", "AQI_actual",
        "aqi_actual", "AQI_t", "aqi_t", "current_aqi"
    ])

    if not aqi_col:
        return None

    df[aqi_col] = pd.to_numeric(df[aqi_col], errors="coerce")
    df = df.dropna(subset=[aqi_col])
    if df.empty:
        return None

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

    row = df.iloc[-1]
    value = float(row[aqi_col])

    return {
        "aqi": round(value, 2),
        "category": get_aqi_category(value),
        "date": row[date_col].strftime("%Y-%m-%d") if date_col else None,
        "source": "historical_observation",
        "column": aqi_col,
    }


def normalize_horizon(value, index):
    if pd.isna(value):
        return f"{index + 1}_day" if index == 0 else f"{index + 1}_days"
    value = str(value).strip().lower()
    if value in {"1_day", "day_1", "1", "day1"}:
        return "1_day"
    if value in {"2_days", "day_2", "2", "day2"}:
        return "2_days"
    if value in {"3_days", "day_3", "3", "day3"}:
        return "3_days"
    return value


def load_forecast():
    df = load_csv(PREDICTION_FILE)
    date_col = find_column(df, ["forecast_date", "date", "Date", "prediction_date"])
    aqi_col = find_column(df, ["predicted_aqi", "aqi", "AQI", "prediction", "predicted_AQI"])
    horizon_col = find_column(df, ["forecast_horizon", "horizon", "day", "prediction_horizon"])

    if not aqi_col:
        raise ValueError(f"Could not find AQI column. Available columns: {df.columns.tolist()}")

    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dates = pd.Series([pd.NaT] * len(df))

    source_date = get_source_date()
    if dates.isna().any():
        if source_date is None:
            raise ValueError("Could not determine forecast dates.")
        base = pd.Timestamp(source_date)
        dates = pd.Series([base + pd.Timedelta(days=i) for i in range(1, len(df) + 1)])

    forecast = []
    for index, row in df.iterrows():
        aqi = round(float(row[aqi_col]), 2)
        horizon = normalize_horizon(row[horizon_col], index) if horizon_col else ("1_day" if index == 0 else "2_days" if index == 1 else "3_days")
        info = MODEL_INFO.get(horizon, {})
        item = {
            "date": dates.iloc[index].strftime("%Y-%m-%d"),
            "aqi": aqi,
            "category": get_aqi_category(aqi),
            "horizon": horizon,
            **info,
        }
        forecast.append(item)

    return forecast[:3]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "Pearls AQI Predictor",
        "service": "Karachi AQI Prediction API",
        "status": "running",
        "city": "Karachi",
        "forecast_horizon": "3 days",
        "model_registry": "Hopsworks",
        "endpoints": ["/health", "/forecast", "/model-info", "/shap/1_day", "/shap/2_days", "/shap/3_days"],
    })


@app.route("/health", methods=["GET"])
def health():
    prediction_available = os.path.exists(PREDICTION_FILE)
    historical_available = os.path.exists(HISTORICAL_FILE)
    shap_available = os.path.exists(SHAP_DIR)
    return jsonify({
        "status": "healthy" if prediction_available else "degraded",
        "service": "Karachi AQI Prediction API",
        "prediction_file": prediction_available,
        "historical_data": historical_available,
        "shap_data": shap_available,
    })


@app.route("/forecast", methods=["GET"])
def forecast():
    try:
        forecast_data = load_forecast()
        current = get_latest_actual_aqi()
        peak = max((x["aqi"] for x in forecast_data), default=None)
        lowest = min((x["aqi"] for x in forecast_data), default=None)
        peak_item = max(forecast_data, key=lambda x: x["aqi"]) if forecast_data else None
        lowest_item = min(forecast_data, key=lambda x: x["aqi"]) if forecast_data else None

        return jsonify({
            "project": "Pearls AQI Predictor",
            "city": "Karachi",
            "current": current,
            "current_available": current is not None,
            "source_date": get_source_date(),
            "forecast_days": len(forecast_data),
            "peak_aqi": peak,
            "peak_date": peak_item["date"] if peak_item else None,
            "lowest_aqi": lowest,
            "lowest_date": lowest_item["date"] if lowest_item else None,
            "forecast": forecast_data,
        })
    except Exception as exc:
        return jsonify({"project": "Pearls AQI Predictor", "error": str(exc)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({"project": "Pearls AQI Predictor", "city": "Karachi", "model_registry": "Hopsworks", "models": MODEL_INFO})


@app.route("/shap/<horizon>", methods=["GET"])
def shap_data(horizon):
    try:
        aliases = {"1": "AQI_t+1", "2": "AQI_t+2", "3": "AQI_t+3", "1_day": "AQI_t+1", "2_days": "AQI_t+2", "3_days": "AQI_t+3"}
        target = aliases.get(horizon.lower())
        if not target:
            return jsonify({"error": "Use 1_day, 2_days or 3_days."}), 400

        path = os.path.join(SHAP_DIR, f"{target}_feature_importance.csv")
        if not os.path.exists(path):
            return jsonify({"error": f"SHAP file not found: {path}"}), 404

        df = pd.read_csv(path)
        required = {"feature", "mean_abs_shap"}
        if not required.issubset(df.columns):
            return jsonify({"error": f"SHAP CSV must contain {sorted(required)}."}), 500

        df = df.sort_values("mean_abs_shap", ascending=False)
        return jsonify({"target": target, "horizon": horizon, "features": df.to_dict(orient="records")})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("=" * 70)
    print("PEARLS AQI PREDICTOR — KARACHI AQI API")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Prediction file: {PREDICTION_FILE}")
    print(f"Historical file: {HISTORICAL_FILE}")
    print(f"SHAP directory: {SHAP_DIR}")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False)
