import os
import pandas as pd

from flask import Flask, jsonify
from flask_cors import CORS


# ============================================================
# PEARLS AQI PREDICTOR
# KARACHI AQI FLASK API
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


PREDICTION_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "latest_predictions.csv"
)


HISTORICAL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "hopsworks_training_data.csv"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {

    "1_day": {
        "registry_name": "karachi_aqi_svr_t_1",
        "version": 1,
        "model": "SVR + StandardScaler + Recency",
        "rmse": 13.6674,
        "mae": 9.3747,
        "r2": 0.7137
    },

    "2_days": {
        "registry_name": "karachi_aqi_svr_t_2",
        "version": 1,
        "model": "SVR + RobustScaler",
        "rmse": 18.3090,
        "mae": 13.5716,
        "r2": 0.4830
    },

    "3_days": {
        "registry_name": "karachi_aqi_catboost_t_3",
        "version": 1,
        "model": "CatBoost Regressor",
        "rmse": 19.7632,
        "mae": 15.4517,
        "r2": 0.4000
    }
}


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_aqi_category(aqi):
    """
    Return AQI health category.
    """

    aqi = float(aqi)

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


def load_predictions():
    """
    Load the latest prediction CSV.
    """

    if not os.path.exists(PREDICTION_FILE):

        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_FILE}"
        )

    df = pd.read_csv(
        PREDICTION_FILE
    )

    if df.empty:

        raise ValueError(
            "Prediction file is empty."
        )

    return df


def get_source_date():
    """
    Determine the latest historical date
    available in the retrieved Hopsworks data.
    """

    if not os.path.exists(HISTORICAL_FILE):

        return None

    historical_df = pd.read_csv(
        HISTORICAL_FILE
    )

    if "date" not in historical_df.columns:

        return None

    dates = pd.to_datetime(
        historical_df["date"],
        errors="coerce"
    )

    dates = dates.dropna()

    if dates.empty:

        return None

    return dates.max().strftime(
        "%Y-%m-%d"
    )


def find_column(df, candidates):
    """
    Find the first matching column from a list
    of possible column names.
    """

    for column in candidates:

        if column in df.columns:
            return column

    return None


def normalize_horizon(value, index):
    """
    Normalize forecast horizon values.

    Supports values such as:
        1_day
        2_days
        3_days
        day_1
        day_2
        day_3
    """

    if pd.isna(value):

        return f"{index + 1}_day" if index == 0 else f"{index + 1}_days"

    value = str(value).strip().lower()

    if value in [
        "1_day",
        "day_1",
        "1",
        "day1"
    ]:

        return "1_day"

    if value in [
        "2_days",
        "day_2",
        "2",
        "day2"
    ]:

        return "2_days"

    if value in [
        "3_days",
        "day_3",
        "3",
        "day3"
    ]:

        return "3_days"

    return value


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "project":
            "Pearls AQI Predictor",

        "service":
            "Karachi AQI Prediction API",

        "status":
            "running",

        "city":
            "Karachi",

        "forecast_horizon":
            "3 days",

        "model_registry":
            "Hopsworks",

        "endpoints": {

            "health":
                "/health",

            "forecast":
                "/forecast",

            "model_info":
                "/model-info"

        }

    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    prediction_available = os.path.exists(
        PREDICTION_FILE
    )

    historical_available = os.path.exists(
        HISTORICAL_FILE
    )

    status = (
        "healthy"
        if prediction_available
        else "degraded"
    )

    return jsonify({

        "status":
            status,

        "service":
            "Karachi AQI Prediction API",

        "prediction_file":
            prediction_available,

        "historical_data":
            historical_available,

        "prediction_file_path":
            PREDICTION_FILE

    })


# ============================================================
# FORECAST
# ============================================================

@app.route(
    "/forecast",
    methods=["GET"]
)
def forecast():

    try:

        # ----------------------------------------------------
        # LOAD PREDICTIONS
        # ----------------------------------------------------

        df = load_predictions()


        # ----------------------------------------------------
        # FIND DATE COLUMN
        # ----------------------------------------------------

        date_column = find_column(
            df,
            [
                "forecast_date",
                "date",
                "Date",
                "prediction_date"
            ]
        )


        # ----------------------------------------------------
        # FIND AQI COLUMN
        # ----------------------------------------------------

        aqi_column = find_column(
            df,
            [
                "predicted_aqi",
                "aqi",
                "AQI",
                "prediction",
                "predicted_AQI"
            ]
        )


        if aqi_column is None:

            raise ValueError(
                "Could not find AQI column. "
                f"Available columns: {df.columns.tolist()}"
            )


        # ----------------------------------------------------
        # FIND HORIZON COLUMN
        # ----------------------------------------------------

        horizon_column = find_column(
            df,
            [
                "forecast_horizon",
                "horizon",
                "day",
                "prediction_horizon"
            ]
        )


        # ----------------------------------------------------
        # PARSE DATES
        # ----------------------------------------------------

        if date_column is not None:

            dates = pd.to_datetime(
                df[date_column],
                errors="coerce"
            )

        else:

            dates = pd.Series(
                [pd.NaT] * len(df)
            )


        # ----------------------------------------------------
        # FALLBACK DATES
        # ----------------------------------------------------

        if dates.isna().any():

            source_date = get_source_date()

            if source_date is None:

                raise ValueError(
                    "Could not determine forecast dates."
                )

            source_date = pd.Timestamp(
                source_date
            )

            dates = pd.Series([

                source_date +
                pd.Timedelta(
                    days=i
                )

                for i in range(
                    1,
                    len(df) + 1
                )

            ])


        # ----------------------------------------------------
        # BUILD FORECAST
        # ----------------------------------------------------

        forecast_data = []


        for index, row in df.iterrows():

            # ------------------------------------------------
            # AQI
            # ------------------------------------------------

            aqi = round(
                float(row[aqi_column]),
                2
            )


            # ------------------------------------------------
            # CATEGORY
            # ------------------------------------------------

            category = get_aqi_category(
                aqi
            )


            # ------------------------------------------------
            # HORIZON
            # ------------------------------------------------

            if horizon_column is not None:

                horizon = normalize_horizon(
                    row[horizon_column],
                    index
                )

            else:

                if index == 0:
                    horizon = "1_day"

                elif index == 1:
                    horizon = "2_days"

                else:
                    horizon = "3_days"


            # ------------------------------------------------
            # MODEL INFORMATION
            # ------------------------------------------------

            model_info = MODEL_INFO.get(
                horizon
            )


            # ------------------------------------------------
            # FORECAST ITEM
            # ------------------------------------------------

            forecast_item = {

                "date":
                    dates.iloc[index].strftime(
                        "%Y-%m-%d"
                    ),

                "aqi":
                    aqi,

                "category":
                    category,

                "horizon":
                    horizon

            }


            # ------------------------------------------------
            # ADD MODEL INFORMATION
            # ------------------------------------------------

            if model_info:

                forecast_item["model"] = (
                    model_info["model"]
                )

                forecast_item["model_registry"] = (
                    model_info["registry_name"]
                )

                forecast_item["model_version"] = (
                    model_info["version"]
                )

                forecast_item["rmse"] = (
                    model_info["rmse"]
                )

                forecast_item["mae"] = (
                    model_info["mae"]
                )

                forecast_item["r2"] = (
                    model_info["r2"]
                )


            forecast_data.append(
                forecast_item
            )


        # ----------------------------------------------------
        # LIMIT TO THREE DAYS
        # ----------------------------------------------------

        forecast_data = forecast_data[:3]


        # ----------------------------------------------------
        # SOURCE DATE
        # ----------------------------------------------------

        source_date = get_source_date()


        # ----------------------------------------------------
        # PEAK AQI
        # ----------------------------------------------------

        peak_aqi = None

        if forecast_data:

            peak_aqi = max(
                item["aqi"]
                for item in forecast_data
            )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "project":
                "Pearls AQI Predictor",

            "city":
                "Karachi",

            "forecast_days":
                len(forecast_data),

            "source_date":
                source_date,

            "peak_aqi":
                peak_aqi,

            "forecast":
                forecast_data

        })


    except Exception as e:

        return jsonify({

            "project":
                "Pearls AQI Predictor",

            "error":
                str(e)

        }), 500


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.route(
    "/model-info",
    methods=["GET"]
)
def model_info():

    return jsonify({

        "project":
            "Pearls AQI Predictor",

        "city":
            "Karachi",

        "model_registry":
            "Hopsworks",

        "models":
            MODEL_INFO

    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("KARACHI AQI API")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Prediction file: {PREDICTION_FILE}")
    print(f"Historical file: {HISTORICAL_FILE}")

    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )