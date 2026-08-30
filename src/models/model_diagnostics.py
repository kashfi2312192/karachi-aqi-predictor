import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PEARLS AQI PREDICTOR
# MODEL DIAGNOSTICS & ERROR ANALYSIS
# ============================================================

DATA_PATH = "data/processed/karachi_ml_dataset.csv"

MODEL_DIR = "models/final"
RESULTS_DIR = "results"
DIAGNOSTICS_DIR = "results/diagnostics"

os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {
    "AQI_t+1": {
        "file": "day1_svr_standard_recency.joblib",
        "name": "SVR + StandardScaler + Recency"
    },

    "AQI_t+2": {
        "file": "day2_svr_robust.joblib",
        "name": "SVR + RobustScaler"
    },

    "AQI_t+3": {
        "file": "day3_catboost.joblib",
        "name": "CatBoost Regressor"
    }
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("MODEL DIAGNOSTICS & ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna().reset_index(drop=True)


# ============================================================
# FEATURES
# ============================================================

target_columns = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]

feature_columns = [
    c
    for c in df.columns
    if c not in target_columns + ["date"]
]

X = df[feature_columns].copy()


# ============================================================
# SAME FINAL TEST SPLIT
# ============================================================

test_size = int(
    np.ceil(len(df) * 0.20)
)

train_end = len(df) - test_size

X_test = X.iloc[train_end:].copy()

test_dates = (
    df["date"]
    .iloc[train_end:]
    .reset_index(drop=True)
)


print(
    f"\nFinal test observations: "
    f"{len(X_test)}"
)

print(
    f"Test period: "
    f"{test_dates.iloc[0].date()} → "
    f"{test_dates.iloc[-1].date()}"
)


# ============================================================
# STORAGE
# ============================================================

comparison_rows = []

all_error_rows = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return rmse, mae, r2


def save_figure(
    filename
):

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            DIAGNOSTICS_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PROCESS EACH MODEL
# ============================================================

for horizon, config in MODELS.items():

    print("\n" + "=" * 70)

    print(
        f"DIAGNOSTICS: {horizon}"
    )

    print(
        f"Model: {config['name']}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        config["file"]
    )


    if not os.path.exists(model_path):

        print(
            f"Model not found: {model_path}"
        )

        continue


    print(
        "\nLoading model..."
    )

    model = joblib.load(
        model_path
    )


    # --------------------------------------------------------
    # ACTUAL TARGET
    # --------------------------------------------------------

    actual = (
        df[horizon]
        .iloc[train_end:]
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print(
        "Generating test predictions..."
    )

    predicted = model.predict(
        X_test
    )

    predicted = pd.Series(
        predicted
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    rmse, mae, r2 = calculate_metrics(
        actual,
        predicted
    )


    print(
        f"\nRMSE: {rmse:.4f}"
    )

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


    # --------------------------------------------------------
    # ERROR DATAFRAME
    # --------------------------------------------------------

    errors = pd.DataFrame({

        "date":
            test_dates,

        "actual_AQI":
            actual,

        "predicted_AQI":
            predicted,

    })


    errors["error"] = (
        errors["predicted_AQI"]
        -
        errors["actual_AQI"]
    )

    errors["absolute_error"] = (
        errors["error"]
        .abs()
    )

    errors["squared_error"] = (
        errors["error"] ** 2
    )


    # --------------------------------------------------------
    # AQI CATEGORIES
    # --------------------------------------------------------

    def aqi_category(aqi):

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


    errors["actual_category"] = (
        errors["actual_AQI"]
        .apply(aqi_category)
    )


    errors["predicted_category"] = (
        errors["predicted_AQI"]
        .apply(aqi_category)
    )


    errors["category_correct"] = (
        errors["actual_category"]
        ==
        errors["predicted_category"]
    )


    # --------------------------------------------------------
    # SAVE DETAILED ERRORS
    # --------------------------------------------------------

    error_path = os.path.join(
        DIAGNOSTICS_DIR,
        f"{horizon}_errors.csv"
    )


    errors.to_csv(
        error_path,
        index=False
    )


    print(
        f"\nSaved error analysis:"
    )

    print(error_path)


    # --------------------------------------------------------
    # CATEGORY ACCURACY
    # --------------------------------------------------------

    category_accuracy = (
        errors["category_correct"]
        .mean()
        * 100
    )


    print(
        f"\nAQI category accuracy: "
        f"{category_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # ERROR BY AQI CATEGORY
    # --------------------------------------------------------

    category_errors = (
        errors
        .groupby(
            "actual_category",
            observed=True
        )
        .agg(
            samples=(
                "absolute_error",
                "count"
            ),

            MAE=(
                "absolute_error",
                "mean"
            ),

            RMSE=(
                "squared_error",
                lambda x:
                np.sqrt(x.mean())
            )
        )
        .reset_index()
    )


    print(
        "\nError by actual AQI category:"
    )

    print(
        category_errors.to_string(
            index=False
        )
    )


    category_errors.to_csv(
        os.path.join(
            DIAGNOSTICS_DIR,
            f"{horizon}_error_by_category.csv"
        ),
        index=False
    )


    # ========================================================
    # 1. ACTUAL VS PREDICTED
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        test_dates,
        actual,
        label="Actual AQI",
        linewidth=2
    )

    plt.plot(
        test_dates,
        predicted,
        label="Predicted AQI",
        linewidth=2
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "AQI"
    )

    plt.title(
        f"{horizon} — Actual vs Predicted AQI"
    )

    plt.legend()

    save_figure(
        f"{horizon}_actual_vs_predicted.png"
    )


    # ========================================================
    # 2. SCATTER: ACTUAL VS PREDICTED
    # ========================================================

    plt.figure(
        figsize=(8, 8)
    )

    plt.scatter(
        actual,
        predicted,
        alpha=0.6
    )


    min_value = min(
        actual.min(),
        predicted.min()
    )

    max_value = max(
        actual.max(),
        predicted.max()
    )


    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )


    plt.xlabel(
        "Actual AQI"
    )

    plt.ylabel(
        "Predicted AQI"
    )

    plt.title(
        f"{horizon} — Actual vs Predicted"
    )


    save_figure(
        f"{horizon}_prediction_scatter.png"
    )


    # ========================================================
    # 3. RESIDUAL OVER TIME
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.scatter(
        test_dates,
        errors["error"],
        alpha=0.6
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Residual (Predicted - Actual)"
    )

    plt.title(
        f"{horizon} — Residuals Over Time"
    )


    save_figure(
        f"{horizon}_residuals_over_time.png"
    )


    # ========================================================
    # 4. RESIDUAL DISTRIBUTION
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        errors["error"],
        bins=30
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Residual"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{horizon} — Residual Distribution"
    )


    save_figure(
        f"{horizon}_residual_distribution.png"
    )


    # ========================================================
    # 5. ABSOLUTE ERROR OVER TIME
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        test_dates,
        errors["absolute_error"],
        linewidth=1.5
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Absolute Error"
    )

    plt.title(
        f"{horizon} — Absolute Prediction Error"
    )


    save_figure(
        f"{horizon}_absolute_error.png"
    )


    # ========================================================
    # STORE COMPARISON
    # ========================================================

    comparison_rows.append({

        "horizon":
            horizon,

        "model":
            config["name"],

        "RMSE":
            rmse,

        "MAE":
            mae,

        "R2":
            r2,

        "AQI_Category_Accuracy":
            category_accuracy
    })


    # ========================================================
    # STORE ERRORS
    # ========================================================

    temp_errors = errors.copy()

    temp_errors["horizon"] = horizon

    temp_errors["model"] = config["name"]

    all_error_rows.append(
        temp_errors
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_path = os.path.join(
    DIAGNOSTICS_DIR,
    "diagnostic_model_comparison.csv"
)


comparison_df.to_csv(
    comparison_path,
    index=False
)


print("\n\n")
print("=" * 70)
print("DIAGNOSTIC MODEL COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# COMBINED ERROR DATA
# ============================================================

if all_error_rows:

    combined_errors = pd.concat(
        all_error_rows,
        ignore_index=True
    )

    combined_errors.to_csv(
        os.path.join(
            DIAGNOSTICS_DIR,
            "all_model_errors.csv"
        ),
        index=False
    )


# ============================================================
# MODEL PERFORMANCE BAR CHART
# ============================================================

if len(comparison_df) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    labels = (
        comparison_df["horizon"]
        .tolist()
    )

    rmse_values = (
        comparison_df["RMSE"]
        .tolist()
    )

    plt.bar(
        labels,
        rmse_values
    )

    plt.xlabel(
        "Forecast Horizon"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "AQI Forecasting RMSE by Horizon"
    )

    save_figure(
        "model_rmse_comparison.png"
    )


    # --------------------------------------------------------
    # R2 CHART
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    r2_values = (
        comparison_df["R2"]
        .tolist()
    )

    plt.bar(
        labels,
        r2_values
    )

    plt.xlabel(
        "Forecast Horizon"
    )

    plt.ylabel(
        "R²"
    )

    plt.title(
        "AQI Forecasting R² by Horizon"
    )

    save_figure(
        "model_r2_comparison.png"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("MODEL DIAGNOSTICS COMPLETE")
print("=" * 70)

print(
    "\nDiagnostics saved to:"
)

print(
    f"  {DIAGNOSTICS_DIR}"
)

print("\nGenerated:")

print(
    "  • Actual vs predicted plots"
)

print(
    "  • Prediction scatter plots"
)

print(
    "  • Residual-over-time plots"
)

print(
    "  • Residual distributions"
)

print(
    "  • Absolute error plots"
)

print(
    "  • Error by AQI category"
)

print(
    "  • AQI category accuracy"
)

print(
    "  • Combined model comparison"
)

print(
    "\nNext step:"
)

print(
    "Build the production 3-day prediction pipeline "
    "using Open-Meteo."
)