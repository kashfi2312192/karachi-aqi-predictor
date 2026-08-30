import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# ============================================================
# PEARLS AQI PREDICTOR
# FAST FINAL HORIZON-SPECIFIC MODELING
# ============================================================

DATA_PATH = "data/processed/karachi_ml_dataset.csv"

MODEL_DIR = "models/final"
RESULTS_DIR = "results"

TARGETS = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3"
]

RANDOM_STATE = 42

TEST_SIZE = 0.20
N_SPLITS = 5


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("FAST FINAL HORIZON-SPECIFIC MODELING")
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


print(f"Dataset shape: {df.shape}")

print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print(
    f"Usable observations: {len(df)}"
)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    c
    for c in df.columns
    if c not in TARGETS + ["date"]
]

X_all = df[feature_columns].copy()

print(
    f"\nNumber of candidate features: "
    f"{len(feature_columns)}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n_total = len(df)

test_size = int(
    np.ceil(n_total * TEST_SIZE)
)

train_end = n_total - test_size


X_development = X_all.iloc[:train_end].copy()
X_test = X_all.iloc[train_end:].copy()


print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"Development observations: {len(X_development)}"
)

print(
    f"Final test observations:  {len(X_test)}"
)

print(
    f"Development period: "
    f"{df['date'].iloc[0].date()} → "
    f"{df['date'].iloc[train_end - 1].date()}"
)

print(
    f"FINAL TEST PERIOD: "
    f"{df['date'].iloc[train_end].date()} → "
    f"{df['date'].iloc[-1].date()}"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, predictions):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# ============================================================
# RECENCY WEIGHTS
# ============================================================

def calculate_recency_weights(n):

    if n <= 1:
        return np.ones(n)

    positions = np.arange(n)

    normalized = (
        positions /
        (n - 1)
    )

    weights = np.exp(
        2.0 * normalized
    )

    weights = (
        weights /
        np.mean(weights)
    )

    return weights


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_day1_model():

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            SVR(
                kernel="rbf"
            )
        )
    ])

    parameter_grid = {
        "model__C": [
            1,
            10,
            50,
            100
        ],

        "model__epsilon": [
            0.05,
            0.1,
            0.2,
            0.5
        ],

        "model__gamma": [
            "scale",
            0.01,
            0.05,
            0.1
        ]
    }

    return pipeline, parameter_grid


def build_day2_model():

    pipeline = Pipeline([
        (
            "scaler",
            RobustScaler()
        ),

        (
            "model",
            SVR(
                kernel="rbf"
            )
        )
    ])

    parameter_grid = {
        "model__C": [
            1,
            10,
            50,
            100
        ],

        "model__epsilon": [
            0.05,
            0.1,
            0.2,
            0.5
        ],

        "model__gamma": [
            "scale",
            0.01,
            0.05,
            0.1
        ]
    }

    return pipeline, parameter_grid


def build_day3_model():

    """
    Fast CatBoost search.

    Only 8 combinations × 5 folds = 40 fits.
    """

    model = CatBoostRegressor(
        loss_function="RMSE",

        random_seed=RANDOM_STATE,

        verbose=False,

        allow_writing_files=False,

        # Use multiple CPU cores.
        thread_count=-1
    )

    parameter_grid = {
        "depth": [
            4,
            6
        ],

        "learning_rate": [
            0.05,
            0.1
        ],

        "iterations": [
            300,
            500
        ],

        "l2_leaf_reg": [
            3
        ]
    }

    return model, parameter_grid


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

def persistence_predictions(X):

    return X["AQI"].values


# ============================================================
# RESULT STORAGE
# ============================================================

all_results = []


# ============================================================
# DAY 1
# ============================================================

print("\n\n")
print("=" * 70)
print("DAY 1 — AQI_t+1")
print("=" * 70)


target = "AQI_t+1"

y_all = df[target].values

y_development = y_all[:train_end]
y_test = y_all[train_end:]


baseline_predictions = (
    persistence_predictions(
        X_test
    )
)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions
)


print("\nPersistence baseline:")

print(
    f"RMSE: {baseline_metrics['RMSE']:.4f}"
)

print(
    f"MAE : {baseline_metrics['MAE']:.4f}"
)

print(
    f"R²  : {baseline_metrics['R2']:.4f}"
)


all_results.append({
    "horizon": target,
    "model": "Persistence Baseline",
    "RMSE": baseline_metrics["RMSE"],
    "MAE": baseline_metrics["MAE"],
    "R2": baseline_metrics["R2"]
})


day1_path = os.path.join(
    MODEL_DIR,
    "day1_svr_standard_recency.joblib"
)


if os.path.exists(day1_path):

    print(
        "\nExisting Day 1 model found."
    )

    print(
        "Loading saved model instead "
        "of retraining..."
    )

    day1_model = joblib.load(
        day1_path
    )

else:

    print(
        "\nDay 1 model not found."
    )

    print(
        "Training SVR + StandardScaler "
        "+ Recency..."
    )

    pipeline, parameter_grid = (
        build_day1_model()
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(
            n_splits=N_SPLITS
        ),
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    weights = calculate_recency_weights(
        len(X_development)
    )

    search.fit(
        X_development,
        y_development,
        model__sample_weight=weights
    )

    day1_model = (
        search.best_estimator_
    )

    joblib.dump(
        day1_model,
        day1_path
    )


day1_predictions = (
    day1_model.predict(
        X_test
    )
)

day1_metrics = calculate_metrics(
    y_test,
    day1_predictions
)


print("\nDay 1 FINAL TEST RESULTS:")

print(
    f"RMSE: {day1_metrics['RMSE']:.4f}"
)

print(
    f"MAE : {day1_metrics['MAE']:.4f}"
)

print(
    f"R²  : {day1_metrics['R2']:.4f}"
)


all_results.append({
    "horizon": target,
    "model":
        "SVR + StandardScaler + Recency",
    "RMSE":
        day1_metrics["RMSE"],
    "MAE":
        day1_metrics["MAE"],
    "R2":
        day1_metrics["R2"]
})


pd.DataFrame({
    "date":
        df["date"].iloc[train_end:].values,

    "actual_AQI":
        y_test,

    "predicted_AQI":
        day1_predictions,

    "horizon":
        target

}).to_csv(
    os.path.join(
        RESULTS_DIR,
        "AQI_t+1_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# DAY 2
# ============================================================

print("\n\n")
print("=" * 70)
print("DAY 2 — AQI_t+2")
print("=" * 70)


target = "AQI_t+2"

y_all = df[target].values

y_development = y_all[:train_end]
y_test = y_all[train_end:]


baseline_predictions = (
    persistence_predictions(
        X_test
    )
)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions
)


print("\nPersistence baseline:")

print(
    f"RMSE: {baseline_metrics['RMSE']:.4f}"
)

print(
    f"MAE : {baseline_metrics['MAE']:.4f}"
)

print(
    f"R²  : {baseline_metrics['R2']:.4f}"
)


all_results.append({
    "horizon": target,
    "model": "Persistence Baseline",
    "RMSE": baseline_metrics["RMSE"],
    "MAE": baseline_metrics["MAE"],
    "R2": baseline_metrics["R2"]
})


day2_path = os.path.join(
    MODEL_DIR,
    "day2_svr_robust.joblib"
)


if os.path.exists(day2_path):

    print(
        "\nExisting Day 2 model found."
    )

    print(
        "Loading saved model instead "
        "of retraining..."
    )

    day2_model = joblib.load(
        day2_path
    )

else:

    print(
        "\nDay 2 model not found."
    )

    print(
        "Training SVR + RobustScaler..."
    )

    pipeline, parameter_grid = (
        build_day2_model()
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(
            n_splits=N_SPLITS
        ),
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(
        X_development,
        y_development
    )

    day2_model = (
        search.best_estimator_
    )

    joblib.dump(
        day2_model,
        day2_path
    )


day2_predictions = (
    day2_model.predict(
        X_test
    )
)

day2_metrics = calculate_metrics(
    y_test,
    day2_predictions
)


print("\nDay 2 FINAL TEST RESULTS:")

print(
    f"RMSE: {day2_metrics['RMSE']:.4f}"
)

print(
    f"MAE : {day2_metrics['MAE']:.4f}"
)

print(
    f"R²  : {day2_metrics['R2']:.4f}"
)


all_results.append({
    "horizon": target,
    "model":
        "SVR + RobustScaler",
    "RMSE":
        day2_metrics["RMSE"],
    "MAE":
        day2_metrics["MAE"],
    "R2":
        day2_metrics["R2"]
})


pd.DataFrame({
    "date":
        df["date"].iloc[train_end:].values,

    "actual_AQI":
        y_test,

    "predicted_AQI":
        day2_predictions,

    "horizon":
        target

}).to_csv(
    os.path.join(
        RESULTS_DIR,
        "AQI_t+2_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# DAY 3
# ============================================================

print("\n\n")
print("=" * 70)
print("DAY 3 — AQI_t+3")
print("=" * 70)


target = "AQI_t+3"

y_all = df[target].values

y_development = y_all[:train_end]
y_test = y_all[train_end:]


baseline_predictions = (
    persistence_predictions(
        X_test
    )
)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions
)


print("\nPersistence baseline:")

print(
    f"RMSE: {baseline_metrics['RMSE']:.4f}"
)

print(
    f"MAE : {baseline_metrics['MAE']:.4f}"
)

print(
    f"R²  : {baseline_metrics['R2']:.4f}"
)


all_results.append({
    "horizon": target,
    "model": "Persistence Baseline",
    "RMSE": baseline_metrics["RMSE"],
    "MAE": baseline_metrics["MAE"],
    "R2": baseline_metrics["R2"]
})


print(
    "\nTraining CatBoost..."
)

print(
    "Search size: "
    "8 combinations × 5 folds = 40 fits"
)


pipeline, parameter_grid = (
    build_day3_model()
)


search = GridSearchCV(
    estimator=pipeline,

    param_grid=parameter_grid,

    scoring="neg_root_mean_squared_error",

    cv=TimeSeriesSplit(
        n_splits=N_SPLITS
    ),

    n_jobs=1,

    refit=True,

    verbose=1
)


search.fit(
    X_development,
    y_development
)


day3_model = (
    search.best_estimator_
)


print(
    "\nBest CatBoost parameters:"
)

print(
    search.best_params_
)

print(
    f"\nBest CV RMSE: "
    f"{-search.best_score_:.4f}"
)


day3_predictions = (
    day3_model.predict(
        X_test
    )
)

day3_metrics = calculate_metrics(
    y_test,
    day3_predictions
)


print("\nDay 3 FINAL TEST RESULTS:")

print(
    f"RMSE: {day3_metrics['RMSE']:.4f}"
)

print(
    f"MAE : {day3_metrics['MAE']:.4f}"
)

print(
    f"R²  : {day3_metrics['R2']:.4f}"
)


baseline_rmse = (
    baseline_metrics["RMSE"]
)

improvement = (
    (
        baseline_rmse -
        day3_metrics["RMSE"]
    )
    /
    baseline_rmse
) * 100


print(
    f"\nRMSE improvement over baseline: "
    f"{improvement:.2f}%"
)


# ============================================================
# SAVE DAY 3 MODEL
# ============================================================

day3_path = os.path.join(
    MODEL_DIR,
    "day3_catboost.joblib"
)


joblib.dump(
    day3_model,
    day3_path
)


print(
    f"\nSaved model:"
)

print(day3_path)


# ============================================================
# SAVE DAY 3 PREDICTIONS
# ============================================================

pd.DataFrame({

    "date":
        df["date"].iloc[train_end:].values,

    "actual_AQI":
        y_test,

    "predicted_AQI":
        day3_predictions,

    "horizon":
        target

}).to_csv(
    os.path.join(
        RESULTS_DIR,
        "AQI_t+3_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# STORE DAY 3 RESULTS
# ============================================================

all_results.append({

    "horizon":
        target,

    "model":
        "CatBoost Regressor",

    "RMSE":
        day3_metrics["RMSE"],

    "MAE":
        day3_metrics["MAE"],

    "R2":
        day3_metrics["R2"]
})


# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "final_model_comparison.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


print(
    f"\nResults saved to:"
)

print(results_path)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)


for target in TARGETS:

    rows = results_df[
        results_df["horizon"] == target
    ]

    print(
        f"\n{target}"
    )

    for _, row in rows.iterrows():

        print(
            f"  {row['model']}: "
            f"RMSE={row['RMSE']:.4f}, "
            f"MAE={row['MAE']:.4f}, "
            f"R²={row['R2']:.4f}"
        )


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nSaved models:")

print(
    "  models/final/"
    "day1_svr_standard_recency.joblib"
)

print(
    "  models/final/"
    "day2_svr_robust.joblib"
)

print(
    "  models/final/"
    "day3_catboost.joblib"
)

print("\nSaved results:")

print(
    "  results/final_model_comparison.csv"
)

print(
    "  results/AQI_t+1_test_predictions.csv"
)

print(
    "  results/AQI_t+2_test_predictions.csv"
)

print(
    "  results/AQI_t+3_test_predictions.csv"
)

print(
    "\nNext step: SHAP analysis + "
    "3-day production prediction pipeline."
)