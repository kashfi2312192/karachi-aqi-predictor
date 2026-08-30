import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")


# ============================================================
# PEARLS AQI PREDICTOR
# SHAP MODEL EXPLAINABILITY ANALYSIS
# ============================================================

DATA_PATH = "data/processed/karachi_ml_dataset.csv"

MODEL_DIR = "models/final"
RESULTS_DIR = "results"
SHAP_DIR = "results/shap"

os.makedirs(SHAP_DIR, exist_ok=True)


TARGETS = {
    "AQI_t+1": (
        "day1_svr_standard_recency.joblib",
        "SVR + StandardScaler + Recency"
    ),

    "AQI_t+2": (
        "day2_svr_robust.joblib",
        "SVR + RobustScaler"
    ),

    "AQI_t+3": (
        "day3_catboost.joblib",
        "CatBoost Regressor"
    )
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("SHAP MODEL EXPLAINABILITY ANALYSIS")
print("=" * 70)

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
# SAME FINAL TEST PERIOD
# ============================================================

test_size = int(
    np.ceil(len(df) * 0.20)
)

train_end = len(df) - test_size

X_test = X.iloc[train_end:].copy()

print(
    f"\nFinal test observations: "
    f"{len(X_test)}"
)

print(
    f"Test period: "
    f"{df['date'].iloc[train_end].date()} → "
    f"{df['date'].iloc[-1].date()}"
)


# ============================================================
# SAMPLE TEST DATA
# ============================================================

# SHAP can be expensive with 104 features.
# Use a representative subset of the untouched test set.

MAX_SAMPLES = 150

if len(X_test) > MAX_SAMPLES:

    X_explain = X_test.sample(
        n=MAX_SAMPLES,
        random_state=42
    ).sort_index()

else:

    X_explain = X_test.copy()


print(
    f"SHAP observations used: "
    f"{len(X_explain)}"
)


# ============================================================
# HELPER
# ============================================================

def save_bar_plot(
    importance,
    title,
    filename,
    top_n=20
):

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .head(top_n)
        .sort_values(
            "mean_abs_shap"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        importance["feature"],
        importance["mean_abs_shap"]
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PROCESS EACH MODEL
# ============================================================

for target, model_info in TARGETS.items():

    model_filename, model_name = model_info

    print("\n" + "=" * 70)

    print(
        f"ANALYZING: {target}"
    )

    print(
        f"Model: {model_name}"
    )

    print("=" * 70)


    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )


    if not os.path.exists(model_path):

        print(
            f"Model not found: "
            f"{model_path}"
        )

        continue


    print(
        "\nLoading model..."
    )

    model = joblib.load(
        model_path
    )


    # --------------------------------------------------------
    # CATBOOST
    # --------------------------------------------------------

    if target == "AQI_t+3":

        print(
            "Using TreeExplainer..."
        )

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer(
            X_explain
        )

        values = shap_values.values

        if values.ndim == 3:

            values = values[:, :, 0]


        # SHAP summary plot

        plt.figure()

        shap.summary_plot(
            shap_values,
            X_explain,
            max_display=20,
            show=False
        )

        plt.title(
            "Day 3 CatBoost — SHAP Feature Importance"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                SHAP_DIR,
                "day3_catboost_shap_summary.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    # --------------------------------------------------------
    # SVR
    # --------------------------------------------------------

    else:

        print(
            "Using KernelExplainer for SVR..."
        )

        # Background sample
        background_size = min(
            50,
            len(X_test)
        )

        background = (
            X_test
            .sample(
                n=background_size,
                random_state=42
            )
        )


        # Prediction function
        def predict_function(data):

            if isinstance(
                data,
                np.ndarray
            ):

                data = pd.DataFrame(
                    data,
                    columns=feature_columns
                )

            return model.predict(
                data
            )


        explainer = shap.KernelExplainer(
            predict_function,
            background
        )


        shap_values = explainer.shap_values(
            X_explain,
            nsamples=100
        )

        values = np.asarray(
            shap_values
        )


        # Summary plot

        plt.figure()

        shap.summary_plot(
            values,
            X_explain,
            max_display=20,
            show=False
        )

        plt.title(
            f"{target} — {model_name} SHAP Feature Importance"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                SHAP_DIR,
                f"{target}_shap_summary.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    # ========================================================
    # FEATURE IMPORTANCE TABLE
    # ========================================================

    mean_abs_shap = np.mean(
        np.abs(values),
        axis=0
    )

    importance_df = pd.DataFrame({

        "feature":
            feature_columns,

        "mean_abs_shap":
            mean_abs_shap

    })


    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )


    importance_path = os.path.join(
        SHAP_DIR,
        f"{target}_feature_importance.csv"
    )


    importance_df.to_csv(
        importance_path,
        index=False
    )


    # ========================================================
    # CUSTOM BAR CHART
    # ========================================================

    save_bar_plot(

        importance_df,

        f"{target} — Top 20 SHAP Features",

        os.path.join(
            SHAP_DIR,
            f"{target}_top20_features.png"
        ),

        top_n=20
    )


    # ========================================================
    # PRINT TOP FEATURES
    # ========================================================

    print(
        "\nTop 15 features:"
    )

    print(
        importance_df
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nSaved SHAP files to:"
)

print(
    f"  {SHAP_DIR}"
)

print(
    "\nGenerated:"
)

print(
    "  • Day 1 SHAP summary"
)

print(
    "  • Day 2 SHAP summary"
)

print(
    "  • Day 3 SHAP summary"
)

print(
    "  • Feature importance CSV files"
)

print(
    "  • Top-20 feature charts"
)

print(
    "\nNext step:"
)

print(
    "Actual-vs-predicted plots + "
    "residual/error analysis."
)