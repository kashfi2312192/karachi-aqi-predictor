import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# ============================================================
# PEARLS AQI PREDICTOR — SHAP MODEL EXPLAINABILITY ANALYSIS
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "karachi_ml_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "final")
SHAP_DIR = os.path.join(PROJECT_ROOT, "results", "shap")
os.makedirs(SHAP_DIR, exist_ok=True)

TARGETS = {
    "AQI_t+1": ("day1_svr_standard_recency.joblib", "SVR + StandardScaler + Recency"),
    "AQI_t+2": ("day2_svr_robust.joblib", "SVR + RobustScaler"),
    "AQI_t+3": ("day3_catboost.joblib", "CatBoost Regressor"),
}

MAX_SAMPLES = 150
BACKGROUND_SIZE = 50
KERNEL_NSAMPLES = 100


def save_bar_plot(importance, title, filename, top_n=20):
    plot_df = importance.sort_values("mean_abs_shap", ascending=False).head(top_n).sort_values("mean_abs_shap")
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("PEARLS AQI PREDICTOR")
    print("SHAP MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    target_columns = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]
    feature_columns = [c for c in df.columns if c not in target_columns + ["date"]]
    X = df[feature_columns].copy()

    test_size = int(np.ceil(len(df) * 0.20))
    train_end = len(df) - test_size
    X_test = X.iloc[train_end:].copy()

    X_explain = X_test.sample(n=min(MAX_SAMPLES, len(X_test)), random_state=42).sort_index()

    print(f"Final test observations: {len(X_test)}")
    print(f"SHAP observations used: {len(X_explain)}")

    for target, (model_filename, model_name) in TARGETS.items():
        print("\n" + "=" * 70)
        print(f"ANALYZING: {target}")
        print(f"MODEL: {model_name}")
        print("=" * 70)

        model_path = os.path.join(MODEL_DIR, model_filename)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        model = joblib.load(model_path)

        if target == "AQI_t+3":
            print("Using TreeExplainer...")
            explainer = shap.TreeExplainer(model)
            explanation = explainer(X_explain)
            values = np.asarray(explanation.values)
            if values.ndim == 3:
                values = values[:, :, 0]

            plt.figure()
            shap.summary_plot(explanation, X_explain, max_display=20, show=False)
            plt.title("Day 3 CatBoost — SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(SHAP_DIR, "day3_catboost_shap_summary.png"), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            print("Using KernelExplainer for SVR...")
            background = X_test.sample(n=min(BACKGROUND_SIZE, len(X_test)), random_state=42)

            def predict_function(data):
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data, columns=feature_columns)
                return model.predict(data)

            explainer = shap.KernelExplainer(predict_function, background)
            shap_values = explainer.shap_values(X_explain, nsamples=KERNEL_NSAMPLES)
            values = np.asarray(shap_values)
            if values.ndim == 3:
                values = values[:, :, 0]

            plt.figure()
            shap.summary_plot(values, X_explain, max_display=20, show=False)
            plt.title(f"{target} — {model_name} SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(SHAP_DIR, f"{target}_shap_summary.png"), dpi=300, bbox_inches="tight")
            plt.close()

        mean_abs_shap = np.mean(np.abs(values), axis=0)
        importance_df = pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs_shap})
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_df.to_csv(os.path.join(SHAP_DIR, f"{target}_feature_importance.csv"), index=False)
        save_bar_plot(importance_df, f"{target} — Top 20 SHAP Features", os.path.join(SHAP_DIR, f"{target}_top20_features.png"))

        print("\nTop 15 features:")
        print(importance_df.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print(f"Saved to: {SHAP_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
