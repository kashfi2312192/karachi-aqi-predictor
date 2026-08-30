import os
import pandas as pd
import hopsworks


# ============================================================
# PEARLS AQI PREDICTOR
# REGISTER FINAL LOCAL MODELS IN HOPSWORKS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "final"
)

RESULTS_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "final_model_comparison.csv"
)

FEATURE_INFO_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "reference",
    "feature_info.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {

    "AQI_t+1": {
        "registry_name": "karachi_aqi_svr_t_1",
        "filename": "day1_svr_standard_recency.joblib",
        "model_description": (
            "Pearls AQI Predictor production model for "
            "Karachi AQI one-day-ahead prediction. "
            "SVR with StandardScaler and recency weighting."
        )
    },

    "AQI_t+2": {
        "registry_name": "karachi_aqi_svr_t_2",
        "filename": "day2_svr_robust.joblib",
        "model_description": (
            "Pearls AQI Predictor production model for "
            "Karachi AQI two-days-ahead prediction. "
            "SVR with RobustScaler."
        )
    },

    "AQI_t+3": {
        "registry_name": "karachi_aqi_catboost_t_3",
        "filename": "day3_catboost.joblib",
        "model_description": (
            "Pearls AQI Predictor production model for "
            "Karachi AQI three-days-ahead prediction. "
            "CatBoost Regressor."
        )
    }
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("REGISTERING FINAL LOCAL MODELS IN HOPSWORKS")
print("=" * 70)

print(
    f"\nProject root:\n"
    f"{PROJECT_ROOT}"
)

print(
    f"\nFinal model directory:\n"
    f"{MODEL_DIR}"
)

print(
    f"\nResults file:\n"
    f"{RESULTS_PATH}"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_DIR):

    raise FileNotFoundError(
        f"\nFinal model directory not found:\n{MODEL_DIR}"
    )


if not os.path.exists(RESULTS_PATH):

    raise FileNotFoundError(
        f"\nFinal model comparison file not found:\n"
        f"{RESULTS_PATH}"
    )


# ============================================================
# LOAD FINAL VALIDATION RESULTS
# ============================================================

print(
    "\nLoading final validated model metrics..."
)

results_df = pd.read_csv(
    RESULTS_PATH
)


print(
    "\nFinal validated models:"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# CONNECT TO HOPSWORKS
# ============================================================

print("\n" + "=" * 70)
print("CONNECTING TO HOPSWORKS")
print("=" * 70)

project = hopsworks.login()

print(
    "\nConnected successfully."
)

print(
    f"Project: {project.name}"
)


# ============================================================
# MODEL REGISTRY
# ============================================================

mr = project.get_model_registry()

print(
    "\nModel Registry connected."
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_model_metrics(target):

    rows = results_df[
        (results_df["horizon"] == target)
        &
        (
            results_df["model"]
            !=
            "Persistence Baseline"
        )
    ]

    if rows.empty:

        raise RuntimeError(
            f"No final trained model metrics "
            f"found for {target}."
        )

    row = rows.iloc[0]

    return {
        "model": str(row["model"]),
        "rmse": float(row["RMSE"]),
        "mae": float(row["MAE"]),
        "r2": float(row["R2"])
    }


# ============================================================
# CHECK EXISTING HOPSWORKS MODELS
# ============================================================

print(
    "\nChecking existing model registry versions..."
)


existing_models = {}

for target, config in MODEL_CONFIG.items():

    registry_name = config["registry_name"]

    try:

        existing_model = mr.get_model(
            registry_name,
            version=1
        )

        if existing_model is not None:

            existing_models[registry_name] = (
                existing_model
            )

            print(
                f"  {registry_name}: "
                f"Version {existing_model.version} exists"
            )

        else:

            print(
                f"  {registry_name}: "
                f"not registered"
            )

    except Exception as e:

        print(
            f"  {registry_name}: "
            f"not found"
        )


# ============================================================
# REGISTER EACH MODEL
# ============================================================

registered_models = []


for target, config in MODEL_CONFIG.items():

    print("\n" + "-" * 70)

    print(
        f"REGISTERING: {target}"
    )

    print("-" * 70)


    registry_name = config[
        "registry_name"
    ]

    filename = config[
        "filename"
    ]

    model_description = config[
        "model_description"
    ]


    # --------------------------------------------------------
    # LOCAL MODEL PATH
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )


    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nExpected model file not found:\n"
            f"{model_path}"
        )


    print(
        f"\nLocal model:\n"
        f"{model_path}"
    )


    # --------------------------------------------------------
    # GET VALIDATED METRICS
    # --------------------------------------------------------

    metrics = get_model_metrics(
        target
    )


    print(
        "\nFinal test metrics:"
    )

    print(
        f"  Model: {metrics['model']}"
    )

    print(
        f"  RMSE: {metrics['rmse']:.4f}"
    )

    print(
        f"  MAE : {metrics['mae']:.4f}"
    )

    print(
        f"  R²  : {metrics['r2']:.4f}"
    )


    # --------------------------------------------------------
    # SKIP IF VERSION 1 ALREADY EXISTS
    # --------------------------------------------------------

    if registry_name in existing_models:

        existing_model = existing_models[
            registry_name
        ]

        print(
            "\nModel already exists in Hopsworks."
        )

        print(
            f"  Registry name: "
            f"{existing_model.name}"
        )

        print(
            f"  Existing version: "
            f"{existing_model.version}"
        )

        print(
            "\nSkipping registration."
        )

        registered_models.append(
            existing_model
        )

        continue


    # --------------------------------------------------------
    # CREATE HOPSWORKS MODEL
    # --------------------------------------------------------

    print(
        "\nCreating Hopsworks model..."
    )


    model = mr.python.create_model(

        name=registry_name,

        metrics={
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"]
        },

        description=model_description
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print(
        "\nUploading model to Hopsworks..."
    )

    registered_model = model.save(
        model_path,
        keep_original_files=True
    )


    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print(
        "\nMODEL REGISTERED SUCCESSFULLY"
    )

    print(
        f"  Registry name: "
        f"{registered_model.name}"
    )

    print(
        f"  Version: "
        f"{registered_model.version}"
    )

    print(
        f"  Model path: "
        f"{registered_model.model_path}"
    )

    print(
        f"  RMSE: "
        f"{metrics['rmse']:.4f}"
    )

    print(
        f"  MAE: "
        f"{metrics['mae']:.4f}"
    )

    print(
        f"  R²: "
        f"{metrics['r2']:.4f}"
    )


    registered_models.append(
        registered_model
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("HOPSWORKS MODEL REGISTRATION COMPLETE")
print("=" * 70)


print(
    "\nRegistered production models:"
)


for target, config in MODEL_CONFIG.items():

    registry_name = config[
        "registry_name"
    ]

    try:

        model = mr.get_model(
            registry_name,
            version=1
        )

        if model is not None:

            print(
                f"  {target} → "
                f"{model.name} "
                f"(Version {model.version})"
            )

    except Exception as e:

        print(
            f"  {target} → "
            f"verification failed: {e}"
        )


print(
    "\nHopsworks Model Registry is ready."
)


print(
    "\nModels:"
)

print(
    "  AQI_t+1 → "
    "karachi_aqi_svr_t_1"
)

print(
    "  AQI_t+2 → "
    "karachi_aqi_svr_t_2"
)

print(
    "  AQI_t+3 → "
    "karachi_aqi_catboost_t_3"
)


print(
    "\nNext step:"
)

print(
    "Update the Flask prediction API to load "
    "these registered Hopsworks models."
)

print(
    "\n" + "=" * 70
)