
import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

import hopsworks

warnings.filterwarnings("ignore")


# ============================================================
# PEARLS AQI PREDICTOR
# HOPSWORKS MODEL REGISTRY INFERENCE
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_NAME = "aqi_predictor_future"

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 3

MODEL_NAMES = {
    "aqi_t_1": "karachi_aqi_svr_t_1",
    "aqi_t_2": "karachi_aqi_svr_t_2",
    "aqi_t_3": "karachi_aqi_catboost_t_3",
}

MODEL_VERSIONS = {
    "aqi_t_1": 1,
    "aqi_t_2": 1,
    "aqi_t_3": 1,
}


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


FEATURE_INFO_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "feature_info.json"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


PREDICTION_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "latest_predictions.csv"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("PEARLS AQI PREDICTOR")
print("HOPSWORKS MODEL REGISTRY INFERENCE")
print("=" * 80)

print("\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# LOAD FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("LOADING FEATURE INFORMATION")
print("=" * 80)


if not os.path.exists(FEATURE_INFO_PATH):

    raise FileNotFoundError(
        f"Feature information not found:\n"
        f"{FEATURE_INFO_PATH}"
    )


with open(
    FEATURE_INFO_PATH,
    "r"
) as f:

    feature_info = json.load(f)


feature_names = feature_info.get(
    "feature_names",
    []
)


print("\nFeature information loaded from:")
print(FEATURE_INFO_PATH)

print(
    f"\nFeatures listed in feature_info.json: "
    f"{len(feature_names)}"
)


# ============================================================
# CONNECT TO HOPSWORKS
# ============================================================

print("\n" + "=" * 80)
print("CONNECTING TO HOPSWORKS")
print("=" * 80)

print("\nConnecting to Hopsworks...")


project = hopsworks.login()


print("\nConnected successfully.")
print(f"Project: {project.name}")


# ============================================================
# CONNECT TO FEATURE STORE
# ============================================================

print("\n" + "=" * 80)
print("CONNECTING TO FEATURE STORE")
print("=" * 80)


fs = project.get_feature_store()


print(
    f"\nFeature Store: {fs.name}"
)


# ============================================================
# GET FEATURE GROUP
# ============================================================

print("\n" + "=" * 80)
print("GETTING FEATURE GROUP")
print("=" * 80)


fg = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)


print(
    f"\nFeature Group: {fg.name}"
)

print(
    f"Version: {fg.version}"
)

print(
    f"Feature Group ID: {fg.id}"
)


# ============================================================
# READ FEATURES
# ============================================================

print("\n" + "=" * 80)
print("READING LATEST FEATURES")
print("=" * 80)

print("\nReading data from Hopsworks...")
print("This may take some time...")


df = fg.select_all().read()


print("\nData retrieved successfully.")

print(
    f"Shape: {df.shape}"
)


# ============================================================
# DATE VALIDATION
# ============================================================

if "date" not in df.columns:

    raise ValueError(
        "Feature Group data does not contain "
        "'date' column."
    )


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


df = (
    df
    .dropna(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)


if df.empty:

    raise ValueError(
        "No valid dated observations found "
        "in the Feature Group."
    )


latest_date = df["date"].iloc[-1]


print(
    f"\nLatest available date: "
    f"{latest_date}"
)

print(
    f"Total rows available: "
    f"{len(df)}"
)


# ============================================================
# LOAD REGISTERED MODELS
# ============================================================

print("\n" + "=" * 80)
print("LOADING REGISTERED HOPSWORKS MODELS")
print("=" * 80)


mr = project.get_model_registry()


loaded_models = {}


for target in [
    "aqi_t_1",
    "aqi_t_2",
    "aqi_t_3"
]:

    print("\n" + "-" * 80)
    print(f"LOADING MODEL: {target}")
    print("-" * 80)


    registry_name = MODEL_NAMES[target]

    version = MODEL_VERSIONS[target]


    print(
        f"Registry name: {registry_name}"
    )

    print(
        f"Version: {version}"
    )


    try:

        model = mr.get_model(
            registry_name,
            version=version
        )


        if model is None:

            raise RuntimeError(
                "Model was not found in "
                "the Hopsworks Model Registry."
            )


        print(
            "Model registry object retrieved."
        )


        print(
            "Downloading model files..."
        )


        model_path = model.download()


        print(
            "Model downloaded successfully."
        )

        print(
            f"Downloaded path:\n{model_path}"
        )


        # ----------------------------------------------------
        # LOCATE MODEL FILE
        # ----------------------------------------------------

        model_file = None


        if os.path.isfile(model_path):

            model_file = model_path


        elif os.path.isdir(model_path):

            possible_files = []


            for root, dirs, files in os.walk(
                model_path
            ):

                for filename in files:

                    if filename.endswith(
                        (
                            ".pkl",
                            ".joblib"
                        )
                    ):

                        possible_files.append(
                            os.path.join(
                                root,
                                filename
                            )
                        )


            if possible_files:

                target_matches = [

                    path
                    for path in possible_files
                    if target.lower()
                    in os.path.basename(path).lower()

                ]


                if target_matches:

                    model_file = target_matches[0]

                else:

                    model_file = possible_files[0]


        if model_file is None:

            raise FileNotFoundError(
                "No .pkl or .joblib model file "
                f"found inside:\n{model_path}"
            )


        print(
            f"\nModel file found:\n"
            f"{model_file}"
        )


        loaded_model = joblib.load(
            model_file
        )


        print(
            "\nModel loaded:"
        )

        print(
            f"  {type(loaded_model).__name__}"
        )


        loaded_models[target] = loaded_model


        print(
            "Model ready for inference."
        )


    except Exception as e:

        print(
            "\nERROR: Could not load model."
        )

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )


# ============================================================
# VERIFY ALL MODELS
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING REGISTERED MODELS")
print("=" * 80)


if len(loaded_models) != 3:

    print(
        "\nLoaded models:"
    )

    print(
        list(loaded_models.keys())
    )


    raise RuntimeError(
        "All three registered models are required "
        "for 3-day AQI forecasting."
    )


print(
    "\nAll three registered models "
    "loaded successfully."
)


# ============================================================
# FEATURE NAME NORMALIZATION
# ============================================================

def normalize_feature_name(name):
    """
    Normalize feature names ONLY for matching.

    This handles differences such as:

        AQI              -> aqi
        aqi              -> aqi

        log_PM2.5        -> logpm25
        log_pm2_5        -> logpm25

        AQI_lag_1        -> aqilag1
        aqi_lag_1        -> aqilag1

    The original model feature name is preserved
    after matching.
    """

    return (
        str(name)
        .lower()
        .replace(".", "")
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


# ============================================================
# FIND CORRESPONDING HOPSWORKS COLUMN
# ============================================================

def find_column_case_insensitive(
    dataframe_columns,
    requested_name
):
    """
    Find the Hopsworks Feature Store column
    corresponding to the feature expected by
    the trained model.

    Matching order:

        1. Exact match
        2. Case-insensitive match
        3. Normalized match
    """

    # --------------------------------------------------------
    # 1. Exact match
    # --------------------------------------------------------

    if requested_name in dataframe_columns:

        return requested_name


    # --------------------------------------------------------
    # 2. Case-insensitive match
    # --------------------------------------------------------

    requested_lower = str(
        requested_name
    ).lower()


    for column in dataframe_columns:

        if str(column).lower() == requested_lower:

            return column


    # --------------------------------------------------------
    # 3. Normalized match
    # --------------------------------------------------------

    requested_normalized = (
        normalize_feature_name(
            requested_name
        )
    )


    matches = []


    for column in dataframe_columns:

        column_normalized = (
            normalize_feature_name(
                column
            )
        )


        if column_normalized == requested_normalized:

            matches.append(column)


    # --------------------------------------------------------
    # Unique match
    # --------------------------------------------------------

    if len(matches) == 1:

        return matches[0]


    # --------------------------------------------------------
    # Ambiguous match
    # --------------------------------------------------------

    if len(matches) > 1:

        raise ValueError(
            f"Ambiguous feature mapping for "
            f"'{requested_name}'. "
            f"Possible Hopsworks columns: "
            f"{matches}"
        )


    # --------------------------------------------------------
    # No match
    # --------------------------------------------------------

    return None


# ============================================================
# GET EXACT FEATURES EXPECTED BY MODEL
# ============================================================

def get_model_features(model):
    """
    Determine the exact feature names used during
    model training.

    Supports:

        - sklearn estimators
        - sklearn Pipelines
        - CatBoost models
    """

    # --------------------------------------------------------
    # sklearn estimator / Pipeline
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )


    # --------------------------------------------------------
    # Pipeline steps
    # --------------------------------------------------------

    if hasattr(
        model,
        "named_steps"
    ):

        for step_name, step in model.named_steps.items():

            if hasattr(
                step,
                "feature_names_in_"
            ):

                return list(
                    step.feature_names_in_
                )


    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_"
    ):

        names = model.feature_names_


        if names:

            return list(names)


    return None


# ============================================================
# PREPARE MODEL INPUTS
# ============================================================

print("\n" + "=" * 80)
print("PREPARING MODEL INPUTS")
print("=" * 80)


latest_row = df.iloc[[-1]].copy()


model_inputs = {}


for target in [
    "aqi_t_1",
    "aqi_t_2",
    "aqi_t_3"
]:

    model = loaded_models[target]


    print("\n" + "-" * 80)
    print(
        f"PREPARING INPUT: {target}"
    )
    print("-" * 80)


    # --------------------------------------------------------
    # Get features directly from trained model
    # --------------------------------------------------------

    expected_features = get_model_features(
        model
    )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if expected_features is None:

        print(
            "Could not determine feature names "
            "directly from model."
        )

        print(
            "Using feature_info.json."
        )

        expected_features = feature_names


    print(
        f"\nModel expects "
        f"{len(expected_features)} features."
    )


    # --------------------------------------------------------
    # Match Feature Store columns
    # --------------------------------------------------------

    selected_columns = []

    rename_map = {}

    missing_features = []


    for expected_feature in expected_features:

        actual_column = (
            find_column_case_insensitive(
                df.columns,
                expected_feature
            )
        )


        if actual_column is None:

            missing_features.append(
                expected_feature
            )

        else:

            selected_columns.append(
                actual_column
            )

            rename_map[
                actual_column
            ] = expected_feature


    # --------------------------------------------------------
    # Missing features
    # --------------------------------------------------------

    if missing_features:

        print(
            "\nERROR: Missing model features:"
        )


        for feature in missing_features:

            print(
                f"  - {feature}"
            )


        raise ValueError(
            f"{len(missing_features)} expected "
            "model features are missing from "
            "the Hopsworks Feature Group."
        )


    # --------------------------------------------------------
    # Build input dataframe
    # --------------------------------------------------------

    X_latest = latest_row[
        selected_columns
    ].copy()


    # Rename Feature Store names to the exact
    # names used during model training.

    X_latest = X_latest.rename(
        columns=rename_map
    )


    # --------------------------------------------------------
    # Force exact feature order
    # --------------------------------------------------------

    X_latest = X_latest[
        expected_features
    ]


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for feature in expected_features:

        X_latest[feature] = pd.to_numeric(
            X_latest[feature],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Missing / invalid values
    # --------------------------------------------------------

    if X_latest.isna().any().any():

        bad_features = (
            X_latest.columns[
                X_latest.isna().any()
            ]
            .tolist()
        )


        print(
            "\nERROR: Missing/non-numeric "
            "values detected:"
        )


        for feature in bad_features:

            print(
                f"  - {feature}"
            )


        raise ValueError(
            "Model input contains missing "
            "or non-numeric values."
        )


    # --------------------------------------------------------
    # Convert to float
    # --------------------------------------------------------

    X_latest = X_latest.astype(float)


    print(
        f"\nInput shape: "
        f"{X_latest.shape}"
    )


    print(
        f"Expected features: "
        f"{len(expected_features)}"
    )


    print(
        "Feature order verified."
    )


    print(
        "Feature name compatibility verified."
    )


    print(
        "Input validation successful."
    )


    model_inputs[target] = X_latest


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

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


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING AQI FORECAST")
print("=" * 80)


predictions = {}


for target in [
    "aqi_t_1",
    "aqi_t_2",
    "aqi_t_3"
]:

    print(
        f"\nRunning prediction for "
        f"{target}..."
    )


    model = loaded_models[target]

    X_latest = model_inputs[target]


    try:

        prediction = model.predict(
            X_latest
        )


    except Exception as e:

        print(
            "\nERROR DURING PREDICTION"
        )

        print(
            f"Target: {target}"
        )

        print(
            f"Model: "
            f"{type(model).__name__}"
        )

        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

        raise


    prediction_value = float(
        np.asarray(
            prediction
        )
        .reshape(-1)[0]
    )


    # AQI cannot be negative

    prediction_value = max(
        0.0,
        prediction_value
    )


    predictions[target] = (
        prediction_value
    )


    print(
        f"Prediction: "
        f"{prediction_value:.2f}"
    )


# ============================================================
# CREATE FORECAST TABLE
# ============================================================

forecast_dates = [

    latest_date
    + pd.Timedelta(days=1),

    latest_date
    + pd.Timedelta(days=2),

    latest_date
    + pd.Timedelta(days=3)

]


forecast_df = pd.DataFrame({

    "forecast_date":
        forecast_dates,

    "forecast_horizon": [

        "1_day",
        "2_days",
        "3_days"

    ],

    "predicted_aqi": [

        predictions["aqi_t_1"],
        predictions["aqi_t_2"],
        predictions["aqi_t_3"]

    ],

    "source_date": [

        latest_date,
        latest_date,
        latest_date

    ]

})


# ============================================================
# AQI CATEGORY
# ============================================================

forecast_df[
    "aqi_category"
] = (
    forecast_df[
        "predicted_aqi"
    ]
    .apply(
        get_aqi_category
    )
)


# ============================================================
# DISPLAY FORECAST
# ============================================================

print("\n" + "=" * 80)
print("KARACHI AQI 3-DAY FORECAST")
print("=" * 80)


print(
    f"\nLatest available data: "
    f"{latest_date}"
)


print()


for _, row in forecast_df.iterrows():

    print(
        f"{row['forecast_date'].date()} | "
        f"AQI: {row['predicted_aqi']:.2f} | "
        f"{row['aqi_category']}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

forecast_df.to_csv(
    PREDICTION_OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 80)
print("PREDICTION SAVED")
print("=" * 80)


print(
    f"\nSaved to:\n"
    f"{PREDICTION_OUTPUT_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)


print("\nModels used:")


print(
    f"  Day 1 → "
    f"{MODEL_NAMES['aqi_t_1']} "
    f"v{MODEL_VERSIONS['aqi_t_1']}"
)


print(
    f"  Day 2 → "
    f"{MODEL_NAMES['aqi_t_2']} "
    f"v{MODEL_VERSIONS['aqi_t_2']}"
)


print(
    f"  Day 3 → "
    f"{MODEL_NAMES['aqi_t_3']} "
    f"v{MODEL_VERSIONS['aqi_t_3']}"
)


print("\nForecast:")


for _, row in forecast_df.iterrows():

    print(
        f"  {row['forecast_date'].date()}: "
        f"AQI {row['predicted_aqi']:.2f} "
        f"({row['aqi_category']})"
    )


print("\n" + "=" * 80)
print("READY FOR APPLICATION / API INTEGRATION")
print("=" * 80)
