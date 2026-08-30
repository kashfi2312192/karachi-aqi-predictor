import os
import sys
import subprocess
from datetime import datetime


# ============================================================
# PEARLS AQI PREDICTOR
# END-TO-END PREDICTION PIPELINE
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


PYTHON_EXECUTABLE = sys.executable


# ============================================================
# PIPELINE SCRIPTS
# ============================================================

GET_TRAINING_DATA = os.path.join(
    PROJECT_ROOT,
    "src",
    "hopsworks",
    "get_training_data.py"
)


PREDICT = os.path.join(
    PROJECT_ROOT,
    "src",
    "models",
    "predict.py"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("PEARLS AQI PREDICTOR")
print("END-TO-END PREDICTION PIPELINE")
print("=" * 80)

print(
    f"\nPipeline started at: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

print(
    f"\nProject root:\n"
    f"{PROJECT_ROOT}"
)

print(
    f"\nPython executable:\n"
    f"{PYTHON_EXECUTABLE}"
)


# ============================================================
# VALIDATE FILES
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING PIPELINE FILES")
print("=" * 80)


required_files = {
    "Hopsworks data retrieval": GET_TRAINING_DATA,
    "Model inference": PREDICT,
}


for name, path in required_files.items():

    print(
        f"\n{name}:"
    )

    print(
        path
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired pipeline file does not exist:\n"
            f"{path}"
        )

    print(
        "✓ Found"
    )


# ============================================================
# RUN COMMAND
# ============================================================

def run_step(
    step_number,
    total_steps,
    name,
    script
):

    print("\n" + "=" * 80)

    print(
        f"STEP {step_number}/{total_steps}: "
        f"{name}"
    )

    print("=" * 80)

    print(
        f"\nRunning:\n"
        f"{script}"
    )

    print()

    result = subprocess.run(
        [
            PYTHON_EXECUTABLE,
            script
        ],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print(
            "\n" + "=" * 80
        )

        print(
            f"✗ STEP FAILED: {name}"
        )

        print("=" * 80)

        raise RuntimeError(
            f"Pipeline step failed: {name}"
        )

    print(
        "\n" + "=" * 80
    )

    print(
        f"✓ STEP COMPLETED: {name}"
    )

    print("=" * 80)


# ============================================================
# STEP 1
# ============================================================

run_step(
    1,
    2,
    "RETRIEVE LATEST HOPSWORKS DATA",
    GET_TRAINING_DATA
)


# ============================================================
# STEP 2
# ============================================================

run_step(
    2,
    2,
    "GENERATE 3-DAY AQI FORECAST",
    PREDICT
)


# ============================================================
# VERIFY OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING PIPELINE OUTPUT")
print("=" * 80)


prediction_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "latest_predictions.csv"
)


if not os.path.exists(prediction_file):

    raise FileNotFoundError(
        "\nPrediction pipeline completed, "
        "but output file was not created:\n"
        f"{prediction_file}"
    )


print(
    "\n✓ Prediction file created:"
)

print(
    prediction_file
)


# ============================================================
# READ FINAL OUTPUT
# ============================================================

import pandas as pd


forecast_df = pd.read_csv(
    prediction_file
)


if forecast_df.empty:

    raise ValueError(
        "Prediction output file is empty."
    )


required_columns = [
    "forecast_date",
    "forecast_horizon",
    "predicted_aqi",
    "aqi_category",
]


missing_columns = [
    column
    for column in required_columns
    if column not in forecast_df.columns
]


if missing_columns:

    raise ValueError(
        "Prediction output is missing required "
        f"columns: {missing_columns}"
    )


# ============================================================
# FINAL FORECAST
# ============================================================

print("\n" + "=" * 80)
print("FINAL 3-DAY KARACHI AQI FORECAST")
print("=" * 80)


for _, row in forecast_df.iterrows():

    print(
        f"\n{row['forecast_date']} "
        f"| {row['forecast_horizon']} "
        f"| AQI: {float(row['predicted_aqi']):.2f} "
        f"| {row['aqi_category']}"
    )


# ============================================================
# PIPELINE COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    f"\nCompleted at: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

print(
    "\nPipeline:"
)

print(
    "  1. Hopsworks Feature Store"
)

print(
    "        ↓"
)

print(
    "  2. Latest Karachi AQI features"
)

print(
    "        ↓"
)

print(
    "  3. Hopsworks Model Registry"
)

print(
    "        ↓"
)

print(
    "  4. Registered production models"
)

print(
    "        ↓"
)

print(
    "  5. 3-day AQI prediction"
)

print(
    "        ↓"
)

print(
    "  6. latest_predictions.csv"
)

print(
    "        ↓"
)

print(
    "  7. Flask API"
)

print(
    "        ↓"
)

print(
    "  8. Streamlit Dashboard"
)

print("\n" + "=" * 80)
print("READY FOR AUTOMATION / DEPLOYMENT")
print("=" * 80)