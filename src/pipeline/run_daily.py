
import os
import sys
import subprocess
from datetime import datetime


# ============================================================
# PEARLS AQI PREDICTOR
# DAILY AUTOMATION RUNNER
# ============================================================


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


PYTHON_EXECUTABLE = sys.executable


PIPELINE_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "src",
    "pipeline",
    "run_pipeline.py"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("PEARLS AQI PREDICTOR")
print("DAILY AUTOMATION RUNNER")
print("=" * 80)

print(
    f"\nStarted at: "
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

print(
    f"\nPipeline script:\n"
    f"{PIPELINE_SCRIPT}"
)


# ============================================================
# VALIDATE PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("VALIDATING PIPELINE")
print("=" * 80)


if not os.path.exists(PIPELINE_SCRIPT):

    print(
        "\nERROR: Pipeline script not found."
    )

    print(
        f"Expected path:\n"
        f"{PIPELINE_SCRIPT}"
    )

    sys.exit(1)


print(
    "\n✓ Pipeline script found."
)


# ============================================================
# RUN PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("STARTING DAILY AQI PIPELINE")
print("=" * 80)

print(
    "\nExecuting end-to-end pipeline..."
)

print(
    "\nPipeline:"
)

print(
    "  1. Retrieve latest Hopsworks data"
)

print(
    "  2. Load registered models"
)

print(
    "  3. Generate 3-day AQI forecast"
)

print(
    "  4. Save latest_predictions.csv"
)

print(
    "\n" + "-" * 80
)


try:

    result = subprocess.run(
        [
            PYTHON_EXECUTABLE,
            PIPELINE_SCRIPT
        ],
        cwd=PROJECT_ROOT,
        check=False
    )


except Exception as e:

    print(
        "\nERROR: Failed to start pipeline."
    )

    print(
        f"Error type: {type(e).__name__}"
    )

    print(
        f"Error: {e}"
    )

    sys.exit(1)


# ============================================================
# CHECK RESULT
# ============================================================

print("\n" + "-" * 80)

if result.returncode != 0:

    print(
        "\n❌ DAILY PIPELINE FAILED"
    )

    print(
        f"Pipeline exit code: "
        f"{result.returncode}"
    )

    sys.exit(
        result.returncode
    )


print(
    "\n✓ DAILY PIPELINE COMPLETED SUCCESSFULLY"
)


# ============================================================
# VERIFY OUTPUT
# ============================================================

prediction_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "latest_predictions.csv"
)


print("\n" + "=" * 80)
print("VERIFYING OUTPUT")
print("=" * 80)


if not os.path.exists(prediction_file):

    print(
        "\n❌ Prediction file was not created."
    )

    print(
        f"Expected:\n"
        f"{prediction_file}"
    )

    sys.exit(1)


print(
    "\n✓ Prediction file exists:"
)

print(
    prediction_file
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("DAILY AUTOMATION COMPLETED")
print("=" * 80)

print(
    f"\nCompleted at: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

print(
    "\nLatest forecast is ready for:"
)

print(
    "  ✓ Flask API"
)

print(
    "  ✓ Streamlit Dashboard"
)

print(
    "  ✓ Application integration"
)

print(
    "  ✓ Scheduled automation"
)

print("\n" + "=" * 80)
print("PEARLS AQI PREDICTOR READY")
print("=" * 80)
