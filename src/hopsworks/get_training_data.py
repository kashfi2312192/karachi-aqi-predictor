import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from src.hopsworks.connect import get_feature_store


# ============================================================
# PEARLS AQI PREDICTOR
# HOPSWORKS TRAINING DATA RETRIEVAL
# ============================================================

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 3

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hopsworks_training_data.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("HOPSWORKS TRAINING DATA RETRIEVAL")
print("=" * 70)


# ============================================================
# CONNECT TO HOPSWORKS
# ============================================================

print("\nConnecting to Hopsworks...")

fs = get_feature_store()

print("\nConnected successfully.")
print(
    f"Feature Store: {fs.name}"
)


# ============================================================
# GET FEATURE GROUP
# ============================================================

print("\nGetting Feature Group...")

fg = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

print(
    f"Feature Group: {FEATURE_GROUP_NAME}"
)

print(
    f"Version: {FEATURE_GROUP_VERSION}"
)

print(
    f"Feature Group ID: {fg.id}"
)


# ============================================================
# READ DATA
# ============================================================

print("\nReading data from Hopsworks...")

try:

    df = fg.select_all().read()

except Exception as error:

    print("\nERROR: Could not read Feature Group.")
    print(f"Reason: {error}")
    raise


# ============================================================
# VALIDATION
# ============================================================

print("\nData retrieved successfully.")

print(
    f"Shape: {df.shape}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# DATE
# ============================================================

if "date" in df.columns:

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print("\nFeature columns:")

for index, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"  {index:03d}. {column}"
    )


# ============================================================
# CHECK DUPLICATES
# ============================================================

if "date" in df.columns:

    duplicate_dates = (
        df["date"]
        .duplicated()
        .sum()
    )

    print(
        f"\nDuplicate dates: "
        f"{duplicate_dates}"
    )

    if duplicate_dates > 0:

        raise ValueError(
            "Duplicate dates found in "
            "Hopsworks Feature Group."
        )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing = (
    df.isna()
    .sum()
)

missing = missing[
    missing > 0
]


if len(missing) > 0:

    print("\nMissing values detected:")

    print(
        missing
    )

else:

    print(
        "\nMissing values: 0"
    )


# ============================================================
# CHECK TARGETS
# ============================================================

TARGETS = [
    "aqi_t_1",
    "aqi_t_2",
    "aqi_t_3"
]


print("\nChecking forecast targets...")

for target in TARGETS:

    if target in df.columns:

        print(
            f"  {target}: FOUND"
        )

    else:

        raise ValueError(
            f"Required target '{target}' "
            "was not found in Feature Group."
        )


# ============================================================
# SAVE LOCAL COPY
# ============================================================

print(
    "\nSaving retrieved dataset locally..."
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"Saved to:\n{OUTPUT_PATH}"
)


# ============================================================
# SAMPLE
# ============================================================

print("\nFirst 5 rows:")

print(
    df.head()
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HOPSWORKS DATA RETRIEVAL COMPLETE")
print("=" * 70)

print(
    f"\nRows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"Feature Group: "
    f"{FEATURE_GROUP_NAME}"
)

print(
    f"Version: "
    f"{FEATURE_GROUP_VERSION}"
)

print(
    "\nForecast targets:"
)

for target in TARGETS:

    print(
        f"  - {target}"
    )

print(
    "\nTraining dataset is ready."
)

print("=" * 70)
