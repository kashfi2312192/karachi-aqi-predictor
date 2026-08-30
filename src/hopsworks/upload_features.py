import sys
import re
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
# HOPSWORKS FEATURE STORE UPLOAD
# ============================================================

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "karachi_ml_dataset.csv"
)

FEATURE_GROUP_NAME = "karachi_aqi_features"

# IMPORTANT:
# Version 1 was created using DELTA and is causing the
# Windows Delta/HDFS RPC problem.
#
# We create VERSION 2 using HUDI instead.
FEATURE_GROUP_VERSION = 3


# ============================================================
# HOPSWORKS FEATURE NAME SANITIZATION
# ============================================================

def sanitize_feature_name(name):
    """
    Convert feature names into names accepted by Hopsworks.

    Hopsworks feature names:
      - lowercase only
      - letters, numbers and underscores
      - must start with a letter
      - maximum 63 characters
    """

    name = str(name)

    # Lowercase
    name = name.lower()

    # Replace everything except letters/numbers/underscore
    name = re.sub(
        r"[^a-z0-9_]",
        "_",
        name
    )

    # Collapse multiple underscores
    name = re.sub(
        r"_+",
        "_",
        name
    )

    # Remove leading underscores
    name = name.lstrip("_")

    # Must start with a letter
    if not name:
        name = "feature"

    if not name[0].isalpha():
        name = "feature_" + name

    # Hopsworks maximum feature-name length
    name = name[:63]

    # Remove trailing underscore after truncation
    name = name.rstrip("_")

    return name


def sanitize_feature_names(df):
    """
    Sanitize all dataframe column names for Hopsworks.

    Returns:
        sanitized dataframe
        rename dictionary
    """

    original_columns = list(df.columns)

    new_columns = []
    rename_map = {}

    used_names = set()

    for column in original_columns:

        sanitized = sanitize_feature_name(column)

        # Prevent duplicate names after sanitization.
        base_name = sanitized
        counter = 2

        while sanitized in used_names:

            suffix = f"_{counter}"

            # Keep total length <= 63
            sanitized = (
                base_name[:63 - len(suffix)]
                + suffix
            )

            counter += 1

        used_names.add(sanitized)
        new_columns.append(sanitized)

        if column != sanitized:
            rename_map[column] = sanitized

    df = df.copy()
    df.columns = new_columns

    return df, rename_map


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("HOPSWORKS FEATURE STORE UPLOAD")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print(
    f"\nLoading dataset:\n{INPUT_PATH}"
)

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {INPUT_PATH}"
    )


df = pd.read_csv(
    INPUT_PATH
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print(
    f"\nDataset shape: {df.shape}"
)


# ============================================================
# DATE
# ============================================================

if "date" not in df.columns:

    raise ValueError(
        "Dataset must contain a 'date' column."
    )


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


if df["date"].isna().any():

    raise ValueError(
        "Invalid dates found in the dataset."
    )


df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)


# ============================================================
# DUPLICATE DATE CHECK
# ============================================================

if df["date"].duplicated().any():

    duplicates = (
        df.loc[
            df["date"].duplicated(keep=False),
            "date"
        ]
        .dt.strftime("%Y-%m-%d")
        .unique()
    )

    raise ValueError(
        "Duplicate dates detected.\n"
        f"Duplicate dates: {duplicates}"
    )


# ============================================================
# REPLACE INFINITE VALUES
# ============================================================

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print(
    "\nConverting feature columns to numeric..."
)

for column in df.columns:

    if column != "date":

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# MISSING VALUE CHECK
# ============================================================

# NOTE:
# AQI_t+1 / AQI_t+2 / AQI_t+3 are future targets. The most
# recent 1-3 rows legitimately have no target yet (tomorrow /
# day-after hasn't happened), because those rows are kept for
# live inference. Only feature columns must be fully populated.

target_columns = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3",
]

feature_columns = [
    col for col in df.columns
    if col not in target_columns
]

missing = (
    df[feature_columns]
    .isna()
    .sum()
)

missing = missing[
    missing > 0
]

if len(missing) > 0:

    print(
        "\nMissing values detected in feature columns:"
    )

    print(missing)

    raise ValueError(
        "\nDataset contains missing feature values. "
        "Fix the processed dataset before uploading."
    )

target_missing = (
    df[target_columns]
    .isna()
    .sum()
)

target_missing = target_missing[target_missing > 0]

if len(target_missing) > 0:

    print(
        "\nNote: target columns below have missing values on "
        "the most recent row(s) — expected for inference rows:"
    )

    print(target_missing)


# ============================================================
# SANITIZE FEATURE NAMES
# ============================================================

print(
    "\nFeature names sanitized for Hopsworks."
)


df, rename_map = sanitize_feature_names(
    df
)


if rename_map:

    print(
        f"Renamed {len(rename_map)} feature(s):"
    )

    for old_name, new_name in rename_map.items():

        print(
            f"  {old_name} -> {new_name}"
        )

else:

    print(
        "No feature names required renaming."
    )


print(
    f"\nTotal features: {len(df.columns)}"
)


# ============================================================
# VERIFY DATE COLUMN AFTER SANITIZATION
# ============================================================

if "date" not in df.columns:

    raise ValueError(
        "The 'date' column was not preserved."
    )


# ============================================================
# FINAL HOPSWORKS NAME VALIDATION
# ============================================================

invalid_names = []

for column in df.columns:

    if not re.match(
        r"^[a-z][a-z0-9_]{0,62}$",
        column
    ):

        invalid_names.append(
            column
        )


if invalid_names:

    raise ValueError(
        "Invalid Hopsworks feature names remain:\n"
        + "\n".join(invalid_names)
    )


# ============================================================
# CONNECT TO HOPSWORKS
# ============================================================

print(
    "\nConnecting to Hopsworks..."
)

fs = get_feature_store()


# ============================================================
# CREATE / GET FEATURE GROUP
# ============================================================

print(
    "\nCreating/getting Feature Group..."
)

print(
    f"Feature Group: {FEATURE_GROUP_NAME}"
)

print(
    f"Version: {FEATURE_GROUP_VERSION}"
)

print(
    "Table format: HUDI"
)


# IMPORTANT:
#
# We explicitly use HUDI instead of DELTA.
#
# This avoids the local Windows deltalake/HDFS path that was
# producing:
#
#   Generic HdfsObjectStore error
#   RPC listener disconnected
#
# Version 2 is intentional because version 1 was already
# created with DELTA.


fg = fs.get_or_create_feature_group(

    name=FEATURE_GROUP_NAME,

    version=FEATURE_GROUP_VERSION,

    description=(
        "Daily engineered weather, pollutant and "
        "historical AQI features for Karachi "
        "3-day AQI forecasting."
    ),

    primary_key=[
        "date"
    ],

    event_time="date",

    online_enabled=True,

    time_travel_format="HUDI",

    stream=True
)


# ============================================================
# INSERT
# ============================================================

print(
    "\nUploading features to Hopsworks..."
)

print(
    "Table format: HUDI"
)

print(
    "This may take some time for the first upload."
)


# Make a clean copy before insertion
upload_df = df.copy()


# Hopsworks/Arrow commonly works more cleanly with
# microsecond timestamps than nanosecond timestamps.

upload_df["date"] = pd.to_datetime(
    upload_df["date"]
).astype(
    "datetime64[us]"
)


# ============================================================
# INSERT INTO FEATURE GROUP
# ============================================================

try:

    fg.insert(
        upload_df,
        wait=True
    )

except Exception as error:

    print(
        "\n" + "=" * 70
    )

    print(
        "HOPSWORKS FEATURE UPLOAD FAILED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nError type: {type(error).__name__}"
    )

    print(
        f"\nError:\n{error}"
    )

    print(
        "\nThe Feature Group metadata may already exist."
    )

    print(
        "If this is a new failure, do not keep changing "
        "the dataset or feature names."
    )

    raise


# ============================================================
# SUCCESS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FEATURE GROUP UPLOAD SUCCESSFUL"
)

print(
    "=" * 70
)


print(
    f"\nFeature Group:"
    f" {FEATURE_GROUP_NAME}"
)

print(
    f"Version:"
    f" {FEATURE_GROUP_VERSION}"
)

print(
    "Table format: HUDI"
)

print(
    f"Rows uploaded:"
    f" {len(upload_df)}"
)

print(
    f"Columns:"
    f" {len(upload_df.columns)}"
)


# ============================================================
# FEATURE GROUP URL
# ============================================================

try:

    print(
        "\nFeature Group ID:"
        f" {fg.id}"
    )

except Exception:

    pass


# ============================================================
# VERIFY FEATURE GROUP
# ============================================================

print(
    "\nVerifying Feature Group..."
)


try:

    sample = (
        fg.select_all()
        .limit(5)
        .read()
    )

    print(
        "\nSample rows from Feature Store:"
    )

    print(
        sample
    )

except Exception as error:

    print(
        "\nWarning:"
    )

    print(
        "Feature Group upload appears to have "
        "completed, but sample verification failed."
    )

    print(
        f"Reason: {error}"
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "HOPSWORKS FEATURE UPLOAD COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nNext step:"
)

print(
    "Use this Feature Group from the training pipeline "
    "to retrieve the historical AQI features and targets."
)

