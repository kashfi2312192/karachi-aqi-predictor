import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "data/processed/karachi_daily_aqi_openmeteo.csv"
OUTPUT_PATH = "data/processed/karachi_ml_dataset.csv"


print("=" * 70)
print("PEARLS AQI PREDICTOR")
print("IMPROVED FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading daily AQI data...")

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"Original shape: {df.shape}")
print(
    f"Date range: "
    f"{df['date'].min().date()} → {df['date'].max().date()}"
)


# ============================================================
# RENAME COLUMNS
# ============================================================

rename_map = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2",
    "ozone": "O3",
    "temperature_mean": "Temperature",
    "humidity_mean": "Humidity",
    "precipitation_sum": "Precipitation",
}

df = df.rename(columns=rename_map)


# ============================================================
# REMOVE OLD NEXT-DAY TARGET IF PRESENT
# ============================================================

if "Next_Day_AQI" in df.columns:
    df.drop(columns=["Next_Day_AQI"], inplace=True)


# ============================================================
# DATE FEATURES
# ============================================================

print("\nCreating date features...")

df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["day_of_week"] = df["date"].dt.dayofweek


def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    else:
        return "Autumn"


df["season"] = df["month"].apply(get_season)


# ============================================================
# CYCLICAL DATE FEATURES
# ============================================================

# Month is cyclical:
# December and January are close to each other.

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# Day of year is also cyclical.

df["day_of_year_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["day_of_year_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)


# Day of week is cyclical.

df["weekday_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["weekday_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)


print(
    "Added month, day_of_year, day_of_week, "
    "season and cyclical date features."
)


# ============================================================
# LOG TRANSFORM SKEWED POLLUTANTS
# ============================================================

print("\nApplying log transformations...")

for col in ["PM2.5", "CO"]:

    if col in df.columns:

        # Ensure no negative values
        df[col] = df[col].clip(lower=0)

        df[f"log_{col}"] = np.log1p(df[col])

        df.drop(columns=[col], inplace=True)

        print(f"Log transformed: {col}")


# ============================================================
# IQR CAPPING
# ============================================================

def iqr_cap(dataframe, column):

    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return dataframe[column].clip(
        lower=lower,
        upper=upper
    )


print("\nApplying IQR capping...")

cap_columns = [
    "PM10",
    "SO2",
    "NO2",
    "O3",
    "Temperature",
    "Humidity",
    "Precipitation",
]

for col in cap_columns:

    if col in df.columns:

        df[col] = iqr_cap(df, col)

        print(f"IQR capped: {col}")


# ============================================================
# ONE-HOT ENCODING
# ============================================================

print("\nEncoding categorical date features...")

df = pd.get_dummies(
    df,
    columns=["season"],
    drop_first=True
)


# Convert boolean columns to integers.

for col in df.columns:

    if df[col].dtype == bool:
        df[col] = df[col].astype(int)


# ============================================================
# AQI HISTORICAL FEATURES
# ============================================================

print("\nCreating historical AQI features...")


# ------------------------------------------------------------
# AQI LAGS
# ------------------------------------------------------------

# IMPORTANT:
# Every lag uses only information available BEFORE
# the prediction date.

for lag in range(1, 8):

    df[f"AQI_lag_{lag}"] = df["AQI"].shift(lag)


print("Added AQI lags 1-7.")


# ============================================================
# AQI ROLLING FEATURES
# ============================================================

# Shift first so today's AQI is NOT included.
# This prevents leakage.

shifted_aqi = df["AQI"].shift(1)


for window in [3, 7, 14]:

    df[f"AQI_roll_mean_{window}"] = (
        shifted_aqi
        .rolling(window)
        .mean()
    )

    df[f"AQI_roll_std_{window}"] = (
        shifted_aqi
        .rolling(window)
        .std()
    )

    df[f"AQI_roll_min_{window}"] = (
        shifted_aqi
        .rolling(window)
        .min()
    )

    df[f"AQI_roll_max_{window}"] = (
        shifted_aqi
        .rolling(window)
        .max()
    )


print(
    "Added AQI rolling mean/std/min/max "
    "for 3, 7 and 14 days."
)


# ============================================================
# AQI MOMENTUM / CHANGE FEATURES
# ============================================================

df["AQI_diff_1"] = (
    df["AQI"]
    .diff(1)
    .shift(1)
)

df["AQI_diff_2"] = (
    df["AQI"]
    .diff(2)
    .shift(1)
)

df["AQI_diff_3"] = (
    df["AQI"]
    .diff(3)
    .shift(1)
)


# Percentage changes.

df["AQI_pct_change_1"] = (
    df["AQI"]
    .pct_change(1)
    .shift(1)
)

df["AQI_pct_change_3"] = (
    df["AQI"]
    .pct_change(3)
    .shift(1)
)


print("Added AQI momentum and percentage-change features.")


# ============================================================
# POLLUTANT HISTORICAL FEATURES
# ============================================================

print("\nCreating pollutant lag features...")


pollutant_columns = [
    "PM10",
    "NO2",
    "SO2",
    "O3",
    "log_PM2.5",
    "log_CO",
]


for col in pollutant_columns:

    if col in df.columns:

        # Short-term history
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_2"] = df[col].shift(2)
        df[f"{col}_lag_3"] = df[col].shift(3)

        # 3-day historical average
        df[f"{col}_roll_mean_3"] = (
            df[col]
            .shift(1)
            .rolling(3)
            .mean()
        )


print("Added pollutant lags 1-3 and 3-day averages.")


# ============================================================
# WEATHER HISTORICAL FEATURES
# ============================================================

print("\nCreating weather history features...")


weather_columns = [
    "Temperature",
    "temperature_max",
    "temperature_min",
    "Humidity",
    "Precipitation",
    "pressure_mean",
    "wind_speed_mean",
    "wind_speed_max",
]


for col in weather_columns:

    if col in df.columns:

        df[f"{col}_lag_1"] = df[col].shift(1)

        df[f"{col}_lag_2"] = df[col].shift(2)

        df[f"{col}_roll_mean_3"] = (
            df[col]
            .shift(1)
            .rolling(3)
            .mean()
        )


print("Added weather lag and rolling features.")


# ============================================================
# POLLUTION RELATIONSHIP FEATURES
# ============================================================

print("\nCreating pollutant interaction features...")


if "PM10" in df.columns and "NO2" in df.columns:

    df["PM10_NO2_ratio"] = (
        df["PM10"] /
        (df["NO2"] + 1e-6)
    )


if "PM10" in df.columns and "O3" in df.columns:

    df["PM10_O3_ratio"] = (
        df["PM10"] /
        (df["O3"] + 1e-6)
    )


if "NO2" in df.columns and "O3" in df.columns:

    df["NO2_O3_ratio"] = (
        df["NO2"] /
        (df["O3"] + 1e-6)
    )


# Historical pollution averages

available_pollutants = [
    col for col in [
        "PM10",
        "NO2",
        "SO2",
        "O3"
    ]
    if col in df.columns
]


if available_pollutants:

    df["pollution_mean"] = (
        df[available_pollutants]
        .shift(1)
        .mean(axis=1)
    )

    df["pollution_max"] = (
        df[available_pollutants]
        .shift(1)
        .max(axis=1)
    )


print("Added pollutant relationship features.")


# ============================================================
# FUTURE TARGETS
# ============================================================

print("\nCreating forecast targets...")

df["AQI_t+1"] = df["AQI"].shift(-1)

df["AQI_t+2"] = df["AQI"].shift(-2)

df["AQI_t+3"] = df["AQI"].shift(-3)


# ============================================================
# HANDLE INF VALUES
# ============================================================

print("\nCleaning infinite values...")

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# FEATURE-SIDE FORWARD FILL
# ============================================================

# DO NOT forward-fill future targets.
#
# The final rows naturally have no t+2/t+3 values.
# Those rows must be removed instead.

target_columns = [
    "AQI_t+1",
    "AQI_t+2",
    "AQI_t+3",
]


feature_columns = [
    col for col in df.columns
    if col not in target_columns
]


df[feature_columns] = (
    df[feature_columns]
    .ffill()
)


# ============================================================
# KEEP LATEST ROWS FOR INFERENCE
# ============================================================

# IMPORTANT:
# The final 1-3 rows naturally have missing future targets
# because AQI_t+1 / AQI_t+2 / AQI_t+3 do not exist yet.
#
# These rows MUST be preserved so the inference pipeline
# can use the most recent real AQI data.
#
# Training will later remove rows with missing targets.

print("\nChecking future targets...")

print(
    f"Rows before target filtering: {len(df)}"
)

print(
    f"Rows with complete targets: "
    f"{df[target_columns].notna().all(axis=1).sum()}"
)

print(
    f"Rows reserved for inference: "
    f"{df[target_columns].isna().any(axis=1).sum()}"
)

# ------------------------------------------------------------
# Only remove rows where FEATURE values are missing.
# Do NOT remove rows because future targets are missing.
# ------------------------------------------------------------

feature_columns = [
    col
    for col in df.columns
    if col not in target_columns
]

df = df.dropna(
    subset=feature_columns
)

# ------------------------------------------------------------
# Sort again
# ------------------------------------------------------------

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

# ============================================================
# SORT
# ============================================================

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# ENSURE NUMERIC FEATURES
# ============================================================

for col in df.columns:

    if col != "date":

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# FINAL NUMERIC VALIDATION
# ============================================================

for col in df.columns:

    if col != "date":

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ------------------------------------------------------------
# Check FEATURE columns only.
# Target columns are allowed to contain NaN
# for the latest inference rows.
# ------------------------------------------------------------

feature_missing = (
    df[feature_columns]
    .isna()
    .sum()
)

feature_missing = feature_missing[
    feature_missing > 0
]

if len(feature_missing) > 0:

    print("\nERROR: Missing feature values detected:")

    print(feature_missing)

    raise ValueError(
        "Feature columns contain missing values."
    )

print(
    "\nFeature validation successful."
)

print(
    "Latest rows retained for inference."
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"\nFinal shape: {df.shape}")

print(f"\nTotal features/columns: {len(df.columns)}")

print("\nColumns:")

print(df.columns.tolist())


print("\nMissing values:")

print(
    df.isna().sum()
)


print("\nTarget statistics:")

print(
    df[target_columns].describe()
)


print("\nSaved to:")

print(OUTPUT_PATH)


print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nAll historical AQI/pollution/weather features "
    "use only information available before the prediction date."
)

print(
    "\nFuture targets AQI_t+1, AQI_t+2 and AQI_t+3 "
    "are never forward-filled."
)

print(
    "\nNext step: retrain the model comparison script "
    "using this dataset."
)
