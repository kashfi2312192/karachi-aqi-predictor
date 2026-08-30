import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# PEARLS AQI PREDICTOR
# OPEN-METEO HISTORICAL BACKFILL
# ============================================================

LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"

START_DATE = "2022-08-01"
END_DATE = (
    datetime.now()
    .astimezone()
    - timedelta(days=1)
).strftime("%Y-%m-%d")

OUTPUT_PATH = "data/processed/karachi_daily_aqi_openmeteo.csv"

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def fetch_json(url, params, name):

    print(f"\nFetching {name}...")

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"{name} API error: "
            f"{data.get('reason', 'Unknown error')}"
        )

    print(f"{name} request successful.")

    return data


# ============================================================
# WEATHER DATA
# ============================================================

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,

    "start_date": START_DATE,
    "end_date": END_DATE,

    "daily": ",".join([
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "pressure_msl_mean",
        "wind_speed_10m_mean",
        "wind_speed_10m_max"
    ]),

    "timezone": TIMEZONE,

    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm"
}


weather_data = fetch_json(
    WEATHER_URL,
    weather_params,
    "historical weather"
)


weather_daily = pd.DataFrame(
    weather_data["daily"]
)


weather_daily = weather_daily.rename(
    columns={
        "time": "date",
        "temperature_2m_mean": "temperature_mean",
        "temperature_2m_max": "temperature_max",
        "temperature_2m_min": "temperature_min",
        "relative_humidity_2m_mean": "humidity_mean",
        "precipitation_sum": "precipitation_sum",
        "pressure_msl_mean": "pressure_mean",
        "wind_speed_10m_mean": "wind_speed_mean",
        "wind_speed_10m_max": "wind_speed_max"
    }
)


# ============================================================
# AIR QUALITY DATA
# ============================================================

air_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,

    "start_date": START_DATE,
    "end_date": END_DATE,

    "hourly": ",".join([
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi"
    ]),

    "timezone": TIMEZONE,

    "domains": "cams_global"
}


air_data = fetch_json(
    AIR_URL,
    air_params,
    "historical air quality"
)


air_hourly = pd.DataFrame(
    air_data["hourly"]
)


air_hourly["time"] = pd.to_datetime(
    air_hourly["time"]
)


# ============================================================
# HOURLY → DAILY
# ============================================================

print("\nAggregating hourly air-quality data to daily values...")


air_hourly["date"] = (
    air_hourly["time"]
    .dt.strftime("%Y-%m-%d")
)


air_daily = (
    air_hourly
    .groupby("date", as_index=False)
    .agg({
        "pm2_5": "mean",
        "pm10": "mean",
        "carbon_monoxide": "mean",
        "nitrogen_dioxide": "mean",
        "sulphur_dioxide": "mean",
        "ozone": "mean",

        # Maximum hourly AQI represents
        # the day's consolidated US AQI.
        "us_aqi": "max"
    })
)


# ============================================================
# MERGE WEATHER + AIR QUALITY
# ============================================================

weather_daily["date"] = pd.to_datetime(
    weather_daily["date"]
).dt.strftime("%Y-%m-%d")


air_daily["date"] = pd.to_datetime(
    air_daily["date"]
).dt.strftime("%Y-%m-%d")


df = pd.merge(
    air_daily,
    weather_daily,
    on="date",
    how="inner"
)


# ============================================================
# RENAME AQI
# ============================================================

df = df.rename(
    columns={
        "us_aqi": "AQI"
    }
)


# ============================================================
# COLUMN ORDER
# ============================================================

columns = [
    "date",

    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",

    "temperature_mean",
    "temperature_max",
    "temperature_min",

    "humidity_mean",
    "precipitation_sum",

    "pressure_mean",
    "wind_speed_mean",
    "wind_speed_max",

    "AQI"
]


df = df[
    [
        col
        for col in columns
        if col in df.columns
    ]
]


# ============================================================
# CLEAN
# ============================================================

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)


df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("OPEN-METEO BACKFILL COMPLETE")
print("=" * 70)

print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"Missing rows: "
    f"{df.isna().any(axis=1).sum()}"
)

print("\nMissing values:")
print(
    df.isna().sum()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())