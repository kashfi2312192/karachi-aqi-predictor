import requests
import pandas as pd
from pathlib import Path


LATITUDE = 24.8607
LONGITUDE = 67.0011


def fetch_historical_data(start_date, end_date):
    """
    Fetch historical weather and air-quality data
    for Karachi from Open-Meteo.
    """

    # ---------------------------------------------------------
    # WEATHER DATA
    # ---------------------------------------------------------

    weather_url = "https://archive-api.open-meteo.com/v1/archive"

    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "timezone": "Asia/Karachi",
    }

    print(f"Fetching weather data: {start_date} → {end_date}")

    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=60
    )

    weather_response.raise_for_status()

    weather_data = weather_response.json()

    weather_df = pd.DataFrame(
        weather_data["hourly"]
    )

    print("Weather rows:", len(weather_df))

    # ---------------------------------------------------------
    # AIR QUALITY DATA
    # ---------------------------------------------------------

    air_quality_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    air_quality_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
        "timezone": "Asia/Karachi",
    }

    print(
        f"Fetching air-quality data: "
        f"{start_date} → {end_date}"
    )

    air_response = requests.get(
        air_quality_url,
        params=air_quality_params,
        timeout=60
    )

    air_response.raise_for_status()

    air_data = air_response.json()

    air_df = pd.DataFrame(
        air_data["hourly"]
    )

    print(
        "Air-quality rows:",
        len(air_df)
    )

    # ---------------------------------------------------------
    # MERGE WEATHER + AIR QUALITY
    # ---------------------------------------------------------

    df = pd.merge(
        weather_df,
        air_df,
        on="time",
        how="inner"
    )

    return df


if __name__ == "__main__":

    start_date = "2025-01-01"
    end_date = "2026-01-07"

    df = fetch_historical_data(
        start_date,
        end_date
    )

    # ---------------------------------------------------------
    # SAVE DATA
    # ---------------------------------------------------------

    output_directory = Path("data/backfill")

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_directory
        / "karachi_2025_2026_raw.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # INFORMATION
    # ---------------------------------------------------------

    print()
    print("Combined data:")
    print(df.head())

    print()
    print("Shape:", df.shape)

    print()
    print("Columns:")
    print(df.columns.tolist())

    print()
    print("Missing values:")
    print(df.isnull().sum())

    print()
    print("Saved to:")
    print(output_path)