import requests
import pandas as pd
from pathlib import Path

LATITUDE = 24.8607
LONGITUDE = 67.0011


def fetch_weather_data():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "timezone": "Asia/Karachi",
        "forecast_days": 3,
    }

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    return df





def fetch_air_quality_data():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
        "timezone": "Asia/Karachi",
        "forecast_days": 3,
    }

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    return df


if __name__ == "__main__":
    weather_df = fetch_weather_data()
    air_quality_df = fetch_air_quality_data()

    df = pd.merge(
        weather_df,
        air_quality_df,
        on="time",
        how="inner"
    )

    output_path = Path("data/raw/karachi_raw_data.csv")

    df.to_csv(output_path, index=False)

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
    print("Data types:")
    print(df.dtypes)

    print()
    print("Saved to:", output_path)