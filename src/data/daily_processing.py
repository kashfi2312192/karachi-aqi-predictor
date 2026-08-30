import pandas as pd
from pathlib import Path

from aqi import calculate_daily_aqi


INPUT_PATH = (
    "data/backfill/"
    "karachi_2025_2026_raw.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "karachi_daily_aqi.csv"
)


def create_daily_dataset(df):
    """
    Convert hourly observations into
    daily observations.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    df["time"] = pd.to_datetime(
        df["time"]
    )

    # --------------------------------------------------------
    # Create date
    # --------------------------------------------------------

    df["date"] = (
        df["time"]
        .dt.date
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "time"
    )

    # --------------------------------------------------------
    # Daily aggregation
    # --------------------------------------------------------

    daily = (
        df.groupby("date")
        .agg(

            # Air pollution
            pm2_5=(
                "pm2_5",
                "mean"
            ),

            pm10=(
                "pm10",
                "mean"
            ),

            carbon_monoxide=(
                "carbon_monoxide",
                "mean"
            ),

            nitrogen_dioxide=(
                "nitrogen_dioxide",
                "mean"
            ),

            sulphur_dioxide=(
                "sulphur_dioxide",
                "mean"
            ),

            ozone=(
                "ozone",
                "mean"
            ),

            # Weather
            temperature_mean=(
                "temperature_2m",
                "mean"
            ),

            temperature_max=(
                "temperature_2m",
                "max"
            ),

            temperature_min=(
                "temperature_2m",
                "min"
            ),

            humidity_mean=(
                "relative_humidity_2m",
                "mean"
            ),

            precipitation_sum=(
                "precipitation",
                "sum"
            ),

            pressure_mean=(
                "surface_pressure",
                "mean"
            ),

            wind_speed_mean=(
                "wind_speed_10m",
                "mean"
            ),

            wind_speed_max=(
                "wind_speed_10m",
                "max"
            ),

        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Calculate AQI
    # --------------------------------------------------------

    daily["AQI"] = daily.apply(
        calculate_daily_aqi,
        axis=1
    )

    return daily


if __name__ == "__main__":

    print("Loading historical data...")

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        "Hourly rows:",
        len(df)
    )

    daily_df = create_daily_dataset(
        df
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    Path(
        "data/processed"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    daily_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    print()
    print(
        "Daily dataset created successfully."
    )

    print()
    print(
        "Daily rows:",
        len(daily_df)
    )

    print()
    print("Columns:")
    print(
        daily_df.columns.tolist()
    )

    print()
    print("Date range:")

    print(
        daily_df["date"].min(),
        "→",
        daily_df["date"].max()
    )

    print()
    print("Missing values:")

    print(
        daily_df.isnull().sum()
    )

    print()
    print("AQI statistics:")

    print(
        daily_df["AQI"].describe()
    )

    print()
    print(
        "Saved to:",
        OUTPUT_PATH
    )