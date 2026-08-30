import pandas as pd
import numpy as np


# ============================================================
# US EPA AQI BREAKPOINTS
# ============================================================

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]


def calculate_sub_index(
    concentration,
    breakpoints
):
    """
    Convert pollutant concentration
    into an AQI sub-index.
    """

    if pd.isna(concentration):
        return np.nan

    for (
        concentration_low,
        concentration_high,
        aqi_low,
        aqi_high
    ) in breakpoints:

        if (
            concentration_low
            <= concentration
            <= concentration_high
        ):

            aqi = (
                (
                    aqi_high - aqi_low
                )
                /
                (
                    concentration_high
                    - concentration_low
                )
            ) * (
                concentration
                - concentration_low
            ) + aqi_low

            return round(aqi)

    return np.nan


def calculate_pm25_aqi(pm25):
    """
    Calculate AQI sub-index for PM2.5.
    """

    return calculate_sub_index(
        pm25,
        PM25_BREAKPOINTS
    )


def calculate_pm10_aqi(pm10):
    """
    Calculate AQI sub-index for PM10.
    """

    return calculate_sub_index(
        pm10,
        PM10_BREAKPOINTS
    )


def calculate_daily_aqi(row):
    """
    Calculate daily AQI from daily PM2.5
    and PM10 concentrations.

    The highest valid pollutant sub-index
    becomes the daily AQI.
    """

    pm25_aqi = calculate_pm25_aqi(
        row["pm2_5"]
    )

    pm10_aqi = calculate_pm10_aqi(
        row["pm10"]
    )

    values = [
        pm25_aqi,
        pm10_aqi
    ]

    values = [
        value
        for value in values
        if not pd.isna(value)
    ]

    if not values:
        return np.nan

    return max(values)