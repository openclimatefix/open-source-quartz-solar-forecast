"""
Feature engineering module for solar power forecasting.

This module provides enhanced features for PV power prediction including:
- Solar position calculations (azimuth, elevation)
- Cyclical time encoding
- Derived weather features
- Interaction features

Author: Raakshass (GSoC 2026 Contributor)
"""

import numpy as np
import pandas as pd


def calculate_solar_position(
    latitude: float,
    longitude: float,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Calculate solar position (azimuth and elevation) for given location and times.

    Uses simplified astronomical calculations suitable for ML features.
    For production, consider using pvlib for more accurate calculations.

    Args:
        latitude: Site latitude in degrees
        longitude: Site longitude in degrees
        timestamps: DatetimeIndex of forecast times

    Returns:
        DataFrame with columns: sun_elevation, sun_azimuth, is_daylight
    """
    # Convert to numpy arrays for vectorized operations
    days_of_year = timestamps.dayofyear.values
    hours = timestamps.hour.values + timestamps.minute.values / 60.0

    # Solar declination (simplified)
    declination = 23.45 * np.sin(np.radians((360 / 365) * (days_of_year - 81)))

    # Hour angle (degrees from solar noon)
    # Approximate solar noon based on longitude
    solar_noon = 12 - longitude / 15
    hour_angle = 15 * (hours - solar_noon)

    # Convert to radians
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    # Solar elevation angle
    sin_elevation = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(
        ha_rad
    )
    elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))

    # Solar azimuth angle
    cos_azimuth = (np.sin(dec_rad) - np.sin(lat_rad) * sin_elevation) / (
        np.cos(lat_rad) * np.cos(np.radians(elevation)) + 1e-10
    )
    azimuth = np.degrees(np.arccos(np.clip(cos_azimuth, -1, 1)))

    # Adjust azimuth for afternoon (hour angle > 0)
    azimuth = np.where(hour_angle > 0, 360 - azimuth, azimuth)

    # Is daylight (elevation > 0)
    is_daylight = elevation > 0

    return pd.DataFrame(
        {
            "sun_elevation": elevation,
            "sun_azimuth": azimuth,
            "is_daylight": is_daylight.astype(int),
        },
        index=timestamps,
    )


def encode_cyclical_time(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Encode time features cyclically using sin/cos transformations.

    This preserves the circular nature of time (23:00 is close to 00:00).

    Args:
        timestamps: DatetimeIndex of forecast times

    Returns:
        DataFrame with cyclical time features
    """
    hours = timestamps.hour + timestamps.minute / 60.0
    days = timestamps.dayofyear
    months = timestamps.month

    return pd.DataFrame(
        {
            # Hour of day (period = 24)
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
            # Day of year (period = 365)
            "day_sin": np.sin(2 * np.pi * days / 365),
            "day_cos": np.cos(2 * np.pi * days / 365),
            # Month (period = 12)
            "month_sin": np.sin(2 * np.pi * months / 12),
            "month_cos": np.cos(2 * np.pi * months / 12),
        },
        index=timestamps,
    )


def calculate_derived_weather_features(
    weather_df: pd.DataFrame,
    sun_elevation: pd.Series,
) -> pd.DataFrame:
    """
    Calculate derived weather features for improved predictions.

    Args:
        weather_df: DataFrame with raw weather features
        sun_elevation: Series with sun elevation angles

    Returns:
        DataFrame with derived features
    """
    features = pd.DataFrame(index=weather_df.index)

    # Weighted cloud cover (low clouds have more impact)
    if all(col in weather_df.columns for col in ["lcc", "mcc", "hcc"]):
        features["cloud_cover_weighted"] = (
            0.6 * weather_df["lcc"] + 0.3 * weather_df["mcc"] + 0.1 * weather_df["hcc"]
        ) / 100  # Normalize to 0-1

    # Clear sky factor
    if "cloud_cover_weighted" in features.columns:
        features["clear_sky_factor"] = 1 - features["cloud_cover_weighted"]

    # Effective radiation (accounting for clouds)
    if "dswrf" in weather_df.columns and "clear_sky_factor" in features.columns:
        features["effective_radiation"] = weather_df["dswrf"] * features["clear_sky_factor"]

    # Temperature effect on panel efficiency
    # PV panels lose ~0.4% efficiency per degree above 25C
    if "t" in weather_df.columns:
        temp_celsius = weather_df["t"]
        features["temp_efficiency_factor"] = np.clip(1 - 0.004 * (temp_celsius - 25), 0.7, 1.0)

    # Visibility factor (normalized)
    if "vis" in weather_df.columns:
        features["visibility_factor"] = np.clip(weather_df["vis"] / 24000, 0, 1)

    # Sun elevation factor (more power at higher elevation)
    features["elevation_factor"] = np.clip(np.sin(np.radians(sun_elevation)), 0, 1)

    return features


def calculate_panel_factors(
    sun_azimuth: pd.Series,
    sun_elevation: pd.Series,
    panel_tilt: float,
    panel_orientation: float,
) -> pd.DataFrame:
    """
    Calculate how well the panel orientation matches the sun position.

    Args:
        sun_azimuth: Solar azimuth angles
        sun_elevation: Solar elevation angles
        panel_tilt: Panel tilt angle (degrees from horizontal)
        panel_orientation: Panel orientation/azimuth (degrees, 180=south)

    Returns:
        DataFrame with panel orientation factors
    """
    # Angle of incidence calculation (simplified)
    # Optimal when sun is perpendicular to panel surface

    # Convert to radians
    tilt_rad = np.radians(panel_tilt)
    orient_rad = np.radians(panel_orientation)
    elev_rad = np.radians(sun_elevation)
    azim_rad = np.radians(sun_azimuth)

    # Angle of incidence on tilted surface
    cos_incidence = np.sin(elev_rad) * np.cos(tilt_rad) + np.cos(elev_rad) * np.sin(
        tilt_rad
    ) * np.cos(azim_rad - orient_rad)

    # Incidence factor (0 to 1, higher is better)
    incidence_factor = np.clip(cos_incidence, 0, 1)

    return pd.DataFrame(
        {
            "panel_incidence_factor": incidence_factor,
        },
        index=sun_azimuth.index,
    )


class FeatureEngineer:
    """
    Feature engineering class for solar power forecasting.

    Combines all feature transformations into a single pipeline.
    """

    def __init__(self):
        """Initialize the feature engineer."""
        pass

    def transform(
        self,
        weather_df: pd.DataFrame,
        latitude: float,
        longitude: float,
        capacity_kwp: float,
        tilt: float = 30.0,
        orientation: float = 180.0,
    ) -> pd.DataFrame:
        """
        Transform raw weather data into ML-ready features.

        Args:
            weather_df: DataFrame with raw weather features (indexed by timestamp)
            latitude: Site latitude
            longitude: Site longitude
            capacity_kwp: PV system capacity in kWp
            tilt: Panel tilt angle (degrees)
            orientation: Panel orientation (degrees, 180=south)

        Returns:
            DataFrame with all engineered features
        """
        timestamps = pd.DatetimeIndex(weather_df.index)

        # 1. Solar position features
        solar_pos = calculate_solar_position(latitude, longitude, timestamps)

        # 2. Cyclical time features
        time_features = encode_cyclical_time(timestamps)

        # 3. Derived weather features
        weather_features = calculate_derived_weather_features(
            weather_df, solar_pos["sun_elevation"]
        )

        # 4. Panel orientation factors
        panel_features = calculate_panel_factors(
            solar_pos["sun_azimuth"],
            solar_pos["sun_elevation"],
            tilt,
            orientation,
        )

        # 5. Combine all features
        all_features = pd.concat(
            [
                weather_df,
                solar_pos,
                time_features,
                weather_features,
                panel_features,
            ],
            axis=1,
        )

        # 6. Add capacity as feature
        all_features["capacity_kwp"] = capacity_kwp
        all_features["tilt"] = tilt
        all_features["orientation"] = orientation

        return all_features


def get_feature_names() -> list:
    """
    Get list of all feature names in order.

    Returns:
        List of feature column names
    """
    return [
        # Raw weather
        "t",
        "prate",
        "lcc",
        "mcc",
        "hcc",
        "si10",
        "dswrf",
        "dlwrf",
        "vis",
        # Solar position
        "sun_elevation",
        "sun_azimuth",
        "is_daylight",
        # Cyclical time
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        # Derived weather
        "cloud_cover_weighted",
        "clear_sky_factor",
        "effective_radiation",
        "temp_efficiency_factor",
        "visibility_factor",
        "elevation_factor",
        # Panel factors
        "panel_incidence_factor",
        # Site info
        "capacity_kwp",
        "tilt",
        "orientation",
    ]
