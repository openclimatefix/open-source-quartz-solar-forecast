"""
Tests for the LightGBM solar predictor (v3) and feature engineering.

Author: Raakshass (GSoC 2026 Contributor)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from quartz_solar_forecast.forecasts.feature_engineering import (
    calculate_solar_position,
    encode_cyclical_time,
    calculate_derived_weather_features,
    calculate_panel_factors,
    FeatureEngineer,
    get_feature_names,
)


class TestSolarPosition:
    """Tests for solar position calculations."""

    def test_solar_position_basic(self):
        """Test that solar position returns expected columns."""
        timestamps = pd.date_range("2024-06-21 00:00", periods=24, freq="h")
        result = calculate_solar_position(51.5, -0.1, timestamps)

        assert "sun_elevation" in result.columns
        assert "sun_azimuth" in result.columns
        assert "is_daylight" in result.columns
        assert len(result) == 24

    def test_solar_position_elevation_range(self):
        """Test that elevation is within valid range."""
        timestamps = pd.date_range("2024-06-21 00:00", periods=24, freq="h")
        result = calculate_solar_position(51.5, -0.1, timestamps)

        assert result["sun_elevation"].min() >= -90
        assert result["sun_elevation"].max() <= 90

    def test_solar_position_azimuth_range(self):
        """Test that azimuth is within valid range."""
        timestamps = pd.date_range("2024-06-21 00:00", periods=24, freq="h")
        result = calculate_solar_position(51.5, -0.1, timestamps)

        assert result["sun_azimuth"].min() >= 0
        assert result["sun_azimuth"].max() <= 360

    def test_is_daylight_at_noon(self):
        """Test that is_daylight is 1 at solar noon."""
        # Summer solstice at noon in London
        timestamps = pd.DatetimeIndex([pd.Timestamp("2024-06-21 12:00")])
        result = calculate_solar_position(51.5, 0, timestamps)

        assert result["is_daylight"].iloc[0] == 1
        assert result["sun_elevation"].iloc[0] > 0

    def test_is_daylight_at_midnight(self):
        """Test that is_daylight is 0 at midnight (in most locations)."""
        timestamps = pd.DatetimeIndex([pd.Timestamp("2024-06-21 00:00")])
        result = calculate_solar_position(51.5, 0, timestamps)

        # At midnight, sun should be below horizon (except polar regions)
        assert result["sun_elevation"].iloc[0] < 20  # Allow some margin for summer


class TestCyclicalTimeEncoding:
    """Tests for cyclical time encoding."""

    def test_cyclical_time_columns(self):
        """Test that cyclical encoding returns expected columns."""
        timestamps = pd.date_range("2024-01-01", periods=48, freq="h")
        result = encode_cyclical_time(timestamps)

        expected_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]
        for col in expected_cols:
            assert col in result.columns

    def test_cyclical_values_range(self):
        """Test that sin/cos values are in [-1, 1]."""
        timestamps = pd.date_range("2024-01-01", periods=365 * 24, freq="h")
        result = encode_cyclical_time(timestamps)

        for col in result.columns:
            assert result[col].min() >= -1
            assert result[col].max() <= 1

    def test_cyclical_continuity(self):
        """Test that encoding is continuous (23:00 close to 00:00)."""
        timestamps = pd.DatetimeIndex([
            pd.Timestamp("2024-01-01 23:00"),
            pd.Timestamp("2024-01-02 00:00"),
        ])
        result = encode_cyclical_time(timestamps)

        # Values should be close (within 0.5 for adjacent hours)
        hour_sin_diff = abs(result["hour_sin"].iloc[0] - result["hour_sin"].iloc[1])
        assert hour_sin_diff < 0.5


class TestDerivedWeatherFeatures:
    """Tests for derived weather features."""

    def test_cloud_cover_weighted(self):
        """Test weighted cloud cover calculation."""
        weather_df = pd.DataFrame({
            "lcc": [50.0],  # Low cloud cover
            "mcc": [30.0],  # Mid cloud cover
            "hcc": [20.0],  # High cloud cover
        }, index=pd.DatetimeIndex([pd.Timestamp.now()]))
        sun_elevation = pd.Series([45.0], index=weather_df.index)

        result = calculate_derived_weather_features(weather_df, sun_elevation)

        # Expected: (0.6*50 + 0.3*30 + 0.1*20) / 100 = 0.41
        assert "cloud_cover_weighted" in result.columns
        assert abs(result["cloud_cover_weighted"].iloc[0] - 0.41) < 0.01

    def test_clear_sky_factor(self):
        """Test clear sky factor is complement of cloud cover."""
        weather_df = pd.DataFrame({
            "lcc": [100.0],
            "mcc": [100.0],
            "hcc": [100.0],
        }, index=pd.DatetimeIndex([pd.Timestamp.now()]))
        sun_elevation = pd.Series([45.0], index=weather_df.index)

        result = calculate_derived_weather_features(weather_df, sun_elevation)

        # Full cloud cover should give clear_sky_factor close to 0
        assert result["clear_sky_factor"].iloc[0] < 0.1


class TestPanelFactors:
    """Tests for panel orientation factors."""

    def test_panel_incidence_factor_range(self):
        """Test that incidence factor is in [0, 1]."""
        sun_azimuth = pd.Series([180.0])
        sun_elevation = pd.Series([45.0])

        result = calculate_panel_factors(sun_azimuth, sun_elevation, 30.0, 180.0)

        assert result["panel_incidence_factor"].iloc[0] >= 0
        assert result["panel_incidence_factor"].iloc[0] <= 1

    def test_optimal_orientation_higher_factor(self):
        """Test that optimal panel orientation has higher factor."""
        sun_azimuth = pd.Series([180.0, 180.0])  # Sun in south
        sun_elevation = pd.Series([45.0, 45.0])

        # Panel facing south vs east
        result_south = calculate_panel_factors(sun_azimuth, sun_elevation, 30.0, 180.0)
        result_east = calculate_panel_factors(sun_azimuth, sun_elevation, 30.0, 90.0)

        # South-facing should have higher factor when sun is in south
        assert result_south["panel_incidence_factor"].iloc[0] > result_east["panel_incidence_factor"].iloc[0]


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_transform_returns_dataframe(self):
        """Test that transform returns a DataFrame."""
        engineer = FeatureEngineer()
        weather_df = pd.DataFrame({
            "t": [20.0],
            "lcc": [30.0],
            "mcc": [20.0],
            "hcc": [10.0],
            "dswrf": [500.0],
            "vis": [20000.0],
        }, index=pd.DatetimeIndex([pd.Timestamp.now()]))

        result = engineer.transform(
            weather_df=weather_df,
            latitude=51.5,
            longitude=-0.1,
            capacity_kwp=4.0,
            tilt=30.0,
            orientation=180.0,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_transform_includes_all_feature_groups(self):
        """Test that transform includes all feature groups."""
        engineer = FeatureEngineer()
        weather_df = pd.DataFrame({
            "t": [20.0],
            "lcc": [30.0],
            "mcc": [20.0],
            "hcc": [10.0],
            "dswrf": [500.0],
            "vis": [20000.0],
        }, index=pd.DatetimeIndex([pd.Timestamp.now()]))

        result = engineer.transform(
            weather_df=weather_df,
            latitude=51.5,
            longitude=-0.1,
            capacity_kwp=4.0,
        )

        # Check for solar position features
        assert "sun_elevation" in result.columns
        assert "sun_azimuth" in result.columns

        # Check for cyclical time features
        assert "hour_sin" in result.columns
        assert "day_cos" in result.columns

        # Check for derived features
        assert "cloud_cover_weighted" in result.columns

        # Check for site info
        assert "capacity_kwp" in result.columns


class TestGetFeatureNames:
    """Tests for get_feature_names function."""

    def test_feature_names_list(self):
        """Test that get_feature_names returns a list."""
        names = get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0

    def test_feature_names_contains_key_features(self):
        """Test that feature names contains key features."""
        names = get_feature_names()

        key_features = ["t", "dswrf", "sun_elevation", "hour_sin", "capacity_kwp"]
        for feature in key_features:
            assert feature in names
