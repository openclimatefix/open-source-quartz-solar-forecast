"""
Forecasts module

This module contains different forecasting models for solar power prediction.
Different models are put in different files like:
- v1.py contains the v1 model
- v2.py contains the v2 model which was developed by Tryolabs
- v3_lightgbm.py contains the v3 LightGBM model with enhanced features
"""

from .v1 import forecast_v1
from .v1_tilt_orientation import forecast_v1_tilt_orientation
from .v2 import TryolabsSolarPowerPredictor
from .v3_lightgbm import LightGBMSolarPredictor
from .feature_engineering import FeatureEngineer

__all__ = [
    "forecast_v1",
    "forecast_v1_tilt_orientation",
    "TryolabsSolarPowerPredictor",
    "LightGBMSolarPredictor",
    "FeatureEngineer",
]

