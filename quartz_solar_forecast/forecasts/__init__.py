"""
Forecasts module

This module contains different forecasting models for solar power prediction.
Different models are put in different files like:
- v1.py contains the v1 model (requires pv-site-prediction, Python <3.12)
- v2.py contains the v2 model which was developed by Tryolabs
"""

from .v2 import TryolabsSolarPowerPredictor

# v1 models depend on pv-site-prediction which requires Python <3.12
# and fsspec<2023.0.0. Import them lazily to avoid breaking on Python 3.12.
try:
    from .v1 import forecast_v1
    from .v1_tilt_orientation import forecast_v1_tilt_orientation
except ImportError:
    forecast_v1 = None
    forecast_v1_tilt_orientation = None

__all__ = ["forecast_v1", "forecast_v1_tilt_orientation", "TryolabsSolarPowerPredictor"]
