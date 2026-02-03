"""
LightGBM-based solar power forecaster (v3).

This module implements an improved PV forecast model using LightGBM
with enhanced feature engineering for Issue #30.

Key improvements over existing models:
- LightGBM for faster training and inference
- Enhanced feature engineering (solar position, cyclical time, derived weather)
- Better handling of panel orientation and tilt

Author: Raakshass (GSoC 2026 Contributor)
"""

import logging
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from quartz_solar_forecast.weather import WeatherService

from .feature_engineering import FeatureEngineer, get_feature_names

logger = logging.getLogger(__name__)

# Try to import lightgbm, provide helpful error if not installed
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")


class LightGBMSolarPredictor:
    """
    LightGBM-based solar power predictor with enhanced features.

    This model uses LightGBM with sophisticated feature engineering
    to predict PV power output for 48 hours ahead at 15-minute intervals.

    Attributes:
        model: LightGBM model instance
        feature_engineer: FeatureEngineer instance for transformations
        model_path: Path to the model file
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialize the LightGBM solar predictor.

        Args:
            model_path: Optional path to a pre-trained model file
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is required for this model. Install with: pip install lightgbm"
            )

        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.model_path = model_path or self._default_model_path()
        self._weather_service = None

    @staticmethod
    def _default_model_path() -> str:
        """Get the default model path."""
        import quartz_solar_forecast

        package_dir = os.path.dirname(quartz_solar_forecast.__file__)
        return os.path.join(package_dir, "models", "lgbm_v3.txt")

    @property
    def weather_service(self) -> WeatherService:
        """Lazy initialization of weather service."""
        if self._weather_service is None:
            self._weather_service = WeatherService()
        return self._weather_service

    def load_model(
        self,
        model_file: str | None = None,
        repo_id: str = "openclimatefix/open-source-quartz-solar-forecast",
        file_path: str = "models/v3/lgbm_model.txt",
    ) -> None:
        """
        Load a pre-trained LightGBM model.

        Args:
            model_file: Local path to model file (optional)
            repo_id: HuggingFace repository ID
            file_path: Path to model file in the repository
        """
        if model_file and os.path.exists(model_file):
            logger.info(f"Loading model from local file: {model_file}")
            self.model = lgb.Booster(model_file=model_file)
        else:
            # TODO: Download from HuggingFace when model is trained and uploaded
            logger.warning(
                "Pre-trained model not found. Using untrained model for now. "
                "Model training will be added in a future update."
            )
            # Create a placeholder model for testing
            self._create_placeholder_model()

    def _create_placeholder_model(self) -> None:
        """Create a simple placeholder model for testing."""
        # This will be replaced with actual trained model
        logger.info("Creating placeholder model for testing")
        # We'll train a simple model when predict is called
        self.model = None

    def _get_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
    ) -> pd.DataFrame:
        """
        Fetch weather data from Open-Meteo.

        Args:
            latitude: Site latitude
            longitude: Site longitude
            start_date: Start date for forecast

        Returns:
            DataFrame with weather data
        """
        start = pd.Timestamp(start_date)
        end = start + timedelta(hours=48)

        # Use weather service to get data
        weather_data = self.weather_service.get_weather(
            latitude=latitude,
            longitude=longitude,
            start=start,
            end=end,
        )

        return weather_data

    def _prepare_features(
        self,
        weather_df: pd.DataFrame,
        latitude: float,
        longitude: float,
        kwp: float,
        orientation: float,
        tilt: float,
    ) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Args:
            weather_df: Raw weather data
            latitude: Site latitude
            longitude: Site longitude
            kwp: System capacity in kWp
            orientation: Panel orientation (degrees)
            tilt: Panel tilt (degrees)

        Returns:
            DataFrame with engineered features
        """
        features = self.feature_engineer.transform(
            weather_df=weather_df,
            latitude=latitude,
            longitude=longitude,
            capacity_kwp=kwp,
            tilt=tilt,
            orientation=orientation,
        )

        return features

    def predict_power_output(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        kwp: float,
        orientation: float = 180,
        tilt: float = 30,
    ) -> pd.DataFrame:
        """
        Predict solar power output for the next 48 hours.

        Args:
            latitude: Site latitude
            longitude: Site longitude
            start_date: Start date for prediction (YYYY-MM-DD)
            kwp: System capacity in kWp
            orientation: Panel orientation (degrees, 180=south)
            tilt: Panel tilt angle (degrees from horizontal)

        Returns:
            DataFrame with columns: date, power_kw
        """
        logger.info(
            f"Predicting power output for lat={latitude}, lon={longitude}, "
            f"kwp={kwp}, start={start_date}"
        )

        # 1. Get weather data
        try:
            weather_df = self._get_weather_data(latitude, longitude, start_date)
        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")
            raise

        # 2. Prepare features
        features = self._prepare_features(
            weather_df=weather_df,
            latitude=latitude,
            longitude=longitude,
            kwp=kwp,
            orientation=orientation,
            tilt=tilt,
        )

        # 3. Make predictions
        if self.model is not None:
            # Use trained model
            feature_cols = get_feature_names()
            X = features[feature_cols].values
            predictions = self.model.predict(X)
        else:
            # Fallback: simple physics-based estimation
            predictions = self._physics_based_prediction(features, kwp)

        # 4. Post-process predictions
        predictions = np.clip(predictions, 0, kwp)  # Can't exceed capacity

        # 5. Create output DataFrame
        result = pd.DataFrame(
            {
                "date": features.index,
                "power_kw": predictions,
            }
        )

        return result

    def _physics_based_prediction(
        self,
        features: pd.DataFrame,
        kwp: float,
    ) -> np.ndarray:
        """
        Simple physics-based prediction as fallback.

        This is used when no trained model is available.
        It provides a reasonable baseline based on solar geometry.

        Args:
            features: Engineered features DataFrame
            kwp: System capacity

        Returns:
            Array of power predictions
        """
        # Base power from radiation
        if "dswrf" in features.columns:
            base_power = features["dswrf"] / 1000 * kwp  # Normalize radiation
        else:
            base_power = np.zeros(len(features))

        # Apply modifiers
        modifiers = np.ones(len(features))

        if "elevation_factor" in features.columns:
            modifiers *= features["elevation_factor"].values

        if "panel_incidence_factor" in features.columns:
            modifiers *= features["panel_incidence_factor"].values

        if "clear_sky_factor" in features.columns:
            modifiers *= features["clear_sky_factor"].values

        if "temp_efficiency_factor" in features.columns:
            modifiers *= features["temp_efficiency_factor"].values

        # Combine
        power = base_power * modifiers

        # Zero at night
        if "is_daylight" in features.columns:
            power = power * features["is_daylight"].values

        return power.values

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        params: dict | None = None,
    ) -> None:
        """
        Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: LightGBM hyperparameters (optional)
        """
        default_params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        if params:
            default_params.update(params)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Train
        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )

        logger.info(f"Model trained with {self.model.num_trees()} trees")

    def save_model(self, path: str) -> None:
        """
        Save the trained model to a file.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
