import datetime
import os
import pandas as pd

from quartz_solar_forecast.weather import WeatherService

from xgboost.sklearn import XGBRegressor

class TryolabsSolarPowerPredictor:
    """
    A class to predict solar power output based on weather data, location, panel orientation,
        and other factors.

    Attributes
    ----------
    model_path : str
        Path to the trained model joblib file.

    Methods
    -------
    predict_power_output(latitude: float, longitude: float, start_date: str, kwp: float,
        orientation: float, tilt: float) -> pd.DataFrame:

        Predicts solar power output for the given parameters.
    """
    DATE_COLUMN = "date"

    def __init__(self, model: XGBRegressor) -> None:
        """
        Parameters
        ----------
        model : XGBRegressor
            Trained XGBoost model for predicting solar power output.
        """
        self.model = model

    def get_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        kwp: float,
        orientation: float,
        tilt: float,
    ) -> pd.DataFrame:
        """
        Fetches weather data for the given location and date range, and prepares it for prediction.

        Parameters
        ----------
        latitude : float
            Latitude of the location.
        longitude : float
            Longitude of the location.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        kwp : float
            Kilowatt peak of the solar panel system.
        orientation : float
            Orientation angle of the solar panel system in degrees.
        tilt : float
            Tilt angle of the solar panel system in degrees.

        Returns
        -------
        pd.DataFrame
            Prepared weather data with additional solar panel parameters.
        """
        start_date_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date_datetime = start_date_datetime + datetime.timedelta(days=2)
        end_date = end_date_datetime.strftime("%Y-%m-%d")

        weather_service = WeatherService()

        # Check if the start date is more than 3 months ago
        three_months_ago = datetime.datetime.today() - datetime.timedelta(days=3 * 30)

        if start_date_datetime < three_months_ago:
            print(
                f"Start date ({start_date}) is more than 3 months ago, no",
                "forecast data available.",
            )
        else:
            weather_data = weather_service.get_minutely_weather(
                latitude, longitude, start_date, end_date
            )

        PANEL_COLUMNS = [
            "latitude_rounded",
            "longitude_rounded",
            "orientation",
            "tilt",
            "kwp",
        ]

        weather_data["latitude_rounded"] = latitude
        weather_data["longitude_rounded"] = longitude
        weather_data["orientation"] = orientation
        weather_data["tilt"] = tilt
        weather_data["kwp"] = kwp

        cols = PANEL_COLUMNS + [
            col for col in weather_data.columns if col not in PANEL_COLUMNS
        ]
        weather_data = weather_data[cols]

        return weather_data

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and transforms the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the weather and solar panel data.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame ready for prediction.
        """
        df.loc[:, self.DATE_COLUMN] = pd.to_datetime(df[self.DATE_COLUMN])
        df["year"] = df[self.DATE_COLUMN].dt.year
        df["month"] = df[self.DATE_COLUMN].dt.month
        df["day"] = df[self.DATE_COLUMN].dt.day
        df["hour"] = df[self.DATE_COLUMN].dt.hour
        df["minute"] = df[self.DATE_COLUMN].dt.minute

        COLUMNS_TO_DROP = ["minute", "day", "year"]

        df = df.drop(columns=COLUMNS_TO_DROP)

        return df

    def predict_power_output(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        kwp: float,
        orientation: float,
        tilt: float,
    ) -> pd.DataFrame:
        """
        Predicts solar power output for the specified parameters.

        Parameters
        ----------
        latitude : float
            Latitude of the location.
        longitude : float
            Longitude of the location.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        kwp : float
            Kilowatt peak of the solar panel system.
        orientation : float
            Orientation angle of the solar panel system in degrees.
        tilt : float
            Tilt angle of the solar panel system in degrees.

        Returns
        -------
        pd.DataFrame
            DataFrame containing timestamps and predicted power output in kW for every 15 minutes.
        """
        data = self.get_data(latitude, longitude, start_date, kwp, orientation, tilt)
        cleaned_data = self.clean(data)
        predictions = self.model.predict(cleaned_data.drop(columns=[self.DATE_COLUMN]))
        predictions_df = pd.DataFrame(predictions, columns=["prediction"])
        final_data = cleaned_data.join(predictions_df)
        df = final_data[[self.DATE_COLUMN, "prediction"]]
        df = df.rename(columns={"prediction": "power_kw"})
        return df
