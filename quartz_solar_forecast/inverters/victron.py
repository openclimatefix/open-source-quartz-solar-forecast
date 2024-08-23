import pandas as pd
from datetime import datetime, timedelta
from quartz_solar_forecast.inverters.inverter import AbstractInverter
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from vrmapi.vrm import VRM_API

class VictronSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    username: str = Field(alias="VICTRON_USER")
    password: str = Field(alias="VICTRON_PASS")


class VictronInverter(AbstractInverter):

    def __init__(self, settings: VictronSettings):
        self.__settings = settings
        self._api = VRM_API(username=settings.username, password=settings.password)

    def get_data(self, ts: pd.Timestamp) -> pd.DataFrame:
        sites = self._api.get_user_sites(self._api.user_id)
        # get first site (bit of a guess)
        first_site_id = sites["records"][0]["idSite"]
        diag = self._api.get_diag(first_site_id)

        end = datetime.now()
        start = end - timedelta(weeks=1)
        stats = self._api.get_kwh_stats(first_site_id, start=start, end=end)

        kwh = stats["records"]["kwh"]

        df = pd.DataFrame(kwh)

        df[0] = pd.to_datetime(df[0], unit='ms')
        df.columns = ["timestamp", "power_kw"]
        return df
        # return pd.DataFrame(columns=['timestamp', 'power_kw'])

