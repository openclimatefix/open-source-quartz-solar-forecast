from datetime import datetime

from pydantic import BaseModel, Field

from quartz_solar_forecast.inverters.enphase import EnphaseInverter, EnphaseSettings
from quartz_solar_forecast.inverters.givenergy import GivEnergyInverter, GivEnergySettings
from quartz_solar_forecast.inverters.mock import MockInverter
from quartz_solar_forecast.inverters.solarman import SolarmanInverter, SolarmanSettings
from quartz_solar_forecast.inverters.solis import SolisInverter, SolisSettings
from quartz_solar_forecast.inverters.victron import VictronInverter, VictronSettings


class PVSite(BaseModel):
    """
    Photovoltaic (PV) site configuration model.
    
    This model represents a solar photovoltaic installation site with all the necessary
    parameters for generating solar power forecasts, including location coordinates,
    system capacity, and panel configuration details.
    """
    
    latitude: float = Field(..., description="the latitude of the site", ge=-90, le=90)
    longitude: float = Field(..., description="the longitude of the site", ge=-180, le=180)
    capacity_kwp: float = Field(..., description="the capacity [kwp] of the site", ge=0)
    tilt: float = Field(
        default=35,
        description=(
            "the tilt of the site [degrees], the panels' angle relative to horizontal ground"
        ),
        ge=0,
        le=90,
    )
    orientation: float = Field(
        default=180,
        description=(
            "the orientation of the site [degrees], the angle between north and "
            "the direction the panels face, measured on the horizontal plane."
        ),
        ge=0,
        le=360,
    )

    def round_latitude_and_longitude(self):
        """Round the latitude and longitude to 2 decimal places

        This is to ensure that the location of the site is not stored exactly.
        """
        self.latitude = round(self.latitude, 2)
        self.longitude = round(self.longitude, 2)


class PVSiteWithInverter(PVSite):
    """
    PV site configuration model with inverter integration support.
    
    This model extends the basic PVSite with optional inverter type specification,
    enabling real-time data integration from supported solar inverter systems for
    more accurate forecasting that can incorporate live generation data.
    """
    
    inverter_type: str = Field(
        default=None,
        description="The type of inverter used (supported: enphase, solis, givenergy, solarman, victron)",
        json_schema_extra=["enphase", "solis", "givenergy", "solarman", "victron", None],
    )

    def get_inverter(self):
        if self.inverter_type == "enphase":
            return EnphaseInverter(EnphaseSettings())
        elif self.inverter_type == "solis":
            return SolisInverter(SolisSettings())
        elif self.inverter_type == "givenergy":
            return GivEnergyInverter(GivEnergySettings())
        elif self.inverter_type == "solarman":
            return SolarmanInverter(SolarmanSettings())
        elif self.inverter_type == "victron":
            return VictronInverter.from_settings(VictronSettings())
        else:
            return MockInverter()


class ForecastRequest(BaseModel):
    """
    Base request model for generating PV solar forecasts.
    
    This model defines the essential parameters required for requesting a 
    photovoltaic solar power generation forecast for a specific site and timestamp.
    """
    
    site: PVSite = Field(
        ..., 
        description="PV site configuration containing location and system details"
    )
    timestamp: datetime | None = Field(
        default=None, 
        description="Optional timestamp for the forecast. If not provided, current time will be used"
    )


class TokenRequest(BaseModel):
    """
    Request model for handling OAuth token exchange with solar inverter systems.
    
    This model is used to process authorization callbacks from inverter OAuth flows,
    containing the redirect URL with authorization codes needed for token exchange.
    """
    
    redirect_url: str = Field(
        ..., 
        description="The redirect URL containing the authorization code from the OAuth flow"
    )
