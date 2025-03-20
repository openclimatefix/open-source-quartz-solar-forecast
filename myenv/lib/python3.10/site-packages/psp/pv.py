import datetime

import pandas as pd
from pvlib import irradiance, location


def get_irradiance(
    *,
    lat: float,
    lon: float,
    timestamps: pd.DatetimeIndex | list[pd.Timestamp] | list[datetime.datetime],
    tilt: float,
    orientation: float,
):
    """Compute the clearsky and irradiance values."""
    # For some reason `pvlib` likes `DatetimeIndex`.
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)

    loc = location.Location(lat, lon)
    # Generate clearsky data using the Ineichen model, which is the default
    # The get_clearsky method returns a dataframe with values for GHI, DNI,
    # and DHI
    clearsky = loc.get_clearsky(timestamps)
    # Get solar azimuth and zenith to pass to the transposition function
    solar_position = loc.get_solarposition(times=timestamps)
    # Use the get_total_irradiance function to transpose the GHI to POA
    irr = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=orientation,
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
    )
    return pd.concat([clearsky, irr], axis=1)
