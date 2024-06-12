import pandas as pd
import xarray as xr
from datetime import datetime

from quartz_solar_forecast.pydantic_models import PVSite

def mock_enphase_data():
    data = pd.DataFrame({
        'timestamp': [
            datetime(2024, 6, 5, 11, 25),
            datetime(2024, 6, 5, 11, 30),
            datetime(2024, 6, 5, 11, 35)
        ],
        'power_kw': [0.5, 0.6, 0.7]
    })
    return data

def test_make_pv_data_with_enphase(mock_enphase_data):
    # Create a PVSite object
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25, tilt=35, orientation=180, inverter_type='enphase')
    ts = datetime(2024, 6, 5, 11, 35)

    # Perform the necessary processing
    recent_pv_data = mock_enphase_data[mock_enphase_data['timestamp'] <= ts]
    power_kw = recent_pv_data['power_kw'].values
    timestamp = recent_pv_data['timestamp'].values

    da = xr.DataArray(
        data=power_kw,
        dims=["pv_id", "timestamp"],
        coords=dict(
            longitude=(["pv_id"], [site.longitude]),
            latitude=(["pv_id"], [site.latitude]),
            timestamp=timestamp,
            pv_id=[1],
            kwp=(["pv_id"], [site.capacity_kwp]),
            tilt=(["pv_id"], [site.tilt]),
            orientation=(["pv_id"], [site.orientation]),
        ),
    )
    pv_data = da.to_dataset(name="generation_kw")

    # Check the output
    assert isinstance(pv_data, xr.Dataset)
    assert pv_data.dims == {'pv_id': 1, 'timestamp': 3}
    assert pv_data['generation_kw'].values.tolist() == [[0.5, 0.6, 0.7]]