from datetime import datetime
from quartz_solar_forecast.data import get_gfs_nwp
from quartz_solar_forecast.pydantic_models import PVSite


def test_get_gfs_nwp():

    # make input data
    site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
    ts = datetime(2023, 10, 30, 0, 0)
    source = "gfs"

    # get data
    data = get_gfs_nwp(site=site, ts=ts, source=source)    
    
    print(data)



