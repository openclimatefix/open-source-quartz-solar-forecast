from pydantic import BaseModel, Field


class PVSite(BaseModel):

    latitude: float = Field(..., description="the longitude of the site", ge=-90, le=90)
    longitude: float = Field(..., description="the latitude of the site", ge=-180, le=180)
    capacity_kwp: float = Field(..., description="the capacity [kwp] of the site", ge=0)
