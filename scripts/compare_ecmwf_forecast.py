"""Compare ECMWF vs ICON (default) NWP forecast sources."""

from datetime import datetime

import matplotlib.pyplot as plt

from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite

site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)
ts = datetime.today()

print("Running ICON forecast...")
pred_icon = run_forecast(site=site, model="gb", ts=ts, nwp_source="icon")

print("Running ECMWF forecast...")
pred_ecmwf = run_forecast(site=site, model="gb", ts=ts, nwp_source="ecmwf")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(pred_icon.index, pred_icon["power_kw"], label="ICON (default)", linewidth=2)
ax.plot(
    pred_ecmwf.index, pred_ecmwf["power_kw"], label="ECMWF IFS 0.25°", linewidth=2, linestyle="--"
)
ax.set_xlabel("Time")
ax.set_ylabel("Power (kW)")
ax.set_title(
    f"Solar Forecast Comparison — {site.latitude}°N, {site.longitude}°E, {site.capacity_kwp} kWp"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("ecmwf_vs_icon_comparison.png", dpi=150)
print("Saved ecmwf_vs_icon_comparison.png")
plt.show()
