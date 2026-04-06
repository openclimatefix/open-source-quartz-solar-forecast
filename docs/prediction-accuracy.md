# Prediction Accuracy & Expected Use Cases

This document describes what accuracy to expect from Quartz Solar Forecast and when the model is (and isn't) a good fit.

## How accuracy is measured

The evaluation module (`quartz_solar_forecast/eval/metrics.py`) calculates:

- **MAE** (Mean Absolute Error) — average absolute difference between forecast and actual generation in kW
- **Normalized MAE** — MAE divided by site capacity, expressed as a percentage

Accuracy is broken down by forecast horizon (hours ahead), since shorter-term forecasts are generally more accurate.

## What to expect

### Forecast horizon

| Horizon | Typical use case | Expected accuracy |
|---|---|---|
| 0-4 hours | Real-time operations, battery charging | Best — model has recent weather data |
| 5-8 hours | Same-day planning | Good |
| 9-24 hours | Next-day planning, grid scheduling | Moderate |
| 24-48 hours | Two-day ahead planning | Lower — weather uncertainty increases |

Shorter horizons benefit from more recent NWP (numerical weather prediction) data. As the horizon increases, forecast accuracy decreases because weather predictions themselves become less certain.

### NWP source

The model supports two NWP sources:

- **ICON** — German Weather Service global model, generally better for European sites
- **GFS** — US Global Forecast System, wider global coverage

Choose the NWP source closest to your site's region for best results.

## When the model works well

- **Unshaded rooftop solar** in open locations
- **UK and European sites** — the model was primarily developed and validated for these regions
- **Standard residential/commercial PV** systems (1-100 kWp)
- **Daily energy yield estimates** when averaged over multiple days

## Known limitations

- **Shading and complex terrain** — the model does not account for local shading from trees, buildings, or mountains. Sites with significant shading will see over-prediction.
- **Coastal and microclimate effects** — localised fog, sea breezes, or valley inversions are not captured by NWP data at typical resolution.
- **Regions outside Europe** — the model has less validation data for non-European sites. Forecasts may be less accurate.
- **Snow cover** — snow on panels is not modelled. Winter forecasts in snowy regions may over-predict.
- **Panel degradation and soiling** — the model assumes panels perform at rated capacity.
- **Very short-term (minutes)** — the model produces hourly forecasts, not sub-hourly. For minute-level nowcasting, consider cloud-camera based approaches.

## Evaluating on your own data

You can compare the forecast against your actual PV output using the built-in evaluation tools:

```python
from quartz_solar_forecast.eval.metrics import metrics

# results_df: DataFrame with columns [timestamp, pv_id, horizon_hour, forecast_power, generation_power]
# pv_metadata: DataFrame with columns [pv_id, capacity]
metrics(results_df, pv_metadata, include_night=False)
```

This prints MAE and normalized MAE for each horizon group. Setting `include_night=False` (default) excludes nighttime hours from the calculation.

## Tips for better accuracy

1. **Use accurate site metadata** — correct latitude, longitude, and capacity (kWp) are essential
2. **Choose the right NWP source** — ICON for Europe, GFS for other regions
3. **Average over multiple days** — single-day forecasts can be off; weekly averages are more reliable
4. **Compare against a baseline** — a simple "clear sky" model or "yesterday's generation" baseline helps put accuracy in context
