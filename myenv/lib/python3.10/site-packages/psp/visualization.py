"""Visualization utils mostly used in notebooks."""

import datetime as dt
from typing import Literal

import altair as alt
import numpy as np
import pandas as pd
import shap
from IPython.display import display

from psp.data_sources.nwp import NwpDataSource
from psp.data_sources.pv import PvDataSource
from psp.gis import approx_add_meters_to_lat_lon
from psp.metrics import Metric, mean_absolute_error
from psp.models.base import PvSiteModel
from psp.pv import get_irradiance
from psp.training import get_y_from_x
from psp.typings import Horizons, Timestamp, X, Y
from psp.utils.maths import safe_div


def _make_feature_chart(
    name: str,
    feature_obj: np.ndarray,
    horizon_idx: int,
    num_horizons: int,
) -> alt.Chart:
    """Make a chart with all the features.

    We try to guess which chart is best depending on the shape of the data.
    """
    shape = feature_obj.shape
    ndim = len(shape)

    vline = (
        alt.Chart(pd.DataFrame(dict(horizon=[horizon_idx])))
        .mark_rule(color="red")
        .encode(x=alt.X("horizon:Q", title="horizon"))
    )

    chart: alt.Chart | None = None

    # Right now all the features are 1D arrays with one value per horizon.
    assert ndim == 1
    assert shape[0] == num_horizons
    chart = (
        alt.Chart(pd.DataFrame({name: feature_obj, "horizon": range(num_horizons)}))
        .mark_circle()
        .encode(x="horizon", y=name)
        .properties(height=75, width=700)
        + vline
    )

    chart = chart.properties(title=name)

    return chart


def time_rule(timestamp: Timestamp, text: str, align: Literal["left", "right"]) -> alt.Chart:
    """Chart of a vertical rule at a timestamp"""
    data = pd.DataFrame(dict(text=[text], timestamp=[timestamp]))
    rule = alt.Chart(data).mark_rule(color="red").encode(x="timestamp")
    text = (
        alt.Chart(data)
        .mark_text(align=align, color="red", dx=5 if align == "left" else -5)
        .encode(text="text", x="timestamp", y=alt.value(0))
    )
    return rule + text


def _make_pv_timeseries_chart(
    x: X,
    all_y: dict[str, Y],
    pred_ts: Timestamp,
    horizons: Horizons,
    horizon_idx: int,
    pv_data_source: PvDataSource,
    padding_hours: float = 12,
    height: int = 200,
    normalize: bool = False,
    colors: list[str] | None = None,
    resample_pv: bool = False,
) -> alt.Chart:
    """Make a timeseries chart for the PV data.

    resample_pv: Resample PV data to match the horizons.
    """
    # Get the ground truth PV data.
    raw_data = pv_data_source.get(
        pv_ids=x.pv_id,
        start_ts=x.ts - dt.timedelta(hours=padding_hours),
        end_ts=x.ts + dt.timedelta(hours=horizons[-1][1] / 60 + min(padding_hours, 12)),
    )["power"]

    capacity = float(
        pv_data_source.get(pv_ids=x.pv_id, start_ts=x.ts - dt.timedelta(days=7), end_ts=x.ts)[
            "power"
        ].quantile(0.99)
    )

    # Extract the meta data for the PV.
    lat = raw_data.coords["latitude"].values
    lon = raw_data.coords["longitude"].values

    # Optinally resample the PV data to the same frequency as our horizons.
    if resample_pv:
        raw_data = raw_data.resample(
            ts=f"{horizons.duration}min",
            loffset=dt.timedelta(minutes=horizons.duration / 2),
            origin="start_day",
        ).mean()

    # Reshape as a pandas dataframe.
    pv_data = raw_data.to_dataframe()[["power"]].reset_index().rename(columns={"ts": "timestamp"})

    irr_kwargs = dict(
        lat=lat,
        lon=lon,
        tilt=35,
        orientation=180,
    )

    if normalize:
        # Normalize the ground truth with respect to pvlib's irradiance.
        irr = get_irradiance(
            timestamps=pv_data["timestamp"],  # type: ignore
            **irr_kwargs,
        )["poa_global"]

        pv_data["power"] = np.clip(pv_data["power"] / (irr.to_numpy() * capacity), 0, 0.003)

    timestamps = [x.ts + dt.timedelta(minutes=h0 + (h1 - h0) / 2) for h0, h1 in horizons]

    all_powers = {m: y.powers for m, y in all_y.items()}

    if normalize:
        # Normalize the predictions with respect to pvlib's irradiance.
        irr = get_irradiance(
            timestamps=timestamps,
            **irr_kwargs,
        )["poa_global"]

        all_powers = {
            m: np.clip(safe_div(powers, irr * capacity), 0, 0.003)
            for m, powers in all_powers.items()
        }

    model_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "current": [1 if i == horizon_idx else 0 for i in range(len(horizons))],
                    "model": model_name,
                    "power": powers,
                }
            )
            for model_name, powers in all_powers.items()
        ]
    )

    x_axis = alt.Axis(tickCount="day", format=("%d %b %Y"))

    pred_chart = (
        alt.Chart(model_data)
        .mark_line(
            # size=14, opacity=0.8
            # opacity=0.8,
            # width=3,
            size=3,
            point=alt.OverlayMarkDef(size=10, opacity=1),
        )
        .encode(
            x=alt.X("timestamp", title="Time", axis=x_axis),
            y=alt.Y("power", title="Power (MW)"),
            color=alt.Color("model:N", scale=alt.Scale(range=colors)),
        )
    )

    ground_truth_chart = (
        alt.Chart(pv_data)
        .mark_line(
            point=alt.OverlayMarkDef(color="black", size=10, opacity=0.8),
            color="gray",
            opacity=1.0,
        )
        .encode(x="timestamp", y="power")
        .properties(height=height, width=800)
    )

    return (
        ground_truth_chart
        + pred_chart
        + time_rule(x.ts, "now", "right")
        + time_rule(pred_ts, "prediction time", "left")
    )


def find_horizon_index(horizon: float, horizons: Horizons) -> int:
    """
    Given a `horizon` in minutes, find the index of which interval it corresponds to,
    in a `Horizons` object.
    """
    horizon_idx = 0
    for h0, h1 in horizons:
        if h0 <= horizon < h1:
            return horizon_idx
        horizon_idx += 1

    raise RuntimeError(f"Horizon {horizon} does not make sense")


def _make_nwp_heatmap(
    ts: Timestamp,
    pred_ts: Timestamp,
    lat: float,
    lon: float,
    nwp_data_source: NwpDataSource,
    radius: float = 20_000.0,
) -> dict[str, alt.Chart]:
    """Make heatmap charts for each nwp features.

    Arguments:
    ---------
    radius: How many meters to look at on each size.
    """
    [min_lat, min_lon], [max_lat, max_lon] = approx_add_meters_to_lat_lon(
        [lat, lon],
        delta_meters=np.array(
            [
                [-radius, -radius],
                [radius, radius],
            ]
        ),
    )

    # TODO: Should this be hard-coded?
    nwp_freq = dt.timedelta(hours=3)

    nwp_data = nwp_data_source.get(
        now=ts,
        # Get data for three steps. 1 before, 1 at prediction time, and 1 after.
        timestamps=[pred_ts - nwp_freq, pred_ts, pred_ts + nwp_freq],
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )

    if nwp_data is None:
        return {}

    df = nwp_data.to_dataframe()[["UKV"]].reset_index()
    df["step"] = df["step"].dt.seconds / 60.0

    charts = {}

    for variable in nwp_data.coords["variable"].values:
        chart = (
            alt.Chart()
            .mark_rect()
            .encode(
                x="x:O",
                y=alt.Y("y:O", scale=alt.Scale(reverse=True)),
                color=alt.Color("UKV:Q", scale=alt.Scale(zero=False)),
            )
            .properties(height=200, width=200)
        )

        df_var = df[df["variable"] == variable]

        charts[variable] = (
            alt.layer(chart, data=df_var).facet(column="step").resolve_scale(color="independent")
        )

        mean_data = df_var[["step", "UKV"]].groupby("step").mean().reset_index()

        charts[variable + "_mean"] = (
            alt.Chart(mean_data)
            .mark_line(point=True)
            .encode(x="step", y=alt.Y("UKV", scale=alt.Scale(zero=False)))
            .properties(height=50, width=800)
        )

    return charts


def _make_explain_chart(x: X, horizon_idx: int, model: PvSiteModel):
    """Make a model explain chart for the sample."""
    shap_values = model.explain(x)
    chart = shap.plots.force(shap_values[horizon_idx])
    return chart


def plot_sample(
    x: X,
    horizon_idx: int,
    horizons: Horizons,
    models: dict[str, PvSiteModel],
    pv_data_source: PvDataSource,
    nwp_data_source: NwpDataSource | None,
    metric: Metric | None = None,
    normalize: bool = False,
    colors: list[str] | None = None,
    resample_pv: bool = False,
):
    """Plot a sample and relevant information

    This is used in notebooks to debug models.
    """
    if metric is None:
        metric = mean_absolute_error

    pv_id = x.pv_id
    ts = x.ts

    # We assume that all the horizons are the same
    horizon = horizons[horizon_idx]
    pred_ts = ts + dt.timedelta(minutes=(horizon[1] + horizon[0]) / 2)

    all_y: dict[str, Y] = {}

    for model_name, model in models.items():
        y = model.predict(x)

        all_y[model_name] = y

        y_true = get_y_from_x(x=x, horizons=horizons, data_source=pv_data_source)

        if y_true is None:
            err = None
        else:
            err = metric(y_true, y)

        lat = float(pv_data_source.get(pv_ids=pv_id).coords["latitude"])
        lon = float(pv_data_source.get(pv_ids=pv_id).coords["longitude"])

        row_as_dict = dict(
            ts=ts,
            pred_ts=pred_ts,
            horizon=horizon,
            lat=lat,
            lon=lon,
            error=err is not None and err[horizon_idx],
            y_true=y_true.powers[horizon_idx] if y_true else None,
            y=y.powers[horizon_idx],
        )

        for key, value in row_as_dict.items():
            print(f"{key:10} {value}")

    for model_name, model in models.items():
        print(model_name)
        try:
            display(_make_explain_chart(x, horizon_idx, model))
        except Exception:
            print("Could not do explain chart")

    for normalize in [False, True]:
        print(f"Normalize = {normalize}")
        display(
            _make_pv_timeseries_chart(
                x=x,
                all_y=all_y,
                pred_ts=pred_ts,
                horizons=horizons,
                horizon_idx=horizon_idx,
                pv_data_source=pv_data_source,
                padding_hours=3 * 24,
                height=300,
                normalize=normalize,
                colors=colors,
                resample_pv=resample_pv,
            )
        )

        display(
            _make_pv_timeseries_chart(
                x=x,
                all_y=all_y,
                pred_ts=pred_ts,
                horizons=horizons,
                horizon_idx=horizon_idx,
                pv_data_source=pv_data_source,
                height=300,
                padding_hours=12,
                normalize=normalize,
                colors=colors,
                resample_pv=resample_pv,
            )
        )

    num_horizons = len(horizons)

    if nwp_data_source is not None:
        print("*** NWP ***")
        for name, chart in _make_nwp_heatmap(
            ts=ts,
            pred_ts=pred_ts,
            lat=lat,
            lon=lon,
            nwp_data_source=nwp_data_source,
        ).items():
            print(name)
            display(chart)

    print("*** FEATURES ***")
    for model_name, model in models.items():
        print(model_name)
        features = model.get_features(X(pv_id=pv_id, ts=ts))
        for key, value in features.items():
            chart = _make_feature_chart(
                name=key,
                feature_obj=value,
                horizon_idx=horizon_idx,
                num_horizons=num_horizons,
            )
            display(chart)
