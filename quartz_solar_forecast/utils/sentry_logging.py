"""Log usage of this package to Sentry"""

import importlib.metadata
import os

import sentry_sdk

from quartz_solar_forecast.pydantic_models import PVSite

version = importlib.metadata.version("quartz_solar_forecast")

quartz_solar_forecast_logging = (
    os.getenv("QUARTZ_SOLAR_FORECAST_LOGGING", "True").lower() != "false"
)

SENTRY_DSN = "https://b2b6f3c97299f81464bc16ad0d516d0b@o400768.ingest.us.sentry.io/4508439933157376"

# Disable default integrations to prevent incompatible huggingface_hub integration
# from causing AttributeError with older huggingface_hub versions
sentry_sdk.init(
    dsn=SENTRY_DSN,
    traces_sample_rate=1.0,
    default_integrations=False,
)


def write_sentry(params):
    """
    Log usage of this package to Sentry
    """

    if not quartz_solar_forecast_logging:
        return

    try:
        for key, value in params.items():
            # we want to make sure we don't store the exact location of the site
            if isinstance(value, PVSite):
                value.round_latitude_and_longitude()

            # set sentry tag
            sentry_sdk.set_tag(key, value)

        if os.getenv("PYTEST_CURRENT_TEST") is not None:
            sentry_sdk.set_tag("CI Test", "True")

        sentry_sdk.set_tag("version", version)

    except Exception as _:  # noqa
        pass
