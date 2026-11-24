import ssl


def test_import_does_not_disable_tls_verification():
    default = ssl._create_default_https_context
    import importlib

    import quartz_solar_forecast.data as data

    importlib.reload(data)
    assert ssl._create_default_https_context is default
    assert ssl._create_default_https_context is not ssl._create_unverified_context
