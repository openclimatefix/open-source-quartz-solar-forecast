import pandas as pd

from psp.pv import get_irradiance


def test_get_irradiance():
    irr = get_irradiance(
        lat=-70,
        lon=45,
        timestamps=[
            pd.Timestamp(2022, 1, 1, 12, 34),
            pd.Timestamp(2023, 1, 1, 12, 34),
        ],
        tilt=30,
        orientation=270,
    )

    assert len(irr) == 2

    # Check that a few expected columns are there.
    some_columns = {"dhi", "ghi", "dni", "poa_global"}
    assert len(some_columns & set(irr.columns)) == len(some_columns)
