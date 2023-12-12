from quartz_solar_forecast.eval.pv import get_pv_truth
import pandas as pd


# can take ~ 1 minute to run
def test_get_nwp():
    # make test dataset file
    test_set_df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2021-01-26 01:15:00"),
                "latitude": 51.5,
                "longitude": 0.0,
                "pv_id": 8215,
            }
        ]
    )

    # Collect NWP data from Hugging Face, ICON. (Peter)
    _ = get_pv_truth(test_set_df)
