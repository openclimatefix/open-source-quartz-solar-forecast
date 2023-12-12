from quartz_solar_forecast.evaluation import run_eval
import tempfile
import pandas as pd


def test_run_eval():

    # create a fake dataframe

    with tempfile.TemporaryDirectory() as tmpdirname:
        print("created temporary directory", tmpdirname)

        test_dataset = pd.DataFrame(
            columns=[
                "pv_id",
                "timestamp",
            ],
            data=[[8215, "2021-01-26 01:15:00"], [8215, "2021-01-30 16:30:00"]],
        )

        testset_filename = tmpdirname + "/test_dataset.csv"
        test_dataset.to_csv(testset_filename, index=False)

        # call the metrics function
        run_eval(testset_filename)
