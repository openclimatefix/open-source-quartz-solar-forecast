"""
Make a random test set

This takes a random subset of times and for various pv ids and makes a test set
"""
import os
from typing import Optional

import numpy as np
import pandas as pd

test_start_date = pd.Timestamp("2021-01-01")
test_end_date = pd.Timestamp("2021-02-01")

# this have been chosen from the entire training set. This ideas
pv_ids = [
    8215,
    8229,
    8253,
    8266,
    8267,
    8281,
    16474,
    16477,
    16480,
    16483,
    8376,
    16570,
    16597,
    8411,
    8412,
    8419,
    8420,
    8453,
    8464,
    8505,
    16715,
    16717,
    8551,
    8558,
    16769,
    16770,
    8587,
    8591,
    8612,
    8648,
    8670,
    8708,
    8713,
    16920,
    16921,
    8801,
    17034,
    17035,
    8856,
    17062,
    8914,
    17166,
    9069,
    9070,
    9101,
    9107,
    17333,
    9153,
    9171,
    9173,
]

np.random.seed(42)


def make_test_set(output_file_name: Optional[str] = None, number_of_samples_per_system: int = 50):
    """
    Make a test set of random times and pv ids

    :param output_file_name: the name of the file to write the test set to
    :param number_of_samples_per_system: the number of samples to take per pv id
    """

    if output_file_name is None:
        # get the folder where this file is
        output_file_name = os.path.dirname(os.path.abspath(__file__)) + "/testset.csv"

    test_set = []
    for pv_id in pv_ids:
        ts = pd.date_range(start=test_start_date, end=test_end_date, freq="15min")
        ts = ts[np.random.choice(len(ts), size=number_of_samples_per_system, replace=False)]
        test_set.append(pd.DataFrame({"pv_id": pv_id, "datetime": ts}))
    test_set = pd.concat(test_set)
    test_set.to_csv(output_file_name, index=False)

    return test_set


# To run the script, un comment the following line and run this file
make_test_set()
