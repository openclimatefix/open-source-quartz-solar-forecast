"""
Make a random test set

This takes a random subset of times and for various pv ids and makes a test set
"""
import os
from typing import Optional

import numpy as np
import pandas as pd

test_start_date = pd.Timestamp("2021-01-01")
test_end_date = pd.Timestamp("2022-01-01")

# this have been chosen from the entire training set. This ideas
pv_ids = [
    9531,
    7174,
    6872,
    7386,
    13607,
    6330,
    26841,
    6665,
    4045,
    26846,
    6494,
    7834,
    3543,
    7093,
    3864,
    8412,
    3454,
    9765,
    10585,
    26942,
    7721,
    26804,
    7551,
    26861,
    7568,
    7338,
    7410,
    6967,
    16480,
    7241,
    7593,
    7557,
    7757,
    3094,
    6800,
    26905,
    5512,
    26840,
    7595,
    5803,
    26876,
    7846,
    26786,
    7580,
    6629,
    16477,
    3489,
    26796,
    12761,
    26903,
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
# make_test_set()
