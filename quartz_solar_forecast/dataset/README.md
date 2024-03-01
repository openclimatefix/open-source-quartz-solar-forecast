## Test set analysis
The test set contains the data used for testing the solar forecast models developed. It contains 2500 data points, with `pv_id` and `timestamps` of when the data was collected. Upon analysing the dataset, the following observations were made:
1. The data is pretty evenly distrbuted throughout the year, with the highest number of data points from the month of May (256 data points)
2. Looking at distribution by hour of the day, the highest number of data points is from 19:00 hrs (132 data points), and the least from 00:00 hrs (87 data points)

By analysing the metadata, available at [Hugging Face](https://huggingface.co/datasets/openclimatefix/uk_pv), along with the test set, it can be observed that:
1. Most of the data in the test set has a tilt angle of 30-34 degrees
2. The maximum kwp is 4.0 & the minmum kwp is 2.25 in the test set.

A detailed anaysis of the test set can be found at quartz_solar_forecast/dataset/dataset_analysis/test_set_analysis.ipynb