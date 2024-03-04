## Test set analysis
The test set contains the data used for testing the solar forecast models developed. It contains 2500 data points, with `pv_id` and `timestamps` of when the data was collected. Upon analysing the dataset, the following observations were made:
1. The data is pretty evenly distrbuted throughout the year, with the highest number of data points from the month of May (256 data points)
2. Looking at distribution by hour of the day, the highest number of data points is from 19:00 hrs (132 data points), and the least from 00:00 hrs (87 data points)

By analysing the metadata, available at [Hugging Face](https://huggingface.co/datasets/openclimatefix/uk_pv), along with the test set, it can be observed that:
1. Most of the data in the test set has a tilt angle of 30-34 degrees
2. The maximum kwp is 4.0 & the minmum kwp is 2.25 in the test set.

A detailed anaysis of the test set can be found at quartz_solar_forecast/dataset/dataset_analysis/test_set_analysis.ipynb

### `test_set_analysis_pv_id_vs_month.ipynb`
This file uses `testset.csv`, which consists of data from 50 photovoltaic systems represented by unique `pv_id`. Each `pv_id` has 50 data points collected at times represented by `timestamp`. The dataset was analyzed to observe the distribution trends of data points during different months of the year for each PV ID.
The following scatter plot shows the distribution of data points for each PV ID across the months of the year.

![PV ID vs. Month Distribution](https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/blob/main/images/test_analysis_output.png?raw=true)

The following observations were made from the plot:

- **Distribution of Data Points**: The plot displays data points for all months across multiple PV IDs. Each dot signifies an instance of electricity generation data recorded from a PV system.

- **Frequency of Data Points**: The color intensity on the scatter plot corresponds to the frequency of data points for each PV ID and month. Lighter shades represent a lower number of data points, whereas darker shades signify a higher frequency. Notably, the months of May, June, July, August, and September are marked by darker shades, indicating a higher frequency of data points compared to the rest of the year.

- **Uniformity Across Months**: Data points are distributed fairly evenly across the months for each PV ID, which implies that data collection is consistent throughout the year without significant lapses.