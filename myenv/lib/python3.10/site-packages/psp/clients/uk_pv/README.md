# uk_pv

## Summary

This subdirectory contains code related to the [`uk_pv`][hf] dataset.


## Parsing the data

1. Download the [5min.parquet][hf5] and [metadata][m] data from Hugging Face.

Note that this version of the meta data has truncated lat/lon values.


2. Parse the data

```
poetry run python psp/clients/uk_pv/scripts/simplify_data.py data/5min.parquet -m data/metadata.csv data
poetry run python psp/clients/uk_pv/scripts/data_to_netcdf.py data/5min_all.parquet data/5min_all.nc --power-conversion-factor 0.012
```

 
[hf]: https://huggingface.co/datasets/openclimatefix/uk_pv/tree/main
[hf5]: https://huggingface.co/datasets/openclimatefix/uk_pv/blob/main/5min.parquet
