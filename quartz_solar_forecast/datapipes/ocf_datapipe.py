import ocf_datapipes  # noqa
from ocf_datapipes.training.example.simple_pv import simple_pv_datapipe
from ocf_datapipes.training.example.pv_nwp import pv_nwp_datapipe
from ocf_datapipes.batch import BatchKey
from datetime import datetime
import numpy as np

import os
import certifi
import ssl

import warnings
warnings.filterwarnings("ignore")

os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

config_file = 'pv_config.yaml'

# Accessing the datapipes
pv_data_pipe = simple_pv_datapipe(configuration_filename=config_file)
pv_nwp_data_pipe = pv_nwp_datapipe(configuration_filename=config_file)

## Pv datapipe
# Store the batch information to analyze
pv_batch = next(iter(pv_data_pipe))

# Access the batch elements through batch keys
# Convert the first set of timestamps to readable dates
pv_times_readable = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in pv_batch[BatchKey.pv_time_utc][0]]
print("Readable PV Times:", pv_times_readable)

# Check the capacity of PV system
observed_capacity = pv_batch[BatchKey.pv_observed_capacity_wp]
nominal_capacity = pv_batch[BatchKey.pv_nominal_capacity_wp]

print("Observed Capacity (Wp):", observed_capacity)
print("Nominal Capacity (Wp):", nominal_capacity)

# Nwp_pv datapipe
pv_nwp_batch = next(iter(pv_nwp_data_pipe))

# Check the NWP data
nwp_data = pv_nwp_batch[BatchKey.nwp]
print("NWP data", nwp_data)