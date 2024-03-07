import ocf_datapipes  # noqa
from ocf_datapipes.training.example.simple_pv import simple_pv_datapipe
from ocf_datapipes.training.example.pv_nwp import pv_nwp_datapipe
from ocf_datapipes.batch import BatchKey
from datetime import datetime
import os
import certifi
import ssl
import warnings

# Suppress warnings and configure SSL certificate for secure connections
warnings.filterwarnings("ignore")
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

def process_ocf_datapipes(config_file):

    """
    This function demonstrates the use of Open Climate Fix's datapipes with the Open Source Quartz Solar Forecast Project.
    It is an exploratory integration and not currently utilized in the Quartz Solar Forecast production pipeline.

    It processes solar power (PV) and numerical weather prediction (NWP) data to prepare it for forecasting tasks.

    :param config_file: The config file that specifies the paths to the data files, data preprocessing parameters, and other configurations necessary for the datapipes to function correctly.
    -  'pv_and_nwp_config.yaml' specifes the paths to necessary data files (e.g., PV output data, NWP data).
    -  The YAML file needs to be edited to reflect the correct paths to your data files and any specific preprocessing requirements for your project.

    This script is meant as a starting point for integrating Open Climate Fix datapipes into the Quartz Solar Forecast Project, serving as an example of how to preprocess and load data for solar power forecasting.
    """

    pv_data_pipe = simple_pv_datapipe(configuration_filename=config_file)
    pv_nwp_data_pipe = pv_nwp_datapipe(configuration_filename=config_file)

    ## Pv datapipe: store the first batch of processed data for inspection
    pv_batch = next(iter(pv_data_pipe))

    # Access the batch elements through batch keys: Convert the first set of timestamps to readable dates
    pv_times_readable = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in pv_batch[BatchKey.pv_time_utc][0]]
    print("Readable PV Times:", pv_times_readable)

    # Check Observed Capacity (Wp), and Nominal Capacity (Wp) of the PV system
    observed_capacity = pv_batch[BatchKey.pv_observed_capacity_wp]
    nominal_capacity = pv_batch[BatchKey.pv_nominal_capacity_wp]

    print("Observed Capacity (Wp):", observed_capacity)
    print("Nominal Capacity (Wp):", nominal_capacity)

    # NWP_PV DataPipe: Retrieve and print NWP data from the batch
    pv_nwp_batch = next(iter(pv_nwp_data_pipe))
    nwp_data = pv_nwp_batch[BatchKey.nwp]
    print("NWP data", nwp_data)


# Load configuration from YAML specifying data sources and processing parameters for datapipes
config_file = 'pv_and_nwp_config.yaml'
process_ocf_datapipes(config_file)