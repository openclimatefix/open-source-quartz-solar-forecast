import pandas as pd
import numpy as np
from huggingface_hub import HfFileSystem

def download_file_from_hf(file_path):
    """
    Download a file from HuggingFace
    
    This is a helper function that will be mocked in tests
    """
    fs = HfFileSystem()
    if file_path.endswith('.csv'):
        return pd.read_csv(f"hf://datasets/openclimatefix/uk_pv/{file_path}")
    elif file_path.endswith('.netcdf'):
        # In a real implementation, this would use xarray to open the netcdf file
        # For now, we'll return a simple dict structure
        return {
            'power_w': np.random.rand(24, 2) * 1000,
            'time': pd.date_range(start=pd.Timestamp.now(), periods=24, freq='H'),
            'site_id': [0, 1]
        }

def get_pv_data():
    """
    Get PV data from HuggingFace
    
    Returns:
        dict: Dictionary containing PV data
    """
    # Download metadata
    metadata = download_file_from_hf('metadata.csv')
    
    # Download PV data
    pv_data = download_file_from_hf('pv.netcdf')
    
    return pv_data 