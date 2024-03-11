import os
import pandas as pd
import subprocess
import random

def test_generate_forecasts():
    # Generate 100 random PV IDs between 1 and 50000
    pv_ids = [random.randint(1, 50000) for _ in range(100)]
    
    # Define the command to run your script
    command = "python scripts/generate_multiple_forecasts.py " + " ".join(map(str, pv_ids))

    # Run your script
    subprocess.run(command, shell=True)

    # Check if the output file is created
    assert os.path.exists("quartz_solar_forecast/dataset/forecasts.csv"), "Forecast CSV file not created."

    # Load the output file
    df = pd.read_csv("quartz_solar_forecast/dataset/forecasts.csv")

    # Check if the output file has the correct columns
    expected_columns = ['latitude', 'longitude', 'capacity_kwp', 'date', 'time', 'power_wh']
    assert all([col in df.columns for col in expected_columns]), "Some expected columns are missing in the output CSV file."

if __name__ == "__main__":
    test_generate_forecasts()