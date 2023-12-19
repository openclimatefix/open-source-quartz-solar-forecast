""" Make script to run the evaluation on the test set.

The idea is to run the model on the test set and then compare the results to the actual PV generation.
The NWP (ICON) and PV data are both pulled Open Climate Fix's Hugging Face page.

Please note it can take hours to pull the NWP data.
The data will be cached locally so next time you run it, itll be much quicker
"""

from quartz_solar_forecast.evaluation import run_eval

if __name__ == '__main__':
    run_eval()
