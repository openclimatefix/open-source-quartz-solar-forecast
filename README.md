# Quartz Solar Forecast

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->

[![All Contributors](https://img.shields.io/badge/all_contributors-15-orange.svg?style=flat-square)](#contributors-)

<!-- ALL-CONTRIBUTORS-BADGE:END -->

The aim of the project is to build an open source PV forecast that is free and easy to use.
The forecast provides the expected generation in `kw` for 0 to 48 hours for a single PV site.

Open Climate Fix also provides a commercial PV forecast, please get in touch at quartz.support@openclimatefix.org

We recently presented the Quartz Solar Forecast project at FOSDEM 2024 (Free and Open source Software Developers' European Meeting), providing an introduction to Open Climate Fix's motivation for this project and its impact on aiding organizations in resource optimization. To learn more about predictive model's functionality, visit here: [Video Recording](https://www.youtube.com/watch?v=NAZ2VeiN1N8)

The current model uses GFS or ICON NWPs to predict the solar generation at a site

```python
from quartz_solar_forecast.forecast import run_forecast
from quartz_solar_forecast.pydantic_models import PVSite
from datetime import datetime

# make a pv site object
site = PVSite(latitude=51.75, longitude=-1.25, capacity_kwp=1.25)

# run model for today
predictions_df = run_forecast(site=site, ts=datetime.today())
```

which should result in a time series similar to this one:

![https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/blob/main/predictions.png?raw=true](https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/blob/main/predictions.png?raw=true)

A colab notebook providing some examples can be found [here](https://colab.research.google.com/drive/1qKDFRpq4Hk-LHgWuDsz_Najc3Zq-GVNY?usp=sharing).

## Generating Forecasts

To generate solar forecasts and save them into a CSV file, follow these steps:

1. Navigate to the scripts directory

```bash
cd scripts
```

2. Run the forecast_csv.py script with desired inputs

```bash
python forecast_csv.py
```

Replace the --init_time_freq, --start_datetime, --end_datetime, and --site_name with your desired forecast initialization frequency (in hours), start datetime, end datetime, and the name of the forecast or site, respectively.

Output

The script will generate solar forecasts at the specified intervals between the start and end datetimes. The results will be combined into a CSV file named using the site name, start and end datetimes, and the frequency of forecasts. This file will be saved in the scripts/csv_forecasts directory.

## Installation

The source code is currently hosted on GitHub at: https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast

Binary installers for the latest released version are available at the Python Package Index (PyPI)

```bash
pip install quartz-solar-forecast
```

You might need to install the following packages first

```bash
conda install -c conda-forge pyresample
```

This can solve the [bug: \_\_\_kmpc_for_static_fini](https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/issues/32).

## Model

The default model is an XGBoost model and uses the following Numerical Weather Predictions (NWP) input features achieved from [open-meteo](https://open-meteo.com/) variables and additional information about the time, location and specifics about the panel. 

* Temperature at 2m (ºC)
* Relative Humidity at 2m (%)
* Dewpoint at 2m (ºC)
* Precipitation (rain + snow) (mm)
* Surface Pressure (hPa)
* Cloud Cover Total (%)
* Cloud Cover Low (%)
* Cloud Cover Mid (%)
* Cloud Cover High (%)
* Wind Speed at 10m (km/h): Wind speed measured at a height of 10 meters above ground level. Important for understanding weather conditions and potential impacts on solar panels.
* Wind Direction (10m)
* Is day or Night
* Direct Solar Radiation (W/m2)
* Diffusive Solar Radiation DHI (W/m2)

The model was trained and evaluated on 1147 solar panels and tested on 37 independent locations. An intensive hyperparameter tuning was performed. The model provides a feature importance list. Different metrics were calculated and analyzed. Finally the model was evaluated using the Mean Absolute Error (MAE). The MAE over the entire test data is $0.12 kW$, when the night times are excluded the MAE is $0.21kW$. A plot with the MAE for each panel in the test set is shown in the figure below.

![images/mae_test.png]
*Mean absolute error for the panels in the test set.*

When using ```model="ocf"``` in ```run_forecast(site=site, model="ocf", ts=datetime.today())```, the previous version is used. This is a Gradient Boosting model and uses 9 NWP variables. It is trained on 25,000 PV sites with 3 years of PV history, which is available [here](https://huggingface.co/datasets/openclimatefix/uk_pv). The training of this model is handled in [pv-site-prediction](https://github.com/openclimatefix/pv-site-prediction). We however recommend using the default model, because the previous version shows less accurate results and additionally has data inconsistencies of the input features between training and validation.

## Known restrictions

* The model was trained and tested only over the UK, applying it to other geographical regions should be done with caution. 
* When using the default model, only predictions within the last 90 days are available. When using ```model="ocf"``` predictions for past data are available, however, different data sources for the NWP data are used than during training. 

In general, we recommend using the default model and restricting predictions for the period of the last 90 days.


## FOSDEM

FOSDEM is a free event for software developers to meet, share ideas and collaborate. Every year, thousands of developers of free and open source software from all over the world gather at the event in Brussels.
OCF presented Quartz Solar Forecast project at FOSDEM 2024. The link to the original FOSDEM video is availble at [Quartz Solar OS: Building an open source AI solar forecast for everyone](https://fosdem.org/2024/schedule/event/fosdem-2024-2960-quartz-solar-os-building-an-open-source-ai-solar-forecast-for-everyone/).
It is also available on [YouTube](https://www.youtube.com/watch?v=NAZ2VeiN1N8)

## Contribution

We welcome other models.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zakwatts"><img src="https://avatars.githubusercontent.com/u/47150349?v=4?s=100" width="100px;" alt="Megawattz"/><br /><sub><b>Megawattz</b></sub></a><br /><a href="#ideas-zakwatts" title="Ideas, Planning, & Feedback">🤔</a> <a href="#talk-zakwatts" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/EdFage"><img src="https://avatars.githubusercontent.com/u/87755165?v=4?s=100" width="100px;" alt="EdFage"/><br /><sub><b>EdFage</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=EdFage" title="Documentation">📖</a> <a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=EdFage" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chloepilonv"><img src="https://avatars.githubusercontent.com/u/136987461?v=4?s=100" width="100px;" alt="Chloe Pilon Vaillancourt"/><br /><sub><b>Chloe Pilon Vaillancourt</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=chloepilonv" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://racheltipton.dev"><img src="https://avatars.githubusercontent.com/u/86949265?v=4?s=100" width="100px;" alt="rachel tipton"/><br /><sub><b>rachel tipton</b></sub></a><br /><a href="#talk-rachel-labri-tipton" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/armenbod"><img src="https://avatars.githubusercontent.com/u/84937223?v=4?s=100" width="100px;" alt="armenbod"/><br /><sub><b>armenbod</b></sub></a><br /><a href="#content-armenbod" title="Content">🖋</a> <a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=armenbod" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shreyasudaya"><img src="https://avatars.githubusercontent.com/u/94735680?v=4?s=100" width="100px;" alt="Shreyas Udaya"/><br /><sub><b>Shreyas Udaya</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=shreyasudaya" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://github.com/aryanbhosale"><img src="https://avatars.githubusercontent.com/u/36108149?v=4?s=100" width="100px;" alt="Aryan Bhosale"/><br /><sub><b>Aryan Bhosale</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=aryanbhosale" title="Documentation">📖</a> <a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=aryanbhosale" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/0xFrama"><img src="https://avatars.githubusercontent.com/u/30957828?v=4?s=100" width="100px;" alt="Francesco"/><br /><sub><b>Francesco</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=0xFrama" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/roshnaeem"><img src="https://avatars.githubusercontent.com/u/47316899?v=4?s=100" width="100px;" alt="Rosheen Naeem"/><br /><sub><b>Rosheen Naeem</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=roshnaeem" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bikramb98"><img src="https://avatars.githubusercontent.com/u/24806286?v=4?s=100" width="100px;" alt="Bikram Baruah"/><br /><sub><b>Bikram Baruah</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=bikramb98" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Hapyr"><img src="https://avatars.githubusercontent.com/u/23398802?v=4?s=100" width="100px;" alt="Jakob Gebler"/><br /><sub><b>Jakob Gebler</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/issues?q=author%3AHapyr" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ombhojane"><img src="https://avatars.githubusercontent.com/u/82753658?v=4?s=100" width="100px;" alt="Om Bhojane"/><br /><sub><b>Om Bhojane</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=ombhojane" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://chrisadams.me.uk"><img src="https://avatars.githubusercontent.com/u/17906?v=4?s=100" width="100px;" alt="Chris Adams"/><br /><sub><b>Chris Adams</b></sub></a><br /><a href="#ideas-mrchrisadams" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mudrap17"><img src="https://avatars.githubusercontent.com/u/76879120?v=4?s=100" width="100px;" alt="Mudra Patel"/><br /><sub><b>Mudra Patel</b></sub></a><br /><a href="https://github.com/openclimatefix/Open-Source-Quartz-Solar-Forecast/commits?author=mudrap17" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
