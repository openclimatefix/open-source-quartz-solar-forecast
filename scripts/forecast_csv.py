from quartz_solar_forecast.utils.forecast_csv import write_out_forecasts
from datetime import datetime

if __name__ == "__main__":
    site_name = input("Enter site name: ")
    start_date_input = input("Enter start date (dd/mm/yyyy): ")
    end_date_input = input("Enter end date (dd/mm/yyyy): ")
    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))
    capacity_kwp = float(input("Enter capacity in kWp: "))

    start_datetime = datetime.strptime(start_date_input, "%d/%m/%Y").strftime("%Y-%m-%d 00:00:00")
    end_datetime = datetime.strptime(end_date_input, "%d/%m/%Y").strftime("%Y-%m-%d 00:00:00")

    write_out_forecasts(
        init_time_freq=6,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        site_name=site_name,
        latitude=latitude,
        longitude=longitude,
        capacity_kwp=capacity_kwp
    )