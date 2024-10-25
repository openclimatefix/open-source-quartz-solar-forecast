if __name__ == "__main__":
    # please change the site name, start_datetime and end_datetime, latitude, longitude and capacity_kwp as per your requirement
    write_out_forecasts(
        init_time_freq=6,
        start_datetime="2024-03-10 00:00:00",
        end_datetime="2024-03-11 00:00:00",
        site_name="Test",
        latitude=51.75,
        longitude=-1.25,
        capacity_kwp=1.25
    )