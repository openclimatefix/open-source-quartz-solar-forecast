from datetime import datetime


def get_file_path(latitude: float,
                  longitude: float,
                  capacity_kwp: float,
                  model: str = "gb",
                  time: datetime = None) -> str:
    return time.strftime(f"data/%Y/%-m/%-d/{model}_{latitude}_{longitude}_{capacity_kwp}_%Y%m%d_%H.csv")
