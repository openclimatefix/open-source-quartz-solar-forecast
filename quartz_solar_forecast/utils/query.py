"""DuckDB query layer for forecast and weather data.

Enables SQL queries directly on CSV and Parquet files without loading
entire datasets into memory. Useful for multi-site forecasts, long
time ranges, and ad hoc analysis.

Closes #323.

Usage:
    from quartz_solar_forecast.utils.query import ForecastQuery

    fq = ForecastQuery("data/")
    df = fq.sql("SELECT * FROM forecasts WHERE site_name = 'my_site' LIMIT 10")
    avg = fq.mean_power_by_site()
    hourly = fq.hourly_average()
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class ForecastQuery:
    """SQL query interface over forecast CSV/Parquet files using DuckDB."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        if not HAS_DUCKDB:
            raise ImportError(
                "DuckDB is required for the query layer. "
                "Install with: pip install duckdb"
            )
        self.data_dir = Path(data_dir)
        self.con = duckdb.connect()
        self._register_files()

    def _register_files(self) -> None:
        """Auto-discover and register CSV/Parquet files as DuckDB views."""
        csv_files = list(self.data_dir.rglob("*.csv"))
        parquet_files = list(self.data_dir.rglob("*.parquet"))

        if csv_files:
            paths = [str(f) for f in csv_files]
            self.con.execute(
                f"CREATE OR REPLACE VIEW forecasts AS SELECT * FROM read_csv_auto({paths})"
            )

        if parquet_files:
            paths = [str(f) for f in parquet_files]
            self.con.execute(
                f"CREATE OR REPLACE VIEW forecasts_parquet AS SELECT * FROM read_parquet({paths})"
            )

    def sql(self, query: str) -> pd.DataFrame:
        """Run an arbitrary SQL query and return a DataFrame."""
        return self.con.execute(query).fetchdf()

    def mean_power_by_site(self) -> pd.DataFrame:
        """Average predicted power output per site."""
        return self.sql("""
            SELECT
                COALESCE(site_name, 'unknown') AS site,
                AVG(power_kw) AS avg_power_kw,
                COUNT(*) AS readings
            FROM forecasts
            GROUP BY site
            ORDER BY avg_power_kw DESC
        """)

    def hourly_average(self) -> pd.DataFrame:
        """Average power output by hour of day across all sites."""
        return self.sql("""
            SELECT
                EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) AS hour,
                AVG(power_kw) AS avg_power_kw,
                COUNT(*) AS readings
            FROM forecasts
            GROUP BY hour
            ORDER BY hour
        """)

    def filter_by_date_range(
        self, start: str, end: str, table: str = "forecasts"
    ) -> pd.DataFrame:
        """Filter forecast data by date range."""
        return self.sql(f"""
            SELECT * FROM {table}
            WHERE CAST(timestamp AS TIMESTAMP) BETWEEN '{start}' AND '{end}'
            ORDER BY timestamp
        """)

    def site_summary(self) -> pd.DataFrame:
        """Summary statistics per site: min, max, avg, count."""
        return self.sql("""
            SELECT
                COALESCE(site_name, 'unknown') AS site,
                MIN(power_kw) AS min_kw,
                MAX(power_kw) AS max_kw,
                AVG(power_kw) AS avg_kw,
                COUNT(*) AS total_readings
            FROM forecasts
            GROUP BY site
            ORDER BY site
        """)

    def tables(self) -> list[str]:
        """List all registered tables/views."""
        df = self.con.execute("SHOW TABLES").fetchdf()
        return df["name"].tolist()

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
