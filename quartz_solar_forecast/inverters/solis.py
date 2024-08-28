from __future__ import annotations
import asyncio
import pandas as pd
from datetime import datetime, timedelta, timezone
from aiohttp import ClientSession, ClientError
import hashlib
import hmac
import base64
import re
from enum import Enum
from http import HTTPStatus
import json
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from quartz_solar_forecast.inverters.inverter import AbstractInverter

try:
    import async_timeout
except:
    print('Could not import `async_timeout`')


# VERSION
RESOURCE_PREFIX = '/v1/api/'

VERB = "POST"

# Endpoints
INVERTER_LIST = RESOURCE_PREFIX + 'inverterList'
INVERTER_DAY = RESOURCE_PREFIX + 'inverterDay'


class SolisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    api_url: str = Field(alias="SOLIS_CLOUD_API_URL", default='https://www.soliscloud.com')
    port: str = Field(alias="SOLIS_CLOUD_API_PORT", default='13333')
    api_key: str = Field(alias="SOLIS_CLOUD_API_KEY")
    client_secret: str = Field(alias="SOLIS_CLOUD_API_KEY_SECRET")


class SolisInverter(AbstractInverter):
    def __init__(self, settings: SolisSettings):
        self.__settings = settings

    def get_data(self, ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        try:
            return asyncio.run(get_solis_data(self.__settings))
        except Exception as e:
            print(f"Error retrieving Solis data: {str(e)}")
            return None

class SoliscloudAPI():
    """Class with functions for reading data from the Soliscloud Portal."""

    class SolisCloudError(Exception):
        """
        Exception raised for timeouts during calls.
        """

        def __init__(self, message="SolisCloud API error"):

            self.message = message
            super().__init__(self.message)

    class HttpError(SolisCloudError):
        """
        Exception raised for HTTP errors during calls.
        """

        def __init__(self, statuscode, message=None):
            self.statuscode = statuscode
            self.message = message
            if not message:
                if statuscode == 408:
                    now = datetime.now().strftime("%d-%m-%Y %H:%M GMT")
                    self.message = f"Your system time is different from server time, your time is {now}"
                else:
                    self.message = f"Http status code: {statuscode}"
            super().__init__(self.message)

    class TimeoutError(SolisCloudError):
        """
        Exception raised for timeouts during calls.
        """

        def __init__(self, message="Timeout error occurred"):

            self.message = message
            super().__init__(self.message)

    class ApiError(SolisCloudError):
        """
        Exception raised for errors during API calls.
        """

        def __init__(self, message="Undefined API error occurred", code="Unknown", response=None):

            self.message = message
            self.code = code
            self.response = response
            super().__init__(self.message)

        def __str__(self):
            return f'API returned an error: {self.message}, error code: {self.code}, response: {self.response}'

    def __init__(self, domain: str, session: ClientSession) -> None:
        self._domain = domain.rstrip("/")
        self._session: ClientSession = session

    class DateFormat(Enum):
        DAY = 0
        MONTH = 1
        YEAR = 2

    @property
    def domain(self) -> str:
        """ Domain name."""
        return self._domain

    @property
    def session(self) -> ClientSession:
        """ aiohttp client session ID."""
        return self._session

    # All methods take key and secret as positional arguments followed by
    # one or more keyword arguments

    async def inverter_list(self, key_id: str, secret: bytes, /, *,
        page_no: int = 1,
        page_size: int = 20,
        station_id: str = None,
        nmi_code: str = None
    ) -> dict[str, str]:
        """Inverter list"""

        if page_size > 100:
            raise SoliscloudAPI.SolisCloudError("PageSize must be <= 100")

        params: dict[str, Any] = {'pageNo': page_no, 'pageSize': page_size}
        if station_id is not None:
            # If not specified all inverters for all stations for key_id are returned
            params['stationId'] = station_id
        if nmi_code is not None:
            params['nmiCode'] = nmi_code
        return await self._get_records(INVERTER_LIST, key_id, secret, params)

    async def inverter_day(self, key_id: str, secret: bytes, /, *,
        currency: str,
        time: str,
        time_zone: int,
        inverter_id: int = None,
        inverter_sn: str = None
    ) -> dict[str, str]:
        """Inverter daily graph"""

        SoliscloudAPI._verify_date(SoliscloudAPI.DateFormat.DAY, time)
        params: dict[str, Any] = {'money': currency, 'time': time, 'timeZone': time_zone}

        if (inverter_id is not None and inverter_sn is None):
            params['id'] = inverter_id
        elif (inverter_id is None and inverter_sn is not None):
            params['sn'] = inverter_sn
        else:
            raise SoliscloudAPI.SolisCloudError("Only pass one of inverter_id or inverter_sn \
                as identifier")

        return await self._get_data(INVERTER_DAY, key_id, secret, params)

    async def _get_records(self, canonicalized_resource: str, key_id: str, secret: bytes, params: dict[str, Any]):
        """
        Return all records from call
        """

        header: dict[str, str] = SoliscloudAPI._prepare_header(key_id, secret,
            params, canonicalized_resource)

        url = f"{self.domain}{canonicalized_resource}"
        try:
            result = await self._post_data_json(url, header, params)
            return result['page']['records']
        except KeyError as err:
            raise SoliscloudAPI.ApiError("Malformed data", result) from err

    async def _get_data(self, canonicalized_resource: str, key_id: str, secret: bytes, params: dict[str, Any]):
        """
        Return data from call
        """

        header: dict[str, str] = SoliscloudAPI._prepare_header(key_id, secret,
            params, canonicalized_resource)

        url = f"{self.domain}{canonicalized_resource}"
        result = await self._post_data_json(url, header, params)

        return result

    @staticmethod
    def _now() -> datetime.datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _prepare_header(
        key_id: str,
        secret: bytes,
        body: dict[str, str],
        canonicalized_resource: str
    ) -> dict[str, str]:
        content_md5 = base64.b64encode(
            hashlib.md5(json.dumps(body, separators=(",", ":")).encode('utf-8')).digest()
        ).decode('utf-8')

        content_type = "application/json"

        date = SoliscloudAPI._now().strftime("%a, %d %b %Y %H:%M:%S GMT")

        encrypt_str = (VERB + "\n"
            + content_md5 + "\n"
            + content_type + "\n"
            + date + "\n"
            + canonicalized_resource
        )
        hmac_obj = hmac.new(
            secret,
            msg=encrypt_str.encode('utf-8'),
            digestmod=hashlib.sha1
        )
        sign = base64.b64encode(hmac_obj.digest())
        authorization = "API " + key_id + ":" + sign.decode('utf-8')

        header: dict[str, str] = {
            "Content-MD5": content_md5,
            "Content-Type": content_type,
            "Date": date,
            "Authorization": authorization
        }
        return header

    async def _post_data_json(self,
        url: str,
        header: dict[str, Any],
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """ Http-post data to specified domain/canonicalized_resource. """

        resp = None
        result = None
        if self._session is None:
            raise SoliscloudAPI.SolisCloudError("aiohttp.ClientSession not set")
        try:
            async with async_timeout.timeout(10):
                resp = await SoliscloudAPI._do_post_aiohttp(self._session, url, params, header)

                result = await resp.json()
                if resp.status == HTTPStatus.OK:
                    if result['code'] != '0':
                        raise SoliscloudAPI.ApiError(result['msg'], result['code'])
                    return result['data']
                else:
                    raise SoliscloudAPI.HttpError(resp.status)
        except asyncio.TimeoutError as err:
            if resp is not None:
                await resp.release()
            raise SoliscloudAPI.TimeoutError() from err
        except ClientError as err:
            if resp is not None:
                await resp.release()
            raise SoliscloudAPI.ApiError(err)
        except (KeyError, TypeError) as err:
            raise SoliscloudAPI.ApiError("Malformed server response",
                response=result) from err

    @staticmethod
    async def _do_post_aiohttp(
        session,
        url: str,
        params: dict[str, Any],
        header: dict[str, Any]
    ) -> dict[str, Any]:
        """ Allows mocking for unit tests."""
        return await session.post(url, json=params, headers=header)

    @staticmethod
    def _verify_date(format: SoliscloudAPI.DateFormat, date: str):
        rex = re.compile("^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
        err = SoliscloudAPI.SolisCloudError("time must be in format YYYY-MM-DD")
        if format == SoliscloudAPI.DateFormat.MONTH:
            rex = re.compile("^[0-9]{4}-[0-9]{2}$")
            err = SoliscloudAPI.SolisCloudError("month must be in format YYYY-MM")
        elif format == SoliscloudAPI.DateFormat.YEAR:
            rex = re.compile("^[0-9]{4}$")
            err = SoliscloudAPI.SolisCloudError("year must be in format YYYY")
        if not rex.match(date):
            raise err
        return

class SolisData:
    def __init__(self, settings: SolisSettings):
        self.domain = f"{settings.api_url}:{settings.port}"
        self.api_key = settings.api_key
        api_secret_str = settings.client_secret
        if not self.api_key or not api_secret_str:
            raise ValueError("SOLIS_CLOUD_API_KEY or SOLIS_CLOUD_API_KEY_SECRET environment variable is not set")
        self.api_secret = api_secret_str.encode('utf-8')  # Convert to binary string

    async def get_inverter_list(self, soliscloud: SoliscloudAPI):
        """Fetch the list of inverters"""
        inverter_list = await soliscloud.inverter_list(
            self.api_key, 
            self.api_secret, 
            page_no=1, 
            page_size=100
        )
        return inverter_list
    
    def process_solis_data(self, live_generation_kw: pd.DataFrame) -> pd.DataFrame:
        """
        Process the Solis data and convert it to a DataFrame with timestamp and power_kw columns.
        
        :param live_generation_kw: DataFrame with original Solis data
        :return: DataFrame with processed data
        """
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        processed_df = live_generation_kw[['timestamp', 'power_kw']].copy()
        
        # Ensure the timestamp is in the correct format
        processed_df.loc[:, 'timestamp'] = pd.to_datetime(processed_df['timestamp'])
        
        # Sort by timestamp
        processed_df = processed_df.sort_values('timestamp')
        
        # Reset the index
        processed_df = processed_df.reset_index(drop=True)
        
        return processed_df

    async def get_solis_data(self) -> pd.DataFrame:
        """
        Get live PV generation data from Solis API for the last 7 days
        :return: DataFrame with timestamp and power_kw columns
        """
        async with ClientSession() as websession:
            soliscloud = SoliscloudAPI(self.domain, websession)
            
            inverter_list = await self.get_inverter_list(soliscloud)
            if not inverter_list:
                raise ValueError("No inverters found")

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=7)
            
            data_list = []
            
            for inverter in inverter_list:
                inverter_sn = inverter['sn']
                for day in range(7):
                    current_date = (end_time - timedelta(days=day)).strftime('%Y-%m-%d')
                    try:
                        inverter_day_data = await soliscloud.inverter_day(
                            self.api_key,
                            self.api_secret,
                            currency='USD',
                            time=current_date,
                            time_zone=0,
                            inverter_sn=inverter_sn
                        )
                        
                        # Check if inverter_day_data is a list of dictionaries
                        if isinstance(inverter_day_data, list) and all(isinstance(item, dict) for item in inverter_day_data):
                            for data_point in inverter_day_data:
                                timestamp = datetime.fromtimestamp(int(data_point['dataTimestamp']) / 1000, tz=timezone.utc)
                                if start_time <= timestamp <= end_time:
                                    data_list.append({
                                        "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                        "power_kw": float(data_point['pac']) / 1000,  # Convert W to kW
                                        "inverter_sn": inverter_sn
                                    })
                        else:
                            print(f"Unexpected data format for inverter {inverter_sn} on {current_date}")
                            print(f"Received data: {inverter_day_data}")
                    
                    except Exception as e:
                        print(f"Error fetching data for inverter {inverter_sn} on {current_date}: {e}")
                        print(f"Received data: {inverter_day_data}")
                    
                    # Avoid rate limiting
                    await asyncio.sleep(0.5)  # 2 times/sec limit
            
            # Convert the list to a DataFrame
            live_generation_kw = pd.DataFrame(data_list)
            
            if live_generation_kw.empty:
                return pd.DataFrame(columns=["timestamp", "power_kw"])

            # Convert to datetime
            live_generation_kw["timestamp"] = pd.to_datetime(live_generation_kw["timestamp"])
            
            # Sort by timestamp
            live_generation_kw = live_generation_kw.sort_values("timestamp")
            
            # Process the data to match the desired format
            processed_df = self.process_solis_data(live_generation_kw)
            processed_df = processed_df.reset_index(drop=True)
            
            return processed_df


async def get_solis_data(settings: SolisSettings):
    solis_data = SolisData(settings)
    return await solis_data.get_solis_data()
