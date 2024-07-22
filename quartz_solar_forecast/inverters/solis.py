import base64
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_gmt_time():
    return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

def get_digest(body):
    md5 = hashlib.md5(body.encode("utf-8")).digest()
    return base64.b64encode(md5).decode("utf-8")

def hmac_sha1_encrypt(encrypt_text, key_secret):
    key = key_secret.encode("utf-8")
    hmac_obj = hmac.new(key, encrypt_text.encode("utf-8"), hashlib.sha1)
    return base64.b64encode(hmac_obj.digest()).decode("utf-8")

def execute_request(url, data, headers, retries=3):
    request = Request(url, data=data.encode("utf-8"), headers=headers)
    for _ in range(retries):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")
    raise Exception("Max retries reached")

def get_solis_data(inverter_identifier: str, is_sn: bool = False) -> pd.DataFrame:
    api_key_id = os.getenv('SOLIS_CLOUD_API_KEY_ID')
    api_key_secret = os.getenv('SOLIS_CLOUD_API_KEY_SECRET')
    domain = os.getenv('SOLIS_CLOUD_API_URL', 'https://www.soliscloud.com')
    port = int(os.getenv('SOLIS_CLOUD_API_PORT', '13333'))

    url = f'{domain}:{port}/v1/api/inverterMonth'
    
    # Get data for the current month
    current_date = datetime.now(timezone.utc)
    
    body = {
        "money": "EUR",  # Change this to your preferred currency
        "month": current_date.strftime("%Y-%m"),
        "timeZone": "0"  # Adjust if needed
    }
    
    if is_sn:
        body["sn"] = inverter_identifier
    else:
        body["id"] = inverter_identifier
    
    body_json = json.dumps(body)
    
    content_md5 = get_digest(body_json)
    date = get_gmt_time()
    
    string_to_sign = f"POST\n{content_md5}\napplication/json\n{date}\n/v1/api/inverterMonth"
    signature = hmac_sha1_encrypt(string_to_sign, api_key_secret)
    
    headers = {
        "Content-MD5": content_md5,
        "Content-Type": "application/json",
        "Date": date,
        "Authorization": f"API {api_key_id}:{signature}",
    }
    
    try:
        data_json = execute_request(url, body_json, headers)
        
        if not data_json or 'data' not in data_json:
            logging.error("Invalid or empty response from Solis API")
            return None

        # Process the data
        inverter_data = []
        for day_data in data_json['data']:
            timestamp = datetime.fromtimestamp(day_data['date'] / 1000, tz=timezone.utc)
            inverter_data.append({
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "energy_kwh": day_data['energy'],
                "power_kw": day_data['energy'] / 24  # Assuming average power over 24 hours
            })
        
        if not inverter_data:
            logging.warning("No data received from Solis API")
            return None

        live_generation_kw = pd.DataFrame(inverter_data)
        live_generation_kw["timestamp"] = pd.to_datetime(live_generation_kw["timestamp"])
        
        return live_generation_kw

    except Exception as e:
        logging.error(f"Error in get_solis_data: {str(e)}")
        return None