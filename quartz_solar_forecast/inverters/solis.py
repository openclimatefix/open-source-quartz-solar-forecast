import base64
import hashlib
import hmac
import json
import logging
import os
import http.client
from datetime import datetime, timezone
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

def execute_request(domain, port, endpoint, body, headers):
    conn = http.client.HTTPSConnection(f"{domain}:{port}")
    try:
        conn.request("POST", endpoint, body, headers)
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        logging.error(f"Request failed: {str(e)}")
        return None
    finally:
        conn.close()

def get_inverter_list():
    api_key_id = os.getenv('SOLIS_CLOUD_API_KEY_ID')
    api_key_secret = os.getenv('SOLIS_CLOUD_API_KEY_SECRET')
    domain = os.getenv('SOLIS_CLOUD_API_URL', 'www.soliscloud.com')
    port = int(os.getenv('SOLIS_CLOUD_API_PORT', '13333'))

    endpoint = "/v1/api/inverterList"
    
    body = {
        "pageNo": "1",
        "pageSize": "10"
    }
    
    body_json = json.dumps(body)
    
    content_md5 = get_digest(body_json)
    date = get_gmt_time()
    
    string_to_sign = f"POST\n{content_md5}\napplication/json\n{date}\n{endpoint}"
    signature = hmac_sha1_encrypt(string_to_sign, api_key_secret)
    
    headers = {
        "Content-MD5": content_md5,
        "Content-Type": "application/json",
        "Date": date,
        "Authorization": f"API {api_key_id}:{signature}",
    }
    
    try:
        data_json = execute_request(domain, port, endpoint, body_json, headers)
        
        if not data_json or 'data' not in data_json or 'page' not in data_json['data'] or 'records' not in data_json['data']['page']:
            logging.error("Invalid or empty response from Solis API for inverter list")
            return None

        inverters = data_json['data']['page']['records']
        if inverters:
            return inverters[0]['id'], inverters[0]['sn']
        else:
            logging.error("No inverters found in the response")
            return None

    except Exception as e:
        logging.error(f"Error in get_inverter_list: {str(e)}")
        return None

def get_solis_data() -> pd.DataFrame:
    inverter_info = get_inverter_list()
    if inverter_info is None:
        logging.error("Failed to retrieve inverter list")
        return None

    inverter_id, inverter_sn = inverter_info

    api_key_id = os.getenv('SOLIS_CLOUD_API_KEY_ID')
    api_key_secret = os.getenv('SOLIS_CLOUD_API_KEY_SECRET')
    domain = os.getenv('SOLIS_CLOUD_API_URL', 'www.soliscloud.com')
    port = int(os.getenv('SOLIS_CLOUD_API_PORT', '13333'))

    endpoint = "/v1/api/inverterMonth"
    
    current_date = datetime.now(timezone.utc)
    
    body = {
        "money": "EUR",
        "month": current_date.strftime("%Y-%m"),
        "timeZone": "0",
        "id": inverter_id,
        "sn": inverter_sn
    }
    
    body_json = json.dumps(body)
    
    content_md5 = get_digest(body_json)
    date = get_gmt_time()
    
    string_to_sign = f"POST\n{content_md5}\napplication/json\n{date}\n{endpoint}"
    signature = hmac_sha1_encrypt(string_to_sign, api_key_secret)
    
    headers = {
        "Content-MD5": content_md5,
        "Content-Type": "application/json",
        "Date": date,
        "Authorization": f"API {api_key_id}:{signature}",
    }
    
    try:
        data_json = execute_request(domain, port, endpoint, body_json, headers)
        
        if not data_json or 'data' not in data_json:
            logging.error("Invalid or empty response from Solis API")
            return None

        inverter_data = []
        for day_data in data_json['data']:
            timestamp = datetime.fromtimestamp(day_data['date'] / 1000, tz=timezone.utc)
            inverter_data.append({
                "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "energy_kwh": day_data['energy'],
                "power_kw": day_data['energy'] / 24
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