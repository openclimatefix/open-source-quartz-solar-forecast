import http.client
import os
import pandas as pd
import json
import base64
import hashlib
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from urllib.parse import urlencode

load_dotenv()

def get_gmt_time():
    cd = datetime.now(timezone.utc)
    return cd.strftime("%a, %d %b %Y %H:%M:%S GMT")

def get_digest(body):
    md5 = hashlib.md5()
    md5.update(body.encode('utf-8'))
    return base64.b64encode(md5.digest()).decode('utf-8')

def hmac_sha1_encrypt(encrypt_text, key_secret):
    key = key_secret.encode('utf-8')
    data = encrypt_text.encode('utf-8')
    hmac_obj = hashlib.sha1(key)
    hmac_obj.update(data)
    return base64.b64encode(hmac_obj.digest()).decode('utf-8')

def get_solis_access_token():
    api_id = os.getenv('SOLIS_API_ID')
    api_secret = os.getenv('SOLIS_API_SECRET')
    
    body = json.dumps({"pageNo": 1, "pageSize": 10})
    content_md5 = get_digest(body)
    date = get_gmt_time()
    canonicalized_resource = "/v1/api/userStationList"
    
    string_to_sign = f"POST\n{content_md5}\napplication/json\n{date}\n{canonicalized_resource}"
    signature = hmac_sha1_encrypt(string_to_sign, api_secret)
    
    headers = {
        "Content-type": "application/json;charset=UTF-8",
        "Authorization": f"API {api_id}:{signature}",
        "Content-MD5": content_md5,
        "Date": date
    }
    
    conn = http.client.HTTPSConnection("www.soliscloud.com", 13333)
    conn.request("POST", canonicalized_resource, body, headers)
    
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    
    if res.status != 200:
        raise Exception(f"Failed to authenticate: {data}")
    
    return json.loads(data)

def get_solis_data(inverter_id: str) -> pd.DataFrame:
    api_id = os.getenv('SOLIS_API_ID')
    api_secret = os.getenv('SOLIS_API_SECRET')
    
    # Get data for the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    body = json.dumps({
        "id": inverter_id,
        "money": "EUR",
        "month": start_date.strftime("%Y-%m"),
        "timeZone": "0"
    })
    
    content_md5 = get_digest(body)
    date = get_gmt_time()
    canonicalized_resource = "/v1/api/inverterMonth"
    
    string_to_sign = f"POST\n{content_md5}\napplication/json\n{date}\n{canonicalized_resource}"
    signature = hmac_sha1_encrypt(string_to_sign, api_secret)
    
    headers = {
        "Content-type": "application/json;charset=UTF-8",
        "Authorization": f"API {api_id}:{signature}",
        "Content-MD5": content_md5,
        "Date": date
    }
    
    conn = http.client.HTTPSConnection("www.soliscloud.com", 13333)
    conn.request("POST", canonicalized_resource, body, headers)
    
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    
    if res.status != 200:
        raise Exception(f"Failed to get inverter data: {data}")
    
    data_json = json.loads(data)
    
    # Process the data
    inverter_data = []
    for day_data in data_json['data']:
        timestamp = datetime.fromtimestamp(day_data['date'] / 1000, tz=timezone.utc)
        if start_date <= timestamp <= end_date:
            inverter_data.append({
                "timestamp": timestamp,
                "power_kw": day_data['energy']
            })
    
    live_generation_kw = pd.DataFrame(inverter_data)
    live_generation_kw["timestamp"] = pd.to_datetime(live_generation_kw["timestamp"])
    
    return live_generation_kw