import os
import requests
import zipfile
import shutil
from urllib.parse import urlparse

def download_and_process_zip(url, download_dir='../quartz_solar_forecast/dataset/TZ-SAM'):
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    zip_path = os.path.join(download_dir, filename)
    
    # Download the zip file
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {zip_path}")
    
    # Extract the zip file
    extract_dir = download_dir
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Delete GPKG and PDF files from the extracted contents
    print("Removing GPKG and PDF files...")
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(('.gpkg', '.pdf')):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    
    # Delete the original zip file
    print(f"Removing original zip file: {zip_path}")
    os.remove(zip_path)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    url = "https://blog.transitionzero.org/hubfs/Data%20Products/TZ-SAM/TZ-SAM%20_%20Solar%20Asset%20Mapper%20-%20Q4%202024.zip"  # Replace with your actual URL
    download_and_process_zip(url)
