import os
import requests
import subprocess
from pathlib import Path

# AI Hub File Keys
# Format: {"filename": "key"}
FILES = {
    # Label Data (Priority)
    "TL.zip": "522641",
    
    # Source Data (100GB chunks)
    "TS.zip": "522640",
    "TS.z01": "522627",
    "TS.z02": "522628",
    "TS.z03": "522629",
    "TS.z04": "522630",
    "TS.z05": "522631",
    "TS.z06": "522632",
    "TS.z07": "522633",
    "TS.z08": "522634",
    "TS.z09": "522635",
    "TS.z10": "522636",
    "TS.z11": "522637",
    "TS.z12": "522638",
    "TS.z13": "522639"
}

BASE_URL = "https://aihub.or.kr/api/file/download"
OUTPUT_DIR = os.environ.get("AIHUB_OUTPUT_DIR", "data/aihub_smart_factory")

def download_aihub_files():
    print(f"Initiating AI Hub Download for {len(FILES)} files...")
    print(f"Target Directory: {OUTPUT_DIR}")
    
    # AI Hub access key is read from the environment. Never commit it.
    AIHUB_ACCESS_KEY = os.environ.get("AIHUB_ACCESS_KEY")
    if not AIHUB_ACCESS_KEY:
        raise SystemExit("Set the AIHUB_ACCESS_KEY environment variable first.")
    
    headers = {
        "id": "aihub_downloader",
        "key": AIHUB_ACCESS_KEY
    }
    
    for filename, file_key in FILES.items():
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(file_path):
            print(f"[SKIP] {filename} already exists.")
            continue
            
        print(f"[START] Downloading {filename} (File Key: {file_key})...")
        
        # Construct the download URL. Note: The exact API endpoint for direct file key download 
        # varies. We'll use the standard pattern often used with these keys.
        # If this specific endpoint requires a different structure, we'd adjust here.
        download_url = f"https://aihub.or.kr/api/file/download?fileKey={file_key}"
        
        try:
            # Using stream=True for large files
            with requests.get(download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Minimal logging to avoid cluttering logs
                        if total_size > 0 and downloaded % (100 * 1024 * 1024) == 0: # Log every 100MB
                            print(f"  ... {downloaded/1024/1024:.0f} MB / {total_size/1024/1024:.0f} MB downloaded")
                            
            print(f"[COMPLETE] Saved {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
        
    print("Download Queue initialized. Please ensure AI Hub Access Token is set if direct download fails.")

if __name__ == "__main__":
    download_aihub_files()
