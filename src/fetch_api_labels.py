import os
import requests
import pandas as pd
from huggingface_hub import get_token

# --- Configuration ---
DATASET_NAME = "histai/SPIDER-skin"
OUTPUT_DIR = "./ml_datasets"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

def run_api_hack():
    print("--- INITIATING HUGGING FACE API HACK ---")
    
    # 1. Catalog the images physically sitting on your Mac
    # Because they downloaded into nested folders, we walk the directory to find them
    downloaded_files = []
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.endswith('.jpg'):
                # Extract the base ID (e.g., 'patch_0825811')
                patch_id = file.replace('.jpg', '') 
                file_path = os.path.join(root, file)
                downloaded_files.append({"patch_id": patch_id, "file_path": file_path})
                
    if not downloaded_files:
        print("[!] Error: No images found. Did the extraction script work?")
        return

    print(f"Found {len(downloaded_files)} unlabeled histology patches on disk.")
    print("Querying Hugging Face's Datasets Server API for the master records...\n")
    
    # Grab your local HF token (required since SPIDER-skin is a gated dataset)
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    # 2. Ping the Datasets Server API
    # The 'first-rows' endpoint bypasses raw files and hits the pre-computed JSON database
    api_url = f"https://datasets-server.huggingface.co/first-rows?dataset={DATASET_NAME}&config=default&split=train"
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"[!] API Hack Failed. Hugging Face rejected the request: {e}")
        print("This means the internal API database hasn't finished indexing the 250GB archive.")
        return
        
    data = response.json()
    rows = data.get('rows', [])
    
    metadata = []
    
    # 3. Match the API data to your local files
    for row in rows:
        row_data = row.get('row', {})
        # The key looks like 'SPIDER-skin/images/patch_0825811'
        full_key = row_data.get('__key__', '')
        patch_id = full_key.split('/')[-1] 
        
        # Check if this patch exists on our hard drive
        for local_file in downloaded_files:
            if local_file["patch_id"] == patch_id:
                # We found a match! Grab the label.
                label = row_data.get('label', 0)
                
                metadata.append({
                    "image_id": patch_id,
                    "file_path": local_file["file_path"],
                    "label": label
                })
                break
            
    print(f"Successfully reconstructed clinical labels for {len(metadata)} patches!")
    
    if len(metadata) == 0:
        print("[!] The API returned rows, but none matched our downloaded images.")
        return

    # 4. Compile the CSV
    df = pd.DataFrame(metadata)
    df.to_csv(CSV_PATH, index=False)
    
    print(f"Master clinical metadata saved to {CSV_PATH}.")

if __name__ == "__main__":
    run_api_hack()