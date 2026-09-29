import os
import json
import shutil
import subprocess
import pandas as pd
import requests

# --- Configuration ---
METADATA_CSV = "./ml_datasets/tcga/tcga_skcm_clinical.csv"
SLIDE_DIR = "./ml_datasets/tcga/slides"
GDC_FILES_API = "https://api.gdc.cancer.gov/files"

def fetch_wsi_files():
    print("--- HistoHelper: GDC Multi-Threaded Extractor ---")
    os.makedirs(SLIDE_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_CSV, keep_default_na=False)
    valid_patients = df[df['Metastasis_Stage'].str.startswith('M1', na=False)]['Patient_ID'].tolist()

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": valid_patients}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id",
        "format": "JSON",
        "size": "500"
    }

    response = requests.post(GDC_FILES_API, json=params)
    hits = response.json().get("data", {}).get("hits", [])
    
    # --- DYNAMIC STORAGE CAPPING ---
    TARGET_GB_LIMIT = 90 # Safety limit in Gigabytes
    MAX_BYTES = TARGET_GB_LIMIT * 1024**3 # Convert GB to Bytes
    
    target_downloads = []
    current_total_bytes = 0
    
    print(f"Calculating download queue up to a strict {TARGET_GB_LIMIT}GB limit...")
    
    for slide in hits:
        # Grab the file size from the API response (defaults to 0 if missing)
        size_in_bytes = slide.get("file_size", 0)
        
        # If adding this next slide blows past our limit, stop queuing
        if current_total_bytes + size_in_bytes > MAX_BYTES:
            break
            
        target_downloads.append(slide)
        current_total_bytes += size_in_bytes
        
    print(f"Queue finalized: {len(target_downloads)} slides totaling ~{current_total_bytes / (1024**3):.2f} GB.")
    # ------------------------------------

    uuids_to_download = []
    rename_map = {}

    print("Phase 1: Rescuing trapped slides...")
    
    for slide in target_downloads:
        file_id = slide["file_id"]
        original_filename = slide["file_name"]
        patient_id = slide["cases"][0]["submitter_id"]
        
        safe_filename = f"{patient_id}_{file_id[:8]}.svs"
        final_path = os.path.join(SLIDE_DIR, safe_filename)
        
        gdc_folder = os.path.join(SLIDE_DIR, file_id)
        gdc_file = os.path.join(gdc_folder, original_filename)
        
        # Rescue the file if it's trapped in a UUID subfolder
        if os.path.exists(gdc_file):
            shutil.move(gdc_file, final_path)
            print(f"Rescued: {safe_filename}")
            
        # Nuke the empty folder and any hidden .log/.partial files inside it
        if os.path.exists(gdc_folder):
            shutil.rmtree(gdc_folder)

        # Now check if it's securely in the main folder
        if not os.path.exists(final_path):
            uuids_to_download.append(file_id)
            rename_map[file_id] = {
                "original": original_filename,
                "target": safe_filename
            }
        else:
            print(f"Verified ready: {safe_filename}")

    if not uuids_to_download:
        print("\nAll target slides are successfully downloaded and formatted!")
        return

    print(f"\nPhase 2: Handing over {len(uuids_to_download)} missing slides to gdc-client...")
    cmd = ["/usr/local/bin/gdc-client", "download", "-n", "8", "-d", SLIDE_DIR] + uuids_to_download

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        pass # Ignore network drops

    print("\nFormatting newly downloaded files...")
    for file_id, data in rename_map.items():
        gdc_folder = os.path.join(SLIDE_DIR, file_id)
        gdc_file = os.path.join(gdc_folder, data["original"])
        final_file = os.path.join(SLIDE_DIR, data["target"])
        
        if os.path.exists(gdc_file):
            shutil.move(gdc_file, final_file)
            print(f"Formatted successfully: {data['target']}")
            
        if os.path.exists(gdc_folder):
            shutil.rmtree(gdc_folder)
            
    print("\n--- Pipeline ingestion complete ---")

if __name__ == "__main__":
    fetch_wsi_files()