import requests
import pandas as pd
import os

# --- Configuration ---
OUTPUT_DIR = "./ml_datasets/tcga"
CSV_PATH = os.path.join(OUTPUT_DIR, "tcga_skcm_clinical.csv")
GDC_API_URL = "https://api.gdc.cancer.gov/cases"

def fetch_tcga_metadata():
    print("--- HistoHelper V3: GDC API Extraction ---")
    print("Querying the Genomic Data Commons for TCGA-SKCM clinical records...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Define our search parameters (Only Melanoma cases)
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": ["TCGA-SKCM"]
        }
    }

    # 2. Define the exact clinical fields we need to predict Tropism
    fields = [
        "submitter_id",
        "diagnoses.primary_diagnosis",
        "diagnoses.tissue_or_organ_of_origin",
        "diagnoses.tumor_stage",
        "samples.sample_type"
        # "diagnoses.ajcc_pathologic_m"
    ]

    params = {
        "filters": str(filters).replace("'", '"'),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "1000" # Grab up to 1000 patients
    }

    # 3. Execute the API Request
    try:
        response = requests.get(GDC_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return

    hits = data.get("data", {}).get("hits", [])
    print(f"Successfully retrieved {len(hits)} patient records.")

   # 4. Clean and Flatten the JSON data
    clinical_data = []
    for hit in hits:
        patient_id = hit.get("submitter_id", "Unknown")
        diagnoses = hit.get("diagnoses", [{}])[0] # Grab the primary diagnosis object
        
        # --- EXTRACTION LOGIC ---
        samples = hit.get("samples", [{}])[0]
        raw_sample_type = samples.get("sample_type", "N/A")
        
        # Translate TCGA language into our M0/M1 pipeline tags
        if "Metastatic" in raw_sample_type:
            meta_stage = "M1"
        elif "Primary Tumor" in raw_sample_type:
            meta_stage = "M0"
        else:
            meta_stage = "N/A"
        # ----------------------------

        record = {
            "Patient_ID": patient_id,
            "Primary_Diagnosis": diagnoses.get("primary_diagnosis", "N/A"),
            "Origin_Tissue": diagnoses.get("tissue_or_organ_of_origin", "N/A"),
            "Tumor_Stage": diagnoses.get("tumor_stage", "N/A"),
            "Metastasis_Stage": meta_stage 
        }
        clinical_data.append(record)

    # 5. Export to CSV
    df = pd.DataFrame(clinical_data)
    df.to_csv(CSV_PATH, index=False)
    
    print(f"Saved TCGA-SKCM metadata to {CSV_PATH}")
    
    # 6. Print a quick data science summary
    print("\n--- Tropism Breakdown ---")
    print(df['Metastasis_Stage'].value_counts())

if __name__ == "__main__":
    fetch_tcga_metadata()