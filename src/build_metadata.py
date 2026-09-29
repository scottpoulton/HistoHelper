import os
import pandas as pd
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "zacharielegault/PatchCamelyon" # Validated, fast-streaming histology dataset
OUTPUT_DIR = "./ml_datasets"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")
TARGET_SAMPLES = 5000 # Reduced to 5,000 per class (10,000 total) for MVP speed

def build_metadata():
    print(f"Initializing streaming connection to the PCam Histopathology dataset...")
    # Streaming=True ensures we don't download the entire 100GB+ archive to your cache again!
    dataset = load_dataset(DATASET_NAME, split='train', streaming=True)
    
    os.makedirs(IMAGE_DIR, exist_ok=True)
    metadata = []
    
    print(f"Extracting {TARGET_SAMPLES} Metastatic and {TARGET_SAMPLES} Normal H&E slides...")
    
    tumor_count = 0
    normal_count = 0
    
    for i, item in enumerate(dataset):
        if tumor_count >= TARGET_SAMPLES and normal_count >= TARGET_SAMPLES:
            break
            
        img = item.get('image')
        label = item.get('label')
        
        if img is None:
            continue
            
        # PCam labels: 0 = Normal, 1 = Metastatic
        if label == 1 and tumor_count >= TARGET_SAMPLES:
            continue
        if label == 0 and normal_count >= TARGET_SAMPLES:
            continue
            
        filename = f"pcam_{i}.jpg"
        filepath = os.path.join(IMAGE_DIR, filename)
        
        img.convert("RGB").save(filepath, format="JPEG")
        
        metadata.append({
            "image_id": f"pcam_{i}",
            "file_path": filepath,
            "label": label
        })
        
        if label == 1:
            tumor_count += 1
        else:
            normal_count += 1
            
        total = tumor_count + normal_count
        if total % 100 == 0:
            print(f"Secured {total} / {TARGET_SAMPLES * 2} H&E slides...")

    print("\nCompiling clinical metadata...")
    df = pd.DataFrame(metadata)
    df.to_csv(CSV_PATH, index=False)
    
    print(f"\nExtraction Complete! Saved {len(df)} records to {CSV_PATH}.")
    print("\n--- Histopathology Class Distribution ---")
    print("Class 1 (Metastatic Tissue):", len(df[df['label'] == 1]))
    print("Class 0 (Normal Tissue):", len(df[df['label'] == 0]))
    print("-----------------------------------------")

if __name__ == "__main__":
    build_metadata()