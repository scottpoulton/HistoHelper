import os
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "histai/SPIDER-skin"
OUTPUT_DIR = "./ml_datasets"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

def download_spider_slides():
    print("--- SPIDER-SKIN HISTOLOGY EXTRACTION ---")
    print("Tapping into the dataset stream to extract 20 real patches...")
    
    # Streaming bypasses the 250GB download
    dataset = load_dataset(DATASET_NAME, split='train', streaming=True)
    
    count = 0
    for item in dataset:
        if count >= 20:
            break
            
        img = item.get('png')
        # The stream key looks like 'SPIDER-skin/images/patch_0825811'
        img_key = item.get('__key__', f"slide_{count}") 
        
        if img is None:
            continue
            
        # THE FIX: Dynamically extract the folder path from the Hugging Face key and create it
        filepath = os.path.join(IMAGE_DIR, f"{img_key}.jpg")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            img.convert("RGB").save(filepath, format="JPEG")
            print(f"[{count+1}/20] Successfully saved: {filepath}")
            count += 1
        except Exception as e:
            print(f"Failed to save {img_key}: {e}")

    print("\nExtraction complete. Your real SPIDER-skin histology slides are on your hard drive.")

if __name__ == "__main__":
    download_spider_slides()