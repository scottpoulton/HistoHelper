import os
from datasets import load_dataset
from PIL import Image

# Configuration
NUM_SAMPLES = 50
OUTPUT_DIR = "spider_mvp_samples"
DATASET_NAME = "histai/SPIDER-skin"

def fetch_samples():
    print(f"--- HistoHelper: Fetching {NUM_SAMPLES} samples from {DATASET_NAME} ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        print("Connecting to Hugging Face...")
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
        
        count = 0
        for item in dataset:
            if count >= NUM_SAMPLES:
                break
                
            # 1. Extract the image from the 'png' key
            img = item['png']
            
            # 2. Grab the unique key to use as the filename
            key_name = item.get('__key__', f"sample_{count}")
            safe_key = key_name.replace('/', '_').replace('\\', '_')
            
            # 3. Save it locally
            filename = f"spider_patch_{count:03d}_{safe_key}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Ensure it is a valid image object before saving
            if isinstance(img, Image.Image):
                # Convert to RGB just in case the PNG has an alpha channel
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                img.save(filepath, "JPEG", quality=95)
                print(f"Saved: {filename}")
                count += 1
            else:
                print(f"Skipping item {count} - Invalid image format.")
                
        print("\n--- Extraction Complete! ---")
        print(f"Successfully downloaded {count} patches to /{OUTPUT_DIR}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_samples()