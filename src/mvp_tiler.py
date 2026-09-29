import os
import cv2
import numpy as np
import openslide
from PIL import Image

# --- Configuration ---
WSI_PATH = "CMU-1.svs"  
OUTPUT_DIR = "extracted_tiles"
PATCH_SIZE = 224              
TARGET_LEVEL = 0 

def process_wsi(wsi_path, output_dir):
    print(f"--- HistoHelper V2: WSI Tiling Engine ---")
    os.makedirs(output_dir, exist_ok=True)

    try:
        slide = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print(f"Error loading WSI. Is it a valid .svs or .tif? Error: {e}")
        return

    # 1. Get slide dimensions at max magnification
    max_w, max_h = slide.dimensions
    print(f"WSI Dimensions (Level 0): {max_w}x{max_h} pixels")

    # 2. Extract a low-res thumbnail for tissue detection (segmentation)
    # We grab a level deep in the pyramid so it easily fits in RAM
    thumb_level = slide.level_count - 1 
    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    downsample_factor = slide.level_downsamples[thumb_level]
    
    print(f"Extracting Thumbnail at Level {thumb_level} for segmentation...")
    thumbnail = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h)).convert("RGB")
    thumb_cv = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)

    # 3. Dynamic Tissue Segmentation using Otsu's Thresholding
    gray = cv2.cvtColor(thumb_cv, cv2.COLOR_BGR2GRAY)
    
    # Blur slightly to remove cellular noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's method dynamically finds the perfect threshold to separate glass from tissue
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find the contours (the borders of the tissue blobs)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a clean mask
    tissue_mask = np.zeros_like(gray)
    cv2.drawContours(tissue_mask, contours, -1, (255), thickness=cv2.FILLED)

    # 4. Grid Extraction using the Mask
    print("Mapping grid and extracting high-res tissue patches...")
    valid_tiles = 0
    
    for y in range(0, max_h, PATCH_SIZE):
        for x in range(0, max_w, PATCH_SIZE):
            
            # Map the high-res coordinate down to our low-res thumbnail mask
            thumb_x = int(x / downsample_factor)
            thumb_y = int(y / downsample_factor)
            thumb_patch_size = int(PATCH_SIZE / downsample_factor)
            
            # Prevent out-of-bounds errors on the edges
            if thumb_x + thumb_patch_size > thumb_w or thumb_y + thumb_patch_size > thumb_h:
                continue

            # Check the mask: How much of this specific patch is tissue?
            mask_patch = tissue_mask[thumb_y:thumb_y+thumb_patch_size, thumb_x:thumb_x+thumb_patch_size]
            tissue_coverage = np.count_nonzero(mask_patch) / mask_patch.size
            
            # If the patch is at least 50% tissue, extract it at full resolution!
            if tissue_coverage > 0.5:
                # read_region safely pulls just this chunk from the hard drive into RAM
                patch = slide.read_region((x, y), TARGET_LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                
                save_path = os.path.join(output_dir, f"patch_{x}_{y}.jpg")
                patch.save(save_path, "JPEG", quality=95)
                valid_tiles += 1
                
                if valid_tiles % 100 == 0:
                    print(f"Extracted {valid_tiles} valid tissue patches...")

    print(f"\n--- Extraction Complete ---")
    print(f"Successfully saved {valid_tiles} high-res tissue patches to /{output_dir}")

if __name__ == "__main__":
    process_wsi(WSI_PATH, OUTPUT_DIR)