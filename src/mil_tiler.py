import os
import torchstain
import cv2
import numpy as np
import openslide
import torch
import pandas as pd
import warnings
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # <--- NEW: The progress bar library
from model import get_histo_model

# --- Configuration ---
SLIDE_DIR = "./ml_datasets/tcga/slides"
OUTPUT_DIR = "./ml_datasets/tcga/mil_bags"
METADATA_CSV = "./ml_datasets/tcga/tcga_skcm_clinical.csv"
WEIGHTS_PATH = "./histohelper_production.pth" 
PATCH_SIZE = 224

def build_mil_bags():
    print("--- HistoHelper V3: Multiple Instance Learning (MIL) Extractor ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Loading V2 ResNet18 and converting to Feature Extractor...")
    model = get_histo_model(num_classes=2)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    target_img = cv2.imread('perfect_reference_patch.jpg')
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(target_img)
    
    model.fc = torch.nn.Identity() 
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(METADATA_CSV)
    valid_slides = [f for f in os.listdir(SLIDE_DIR) if f.endswith('.svs')]
    
    if not valid_slides:
        print("No .svs slides found! Waiting for GDC download...")
        return
        
    for slide_name in valid_slides:
        patient_id = slide_name.split('_')[0]
        
        patient_record = df[df['Patient_ID'] == patient_id]
        if patient_record.empty: continue
            
        tropism_label = patient_record.iloc[0]['Metastasis_Stage']
        save_path = os.path.join(OUTPUT_DIR, f"{patient_id}_{tropism_label}.pt")
        
        if os.path.exists(save_path):
            print(f"Skipping {patient_id} - MIL bag already exists.")
            continue
            
        print(f"\nProcessing {patient_id} [{tropism_label}]...")
        
        slide_path = os.path.join(SLIDE_DIR, slide_name)
        
        try:
            slide = openslide.OpenSlide(slide_path)
        except Exception as e:
            print(f"⚠️  CORRUPTION ERROR: Cannot read {slide_name}. Skipping. Details: {e}")
            continue
        
        thumb_level = slide.level_count - 1 
        thumb_w, thumb_h = slide.level_dimensions[thumb_level]
        downsample_factor = slide.level_downsamples[thumb_level]
        thumbnail = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h)).convert("RGB")
        
        thumb_cv = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)
        
        # 1. Base Otsu's Thresholding (Finds everything darker than glass)
        gray = cv2.cvtColor(thumb_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, base_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. HSV Artifact Filtration (Isolate Marker Pens)
        hsv = cv2.cvtColor(thumb_cv, cv2.COLOR_BGR2HSV)
        
        # OpenCV HSV ranges: Hue(0-179), Saturation(0-255), Value(0-255)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([179, 255, 50]) # Targets very dark pixels

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        # Combine all artifacts into one exclusion mask
        artifact_mask = cv2.bitwise_or(mask_blue, mask_green)
        artifact_mask = cv2.bitwise_or(artifact_mask, mask_black)

        # 3. Subtract the artifacts from the base tissue mask
        clean_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(artifact_mask))

        # 4. Cleanup with Contours (Fills holes and drops microscopic noise)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tissue_mask = np.zeros_like(gray)
        cv2.drawContours(tissue_mask, contours, -1, (255), thickness=cv2.FILLED)

        # 4. Extract Features
        max_w, max_h = slide.dimensions
        features_bag = []
        coords_bag = [] 
        
        # --- NEW: Flatten the coordinate grid so tqdm can track it ---
        patch_coords = [(x, y) for y in range(0, max_h, PATCH_SIZE) for x in range(0, max_w, PATCH_SIZE)]
        
        with torch.no_grad():
            # --- NEW: Wrap the loop in the progress bar ---
            for x, y in tqdm(patch_coords, desc=f"Tiling {patient_id}", unit="patch"):
                
                thumb_x = int(x / downsample_factor)
                thumb_y = int(y / downsample_factor)
                thumb_patch_size = int(PATCH_SIZE / downsample_factor)
                
                if thumb_x + thumb_patch_size > thumb_w or thumb_y + thumb_patch_size > thumb_h:
                    continue
                    
                mask_patch = tissue_mask[thumb_y:thumb_y+thumb_patch_size, thumb_x:thumb_x+thumb_patch_size]
                
                if np.count_nonzero(mask_patch) / mask_patch.size > 0.5:
                    patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                    
                    np_patch = np.array(patch)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore") 
                        try:
                            norm_patch, _, _ = normalizer.normalize(I=np_patch, stains=False)
                            if np.isnan(norm_patch).any():
                                norm_patch = np_patch
                        except Exception:
                            norm_patch = np_patch
                        
                    tensor = transform(Image.fromarray(norm_patch)).unsqueeze(0).to(device)
                    
                    feature_vector = model(tensor).squeeze(0)
                    features_bag.append(feature_vector.cpu())
                    coords_bag.append(torch.tensor([x, y])) 
        
        # 5. Save the Bag
        if features_bag:
            patient_data = {
                "features": torch.stack(features_bag),
                "coords": torch.stack(coords_bag)
            }
            torch.save(patient_data, save_path)

if __name__ == "__main__":
    build_mil_bags()