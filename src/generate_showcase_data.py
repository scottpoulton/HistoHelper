import os
import sys
import ctypes

# --- THE APPLE SILICON SIP BYPASS ---
# We forcefully load the Homebrew C-library into memory using its absolute path
# BEFORE importing openslide, completely bypassing macOS security stripping.
try:
    ctypes.cdll.LoadLibrary('/opt/homebrew/lib/libopenslide.1.dylib')
except Exception:
    pass

import torch
import numpy as np
import pandas as pd
import openslide
import json
from PIL import Image
from attention_net import MultimodalAttentionMIL

# --- Configuration ---
WEIGHTS_PATH = "histohelper_ab_mil_best.pth"
BAG_DIR = "./ml_datasets/tcga/mil_bags"
SLIDE_DIR = "./ml_datasets/tcga/slides"
METADATA_CSV = "./ml_datasets/tcga/tcga_skcm_clinical.csv"
OUTPUT_DIR = "../../public/slides" # Saves straight to your Next.js public folder
PATCH_SIZE = 224

TARGET_PATIENTS = ['TCGA-3N-A9WC', 'TCGA-BF-A5EQ', 'TCGA-D3-A2J9']

def generate_true_showcase():
    print("--- Extracting Ground-Truth XAI Data for Portfolio ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load your actual trained model
    model = MultimodalAttentionMIL(img_input_dim=512, clinical_dim=5).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    
    clinical_df = pd.read_csv(METADATA_CSV)
    bag_files = [f for f in os.listdir(BAG_DIR) if f.endswith('.pt')]
    
    react_db = {}

    for patient_id in TARGET_PATIENTS:
        print(f"\nProcessing {patient_id}...")
        bag_file = next((f for f in bag_files if f.startswith(patient_id)), None)
        slide_file = next((f for f in os.listdir(SLIDE_DIR) if f.startswith(patient_id) and f.endswith('.svs')), None)
        
        if not bag_file or not slide_file:
            print(f"[!] Missing data for {patient_id}. Skipping.")
            continue
            
        # 2. Run the forward pass to get real predictions and attention weights
        patient_data = torch.load(os.path.join(BAG_DIR, bag_file), map_location=device)
        features = patient_data["features"].to(device)
        coords = patient_data["coords"].cpu().numpy()
        
        # Build dummy clinical tensor for the pass
        clinical_tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits, A = model(features, clinical_tensor)
            
        attention_scores = A.squeeze().cpu().numpy()
        normalized_attention = attention_scores / np.max(attention_scores)
        prob = torch.sigmoid(logits).item()
        diagnosis = "Metastatic" if prob >= 0.5 else "Benign"
        
        # 3. Find the Top 3 Highest Attention Coordinates
        top_indices = np.argsort(normalized_attention)[-3:][::-1]
        
        slide = openslide.OpenSlide(os.path.join(SLIDE_DIR, slide_file))
        max_w, max_h = slide.dimensions
        
        regions = []
        for rank, idx in enumerate(top_indices):
            x, y = coords[idx]
            weight = normalized_attention[idx]
            
            # Convert raw pixel coords to percentage for CSS positioning
            css_x = float(x) / max_w
            css_y = float(y) / max_h
            
            # 4. Crop the exact 224x224 high-res patch from the raw SVS file
            patch_filename = f"{patient_id}_patch_{rank+1}.jpg"
            patch_img = slide.read_region((int(x), int(y)), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            patch_img.save(os.path.join(OUTPUT_DIR, patch_filename), "JPEG", quality=95)
            
            regions.append({
                "x": round(css_x, 3),
                "y": round(css_y, 3),
                "weight": round(float(weight), 3),
                "patch": f"/slides/{patch_filename}",
                "note": "Genuine high-attention feature driving AI classification."
            })
            
        react_db[patient_id] = {
            "prediction": diagnosis,
            "confidence": round(prob * 100, 1),
            "regions": regions
        }
        print(f"Saved 3 high-res patches for {patient_id}")

    # 5. Output the exact JavaScript needed for HistoHelperApp.tsx
    print("\n\n=== COPY THIS INTO YOUR HistoHelperApp.tsx MOCK_DB ===")
    print(json.dumps(react_db, indent=4))
    print("========================================================\n")

if __name__ == "__main__":
    generate_true_showcase()