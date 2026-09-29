import os
import torch
import cv2
import numpy as np
import pandas as pd
import openslide
from attention_net import MultimodalAttentionMIL

# --- Configuration ---
WEIGHTS_PATH = "histohelper_ab_mil_best.pth"
BAG_DIR = "./ml_datasets/tcga/mil_bags"
SLIDE_DIR = "./ml_datasets/tcga/slides"
METADATA_CSV = "./ml_datasets/tcga/tcga_skcm_clinical.csv"
PATCH_SIZE = 224

def generate_clinical_overlay(patient_pt_file, clinical_df):
    print(f"--- HistoHelper V3: Multimodal Attention Overlay ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load the trained Multimodal AB-MIL Brain
    model = MultimodalAttentionMIL(img_input_dim=512, clinical_dim=5).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    
    # 2. Load the WSI Data
    bag_path = os.path.join(BAG_DIR, patient_pt_file)
    patient_data = torch.load(bag_path)
    
    if not isinstance(patient_data, dict) or "coords" not in patient_data:
        print(f"[!] Skipping {patient_pt_file}: Legacy bag format detected.")
        return
    
    features = patient_data["features"].to(device)
    coords = patient_data["coords"].numpy() 
    
    patient_id = patient_pt_file.split('_')[0]
    
    # 3. Build the Clinical Tensor
    patient_record = clinical_df[clinical_df['Patient_ID'] == patient_id]
    clinical_vector = [0.0, 0.0, 0.0, 0.0, 0.0]

    if not patient_record.empty:
        origin = str(patient_record.iloc[0]['Origin_Tissue']).lower()
        if 'skin' in origin: clinical_vector[0] = 1.0
        elif 'lymph' in origin: clinical_vector[1] = 1.0
        elif 'lung' in origin: clinical_vector[2] = 1.0
        elif 'brain' in origin: clinical_vector[3] = 1.0
        else: clinical_vector[4] = 1.0
    else:
        clinical_vector[4] = 1.0

    # Add batch dimension: [1, 5]
    clinical_tensor = torch.tensor(clinical_vector, dtype=torch.float32).unsqueeze(0).to(device)

    # 4. Locate the corresponding .svs file
    slide_files = [f for f in os.listdir(SLIDE_DIR) if f.startswith(patient_id) and f.endswith('.svs')]
    if not slide_files:
        print(f"[!] Error: Could not find original .svs slide for {patient_id}")
        return
        
    slide_path = os.path.join(SLIDE_DIR, slide_files[0])
    slide = openslide.OpenSlide(slide_path)
    max_w, max_h = slide.dimensions
    
    thumb_level = slide.level_count - 1 
    thumb_w, thumb_h = slide.level_dimensions[thumb_level]
    thumbnail = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h)).convert("RGB")
    cv_thumb = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)

    # 5. Extract Multimodal Attention Weights
    with torch.no_grad():
        logits, A = model(features, clinical_tensor)
        
    attention_scores = A.squeeze().cpu().numpy() 
    prob = torch.sigmoid(logits).item()
    diagnosis = "Metastatic" if prob >= 0.5 else "Benign"
    print(f"-> Multimodal Diagnosis: {diagnosis} ({prob*100:.2f}% Confidence)")
    
    # 6. Map Attention to Spatial Coordinates
    grid_w = (max_w // PATCH_SIZE) + 1
    grid_h = (max_h // PATCH_SIZE) + 1
    prob_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    normalized_attention = attention_scores / np.max(attention_scores)
    
    for score, (x, y) in zip(normalized_attention, coords):
        grid_x = int(x) // PATCH_SIZE
        grid_y = int(y) // PATCH_SIZE
        prob_map[grid_y, grid_x] = score
        
    # 7. Render and Blend
    heatmap_8bit = np.uint8(255 * prob_map)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_color, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(cv_thumb, 0.6, heatmap_resized, 0.4, 0)
    
    side_by_side = np.hstack((cv_thumb, overlay))

    output_filename = f"multimodal_overlay_{patient_pt_file.replace('.pt', '.jpg')}"
    cv2.imwrite(output_filename, side_by_side)
    print(f"SUCCESS! Overlay saved to {output_filename}")

if __name__ == "__main__":
    print("--- Scanning for MIL Bags ---")
    bag_files = [f for f in os.listdir(BAG_DIR) if f.endswith('.pt')]
    clinical_df = pd.read_csv(METADATA_CSV)
    
    if not bag_files:
        print(f"[!] Error: No .pt bags found.")
    else:
        print(f"Found {len(bag_files)} patient bags. Generating multimodal overlays...")
        for target_bag in bag_files:
            generate_clinical_overlay(target_bag, clinical_df)
            
    print("\n--- Batch Overlay Generation Complete! ---")