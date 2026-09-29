import os
import cv2
import numpy as np
import base64
import torch
import openslide
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from attention_net import AttentionMIL

# --- Configuration ---
BAG_DIR = "./ml_datasets/tcga/mil_bags"
SLIDE_DIR = "./ml_datasets/tcga/slides"
WEIGHTS_PATH = "histohelper_ab_mil_best.pth"
PATCH_SIZE = 224

# --- SECURITY SETUP ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# In production, set this in your Google Cloud Run environment variables.
# For local dev, it falls back to 'dev_key_123'.
EXPECTED_API_KEY = os.environ.get("HISTOHELPER_API_KEY", "dev_key_123")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials. Unauthorized access.")
    return api_key

app = FastAPI(title="HistoHelper V3 API", version="3.0")

# --- STRICT CORS LOCKDOWN ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.scottpoulton.com",
        "https://scottpoulton.com",
        "http://localhost:1313",
        "http://127.0.0.1:1313"
    ],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"], # Only allow required headers
)

print("Loading V3 AB-MIL Model...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AttentionMIL(input_dim=512).to(device)

try:
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
except FileNotFoundError:
    print("[!] Warning: V3 Weights not found. Awaiting training completion.")

model.eval()

class PatientRequest(BaseModel):
    patient_id: str

# Inject the security dependency directly into the endpoint
@app.post("/predict_mil")
async def predict_mil(request: PatientRequest, api_key: str = Depends(verify_api_key)):
    patient_id = request.patient_id
    
    try:
        bag_files = [f for f in os.listdir(BAG_DIR) if f.startswith(patient_id)]
        if not bag_files:
            raise HTTPException(status_code=404, detail="Patient feature bag not found.")
            
        bag_path = os.path.join(BAG_DIR, bag_files[0])
        patient_data = torch.load(bag_path)
        features = patient_data["features"].to(device)
        coords = patient_data["coords"].numpy()
        
        slide_files = [f for f in os.listdir(SLIDE_DIR) if f.startswith(patient_id) and f.endswith('.svs')]
        if not slide_files:
            raise HTTPException(status_code=404, detail="Original WSI slide not found.")
            
        slide_path = os.path.join(SLIDE_DIR, slide_files[0])
        slide = openslide.OpenSlide(slide_path)
        max_w, max_h = slide.dimensions
        
        thumb_level = slide.level_count - 1 
        thumb_w, thumb_h = slide.level_dimensions[thumb_level]
        thumbnail = slide.read_region((0, 0), thumb_level, (thumb_w, thumb_h)).convert("RGB")
        cv_thumb = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            logits, A = model(features)
            
        attention_scores = A.squeeze().cpu().numpy() 
        prob = torch.sigmoid(logits).item()
        diagnosis = "Metastatic" if prob >= 0.5 else "Benign"
        
        grid_w = (max_w // PATCH_SIZE) + 1
        grid_h = (max_h // PATCH_SIZE) + 1
        prob_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        normalized_attention = attention_scores / np.max(attention_scores)
        
        triage_regions = []
        for i, (score, (x, y)) in enumerate(zip(normalized_attention, coords)):
            grid_x = int(x) // PATCH_SIZE
            grid_y = int(y) // PATCH_SIZE
            prob_map[grid_y, grid_x] = score
            
            if score > 0.85:
                triage_regions.append({"x": float(x)/max_w, "y": float(y)/max_h, "weight": float(score)})

        triage_regions = sorted(triage_regions, key=lambda k: k['weight'], reverse=True)[:5]
            
        heatmap_8bit = np.uint8(255 * prob_map)
        heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap_color, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(cv_thumb, 0.6, heatmap_resized, 0.4, 0)
        
        _, orig_buffer = cv2.imencode('.jpg', cv_thumb)
        _, heat_buffer = cv2.imencode('.jpg', overlay)
        
        return {
            "success": True,
            "patient_id": patient_id,
            "prediction": diagnosis,
            "confidence": round(prob * 100, 2),
            "original_base64": f"data:image/jpeg;base64,{base64.b64encode(orig_buffer).decode('utf-8')}",
            "heatmap_base64": f"data:image/jpeg;base64,{base64.b64encode(heat_buffer).decode('utf-8')}",
            "triage_regions": triage_regions
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)