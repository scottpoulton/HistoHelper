import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model import get_histo_model

# --- Configuration ---
TILE_DIR = "extracted_tiles"
WEIGHTS_PATH = "histohelper_production.pth"
PATCH_SIZE = 224

# 1. Custom Dataset to parse the X, Y coordinates from your filenames
class WSITileDataset(Dataset):
    def __init__(self, tile_dir, transform):
        self.tile_dir = tile_dir
        # Only grab the patches we successfully extracted
        self.tiles = [f for f in os.listdir(tile_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        filename = self.tiles[idx]
        # Filename format from our tiler: patch_X_Y.jpg
        parts = filename.replace('.jpg', '').split('_')
        x, y = int(parts[1]), int(parts[2])
        
        img_path = os.path.join(self.tile_dir, filename)
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        
        return tensor, x, y

def stitch_macroscopic_heatmap():
    print("--- HistoHelper V2: Macroscopic WSI Inference ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device.type.upper()}")

    # 2. Load your plateaued (safely stopped) model
    print("Loading production weights...")
    model = get_histo_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval() # CRITICAL: Freeze weights so it doesn't learn, only predicts

    # Mathematical filters
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Initialize the Batch Loader
    dataset = WSITileDataset(TILE_DIR, transform)
    # Batch size 64 maximizes the M-series GPU memory bandwidth
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    if len(dataset) == 0:
        print("No patches found! Did the tiler run successfully?")
        return

    # Find the maximum X and Y to know how big to make our blank heatmap canvas
    max_x = max([dataset[i][1] for i in range(len(dataset))])
    max_y = max([dataset[i][2] for i in range(len(dataset))])
    
    # Calculate grid dimensions (1 pixel in our heatmap = 1 entire 224x224 patch)
    grid_w = (max_x // PATCH_SIZE) + 1
    grid_h = (max_y // PATCH_SIZE) + 1
    
    # Create a blank probability map (0.0 means empty glass)
    prob_map = np.zeros((grid_h, grid_w), dtype=np.float32)

    print(f"Commencing batch inference on {len(dataset)} patches...")
    
    # 4. Fast Inference Loop
    with torch.no_grad(): # Don't track gradients (saves massive amounts of memory)
        for batch_idx, (tensors, xs, ys) in enumerate(dataloader):
            tensors = tensors.to(device)
            
            # Get raw predictions
            logits = model(tensors)
            
            # Convert to percentages and grab the probability of Class 1 (Metastatic/Malignant)
            probs = F.softmax(logits, dim=1)[:, 1] 
            
            # Move data back to CPU for numpy operations
            probs = probs.cpu().numpy()
            xs = xs.numpy()
            ys = ys.numpy()
            
            # Map each patch's probability back to its specific X, Y grid coordinate
            for p, x, y in zip(probs, xs, ys):
                grid_x = x // PATCH_SIZE
                grid_y = y // PATCH_SIZE
                prob_map[grid_y, grid_x] = p
                
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * 64} / {len(dataset)} patches...")

    # 5. Render the Heatmap via OpenCV
    print("Stitching macroscopic heatmap...")
    
    # Normalize probabilities (0.0 to 1.0) into 8-bit image pixels (0 to 255)
    heatmap_8bit = np.uint8(255 * prob_map)
    
    # Apply JET colormap (Blue = 0% Malignant, Red = 100% Malignant)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    
    # Resize the tiny grid up a bit so it's easier to view on your monitor
    # (Using NEAREST interpolation keeps the exact grid squares crisp, rather than blurring them)
    display_scale = 10 
    heatmap_display = cv2.resize(heatmap_color, 
                                 (grid_w * display_scale, grid_h * display_scale), 
                                 interpolation=cv2.INTER_NEAREST)

    cv2.imwrite("wsi_heatmap_result.jpg", heatmap_display)
    print("SUCCESS! Map saved to wsi_heatmap_result.jpg")

if __name__ == "__main__":
    stitch_macroscopic_heatmap()