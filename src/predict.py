import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model import get_histo_model

# --- Configuration ---
IMG_DIR = "spider_mvp_samples"
MODEL_WEIGHTS = "histohelper_mvp.pth"
CLASSES = ["Primary Melanoma (0)", "Metastatic Melanoma (1)"]

def predict_random_image():
    print("--- HistoHelper Inference Engine MVP ---")
    
    # 1. Hardware Check
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. Load the Brain & The Memories (Weights)
    print("Loading trained weights into architecture...")
    model = get_histo_model(num_classes=2)
    
    # Load the .pth file. 'map_location' ensures it loads correctly whether on MPS or CPU
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    
    # CRITICAL: Put the model in "test" mode. This freezes the weights.
    model.eval() 
    
    # 3. Define the exact same mathematical filter used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Pick a random image from our downloaded samples
    all_images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    if not all_images:
        print("No images found to test!")
        return
        
    test_image_name = random.choice(all_images)
    test_image_path = os.path.join(IMG_DIR, test_image_name)
    print(f"\nAnalyzing Image: {test_image_name}")
    
    # 5. Process the image
    image = Image.open(test_image_path).convert("RGB")
    # A neural network expects a "batch". We use unsqueeze(0) to turn 1 image into a batch of 1.
    input_tensor = transform(image).unsqueeze(0) 
    input_tensor = input_tensor.to(device)
    
    # 6. The Prediction!
    # torch.no_grad() tells PyTorch not to track math for training, saving massive memory
    with torch.no_grad(): 
        logits = model(input_tensor)
        
        # Convert raw logits into percentages (0.0 to 1.0)
        probabilities = F.softmax(logits, dim=1)[0]
        
        # Find the highest percentage and its index (0 or 1)
        confidence, predicted_class_idx = torch.max(probabilities, dim=0)
        
    # 7. Output the Result
    prediction_label = CLASSES[predicted_class_idx.item()]
    confidence_pct = confidence.item() * 100
    
    print("\n--- Diagnostic Output ---")
    print(f"Prediction: {prediction_label}")
    print(f"Confidence: {confidence_pct:.2f}%")
    print("-------------------------")

if __name__ == "__main__":
    predict_random_image()