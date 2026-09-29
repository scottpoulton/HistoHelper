import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model import get_histo_model

class HistoGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 1. Register Hooks to intercept the math during forward/backward passes
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Grab the gradients flowing back from the classification layer
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 2. Forward pass (No torch.no_grad() here!)
        logits = self.model(x)
        
        if class_idx is None:
            # Default to the model's highest confidence prediction
            class_idx = logits.argmax(dim=1).item()

        # 3. Backward pass to calculate gradients for the target class
        self.model.zero_grad()
        class_score = logits[0, class_idx]
        class_score.backward()

        # 4. The Grad-CAM Math
        # Average the gradients spatially
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weigh the activations by the pooled gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels to create a single 2D heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # Apply ReLU to only keep features with a positive influence on the prediction
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap between 0 and 1
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().detach().numpy(), class_idx

def generate_heatmap(image_path, model_weights_path="histohelper_production.pth"):
    print("--- HistoHelper Grad-CAM Interpreter ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load your existing Brain
    model = get_histo_model(num_classes=2)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval() # Keep dropout/batchnorm frozen

    # 2. Initialize Grad-CAM on ResNet18's final convolutional layer ('layer4')
    target_layer = model.layer4[-1]
    cam = HistoGradCAM(model, target_layer)

    # 3. Prepare the image using your exact clinical pipeline math
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_img = Image.open(image_path).convert("RGB")
    original_img_resized = original_img.resize((224, 224))
    
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 4. Generate the Heatmap Data
    # Note: We must allow gradient tracking, so no `with torch.no_grad():` block
    print(f"Analyzing spatial features for: {image_path}")
    heatmap, predicted_class = cam(input_tensor)
    
    CLASSES = ["Benign Tissue", "Malignant Melanoma"]
    print(f"Model Prediction: {CLASSES[predicted_class]}")

    # 5. Process the Overlay using OpenCV
    # Convert PIL image to OpenCV format (RGB to BGR)
    cv_img = cv2.cvtColor(np.array(original_img_resized), cv2.COLOR_RGB2BGR)
    
    # Resize the 7x7 ResNet feature map back up to 224x224
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # Convert heatmap to 8-bit integer (0-255) and apply the JET colormap (Blue=Cold, Red=Hot)
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    
    # Blend the heatmap with the original image (0.4 opacity for the heatmap)
    overlay = cv2.addWeighted(cv_img, 0.6, heatmap_color, 0.4, 0)

    # Save the output
    output_filename = "gradcam_output.jpg"
    cv2.imwrite(output_filename, overlay)
    print(f"Success! Heatmap saved as {output_filename}")

import os
import random

if __name__ == "__main__":
    # Get the absolute path to the scripts/histohelper/ directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Step up two levels to the root of the repository
    REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    
    # Target the samples folder at the root
    SAMPLES_DIR = os.path.join(REPO_ROOT, "spider_mvp_samples")
    
    # 1. Automatically find all JPGs in the folder
    try:
        all_images = [f for f in os.listdir(SAMPLES_DIR) if f.endswith('.jpg')]
        if not all_images:
            raise FileNotFoundError(f"No JPG images found in {SAMPLES_DIR}")
            
        # 2. Pick the first image found (or use random.choice(all_images))
        target_filename = all_images[0] 
        
        IMAGE_PATH = os.path.join(SAMPLES_DIR, target_filename)
        WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "histohelper_production.pth") 
        
        print(f"Auto-selected image: {target_filename}")
        
        # Run the interpreter
        generate_heatmap(IMAGE_PATH, WEIGHTS_PATH)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Did you run 'python3 scripts/histohelper/fetch_spider_sample.py' to download the images first?")