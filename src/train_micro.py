import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# --- Configuration ---
OUTPUT_DIR = "./ml_datasets"

# 1. Create a Custom Micro-Dataset
class MicroHistoDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        # Grab all the images we just downloaded
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        # Mathematical filters required by ResNet18
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        tensor_image = self.transform(image)
        
        # For the micro-overfit test, we don't care about true medical labels. 
        # We just assign alternating 0s and 1s to prove the math can learn differences.
        dummy_label = idx % 2 
        return tensor_image, dummy_label

def run_micro_overfit():
    print("--- Initiating Phase 1: The Micro-Overfit Test ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Utilizing Silicon Compute: {device}")

    # Load Data
    dataset = MicroHistoDataset(IMAGE_DIR)
    
    if len(dataset) == 0:
        print("Error: No images found in ./data/images. Check your download script.")
        return
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Loaded {len(dataset)} SPIDER-skin slides into memory.")

    # Initialize Brain
    print("Loading ResNet18 architecture...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Commencing Memorization Phase ---")
    # Train for 10 epochs. We WANT it to memorize.
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch}/10] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    print("\nPhase 1 Complete! If Accuracy hit 100%, the SPIDER-skin data format is fully compatible.")

if __name__ == "__main__":
    run_micro_overfit()