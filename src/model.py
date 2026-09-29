import torch
import torch.nn as nn
from torchvision import models

def get_histo_model(num_classes=2):
    """
    Loads a pre-trained ResNet18 model and modifies the final layer
    for our specific binary classification task (Primary vs Metastatic).
    """
    print("Downloading pre-trained ResNet18 weights...")
    # 1. Load the pre-trained ResNet18 architecture
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Find out how many inputs the final Fully Connected (fc) layer takes
    num_ftrs = model.fc.in_features
    
    # 3. Replace the final layer with a new one that outputs exactly 'num_classes'
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), # Randomly drops 50% of connections to prevent overfitting
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model

if __name__ == "__main__":
    print("--- Initializing HistoHelper Architecture MVP ---")
    
    # Instantiate our modified model
    histo_model = get_histo_model(num_classes=2)
    
    # Create a dummy "batch" of 8 images (just like the ones from our DataLoader)
    # Shape: (Batch Size=8, Channels=3, Height=224, Width=224)
    dummy_tensor = torch.randn(8, 3, 224, 224)
    
    print("\nFeeding dummy batch to the network...")
    
    # Pass the dummy images through the model
    predictions = histo_model(dummy_tensor)
    
    print("\n--- Network Output ---")
    print(f"Output Tensor Shape: {predictions.shape}")
    print("   -> (Batch Size, Number of Classes)")
    print("\nExample raw 'logits' for the first image:", predictions[0].detach().numpy())
    print("(These numbers will be fed into a probability function later!)")