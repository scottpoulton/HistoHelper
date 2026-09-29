import os
import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_haem_dataloaders(data_dir="./data/haematology", batch_size=32):
    print("Initialising Haematology ImageFolder pipeline...")
    
    # 1. Define the mathematical filters (Same as your histology model!)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the dataset automatically from the folder structure
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
        classes = full_dataset.classes
        print(f"Successfully mapped {len(classes)} cell types: {classes}")
    except Exception as e:
        print(f"Error loading images. Did you download them to {data_dir} yet?")
        return None, None, None

    # 3. Split the data (80% Training, 20% Validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset = copy.copy(full_dataset)
    
    # Safely overrides ONLY the validation data
    val_dataset.dataset.transform = val_transforms

    # 4. Build the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Blood Smear DataLoaders ready! Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, classes

if __name__ == "__main__":
    get_haem_dataloaders()