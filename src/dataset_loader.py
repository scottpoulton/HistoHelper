import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

class SkinLesionDataset(Dataset):
    """Custom PyTorch Dataset for HistoHelper using clinical CSV metadata."""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['file_path']
        # Convert to RGB to prevent 1-channel grayscale or 4-channel RGBA crashes
        image = Image.open(img_path).convert("RGB")
        label = int(self.dataframe.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(csv_path="./ml_datasets/metadata.csv", batch_size=8):
    print("Loading clinical metadata from CSV...")
    df = pd.read_csv(csv_path)

    df['file_path'] = df['file_path'].str.replace('./data/images', './ml_datasets/images', regex=False)

    # 1. Train / Validation Split (80% / 20%) via Pandas
    # Splitting the dataframe first allows us to apply different transforms easily
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 2. Data Augmentation (Morphological & Staining Variance)
    # Train transforms get flipping, rotation, and color jitter to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation MUST NOT be augmented, only resized and normalized
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SkinLesionDataset(train_df, transform=train_transforms)
    val_dataset = SkinLesionDataset(val_df, transform=val_transforms)

    # 3. Class Imbalance Strategy (WeightedRandomSampler)
    print("Calculating class weights for balanced sampling...")
    train_labels = train_df['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    
    # Assign a weight to every individual sample in the training set
    sample_weights = np.array([class_weights[t] for t in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    # 4. Build DataLoaders
    # Note: When using a custom sampler, shuffle MUST be False.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"DataLoaders ready! Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    return train_loader, val_loader

if __name__ == "__main__":
    print("--- Testing HistoHelper Data Pipeline ---")
    # Execute the loader to verify the mathematical pipeline works
    train_loader, val_loader = get_dataloaders()
    
    images, labels = next(iter(train_loader))
    print("\n--- GPU Feed Ready ---")
    print(f"Batch Image Tensor Shape: {images.shape}")
    print(f"Balanced Clinical Labels for this batch: {labels.tolist()}")
    print("-----------------------------------------")