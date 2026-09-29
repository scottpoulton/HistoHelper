import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve, f1_score
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import get_histo_model

# --- Configuration ---
CSV_PATH = "ml_datasets/metadata.csv"
WEIGHTS_PATH = "histohelper_production.pth"
BATCH_SIZE = 64

class PCamEvalDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['file_path']
        label = self.data.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_clinical_metrics():
    print("--- HistoHelper V3: Advanced Clinical Evaluation Suite ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluation Device: {device.type.upper()}")

    model = get_histo_model(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("Successfully loaded production weights.")
    except FileNotFoundError:
        print(f"Error: {WEIGHTS_PATH} not found. Wait for train.py to finish!")
        return
        
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PCamEvalDataset(CSV_PATH, transform=transform)
    eval_size = int(len(dataset) * 0.2)
    _, eval_subset = torch.utils.data.random_split(dataset, [len(dataset) - eval_size, eval_size])
    
    dataloader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = [] # NEW: We need raw probabilities for PR-AUC

    print(f"Evaluating on {len(eval_subset)} clinical patches...")

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # Get raw probabilities via Softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 1. Calculate Advanced Metrics ---
    pr_auc = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n--- Clinical Classification Report ---")
    target_names = ['Benign (0)', 'Malignant (1)']
    print(classification_report(all_labels, all_preds, target_names=target_names))
    print(f"-> Model PR-AUC Score: {pr_auc:.4f}")
    print(f"-> Model F1-Score: {f1:.4f}")

    # --- 2. Plotting the Data ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot A: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax1)
    ax1.set_ylabel('Actual Pathologist Diagnosis')
    ax1.set_xlabel('AI Prediction')
    ax1.set_title('Confusion Matrix')

    # Plot B: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    ax2.plot(recall, precision, color='purple', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision (Positive Predictive Value)')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    output_img = "clinical_evaluation_matrix.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\nSaved clinical matrix and PR curve to {output_img}")

if __name__ == "__main__":
    evaluate_clinical_metrics()