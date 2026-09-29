import torch
from dataset_loader import get_dataloaders
from model import get_histo_model

# --- Configuration ---
WEIGHTS_PATH = "histohelper_production.pth"
BATCH_SIZE = 32

def evaluate_model():
    print("--- HistoHelper Clinical Evaluation ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device.type.upper()}")

    # 1. Load Data & Model
    print("Loading Validation Data...")
    _, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    print("Loading Production Brain...")
    model = get_histo_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval() # Freeze weights

    # 2. Trackers
    tp = 0  # True Positives (Caught the Melanoma)
    tn = 0  # True Negatives (Correctly identified Benign)
    fp = 0  # False Positives (Called Benign tissue Melanoma - False Alarm)
    fn = 0  # False Negatives (Missed the Melanoma - Fatal Error)

    print("\nRunning inference on validation set. This may take a moment...")
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Class 1 = Metastatic/Melanoma, Class 0 = Normal/Benign
                if true_label == 1 and pred_label == 1:
                    tp += 1
                elif true_label == 0 and pred_label == 0:
                    tn += 1
                elif true_label == 0 and pred_label == 1:
                    fp += 1
                elif true_label == 1 and pred_label == 0:
                    fn += 1

    # 3. Calculate Clinical Metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100
    
    # Avoid division by zero
    sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0

    print("\n--- Diagnostic Report ---")
    print(f"Total Slides Evaluated: {total}")
    print(f"Overall Accuracy:       {accuracy:.2f}%")
    print("-" * 25)
    print(f"True Positives:  {tp} (Correctly flagged Melanoma)")
    print(f"True Negatives:  {tn} (Correctly ignored Benign)")
    print(f"False Positives: {fp} (False Alarms)")
    print(f"False Negatives: {fn} (MISSED MELANOMAS 🚨)")
    print("-" * 25)
    print(f"Sensitivity (Recall): {sensitivity:.2f}%")
    print(f"Specificity:          {specificity:.2f}%")

if __name__ == "__main__":
    evaluate_model()