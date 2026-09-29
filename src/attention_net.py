import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================================
# 1. The AB-MIL Architecture (Multimodal)
# ==========================================
class MultimodalAttentionMIL(nn.Module):
    def __init__(self, img_input_dim=512, hidden_dim=256, clinical_dim=5, clinical_hidden=32, num_classes=1):
        super(MultimodalAttentionMIL, self).__init__()
        
        self.L = img_input_dim
        self.D = hidden_dim
        self.C = clinical_dim
        self.C_H = clinical_hidden

        # --- STREAM 1: Image Feature Compressor ---
        self.feature_compressor = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Attention Mechanism (Image Only)
        self.attention = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1) 
        )

        # --- STREAM 2: Clinical Data Processor ---
        self.clinical_network = nn.Sequential(
            nn.Linear(self.C, self.C_H),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # --- FUSION: Final Multimodal Classifier ---
        self.classifier = nn.Linear(self.D + self.C_H, num_classes)

    def forward(self, bag_features, clinical_features):
        # 1. Process the WSI Bag
        compressed_features = self.feature_compressor(bag_features) 
        A_raw = self.attention(compressed_features) 
        A_raw = torch.transpose(A_raw, 1, 0) 
        A = F.softmax(A_raw, dim=1) 
        M = torch.mm(A, compressed_features) # Shape: [1, 256]

        # 2. Process Clinical Metadata
        C_out = self.clinical_network(clinical_features) # Shape: [1, 32]

        # 3. Fusion (Concatenation)
        fusion = torch.cat((M, C_out), dim=1) # Shape: [1, 288]

        # 4. Final Prediction
        logits = self.classifier(fusion) 
        return logits, A

# ==========================================
# 2. The DataLoader Strategy
# ==========================================
class WSIBagDataset(Dataset):
    def __init__(self, bag_dir, csv_path="./ml_datasets/tcga/tcga_skcm_clinical.csv"):
        self.bag_dir = bag_dir
        self.bag_files = [f for f in os.listdir(bag_dir) if f.endswith('.pt')]
        self.clinical_df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.bag_files)

    def __getitem__(self, idx):
        file_name = self.bag_files[idx]
        file_path = os.path.join(self.bag_dir, file_name)

        # 1. Load the WSI Bag
        patient_data = torch.load(file_path)
        if isinstance(patient_data, dict):
            bag_tensor = patient_data["features"]
        else:
            bag_tensor = patient_data

        # 2. Extract Label
        tropism_label = file_name.split('_')[1].replace('.pt', '')
        label = 0.0 if tropism_label == 'M0' else 1.0
        label_tensor = torch.tensor([label], dtype=torch.float32)

        # 3. Extract Clinical Data (The New Multimodal Stream)
        patient_id = file_name.split('_')[0]
        patient_record = self.clinical_df[self.clinical_df['Patient_ID'] == patient_id]

        # Default 5-dim tensor: [Skin, Lymph, Lung, Brain, Other]
        clinical_vector = [0.0, 0.0, 0.0, 0.0, 0.0]

        if not patient_record.empty:
            origin = str(patient_record.iloc[0]['Origin_Tissue']).lower()
            if 'skin' in origin:
                clinical_vector[0] = 1.0
            elif 'lymph' in origin:
                clinical_vector[1] = 1.0
            elif 'lung' in origin:
                clinical_vector[2] = 1.0
            elif 'brain' in origin:
                clinical_vector[3] = 1.0
            else:
                clinical_vector[4] = 1.0
        else:
            clinical_vector[4] = 1.0

        clinical_tensor = torch.tensor(clinical_vector, dtype=torch.float32)

        return bag_tensor, clinical_tensor, label_tensor

def get_mil_dataloaders(bag_dir, split_ratio=0.8):
    """Splits the WSI bags using Stratified Logic to prevent NaN AUC."""
    dataset = WSIBagDataset(bag_dir)
    
    m0_indices = [i for i, f in enumerate(dataset.bag_files) if 'M0' in f]
    m1_indices = [i for i, f in enumerate(dataset.bag_files) if 'M0' not in f]
    
    import random
    random.seed(42)
    random.shuffle(m0_indices)
    random.shuffle(m1_indices)
    
    val_m1_count = max(1, int(len(m1_indices) * (1 - split_ratio)))
    val_m0_count = int(len(dataset) * (1 - split_ratio)) - val_m1_count
    
    val_indices = m1_indices[:val_m1_count] + m0_indices[:val_m0_count]
    train_indices = m1_indices[val_m1_count:] + m0_indices[val_m0_count:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    return train_loader, val_loader


# ==========================================
# 3. The Training Loop Skeleton
# ==========================================
def train_ab_mil(bag_dir="./ml_datasets/tcga/mil_bags", epochs=200, accum_steps=8):
    print("--- HistoHelper V3: AB-MIL Training & Validation Phase ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device.type.upper()}")

    model = MultimodalAttentionMIL(img_input_dim=512, clinical_dim=5).to(device)
    train_loader, val_loader = get_mil_dataloaders(bag_dir)
    
    print(f"Dataset Split: {len(train_loader)} Train Bags | {len(val_loader)} Val Bags")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)

    best_val_auc = 0.0  
    epochs_without_improvement = 0

    for epoch in range(epochs):

        # ==========================
        # TRAINING PHASE
        # ==========================
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (bag, clinical, label) in enumerate(train_loader):
            bag = bag.squeeze(0).to(device) 
            clinical = clinical.to(device) 
            label = label.to(device)

            noise = torch.randn_like(bag) * 0.01
            bag = bag + noise

            logits, attention_weights = model(bag, clinical)
            loss = criterion(logits, label)
            
            loss = loss / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps

        if len(train_loader) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / len(train_loader)

        # ==========================
        # VALIDATION PHASE
        # ==========================
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for bag, clinical, label in val_loader:
                bag = bag.squeeze(0).to(device)
                clinical = clinical.to(device)
                label = label.to(device)

                logits, _ = model(bag, clinical)
                loss = criterion(logits, label)
                val_loss += loss.item()

                prob = torch.sigmoid(logits).item()
                pred = 1.0 if prob >= 0.5 else 0.0

                all_labels.append(label.item())
                all_probs.append(prob)
                all_preds.append(pred)

        val_loss /= len(val_loader)
        
        val_acc = accuracy_score(all_labels, all_preds)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

       # --- SMART EARLY STOPPING ---
        patience_limit = 25 

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "histohelper_ab_mil_best.pth")
            print(f" -> New best model saved! (AUC: {best_val_auc:.4f})")
            epochs_without_improvement = 0 
        else:
            epochs_without_improvement += 1
            print(f" -> No improvement. Patience: {epochs_without_improvement}/{patience_limit}")

        if epochs_without_improvement >= patience_limit:
            print(f"\n Early stopping triggered! The model has peaked at AUC {best_val_auc:.4f}.")
            break

    print("\nTraining complete. Best AB-MIL weights secured.")

if __name__ == "__main__":
    train_ab_mil()