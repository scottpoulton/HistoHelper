import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_dataloaders
from model import get_histo_model

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001      # Slightly lower than before for stability with massive datasets
# MINI_EPOCH_BATCHES = 500  # Safety valve: Train on 500 batches per epoch (~4000 images)
# VAL_BATCHES = 100         # Safety valve: Validate on 100 batches

def train_model():
    print("--- Starting HistoHelper Production Training ---")
    
    # 1. Hardware Check
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Utilizing Compute Device: {device.type.upper()}")

    # 2. Load DataLoaders (from our new CSV pipeline)
    print("Loading DataLoaders...")
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    print("Initializing Model...")
    model = get_histo_model(num_classes=2).to(device)

    # 3. Define the Teacher (Loss) and the Mechanic (Optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Track the best validation score to prevent overfitting
    best_val_loss = float('inf')

    print("\n--- Beginning Training Phase ---")
    for epoch in range(EPOCHS):
        
        # ==========================================
        # 1. TRAINING PHASE (Learning)
        # ==========================================
        model.train() # Unfreeze weights
        running_train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
            # Print an update every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)

        # ==========================================
        # 2. VALIDATION PHASE (Testing)
        # ==========================================
        print("\n--- Evaluating on Validation Set ---")
        model.eval() # Freeze weights (no learning, just guessing)
        running_val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad(): # Don't track gradients, saves memory
            for val_idx, (val_images, val_labels) in enumerate(val_loader):
                    
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                
                v_loss = criterion(val_outputs, val_labels)
                running_val_loss += v_loss.item()
                
                # Calculate basic accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                total_preds += val_labels.size(0)
                correct_preds += (predicted == val_labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_preds / total_preds

        scheduler.step(avg_val_loss)

        print(f"-> Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        # ==========================================
        # 3. SAVE THE BEST BRAIN
        # ==========================================
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving weights!")
            best_val_loss = avg_val_loss
            # Note: Saving as a new file so we don't accidentally overwrite your MVP mock
            torch.save(model.state_dict(), "histohelper_production.pth")
        else:
            print(f"Validation loss did not improve. Weights not saved to prevent overfitting.")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    train_model()