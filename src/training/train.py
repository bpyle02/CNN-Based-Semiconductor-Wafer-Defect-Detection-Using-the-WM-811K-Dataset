import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import time

# Add src to path
sys.path.append(str(Path(__file__).parents[2]))

from src.data.preprocessing import preprocess_data, WaferMapDataset
from src.models.wafer_cnn import get_model

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def get_class_weights(labels):
    class_counts = np.bincount(labels)
    weights = 1. / class_counts
    samples_weights = weights[labels]
    return torch.from_numpy(samples_weights)

def train_hybrid_model(model_name="resnet18", epochs=10):
    print(f"\n{'='*20} Training Hybrid Model: {model_name} {'='*20}")
    
    # 1. Load data with geometric features
    preprocessed_maps, labels, label_encoder, train_transform, geom_features = preprocess_data()
    
    # 2. Split data
    indices = np.arange(len(labels))
    train_idx_val, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx_val, test_size=0.15, random_state=42, stratify=labels[train_idx_val])

    # 3. Create Datasets
    train_dataset = WaferMapDataset([preprocessed_maps[i] for i in train_idx], labels[train_idx], geom_features[train_idx], transform=train_transform)
    val_dataset = WaferMapDataset([preprocessed_maps[i] for i in val_idx], labels[val_idx], geom_features[val_idx])
    test_dataset = WaferMapDataset([preprocessed_maps[i] for i in test_idx], labels[test_idx], geom_features[test_idx])

    # 4. Handle imbalance
    weights = get_class_weights(labels[train_idx])
    sampler = WeightedRandomSampler(weights, len(weights))

    # 5. DataLoaders
    BATCH_SIZE = 256
    NUM_WORKERS = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 6. Device and Model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = get_model(model_name=model_name, num_classes=len(label_encoder.classes_), num_geom_features=6).to(device)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=epochs)

    best_val_acc = 0.0
    model_save_path = Path(__file__).parents[2] / "models" / f"best_{model_name}_hybrid.pth"
    model_save_path.parent.mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for (img, geom), target in pbar:
            img, geom, target = img.to(device), geom.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model((img, geom))
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for (img, geom), target in val_loader:
                img, geom, target = img.to(device), geom.to(device), target.to(device)
                outputs = model((img, geom))
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print("New best model saved!")

    # Final test
    print("\n--- Final Hybrid Model Test ---")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for (img, geom), target in test_loader:
            img, geom, target = img.to(device), geom.to(device), target.to(device)
            outputs = model((img, geom))
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    print(f"Hybrid Test Accuracy: {100 * test_correct / test_total:.2f}%")

    with open(model_save_path.parent / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    train_hybrid_model(model_name="resnet18", epochs=10)
