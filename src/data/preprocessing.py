# ======================================================================================
#  Written by Brandon Pyle
#  This file manages the preprocessing of the dataset.
# ======================================================================================

from dataset import load_dataset
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms

KNOWN_CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
TARGET_SIZE = (96, 96)

class WaferMapDataset(Dataset):
    """
        This is the class for the WM-811K wafer map dataset.
        
        Expects resized and normalized wafer maps.
        Stacks into 3 channels for pretrained model compatibility.
    """
    def __init__(self, preprocessed_maps, labels, transform=None):
        self.maps = preprocessed_maps
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        wm = self.maps[idx]
        
        # Stack to 3 channels  -> (3, H, W)
        img = np.stack([wm] * 3, axis=0)
        img = torch.tensor(img, dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

def preprocess_wafer_maps(wafer_maps, target_size=TARGET_SIZE):
    """
        Resizes all wafer maps to a uniform size and convert to float32.
    """
    preprocessed = []
    for i, wm in enumerate(wafer_maps):
        arr = wm.astype(np.float32)
        arr = resize(arr, target_size, anti_aliasing=True,
                     preserve_range=True).astype(np.float32)
        arr = arr / 2.0  # Normalise to [0, 1]
        preprocessed.append(arr)
        if (i + 1) % 25000 == 0:
            print(f"  Preprocessed {i+1:,} / {len(wafer_maps):,} maps...")
    return preprocessed

def preprocess_data():
    data = load_dataset()

    labeled_data_only = data['failureClass'].isin(KNOWN_CLASSES)
    data_subset = data[labeled_data_only].reset_index(drop=True)

    print(f"Labeled wafers: {len(data_subset):,}  (out of {len(data):,} total)")
    print(f"Dropped: {len(data) - len(data_subset):,} unlabeled / unknown wafers")

    class_dist = data_subset['failureClass'].value_counts()

    print("\n--- Failure Class Distribution (After Preprocessing) ---")
    print(class_dist.to_string())

    majority = class_dist.max()
    minority = class_dist.min()

    print("\n--- Imbalance Check ---")
    print(f"Imbalance ratio (majority / minority): {majority / minority:.1f}x")

    train_transform = image_transform()

    return data_subset, train_transform

def image_transform():
    # Data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])

    print("\n--- Image Transform Info ---")
    print(f"Target image size: {TARGET_SIZE}")
    print(f"Channels: 3 (replicated grayscale)")
    print(f"Normalization: pixel / 2.0  ->  [0, 1]")
    print(f"Train augmentations: HFlip, VFlip, Rotation(+/-15), Translate(5%)")
    print(f"Resize strategy: All maps resized ONCE upfront (not per-batch)")

    return train_transform