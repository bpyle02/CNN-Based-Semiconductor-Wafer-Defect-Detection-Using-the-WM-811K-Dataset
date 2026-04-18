# ======================================================================================
#  Written by Brandon Pyle
#  This file manages the preprocessing of the dataset.
# ======================================================================================

from src.data.dataset import load_dataset
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter
import albumentations as A
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

KNOWN_CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
TARGET_SIZE = (96, 96)

def extract_geometric_features(wm):
    """
    Extracts geometric features from a wafer map to help classify thin defects like scratches.
    """
    mask = (wm == 2).astype(np.uint8)
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    if not regions:
        return np.zeros(6, dtype=np.float32)
    
    largest_region = max(regions, key=lambda r: r.area)
    features = [
        largest_region.area,
        largest_region.eccentricity,
        largest_region.orientation,
        largest_region.solidity,
        largest_region.extent,
        largest_region.perimeter
    ]
    return np.array(features, dtype=np.float32)

class WaferMapDataset(Dataset):
    def __init__(self, preprocessed_maps, labels, geometric_features=None, transform=None):
        self.maps = preprocessed_maps
        self.labels = labels
        self.geometric_features = geometric_features
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        wm = self.maps[idx]
        
        # Albumentations expects HWC format, but we have HW
        # We'll treat it as a single channel image first
        img = np.stack([wm] * 3, axis=-1) # (H, W, 3)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        # Convert to (C, H, W) for PyTorch
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.geometric_features is not None:
            geom = torch.tensor(self.geometric_features[idx], dtype=torch.float32)
            return (img, geom), label
            
        return img, label

def preprocess_wafer_maps(wafer_maps, target_size=TARGET_SIZE):
    preprocessed = []
    for i, wm in enumerate(wafer_maps):
        wm_float = wm.astype(np.float32)
        filtered = median_filter(wm_float, size=3)
        arr = 0.7 * filtered + 0.3 * wm_float
        arr = resize(arr, target_size, anti_aliasing=True, preserve_range=True).astype(np.float32)
        arr = arr / 2.0  
        preprocessed.append(arr)
        if (i + 1) % 25000 == 0:
            print(f"  Preprocessed {i+1:,} / {len(wafer_maps):,} maps...")
    return preprocessed

def encode_labels(labels):
    le = LabelEncoder()
    le.fit(KNOWN_CLASSES)
    return le.transform(labels), le

def preprocess_data():
    data = load_dataset()
    labeled_data_only = data['failureClass'].isin(KNOWN_CLASSES)
    data_subset = data[labeled_data_only].reset_index(drop=True)

    print(f"Labeled wafers: {len(data_subset):,}")

    wafer_maps = data_subset['waferMap'].values
    preprocessed_maps = preprocess_wafer_maps(wafer_maps)
    
    print("\n--- Extracting Geometric Features ---")
    geometric_features = [extract_geometric_features(wm) for wm in wafer_maps]
    geometric_features = np.array(geometric_features, dtype=np.float32)
    
    geom_mean = geometric_features.mean(axis=0)
    geom_std = geometric_features.std(axis=0) + 1e-6
    geometric_features = (geometric_features - geom_mean) / geom_std

    labels, label_encoder = encode_labels(data_subset['failureClass'])
    train_transform = get_train_transforms()

    return preprocessed_maps, labels, label_encoder, train_transform, geometric_features

def get_train_transforms():
    """
    Industry standard augmentations using Albumentations.
    Includes ElasticTransform and GridDistortion which are great for simulating 
    physical defect variations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    ])
