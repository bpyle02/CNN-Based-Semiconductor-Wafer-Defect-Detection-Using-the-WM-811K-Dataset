import os
import pickle
import random
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from skimage import feature, filters
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# Configuration
# ==================================================
DATA_PATH = r"G:\My Drive\AI_570\Final_project_Version15April\data\raw\LSWMD_new.pkl"
OUTPUT_PICKLE = r"G:\My Drive\AI_570\Final_project_Version15April\data\processed\preprocessed_wm811k_improved.pkl"

WAFERMAP_COLUMN = "waferMap"
TARGET_COLUMN = "failureType"
SPLIT_COLUMN = "trianTestLabel"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Wafer map resizing configuration
RESIZE_WAFER_MAPS = True
TARGET_WAFER_SIZE = 64

# Class imbalance handling
REMOVE_UNKNOWN_LABELS = True
KEEP_NONE_CLASS = True
DOWNSAMPLE_NONE_CLASS = True
NONE_RATIO_TO_MAX_DEFECT = 1 #  More aggressive downsampling
AUGMENT_MINORITY_CLASSES = True
#MAX_AUGMENT_PER_CLASS = 5000
MAX_AUGMENT_PER_CLASS = 25000

#   Advanced feature extraction
EXTRACT_MORPHOLOGICAL_FEATURES =  True 
EXTRACT_TEXTURE_FEATURES = True
NORMALIZE_WAFER_MAPS = True
USE_WEIGHTED_SAMPLING = True

# ==================================================
# Enhanced Utility Functions
# ==================================================
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)

def is_array_like(x):
    return isinstance(x, (np.ndarray, list, tuple))

def flatten_scalar_cell(x):
    if is_array_like(x):
        arr = np.asarray(x, dtype=object).reshape(-1)
        if arr.size == 0:
            return np.nan
        if arr.size == 1:
            return arr[0]
        return x
    return x

def normalize_label(x):
    x = flatten_scalar_cell(x)
    if isinstance(x, bytes):
        try:
            x = x.decode("utf-8")
        except Exception:
            x = str(x)
    if x is None:
        return "unknown"
    try:
        if pd.isna(x):
            return "unknown"
    except Exception:
        pass
    text = str(x).strip().lower()
    return text if text else "unknown"

def summarize_unique(series, max_items=20):
    values = series.map(lambda x: "unknown" if pd.isna(x) else str(x)).unique().tolist()
    values = sorted(values)
    preview = values[:max_items]
    return preview, len(values)

def safe_to_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def scan_raw_array_columns(df, target_column, split_column):
    raw_array_cols = []
    print("\nScanning for raw array columns...")
    for col in df.columns:
        if col in [target_column, split_column]:
            continue
        sample = df[col].dropna().head(20)
        if sample.empty:
            continue
        has_array = sample.apply(is_array_like).any()
        if not has_array:
            continue
        multi_element = sample.apply(
            lambda x: is_array_like(x) and np.asarray(x, dtype=object).size > 1
        ).any()
        if multi_element:
            raw_array_cols.append(col)
            print(f"  [{col}] raw multi-element array column")
        else:
            df[col] = df[col].apply(flatten_scalar_cell)
            print(f"  [{col}] single-element arrays flattened")
    print(f"\nRaw array columns kept as-is: {raw_array_cols}")
    return df, raw_array_cols

# ==================================================
#  changed did:  Wafer Map Normalization
# ==================================================
def normalize_wafer_map(wafer_map):
    """
    Normalize wafer map to [0, 1] range using min-max scaling.
    Improves feature consistency across different wafer patterns.
    """
    try:
        arr = np.array(wafer_map, dtype=np.float32)
        if arr.size == 0:
            return arr
        
        arr_min = arr.min()
        arr_max = arr.max()
        
        if arr_max == arr_min:
            return np.zeros_like(arr)
        
        normalized = (arr - arr_min) / (arr_max - arr_min)
        return normalized.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to normalize wafer map. Error: {e}")
        return np.array(wafer_map, dtype=np.float32)

# ==================================================
# Changed did: Enhanced Morphological Feature Extraction
# ==================================================
def extract_morphological_features(wafer_map):
    """
    Extract spatial defect features:
    - Defect density
    - Defect spread (center of mass distance)
    - Connectivity (number of defect clusters)
    - Defect concentration in regions
    """
    try:
        arr = np.array(wafer_map, dtype=np.float32)
        if arr.size == 0:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Binary defect map (threshold at median)
        threshold = np.median(arr[arr > 0]) if np.any(arr > 0) else 0
        binary_defect = arr > threshold
        
        # Defect density
        defect_density = binary_defect.sum() / arr.size
        
        # Connectivity: number of separate defect clusters
        labeled_array, num_clusters = ndimage.label(binary_defect)
        
        # Center of mass analysis
        if num_clusters > 0:
            centers = ndimage.center_of_mass(binary_defect)
            if isinstance(centers[0], (list, tuple, np.ndarray)):
                center_y = np.mean([c[0] for c in centers])
                center_x = np.mean([c[1] for c in centers])
            else:
                center_y, center_x = centers
            
            # Distance from center
            center_dist = np.sqrt((center_y - arr.shape[0]/2)**2 + (center_x - arr.shape[1]/2)**2)
        else:
            center_dist = 0
        
        # Spread (standard deviation of defect positions)
        if defect_density > 0:
            y_coords, x_coords = np.where(binary_defect)
            spread = np.std(y_coords) + np.std(x_coords)
        else:
            spread = 0
        
        # Defect concentration in quadrants
        h, w = arr.shape
        quadrants = [
            binary_defect[:h//2, :w//2].sum(),
            binary_defect[:h//2, w//2:].sum(),
            binary_defect[h//2:, :w//2].sum(),
            binary_defect[h//2:, w//2:].sum(),
        ]
        quadrant_std = np.std(quadrants)
        quadrant_max = np.max(quadrants)
        
        return [
            defect_density,
            num_clusters,
            center_dist,
            spread,
            quadrant_std,
            quadrant_max,
            np.mean(quadrants),
            np.max(arr) - np.min(arr)  # Dynamic range
        ]
    except Exception as e:
        print(f"Warning: Failed to extract morphological features. Error: {e}")
        return [0, 0, 0, 0, 0, 0, 0, 0]

# ==================================================
# Changed : Texture Feature Extraction
# ==================================================
def extract_texture_features(wafer_map):
    """
    Extract texture and statistical features:
    - Entropy (disorder measure)
    - Contrast (local intensity variations)
    - Homogeneity
    - Energy (uniformity)
    - Skewness and Kurtosis
    """
    try:
        arr = np.array(wafer_map, dtype=np.float32)
        if arr.size == 0:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Normalize to 0-1 for texture analysis
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        
        # Entropy (disorder)
        hist, _ = np.histogram(arr_norm, bins=256, range=(0, 1))
        hist = hist / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        
        # Edge detection (Sobel)
        sx = ndimage.sobel(arr_norm, axis=0)
        sy = ndimage.sobel(arr_norm, axis=1)
        edges = np.sqrt(sx**2 + sy**2)
        
        # Contrast (edge energy)
        contrast = edges.mean()
        
        # Local variance (texture roughness)
        local_var = ndimage.generic_filter(arr_norm, np.var, size=3)
        homogeneity = 1 / (1 + local_var.mean())
        
        # Energy (uniformity)
        energy = np.sum(arr_norm**2) / arr.size
        
        # Statistical moments
        skewness = stats.skew(arr.flatten())
        kurtosis = stats.kurtosis(arr.flatten())
        
        # Gradient magnitude
        gradient_mag = np.sqrt(sx**2 + sy**2).mean()
        
        return [
            entropy,
            contrast,
            homogeneity,
            energy,
            skewness,
            kurtosis,
            gradient_mag,
            arr_norm.std()
        ]
    except Exception as e:
        print(f"Warning: Failed to extract texture features. Error: {e}")
        return [0, 0, 0, 0, 0, 0, 0, 0]

# ==================================================
# Resize Wafer Map
# ==================================================
def resize_wafer_map(wafer_map, target_size=TARGET_WAFER_SIZE):
    if not RESIZE_WAFER_MAPS or target_size is None:
        return np.array(wafer_map, dtype=np.float32)
    try:
        arr = np.array(wafer_map, dtype=np.float32)
        if arr.ndim != 2:
            return arr
        current_h, current_w = arr.shape
        zoom_h = target_size / current_h
        zoom_w = target_size / current_w
        resized = ndimage.zoom(arr, zoom=(zoom_h, zoom_w), order=1)
        return resized.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to resize wafer map. Error: {e}")
        return np.array(wafer_map, dtype=np.float32)

# ==================================================
# Changed: Advanced Augmentation with Elastic Distortion
# ==================================================
def augment_wafer_map(wafer_map):
    """
    Advanced augmentation with:
    - Rotation
    - Flips
    - Elastic distortion
    - Gaussian blur
    - Noise injection
    """
    arr = np.array(wafer_map, dtype=np.float32)
    if arr.ndim != 2:
        return arr
    
    # Rotation (90° increments for wafer maps)
    k = np.random.choice([0, 1, 2, 3])
    arr = np.rot90(arr, k=k)
    
    # Flips
    if np.random.rand() < 0.5:
        arr = np.flipud(arr)
    if np.random.rand() < 0.5:
        arr = np.fliplr(arr)
    
    #  Elastic distortion
    if np.random.rand() < 0.4:
        alpha = np.random.uniform(5, 15)
        sigma = np.random.uniform(1, 3)
        
        x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
        dx = gaussian_filter((np.random.random(arr.shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.random(arr.shape) * 2 - 1), sigma) * alpha
        
        x_new = x + dx
        y_new = y + dy
        
        x_new = np.clip(x_new, 0, arr.shape[1] - 1)
        y_new = np.clip(y_new, 0, arr.shape[0] - 1)
        
        arr = ndimage.map_coordinates(arr, (y_new, x_new), order=1)
    
    #  Gaussian blur
    if np.random.rand() < 0.2:
        sigma = np.random.uniform(0.5, 1.5)
        arr = gaussian_filter(arr, sigma=sigma)
    
    #  Noise injection
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.01 * arr.std(), arr.shape)
        arr = arr + noise
    
    return np.clip(arr, 0, arr.max()).astype(np.float32)

# ==================================================
#  ENHANCED: Downsampling with Class Weights
# ==================================================
def downsample_none_class(train_df, target_column):
    """
     Downsample the 'none' class to match augmented defect classes
    This creates a more balanced dataset for better Macro F1-score
    
    Key Changes:
    - Uses MAX_AUGMENT_PER_CLASS (25,000) as the target
    - Shows before/after class distribution
    - Calculates balanced class weights
    """
    
    # Skip if downsampling is disabled
    if not DOWNSAMPLE_NONE_CLASS:
        print("  Downsampling disabled. Skipping...")
        return train_df, {}
    
    # Get class distribution
    value_counts = train_df[target_column].value_counts()
    
    # Check if 'none' class exists
    if "none" not in value_counts.index:
        print("  'none' class not found. Skipping downsampling.")
        return train_df, {}
    
    # Get all defect classes (non-none)
    defect_counts = value_counts.drop(labels=["none"], errors="ignore")
    if defect_counts.empty:
        print("  No defect classes found. Skipping downsampling.")
        return train_df, {}
    
    #  CRITICAL: Target the augmented class count, not the original max
    target_none_count = min(
        value_counts["none"],                              # Don't exceed original 'none' count
        int(NONE_RATIO_TO_MAX_DEFECT * MAX_AUGMENT_PER_CLASS)  #  Use MAX_AUGMENT_PER_CLASS!
    )
    
    # ========== BEFORE DOWNSAMPLING ==========
    print("\n" + "="*60)
    print(" CLASS DISTRIBUTION BEFORE DOWNSAMPLING")
    print("="*60)
    print(f"{'Class':<20} {'Count':<15} {'Ratio':<15}")
    print("-"*60)
    for label in sorted(value_counts.index):
        count = value_counts[label]
        ratio = count / value_counts["none"] if label != "none" else 1.0
        print(f"{label:<20} {count:<15} {ratio:<15.2f}x")
    print("-"*60)
    print(f"{'Total':<20} {len(train_df):<15}")
    print("="*60)
    
    # Split into 'none' and other classes
    none_df = train_df[train_df[target_column] == "none"]
    other_df = train_df[train_df[target_column] != "none"]
    
    # Downsample 'none' class
    if len(none_df) > target_none_count:
        none_df = none_df.sample(n=target_none_count, random_state=RANDOM_STATE)
        removed = value_counts["none"] - target_none_count
        print(f"\n Downsampled 'none' class:")
        print(f"   Before: {value_counts['none']:,} samples")
        print(f"   After:  {target_none_count:,} samples")
        print(f"   Removed: {removed:,} samples")
    else:
        print(f"\n  'none' class already ≤ {target_none_count:,} samples. No downsampling needed.")
    
    # Combine back and shuffle
    result = pd.concat([none_df, other_df], axis=0).sample(
        frac=1, 
        random_state=RANDOM_STATE
    ).reset_index(drop=True)
    
    # ========== AFTER DOWNSAMPLING ==========
    print("\n" + "="*60)
    print(" CLASS DISTRIBUTION AFTER DOWNSAMPLING")
    print("="*60)
    print(f"{'Class':<20} {'Count':<15} {'Ratio':<15}")
    print("-"*60)
    
    new_counts = result[target_column].value_counts()
    for label in sorted(new_counts.index):
        count = new_counts[label]
        ratio = count / new_counts["none"] if label != "none" else 1.0
        print(f"{label:<20} {count:<15} {ratio:<15.2f}x")
    print("-"*60)
    print(f"{'Total':<20} {len(result):<15}")
    print("="*60)
    
    #  Calculate balanced class weights for weighted loss function
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(result[target_column])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=result[target_column].values
    )
    class_weights_dict = dict(zip(unique_classes, class_weights))
    
    print(f"\n  CLASS WEIGHTS (for Focal Loss):")
    print("="*60)
    for label, weight in sorted(class_weights_dict.items()):
        print(f"  {label:<20}: {weight:.4f}")
    print("="*60 + "\n")
    
    return result, class_weights_dict


# ==================================================
# Augment Minority Classes
# ==================================================
def augment_minority_classes(train_df, target_column, wafermap_column):
    if not AUGMENT_MINORITY_CLASSES:
        return train_df
    
    if wafermap_column not in train_df.columns:
        print("Wafer map column not found. Skipping augmentation.")
        return train_df
    
    counts = train_df[target_column].value_counts()
    defect_counts = counts.drop(labels=["none"], errors="ignore")
    if defect_counts.empty:
        return train_df
    
    target_count = defect_counts.max()
    print(f"\n Augmenting minority classes to {target_count} samples per class...")
    
    augmented_rows = []
    
    for label, count in defect_counts.items():
        if count >= target_count:
            continue
        
        subset = train_df[train_df[target_column] == label]
        n_to_add = min(target_count - count, MAX_AUGMENT_PER_CLASS)
        
        print(f"  Augmenting '{label}': current={count}, adding={n_to_add}")
        
        for _ in range(n_to_add):
            row = subset.sample(n=1, replace=True, random_state=np.random.randint(0, 1_000_000)).iloc[0].copy()
            row[wafermap_column] = augment_wafer_map(row[wafermap_column])
            augmented_rows.append(row)
    
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        train_df = pd.concat([train_df, aug_df], axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    return train_df

# ==================================================
#  changed: Enhanced Metadata Feature Building
# ==================================================
def build_metadata_features(train_df, test_df, target_column, split_column, wafermap_column):
    """
    Build metadata + morphological + texture features from wafer maps.
    """
    feature_drop_cols = [target_column]
    if split_column in train_df.columns:
        feature_drop_cols.append(split_column)
    if wafermap_column in train_df.columns:
        feature_drop_cols.append(wafermap_column)
    
    X_train_meta = train_df.drop(columns=feature_drop_cols, errors="ignore").copy()
    X_test_meta = test_df.drop(columns=feature_drop_cols, errors="ignore").copy()
    
    y_train = train_df[target_column].copy().reset_index(drop=True)
    y_test = test_df[target_column].copy().reset_index(drop=True)
    
    # Fill missing metadata values
    for col in X_train_meta.columns:
        if pd.api.types.is_numeric_dtype(X_train_meta[col]):
            median_val = X_train_meta[col].median()
            X_train_meta[col] = X_train_meta[col].fillna(median_val)
            X_test_meta[col] = X_test_meta[col].fillna(median_val)
        else:
            mode_val = X_train_meta[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
            X_train_meta[col] = X_train_meta[col].fillna(fill_val)
            X_test_meta[col] = X_test_meta[col].fillna(fill_val)
    
    # One-hot encode categorical columns
    categorical_cols = X_train_meta.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        combined = pd.concat([X_train_meta, X_test_meta], axis=0, ignore_index=True)
        combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=False)
        X_train_meta = combined.iloc[:len(X_train_meta)].reset_index(drop=True)
        X_test_meta = combined.iloc[len(X_train_meta):].reset_index(drop=True)
    
    #  Extract morphological and texture features from wafer maps
    print("\n Extracting morphological features from wafer maps...")
    morpho_features_train = np.array([extract_morphological_features(w) for w in train_df[wafermap_column]])
    morpho_features_test = np.array([extract_morphological_features(w) for w in test_df[wafermap_column]])
    
    print(" Extracting texture features from wafer maps...")
    texture_features_train = np.array([extract_texture_features(w) for w in train_df[wafermap_column]])
    texture_features_test = np.array([extract_texture_features(w) for w in test_df[wafermap_column]])
    
    # Combine with metadata
    morpho_df_train = pd.DataFrame(morpho_features_train, columns=[f"morpho_{i}" for i in range(8)])
    morpho_df_test = pd.DataFrame(morpho_features_test, columns=[f"morpho_{i}" for i in range(8)])
    
    texture_df_train = pd.DataFrame(texture_features_train, columns=[f"texture_{i}" for i in range(8)])
    texture_df_test = pd.DataFrame(texture_features_test, columns=[f"texture_{i}" for i in range(8)])
    
    X_train_meta = pd.concat([X_train_meta, morpho_df_train, texture_df_train], axis=1)
    X_test_meta = pd.concat([X_test_meta, morpho_df_test, texture_df_test], axis=1)
    
    # Standardize all numeric features
    scaler = StandardScaler()
    numeric_cols = X_train_meta.select_dtypes(include=[np.number]).columns
    X_train_meta[numeric_cols] = scaler.fit_transform(X_train_meta[numeric_cols])
    X_test_meta[numeric_cols] = scaler.transform(X_test_meta[numeric_cols])
    
    return X_train_meta, X_test_meta, y_train, y_test

# ==================================================
# Label Encoding
# ==================================================
def encode_labels(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_safe = y_test.map(lambda x: x if x in le.classes_ else le.classes_[0])
    y_test_enc = le.transform(y_test_safe)
    
    print(f"\n Label encoding complete:")
    print(f"  Classes : {list(le.classes_)}")
    print(f"  Num classes : {len(le.classes_)}")
    
    return y_train_enc, y_test_enc, le

# ==================================================
# Main Preprocessing Function
# ==================================================
def load_and_preprocess_data(
    data_path=DATA_PATH,
    output_pickle=OUTPUT_PICKLE,
    wafermap_column=WAFERMAP_COLUMN,
    target_column=TARGET_COLUMN,
    split_column=SPLIT_COLUMN,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    set_seed(random_state)
    
    print("=" * 70)
    print(" ENHANCED Data Preprocessing with Advanced Feature Extraction")
    print("=" * 70)
    
    output_dir = os.path.dirname(output_pickle)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"\n Loading data from: {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    df = safe_to_dataframe(data)
    print(f"   Initial Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    if df.empty:
        raise ValueError("Loaded data is empty.")
    
    # Step 1: Normalize label columns
    print("\n[Step 1] Normalizing label columns...")
    for col in [split_column, target_column]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_label)
            print(f"    [{col}] normalized")
        else:
            print(f"     [{col}] not found")
    
    # Step 2: Detect raw array columns
    print("\n[Step 2] Detecting raw array columns...")
    df, raw_array_cols = scan_raw_array_columns(df, target_column, split_column)
    
    # Step 3: Drop duplicates
    print("\n[Step 3] Removing duplicates...")
    dedup_cols = [c for c in df.columns if c not in raw_array_cols]
    before_rows = len(df)
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(f"   Shape after deduplication: {df.shape} (removed {before_rows - len(df)})")
    
    # Step 4: Validate required columns
    print("\n[Step 4] Validating required columns...")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    if wafermap_column not in df.columns:
        raise ValueError(f"Wafer map column '{wafermap_column}' not found.")
    print("    All required columns present")
    
    # Step 5: Clean target labels
    print("\n[Step 5] Cleaning target labels...")
    df[target_column] = df[target_column].apply(normalize_label)
    
    if REMOVE_UNKNOWN_LABELS:
        before = len(df)
        df = df[df[target_column] != "unknown"].reset_index(drop=True)
        print(f"   Removed 'unknown': {df.shape} (removed {before - len(df)})")
    
    if not KEEP_NONE_CLASS:
        before = len(df)
        df = df[df[target_column] != "none"].reset_index(drop=True)
        print(f"   Removed 'none': {df.shape} (removed {before - len(df)})")
    
    target_preview, target_count = summarize_unique(df[target_column])
    print(f"\n   Classes ({target_count}): {target_preview}")
    print(f"   Value counts:\n{df[target_column].value_counts(dropna=False)}\n")
    
    # Step 6: Train/Test split
    print("\n[Step 6] Splitting train/test sets...")
    if split_column in df.columns:
        df[split_column] = df[split_column].apply(normalize_label)
        split_lower = df[split_column].str.lower()
        train_mask = split_lower.isin(["training", "train"])
        test_mask = split_lower.isin(["test", "testing"])
        
        if train_mask.any() and test_mask.any():
            train_df = df.loc[train_mask].reset_index(drop=True)
            test_df = df.loc[test_mask].reset_index(drop=True)
            print("    Using existing split from dataset")
        else:
            print("     No valid split found, using stratified split")
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state,
                stratify=df[target_column] if df[target_column].nunique() > 1 else None
            )
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
    else:
        print("     No split column found, using stratified split")
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df[target_column] if df[target_column].nunique() > 1 else None
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Step 7: Class imbalance handling
    print("\n[Step 7] Handling class imbalance...")
    print("   Before balancing:")
    print(train_df[target_column].value_counts())
    
    train_df, class_weights = downsample_none_class(train_df, target_column)
    
    if AUGMENT_MINORITY_CLASSES:
        train_df = augment_minority_classes(train_df, target_column, wafermap_column)
    
    print("\n   After balancing:")
    print(train_df[target_column].value_counts())
    
    # Step 8:  Normalize wafer maps
    print(f"\n[Step 8] Normalizing wafer maps...")
    if NORMALIZE_WAFER_MAPS:
        train_df[wafermap_column] = train_df[wafermap_column].apply(normalize_wafer_map)
        test_df[wafermap_column] = test_df[wafermap_column].apply(normalize_wafer_map)
        print("    Wafer maps normalized to [0, 1] range")
    
    # Step 9: Resize wafer maps
    print(f"\n[Step 9] Resizing wafer maps to {TARGET_WAFER_SIZE}x{TARGET_WAFER_SIZE}...")
    if RESIZE_WAFER_MAPS and TARGET_WAFER_SIZE is not None:
        train_df[wafermap_column] = train_df[wafermap_column].apply(
            lambda x: resize_wafer_map(x, target_size=TARGET_WAFER_SIZE)
        )
        test_df[wafermap_column] = test_df[wafermap_column].apply(
            lambda x: resize_wafer_map(x, target_size=TARGET_WAFER_SIZE)
        )
        sample_shape = np.array(train_df[wafermap_column].iloc[0]).shape
        print(f"    Resized — sample shape: {sample_shape}")
    
    # Step 10: Build enhanced metadata + morphological + texture features
    print(f"\n[Step 10] Building enhanced feature set...")
    X_train_meta, X_test_meta, y_train, y_test = build_metadata_features(
        train_df, test_df, target_column, split_column, wafermap_column
    )
    print(f"   X_train_meta: {X_train_meta.shape}, X_test_meta: {X_test_meta.shape}")
    
    # Step 11: Encode labels
    print(f"\n[Step 11] Encoding labels...")
    y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)
    
    # Step 12: Preserve wafer maps
    waferMap_train = train_df[wafermap_column].tolist()
    waferMap_test = test_df[wafermap_column].tolist()
    print(f"\n   Wafer maps → train: {len(waferMap_train)}, test: {len(waferMap_test)}")
    
    # Step 13: Save processed output
    print(f"\n[Step 12] Saving preprocessed data...")
    processed_data = {
        "train_df"              : train_df,
        "test_df"               : test_df,
        "X_train_meta"          : X_train_meta,
        "X_test_meta"           : X_test_meta,
        "y_train"               : y_train,
        "y_test"                : y_test,
        "y_train_enc"           : y_train_enc,
        "y_test_enc"            : y_test_enc,
        "label_encoder"         : label_encoder,
        "waferMap_train"        : waferMap_train,
        "waferMap_train_resized": waferMap_train,
        "waferMap_test"         : waferMap_test,
        "waferMap_test_resized" : waferMap_test,
        "target_column"         : target_column,
        "split_column"          : split_column if split_column in df.columns else None,
        "wafermap_column"       : wafermap_column,
        "raw_array_columns"     : raw_array_cols,
        "class_names"           : list(label_encoder.classes_),
        "num_classes"           : len(label_encoder.classes_),
        "wafer_size"            : TARGET_WAFER_SIZE if RESIZE_WAFER_MAPS else "original",
        "class_weights"         : class_weights,  #  For weighted loss functions
        "feature_names"         : list(X_train_meta.columns),  #  Feature tracking
    }
    
    with open(output_pickle, "wb") as f:
        pickle.dump(processed_data, f)
    
    print(f"\n Saved to: {output_pickle}")
    print(f"   Keys: {list(processed_data.keys())}")
    print("\n" + "=" * 70)
    print(" Preprocessing Complete!")
    print("=" * 70)
    
    return processed_data

def main():
    try:
        load_and_preprocess_data()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
