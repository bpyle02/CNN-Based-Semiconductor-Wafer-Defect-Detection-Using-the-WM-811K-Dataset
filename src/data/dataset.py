# ======================================================================================
#  Written by Brandon Pyle
#  This file manages the import and loading of the dataset.
# ======================================================================================

import pandas as pd
import numpy as np
from pathlib import Path

def load_dataset():
    BASE_DIR = Path(__file__).parents[2]
    filepath = BASE_DIR / "data" / "LSWMD_new.pkl"

    df = pd.read_pickle(filepath)

    df['failureClass'] = df['failureType'].apply(extract_failure_label)

    print(f"Dataset Shape: {df.shape}")
    print(f"Dataset Columns: {df.columns.tolist()}")
    print(f"Dataset Memory Usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB\n")
    print(f"{df.head()}\n")

    class_counts = df['failureClass'].value_counts()

    print("\n--- Failure Class Distribution (all data) ---")
    print(class_counts)
    print(f"\nUnique classes: {df['failureClass'].nunique()}")
    print(f"Total wafers : {len(df):,}")
    
    return df

def extract_failure_label(failure_label):
    try:
        if isinstance(failure_label, np.ndarray):
            if failure_label.size > 0:
                val = failure_label.flatten()[0]
                if isinstance(val, (str, np.str_, bytes)):
                    return val.decode('latin1').strip if isinstance(val, bytes) else str(val)
        if isinstance(failure_label, list) and len(failure_label) > 0:
            inner = failure_label[0]
            if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
                val = inner[0]
                return val.decode('latin1').strip() if isinstance(val, bytes) else str(val)
            return inner.decode('latin1').strip() if isinstance(inner, bytes) else str(inner)
    except Exception:
        pass
    return 'unknown'

if __name__ == "__main__":
    load_dataset()