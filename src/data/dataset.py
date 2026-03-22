"""
Dataset loading and management for WM-811K semiconductor wafer defect detection.

Handles loading, parsing, and basic quality checks for the WM-811K dataset,
including extraction of failure class labels from nested structures.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


KNOWN_CLASSES = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
]


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the WM-811K dataset from a pickle file and extract failure classes.

    Args:
        path: Optional path to dataset pickle file. Defaults to data/LSWMD_new.pkl.

    Returns:
        DataFrame with columns: waferMap, failureType, failureClass, etc.

    Raises:
        FileNotFoundError: If dataset file not found at specified path.
    """
    if path is None:
        base_dir = Path(__file__).parents[2]
        path = base_dir / "data" / "LSWMD_new.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_pickle(path)
    df['failureClass'] = df['failureType'].apply(extract_failure_label)

    print("\n--- Dataset Info ---")
    print(f"Dataset Shape: {df.shape}")
    print(f"Dataset Columns: {df.columns.tolist()}")
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Dataset Memory Usage: {memory_mb:.2f} MB\n")
    print(f"{df.head()}")

    class_counts = df['failureClass'].value_counts()
    print("\n--- Failure Class Distribution (Before Filtering) ---")
    print(class_counts.to_string())

    print("\n--- Data Quality Checks ---")
    unique_classes = df['failureClass'].unique().tolist()
    print(f"Unique failure classes: {unique_classes}")
    print(f"Total wafers: {len(df):,}")

    return df


def extract_failure_label(failure_label) -> str:
    """
    Extract a clean failure class label from the raw 'failureType' field.

    The failureType field contains nested structures (arrays/lists) with various
    encoding formats. This function unwraps and decodes them robustly.

    Args:
        failure_label: Raw value from failureType column (may be array, list, bytes, str).

    Returns:
        Clean string label (one of KNOWN_CLASSES or 'unknown').
    """
    try:
        # Handle numpy arrays
        if isinstance(failure_label, np.ndarray):
            if failure_label.size > 0:
                val = failure_label.flatten()[0]
                if isinstance(val, bytes):
                    return val.decode('latin1').strip()
                if isinstance(val, (str, np.str_)):
                    return str(val).strip()

        # Handle lists
        if isinstance(failure_label, list) and len(failure_label) > 0:
            inner = failure_label[0]

            # Nested list/array
            if isinstance(inner, (list, np.ndarray)) and len(inner) > 0:
                val = inner[0]
                if isinstance(val, bytes):
                    return val.decode('latin1').strip()
                return str(val).strip()

            # Direct list element
            if isinstance(inner, bytes):
                return inner.decode('latin1').strip()
            return str(inner).strip()

        # Handle direct strings/bytes
        if isinstance(failure_label, bytes):
            return failure_label.decode('latin1').strip()
        if isinstance(failure_label, str):
            return failure_label.strip()

    except Exception:
        pass

    return 'unknown'


if __name__ == "__main__":
    df = load_dataset()
