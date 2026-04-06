"""
Dataset loading and management for WM-811K semiconductor wafer defect detection.

Handles loading, parsing, and basic quality checks for the WM-811K dataset,
including extraction of failure class labels from nested structures.
"""

from pathlib import Path
from typing import Any, Optional
import pandas as pd
import numpy as np

from src.exceptions import DataLoadError
import logging

logger = logging.getLogger(__name__)

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
        DataLoadError: If dataset file cannot be found or loaded.
    """
    if path is None:
        base_dir = Path(__file__).parents[2]
        path = base_dir / "data" / "LSWMD_new.pkl"
    else:
        path = Path(path)

    if not path.exists():
        raise DataLoadError(f"Dataset not found at {path}")

    try:
        df = pd.read_pickle(path)
    except Exception as e:
        raise DataLoadError(f"Failed to load dataset from {path}: {e}") from e

    df['failureClass'] = df['failureType'].apply(extract_failure_label)

    logger.info("\n--- Dataset Info ---")
    logger.info(f"Dataset Shape: {df.shape}")
    logger.info(f"Dataset Columns: {df.columns.tolist()}")
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"Dataset Memory Usage: {memory_mb:.2f} MB\n")
    logger.info(f"{df.head()}")

    class_counts = df['failureClass'].value_counts()
    logger.info("\n--- Failure Class Distribution (Before Filtering) ---")
    logger.info(class_counts.to_string())

    logger.info("\n--- Data Quality Checks ---")
    unique_classes = df['failureClass'].unique().tolist()
    logger.info(f"Unique failure classes: {unique_classes}")
    logger.info(f"Total wafers: {len(df):,}")

    return df


def extract_failure_label(failure_label: Any) -> str:
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

    except Exception as e:
        logger.debug(f"Could not parse failure label: {e}")

    return 'unknown'


if __name__ == "__main__":
    df = load_dataset()
