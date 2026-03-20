# Written by Brandon Pyle
# This file manages the import and loading of the dataset.

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parents[2]
filepath = BASE_DIR / "data" / "LSWMD_new.pkl"

df = pd.read_pickle(filepath)
print(df.head())
print(df.shape)