import pickle, numpy as np, pandas as pd
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

class PandasCompat(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('pandas.indexes'):
            module = module.replace('pandas.indexes', 'pandas.core.indexes', 1)
        return super().find_class(module, name)

def main() -> None:
    dataset_path = Path('General/Dataset/LSWMD.pkl')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with dataset_path.open('rb') as f:
        data = PandasCompat(f, encoding='latin1').load()

    print(f'Shape: {data.shape}')
    print(f'Columns: {data.columns.tolist()}')

    # Check failureType format
    sample = data['failureType'].iloc[0]
    print(f'\nfailureType[0]: type={type(sample)}, repr={repr(sample)[:200]}')
    sample2 = data['failureType'].iloc[100000]
    print(f'failureType[100000]: type={type(sample2)}, repr={repr(sample2)[:200]}')

    # Check waferMap format
    wm = data['waferMap'].iloc[0]
    print(f'\nwaferMap[0]: type={type(wm)}, shape={wm.shape if hasattr(wm, "shape") else "N/A"}')
    if hasattr(wm, 'shape'):
        print(f'waferMap unique values: {np.unique(wm)}')


if __name__ == "__main__":
    main()
