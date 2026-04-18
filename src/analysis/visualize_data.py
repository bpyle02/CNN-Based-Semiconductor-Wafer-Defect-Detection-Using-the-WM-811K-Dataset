import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parents[2]))
from src.data.dataset import load_dataset

def generate_comprehensive_data_plots():
    df = load_dataset()
    analysis_dir = Path(__file__).parents[2] / "src" / "analysis"
    
    # Set the aesthetic style
    plt.style.use('dark_background')
    sns.set_palette("husl")

    # --- Plot 1: Class Distribution (Professional & Log Scale) ---
    plt.figure(figsize=(12, 6))
    counts = df['failureClass'].value_counts()
    
    # Use a log scale because 'none' is so massive
    ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis", hue=counts.index, legend=False)
    ax.set_yscale("log")
    plt.title("Wafer Defect Distribution (Log Scale)", fontsize=16, color='cyan')
    plt.ylabel("Count (Log)", fontsize=12)
    plt.xticks(rotation=45)
    
    # Add actual counts as labels
    for i, v in enumerate(counts.values):
        ax.text(i, v, f'{v:,}', color='white', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(analysis_dir / "defect_class_distribution.png")
    print(f"Distribution plot saved to {analysis_dir / 'defect_class_distribution.png'}")

    # --- Plot 2: Defect Sample Gallery ---
    failure_classes = [c for c in df['failureClass'].unique() if c != 'unknown']
    n_classes = len(failure_classes)
    n_samples = 4
    
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(15, n_classes * 3))
    fig.suptitle("Wafer Map Gallery: Defect Signatures", fontsize=20, color='yellow', y=1.02)

    for i, cls in enumerate(failure_classes):
        samples = df[df['failureClass'] == cls].sample(min(n_samples, len(df[df['failureClass'] == cls])))
        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(samples):
                wm = samples.iloc[j]['waferMap']
                # Use high-contrast 'magma' colormap
                im = ax.imshow(wm, cmap='magma', interpolation='nearest')
                if j == 0:
                    ax.set_ylabel(cls, fontsize=14, color='cyan', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            # Add a subtle neon border
            for spine in ax.spines.values():
                spine.set_edgecolor('lime')
                spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(analysis_dir / "wafer_defect_gallery.png")
    print(f"Gallery plot saved to {analysis_dir / 'wafer_defect_gallery.png'}")

if __name__ == "__main__":
    generate_comprehensive_data_plots()
