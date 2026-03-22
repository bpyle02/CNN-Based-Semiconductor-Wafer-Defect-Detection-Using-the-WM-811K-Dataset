"""GradCAM visualization utilities."""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from .gradcam import GradCAM


def plot_gradcam_grid(
    model: nn.Module,
    test_dataset,
    target_layer: nn.Module,
    class_names: List[str],
    num_samples: int = 9,
    figsize: Tuple[int, int] = (20, 10),
    device: str = "cpu",
) -> None:
    """
    Visualize GradCAM for multiple test samples (one per class).

    Creates a grid with original image and GradCAM heatmap overlay for each class.

    Args:
        model: Trained model
        test_dataset: Test dataset object with __getitem__ returning (img, label)
        target_layer: Layer to visualize
        class_names: List of class names
        num_samples: Number of samples to visualize (min 1, max len(class_names))
        figsize: Figure size (width, height)
        device: Device to run on
    """
    model.eval()
    gradcam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(3, 6, figsize=figsize)
    fig.suptitle(f'Grad-CAM Visualization', fontsize=16, fontweight='bold')

    shown = 0
    classes_shown = set()

    for i in range(len(test_dataset)):
        if shown >= num_samples:
            break

        img, label = test_dataset[i]
        cls_idx = label.item()

        if cls_idx in classes_shown:
            continue
        classes_shown.add(cls_idx)

        # Generate GradCAM
        input_tensor = img.unsqueeze(0).to(device)
        cam, pred_class = gradcam.generate(input_tensor, device=device)

        row = shown // 3
        col_base = (shown % 3) * 2

        # Original image
        ax_img = axes[row, col_base]
        original = img[0].cpu().numpy()
        ax_img.imshow(original, cmap='gray')
        ax_img.set_title(f'{class_names[cls_idx]}\n(True)', fontweight='bold')
        ax_img.axis('off')

        # GradCAM overlay
        ax_cam = axes[row, col_base + 1]
        ax_cam.imshow(original, cmap='gray', alpha=0.6)
        ax_cam.imshow(cam, cmap='jet', alpha=0.4)
        pred_name = class_names[pred_class] if pred_class < len(class_names) else 'Unknown'
        ax_cam.set_title(f'{pred_name}\n(Pred)', fontweight='bold')
        ax_cam.axis('off')

        shown += 1

    plt.tight_layout()
    plt.show()

    # Cleanup
    gradcam.remove_hooks()


if __name__ == "__main__":
    print("GradCAM visualization module loaded.")
