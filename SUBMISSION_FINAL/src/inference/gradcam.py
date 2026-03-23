"""
Gradient-weighted Class Activation Mapping (GradCAM) for interpretability.

Based on: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).

Highlights spatial regions most important for model predictions, providing
visual interpretability for black-box neural networks.
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from skimage.transform import resize as skimage_resize
import numpy as np


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for neural networks.

    Computes activation maps that highlight regions most important for
    predicting a given class. Uses forward and backward hooks to capture
    activations and gradients at a target layer.

    Attributes:
        model: PyTorch model
        target_layer: Layer to compute CAM for
        gradients: Stored gradients from backward hook
        activations: Stored activations from forward hook
        hooks: List of hook handles (for cleanup)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Initialize GradCAM.

        Args:
            model: Trained PyTorch model
            target_layer: Target layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks: List = []

        # Register forward and backward hooks
        self.hooks.append(target_layer.register_forward_hook(self._forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store activations during forward pass."""
        self.activations = output.clone().detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple,
        grad_output: Tuple
    ) -> None:
        """Store gradients during backward pass."""
        self.gradients = grad_output[0].clone().detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Grad-CAM heatmap for an input.

        Algorithm:
            1. Forward pass to get output
            2. Backward pass from target class logit
            3. Compute average gradient across spatial dims: (C, H, W) -> (C, 1, 1)
            4. Weight activations by gradients
            5. Sum weighted activations
            6. ReLU to keep only positive contributions
            7. Normalize to [0, 1]
            8. Upsample to input size

        Args:
            input_tensor: Input image (1, C, H, W)
            target_class: Class to visualize. If None, uses argmax of output.
            device: Device to run on

        Returns:
            Tuple of (heatmap [H, W] normalized to [0, 1], predicted_class)
        """
        self.model.eval()
        input_tensor = input_tensor.to(device)

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients and backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Compute weights: average gradients over spatial dimensions
        # (B, C, H, W) -> (B, C, 1, 1)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activations
        # (B, C, H, W) * (B, C, 1, 1) -> sum over channels -> (B, 1, H, W)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize heatmap to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image dimensions
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = skimage_resize(cam, (h_in, w_in), anti_aliasing=True)

        return cam_resized, target_class

    def remove_hooks(self) -> None:
        """Remove all registered hooks for cleanup."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self) -> None:
        """Cleanup hooks on deletion."""
        self.remove_hooks()


if __name__ == "__main__":
    print("GradCAM module loaded. Use with trained PyTorch models.")
