import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[2]))

from src.models.wafer_cnn import get_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx=None):
        self.model.eval()
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()

def visualize_gradcam(model, image_tensor, original_image, class_names, target_layer, save_path):
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate_heatmap(image_tensor)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Overlay
    original_colored = cv2.cvtColor(np.uint8(original_image * 255), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original_colored, 0.6, heatmap_colored, 0.4, 0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Wafer")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    
    plt.savefig(save_path)
    print(f"Explainability plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage would go here, loading a sample and the best model
    pass
