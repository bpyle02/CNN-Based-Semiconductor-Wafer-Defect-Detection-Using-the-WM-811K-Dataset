import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from skimage.transform import resize
import torchvision.transforms.functional as TF

# Add src to path
sys.path.append(str(Path(__file__).parents[2]))

from src.models.wafer_cnn import get_model

def predict(wafer_map, use_tta=True):
    """
    Predicts the defect class for a single wafer map with Test-Time Augmentation.
    """
    # 1. Load label encoder and setup model
    model_dir = Path(__file__).parents[2] / "models"
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    num_classes = len(label_encoder.classes_)
    # Defaulting to resnet18 as it benchmarked better earlier
    model = get_model(model_name="resnet18", num_classes=num_classes).to(device)
    
    # Try loading the benchmarked model first
    model_path = model_dir / "best_resnet18_bench.pth"
    if not model_path.exists():
        model_path = model_dir / "best_wafer_model_enhanced.pth"
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Preprocess
    target_size = (96, 96)
    arr = wafer_map.astype(np.float32)
    arr = resize(arr, target_size, anti_aliasing=True, preserve_range=True).astype(np.float32)
    arr = arr / 2.0
    
    # (3, H, W)
    img = np.stack([arr] * 3, axis=0)
    img_tensor = torch.tensor(img, dtype=torch.float32).to(device).unsqueeze(0)

    # 3. Predict with TTA
    with torch.no_grad():
        if use_tta:
            # TTA: Original, HFlip, VFlip, Rotations
            tta_inputs = [
                img_tensor,
                TF.hflip(img_tensor),
                TF.vflip(img_tensor),
                TF.rotate(img_tensor, 90),
                TF.rotate(img_tensor, 180),
                TF.rotate(img_tensor, 270)
            ]
            
            # Sum the softmax scores for each augmentation
            tta_outputs = [F.softmax(model(x), dim=1) for x in tta_inputs]
            final_output = torch.mean(torch.stack(tta_outputs), dim=0)
        else:
            final_output = F.softmax(model(img_tensor), dim=1)
            
        _, predicted = torch.max(final_output, 1)
        confidence = torch.max(final_output).item()
        
    predicted_label = label_encoder.classes_[predicted.item()]
    
    return predicted_label, confidence

if __name__ == "__main__":
    pass
