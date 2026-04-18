import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from captum.attr import LayerGradCam

# Add src to path
sys.path.append(str(Path(__file__).parents[2]))

from src.data.preprocessing import preprocess_data, WaferMapDataset
from src.models.wafer_cnn import get_model

def evaluate_professional():
    # 1. Load Data
    preprocessed_maps, labels, label_encoder, _, geom_features = preprocess_data()
    indices = np.arange(len(labels))
    _, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=labels)
    
    test_dataset = WaferMapDataset([preprocessed_maps[i] for i in test_idx], labels[test_idx], geom_features[test_idx])
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 2. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 3. Load Model
    num_classes = len(label_encoder.classes_)
    model = get_model(model_name="resnet18", num_classes=num_classes, num_geom_features=6).to(device)
    model_path = Path(__file__).parents[2] / "models" / "best_resnet18_hybrid.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Predict
    all_preds, all_labels = [], []
    all_probs = []
    with torch.no_grad():
        for (img, geom), target in test_loader:
            img, geom = img.to(device), geom.to(device)
            outputs = model((img, geom))
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))

    # 5. Advanced Metrics
    print("\n" + "="*50)
    print("INDUSTRIAL PERFORMANCE METRICS")
    print("="*50)
    macro_auc = roc_auc_score(all_labels_bin, all_probs, multi_class='ovr', average='macro')
    macro_ap = average_precision_score(all_labels_bin, all_probs, average='macro')
    print(f"Overall Macro AUC-ROC: {macro_auc:.4f}")
    print(f"Overall Macro Average Precision (mAP): {macro_ap:.4f}")
    print("-" * 50)

    # 6. Normalized Confusion Matrix
    plt.style.use('dark_background')
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='rocket', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                annot_kws={"size": 10, "weight": "bold"})
    plt.title('Normalized Performance Matrix', fontsize=18, color='yellow')
    plt.xlabel('Predicted', fontsize=14, color='cyan')
    plt.ylabel('Actual', fontsize=14, color='cyan')
    plt.tight_layout()
    plt.savefig(Path(__file__).parents[2] / "src" / "analysis" / "normalized_performance_matrix.png")
    print(f"Normalized performance matrix saved.")

    # 7. Master Explanation Gallery (Explain every class)
    print("\n--- Generating Master Explanation Gallery (Captum) ---")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Captum AI Attribution: Defect Feature Mapping", fontsize=24, color='yellow', y=1.02)
    
    class_names = label_encoder.classes_
    found_classes = set()
    
    class CaptumWrapper(torch.nn.Module):
        def __init__(self, model, geom):
            super().__init__()
            self.model = model
            self.geom = geom
        def forward(self, img):
            return self.model((img, self.geom))

    for idx_in_test in range(len(test_idx)):
        i = test_idx[idx_in_test]
        c_idx = int(labels[i])
        if c_idx not in found_classes and class_names[c_idx] != 'none':
            row, col = divmod(len(found_classes), 3)
            img = torch.tensor(np.stack([preprocessed_maps[i]]*3, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            geom = torch.tensor(geom_features[i], dtype=torch.float32).unsqueeze(0).to(device)
            lgc = LayerGradCam(CaptumWrapper(model, geom), model.backbone.layer4[-1])
            attribution = lgc.attribute(img, target=c_idx)
            upsampled = LayerGradCam.interpolate(attribution, (96, 96)).squeeze().cpu().detach().numpy()
            if upsampled.ndim == 3: upsampled = np.transpose(upsampled, (1, 2, 0))[..., 0]
            upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-9)
            ax = axes[row, col]
            ax.imshow(preprocessed_maps[i], cmap='gray')
            ax.imshow(upsampled, cmap='jet', alpha=0.5)
            ax.set_title(f"Class: {class_names[c_idx]}", fontsize=14, color='cyan')
            ax.axis('off')
            found_classes.add(c_idx)
            if len(found_classes) >= 9: break

    plt.tight_layout()
    plt.savefig(Path(__file__).parents[2] / "src" / "analysis" / "master_explanation_gallery.png")
    print(f"Master gallery saved to src/analysis/master_explanation_gallery.png")

if __name__ == "__main__":
    evaluate_professional()
