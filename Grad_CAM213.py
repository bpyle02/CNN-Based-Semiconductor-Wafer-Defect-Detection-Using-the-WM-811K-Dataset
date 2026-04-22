# ============================================================
# UPDATED GRAD-CAM PROGRAM
# Based on the attached training code
#
# Output:
#   For each of the 9 classes:
#   - picks 3 samples
#   - shows Original + Grad-CAM for 3 models side by side
#   - saves one PNG per class
# ============================================================

import os
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# Match your attached program
# ============================================================
PREPROCESSED_PICKLE = r"G:\My Drive\AI_570\Final_project_Version15April\data\processed\preprocessed_wm811k_improved.pkl"
RESULTS_SAVE_PATH   = r"G:\My Drive\AI_570\Final_project_Version15April\results_comparison"
GRADCAM_SAVE_PATH   = os.path.join(RESULTS_SAVE_PATH, "gradcam_side_by_side_3samples")

WAFER_SIZE     = 64
DROPOUT_RATE   = 0.30
RANDOM_STATE   = 42
VAL_SIZE       = 0.15
NUM_CLASSES    = 9

CLASS_NAMES_GLOBAL = [
    "center", "donut", "edge-loc", "edge-ring",
    "loc", "near-full", "none", "random", "scratch"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ============================================================
# DATA LOADING
# Based on the attached code
# ============================================================
def load_preprocessed_data(pickle_path):
    print("=" * 70)
    print("Loading preprocessed data...")
    print("=" * 70)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    wafer_train_raw = data["waferMap_train_resized"]
    wafer_test_raw  = data["waferMap_test_resized"]
    X_train_meta    = np.array(data["X_train_meta"], dtype=np.float32)
    X_test_meta     = np.array(data["X_test_meta"], dtype=np.float32)
    y_train         = np.array(data["y_train_enc"], dtype=np.int64)
    y_test          = np.array(data["y_test_enc"], dtype=np.int64)
    class_names     = list(data["class_names"])
    num_classes     = int(data["num_classes"])

    print(f"Train samples : {len(wafer_train_raw)}")
    print(f"Test samples  : {len(wafer_test_raw)}")
    print(f"Classes       : {class_names}")
    print(f"Meta features : {X_train_meta.shape[1]}")

    return (
        wafer_train_raw, wafer_test_raw,
        X_train_meta, X_test_meta,
        y_train, y_test,
        class_names, num_classes
    )


def prepare_wafer_arrays(wafer_list, wafer_size=WAFER_SIZE):
    processed = []
    for wafer in wafer_list:
        arr = np.asarray(wafer, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3:
            if arr.shape[0] == 1:
                pass
            elif arr.shape[-1] == 1:
                arr = np.transpose(arr, (2, 0, 1))

        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val

        processed.append(arr.astype(np.float32))

    return np.stack(processed, axis=0)


# ============================================================
# DATASET
# ============================================================
class WaferDefectDataset(Dataset):
    def __init__(self, wafer_maps, meta_features, labels, augment=False):
        self.wafer_maps    = wafer_maps
        self.meta_features = meta_features
        self.labels        = labels
        self.augment       = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wafer = self.wafer_maps[idx].copy()
        meta  = self.meta_features[idx]
        label = self.labels[idx]

        return (
            torch.tensor(wafer, dtype=torch.float32),
            torch.tensor(meta, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# ============================================================
# MODEL 1: BASELINE CNN
# ============================================================
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, wafer, meta=None):
        x = self.features(wafer)
        return self.classifier(x)


# ============================================================
# MODEL 2: RESNET18 WAFER
# ============================================================
class ResNet18Wafer(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        base.maxpool = nn.Identity()

        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.resnet = base

    def forward(self, wafer, meta=None):
        return self.resnet(wafer)


# ============================================================
# MODEL 3: WAFERMETANET
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se       = SEBlock(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.pool     = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.se(out)
        out = self.relu(out + identity)
        return self.pool(out)


class WaferMetaNet(nn.Module):
    def __init__(self, num_meta_features, num_classes, dropout=DROPOUT_RATE):
        super().__init__()

        self.wafer_encoder = nn.Sequential(
            ResConvBlock(1,   32,  pool=True),
            ResConvBlock(32,  64,  pool=True),
            ResConvBlock(64,  128, pool=True),
            ResConvBlock(128, 256, pool=True),
            ResConvBlock(256, 512, pool=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.meta_encoder = nn.Sequential(
            nn.Linear(num_meta_features, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.GELU(), nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, wafer, meta):
        wafer_feat = torch.flatten(self.wafer_encoder(wafer), 1)
        meta_feat  = self.meta_encoder(meta)
        fused      = torch.cat([wafer_feat, meta_feat], dim=1)
        return self.classifier(fused)


# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def generate(self, wafer_tensor, meta_tensor, class_idx):
        self.model.zero_grad(set_to_none=True)

        if meta_tensor is None:
            logits = self.model(wafer_tensor)
        else:
            logits = self.model(wafer_tensor, meta_tensor)

        pred_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        grads = self.gradients[0]
        acts  = self.activations[0]

        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), pred_idx


# ============================================================
# HELPERS
# ============================================================
def overlay_cam_on_wafer(wafer_np, cam, alpha=0.45):
    h, w = wafer_np.shape
    cam = cv2.resize(cam, (w, h))

    wafer_uint8 = (wafer_np * 255).astype(np.uint8)
    wafer_rgb = cv2.cvtColor(wafer_uint8, cv2.COLOR_GRAY2RGB)

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(wafer_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


def pick_n_samples_per_class(dataset, num_classes, n=3):
    class_to_indices = {i: [] for i in range(num_classes)}

    for idx in range(len(dataset)):
        label = int(dataset.labels[idx])
        if len(class_to_indices[label]) < n:
            class_to_indices[label].append(idx)

        done = all(len(v) >= n for v in class_to_indices.values())
        if done:
            break

    return class_to_indices


def sanitize_filename(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


# ============================================================
# MAIN VISUALIZATION
# One figure per class:
#   rows    = 3 samples
#   columns = original + 3 models
# ============================================================
def generate_side_by_side_gradcam_figures(
    models_dict,
    test_dataset,
    class_names,
    num_classes,
    save_dir,
    samples_per_class=3,
    alpha=0.45
):
    os.makedirs(save_dir, exist_ok=True)

    sample_map = pick_n_samples_per_class(
        test_dataset,
        num_classes=num_classes,
        n=samples_per_class
    )

    gradcam_objects = {}
    for model_name, info in models_dict.items():
        gradcam_objects[model_name] = GradCAM(info["model"], info["target_layer"])

    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        indices = sample_map.get(class_idx, [])

        if len(indices) == 0:
            print(f"[!] No samples found for class: {class_name}")
            continue

        n_rows = samples_per_class
        n_cols = 1 + len(models_dict)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 4.0 * n_rows),
            dpi=140
        )

        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        headers = ["Original"] + list(models_dict.keys())
        for c in range(n_cols):
            axes[0, c].set_title(headers[c], fontsize=12, fontweight="bold")

        for row in range(n_rows):
            if row >= len(indices):
                for c in range(n_cols):
                    axes[row, c].axis("off")
                continue

            ds_idx = indices[row]
            wafer_t, meta_t, label_t = test_dataset[ds_idx]

            wafer_np = wafer_t.squeeze(0).numpy()
            true_idx = int(label_t.item())
            true_name = class_names[true_idx]

            wafer_in = wafer_t.unsqueeze(0).to(DEVICE)
            meta_in  = meta_t.unsqueeze(0).to(DEVICE)

            # ------------------------------------------------
            # Column 0: Original wafer
            # ------------------------------------------------
            axes[row, 0].imshow(wafer_np, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_ylabel(
                f"{true_name}\nSample {row + 1}\nIdx {ds_idx}",
                fontsize=10,
                fontweight="bold",
                rotation=0,
                labelpad=55,
                va="center"
            )
            axes[row, 0].text(
                0.02, 0.02,
                f"True: {true_name}",
                transform=axes[row, 0].transAxes,
                fontsize=9,
                color="yellow",
                bbox=dict(facecolor="black", alpha=0.6, pad=2)
            )
            axes[row, 0].axis("off")

            # ------------------------------------------------
            # Other columns: Grad-CAM for each model
            # ------------------------------------------------
            for col, model_name in enumerate(models_dict.keys(), start=1):
                model_info = models_dict[model_name]
                model = model_info["model"]
                needs_meta = model_info["needs_meta"]
                gradcam = gradcam_objects[model_name]

                try:
                    cam, pred_idx = gradcam.generate(
                        wafer_tensor=wafer_in,
                        meta_tensor=meta_in if needs_meta else None,
                        class_idx=true_idx
                    )

                    overlay = overlay_cam_on_wafer(wafer_np, cam, alpha=alpha)
                    pred_name = class_names[pred_idx]

                    axes[row, col].imshow(overlay)
                    axes[row, col].text(
                        0.02, 0.02,
                        f"True: {true_name}\nPred: {pred_name}",
                        transform=axes[row, col].transAxes,
                        fontsize=9,
                        color="white",
                        bbox=dict(
                            facecolor="green" if pred_idx == true_idx else "red",
                            alpha=0.65,
                            pad=3
                        )
                    )
                    axes[row, col].axis("off")

                except Exception as e:
                    axes[row, col].text(
                        0.5, 0.5,
                        f"Error:\n{str(e)}",
                        ha="center", va="center",
                        fontsize=8, color="red",
                        transform=axes[row, col].transAxes
                    )
                    axes[row, col].axis("off")

        fig.suptitle(
            f"Grad-CAM Side-by-Side | Class: {class_name} | 3 Samples",
            fontsize=15,
            fontweight="bold",
            y=0.995
        )

        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"class_{class_idx}_{sanitize_filename(class_name)}_side_by_side.png"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=180)
        plt.close(fig)

        print(f" Saved: {save_path}")

    for gc in gradcam_objects.values():
        gc.remove_hooks()


# ============================================================
# LOAD TEST SET EXACTLY LIKE TRAINING PIPELINE
# ============================================================
def build_test_dataset():
    (
        wafer_train_raw, wafer_test_raw,
        X_train_meta, X_test_meta,
        y_train_full, y_test,
        class_names, num_classes
    ) = load_preprocessed_data(PREPROCESSED_PICKLE)

    all_wafer  = np.concatenate([wafer_train_raw, wafer_test_raw], axis=0)
    all_meta   = np.concatenate([X_train_meta, X_test_meta], axis=0)
    all_labels = np.concatenate([y_train_full, y_test], axis=0)

    X_wafer_tr_raw, X_wafer_te_raw, \
    X_meta_tr_full, X_meta_te, \
    y_tr_full, y_te = train_test_split(
        all_wafer, all_meta, all_labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=all_labels
    )

    X_wafer_te = prepare_wafer_arrays(X_wafer_te_raw)

    test_ds = WaferDefectDataset(
        wafer_maps=X_wafer_te,
        meta_features=X_meta_te,
        labels=y_te,
        augment=False
    )

    return test_ds, class_names, num_classes, X_meta_te.shape[1]


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(GRADCAM_SAVE_PATH, exist_ok=True)

    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------
    test_ds, class_names, num_classes, num_meta = build_test_dataset()

    print(f"Test dataset size: {len(test_ds)}")
    print(f"Number of classes: {num_classes}")
    print(f"Meta feature size: {num_meta}")

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------
    print("\nLoading trained models...")

    cnn_model = BaselineCNN(num_classes=num_classes).to(DEVICE)
    cnn_model.load_state_dict(
        torch.load(
            os.path.join(RESULTS_SAVE_PATH, "best_baseline_cnn.pt"),
            map_location=DEVICE
        )
    )
    cnn_model.eval()

    rn18_model = ResNet18Wafer(num_classes=num_classes).to(DEVICE)
    rn18_model.load_state_dict(
        torch.load(
            os.path.join(RESULTS_SAVE_PATH, "best_resnet18.pt"),
            map_location=DEVICE
        )
    )
    rn18_model.eval()

    wmn_model = WaferMetaNet(
        num_meta_features=num_meta,
        num_classes=num_classes,
        dropout=DROPOUT_RATE
    ).to(DEVICE)
    wmn_model.load_state_dict(
        torch.load(
            os.path.join(RESULTS_SAVE_PATH, "best_wafermetanet.pt"),
            map_location=DEVICE
        )
    )
    wmn_model.eval()

    # --------------------------------------------------------
    # Target layers for Grad-CAM
    # --------------------------------------------------------
    # BaselineCNN:
    # last convolution layer is features[24]
    cnn_target_layer = cnn_model.features[24]

    # ResNet18:
    # final high-level block
    rn18_target_layer = rn18_model.resnet.layer4[-1]

    # WaferMetaNet:
    # last encoder residual block before adaptive pooling
    wmn_target_layer = wmn_model.wafer_encoder[4]

    models_dict = {
        "Baseline CNN": {
            "model": cnn_model,
            "target_layer": cnn_target_layer,
            "needs_meta": False
        },
        "ResNet18": {
            "model": rn18_model,
            "target_layer": rn18_target_layer,
            "needs_meta": False
        },
        "WaferMetaNet": {
            "model": wmn_model,
            "target_layer": wmn_target_layer,
            "needs_meta": True
        }
    }

    # --------------------------------------------------------
    # Generate side-by-side Grad-CAM figures
    # --------------------------------------------------------
    print("\nGenerating Grad-CAM side-by-side figures...")
    generate_side_by_side_gradcam_figures(
        models_dict=models_dict,
        test_dataset=test_ds,
        class_names=class_names,
        num_classes=num_classes,
        save_dir=GRADCAM_SAVE_PATH,
        samples_per_class=3,
        alpha=0.45
    )

    print("\n All Grad-CAM figures generated successfully!")
    print(f" Saved in: {GRADCAM_SAVE_PATH}")


if __name__ == "__main__":
    main()
