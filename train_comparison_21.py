# ==================================================
# WAFER DEFECT CLASSIFICATION — MODEL COMPARISON
# Models:
#   1. Baseline CNN
#   2. ResNet18 (Transfer Learning / Adapted)
#   3. Proposed WaferMetaNet (SE + ResConv + Meta)
#
# Metrics:
#   - F1 (macro + per-class)
#   - Accuracy
#   - ROC AUC (per-class + macro)
#   - Confusion Matrix (raw + normalized, high-res)
#   - Training Time
#   - Training History & Curves
# ==================================================

import os
import csv
import time
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
    precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import models
from tqdm import tqdm


# ==================================================
# Configuration
# ==================================================
PREPROCESSED_PICKLE = r"G:\My Drive\AI_570\Final_project_Version15April\data\processed\preprocessed_wm811k_improved.pkl"
RESULTS_SAVE_PATH   = r"G:\My Drive\AI_570\Final_project_Version15April\results_comparison"



BOOST_CLASSES    = (2, 4)          # edge-loc, loc
BOOST_FACTOR     = 4.0
NONE_CLASS       = 6
NONE_SAMPLE_MULT = 0.35
FOCUS_CLASSES    = (2, 4)

WAFER_SIZE       = 64
BATCH_SIZE       = 64
EPOCHS           = 120 #120
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 2e-4
DROPOUT_RATE     = 0.30
RANDOM_STATE     = 42
VAL_SIZE         = 0.15
PATIENCE         = 25
NUM_WORKERS      = 0
USE_AMP          = True
FOCAL_GAMMA      = 4.0
LABEL_SMOOTHING  = 0.03
MIN_LR           = 1e-7
GRAD_CLIP        = 0.5
COSINE_T0        = 15
COSINE_T_MULT    = 2
#BOOST_CLASSES    = (0, 2, 4, 8)
#BOOST_FACTOR     = 2.5
NUM_CLASSES_GLOBAL = 9

CLASS_NAMES_GLOBAL = [
    "center", "donut", "edge-loc", "edge-ring",
    "loc", "near-full", "none", "random", "scratch"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# Reproducibility
# ==================================================
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ==================================================
# Data Loading
# ==================================================
def load_preprocessed_data(pickle_path):
    print("=" * 70)
    print("Loading Preprocessed Data...")
    print("=" * 70)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    print(f"Keys found: {list(data.keys())}")
    wafer_train_raw = data["waferMap_train_resized"]
    wafer_test_raw  = data["waferMap_test_resized"]
    X_train_meta    = np.array(data["X_train_meta"], dtype=np.float32)
    X_test_meta     = np.array(data["X_test_meta"],  dtype=np.float32)
    y_train         = np.array(data["y_train_enc"],  dtype=np.int64)
    y_test          = np.array(data["y_test_enc"],   dtype=np.int64)
    class_names     = list(data["class_names"])
    num_classes     = int(data["num_classes"])
    print(f"Train samples : {len(wafer_train_raw)}")
    print(f"Test  samples : {len(wafer_test_raw)}")
    print(f"Classes       : {class_names}")
    print(f"Meta features : {X_train_meta.shape[1]}")
    return (wafer_train_raw, wafer_test_raw,
            X_train_meta, X_test_meta,
            y_train, y_test, class_names, num_classes)


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


# ==================================================
# Dataset
# ==================================================
class WaferDefectDataset(Dataset):
    def __init__(self, wafer_maps, meta_features, labels, augment=False):
        self.wafer_maps    = wafer_maps
        self.meta_features = meta_features
        self.labels        = labels
        self.augment       = augment

    def __len__(self):
        return len(self.labels)

    def _augment_wafer(self, wafer):
        if random.random() < 0.5:
            wafer = np.flip(wafer, axis=2).copy()
        if random.random() < 0.5:
            wafer = np.flip(wafer, axis=1).copy()
        k = random.randint(0, 3)
        if k > 0:
            wafer = np.rot90(wafer, k=k, axes=(1, 2)).copy()
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.03, wafer.shape).astype(np.float32)
            wafer = np.clip(wafer + noise, 0.0, 1.0)
        if random.random() < 0.35:
            _, h, w = wafer.shape
            eh = random.randint(4, h // 3)
            ew = random.randint(4, w // 3)
            y0 = random.randint(0, h - eh)
            x0 = random.randint(0, w - ew)
            wafer[:, y0:y0 + eh, x0:x0 + ew] = 0.0
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            wafer = np.clip(wafer * factor, 0.0, 1.0)
        return wafer

    def __getitem__(self, idx):
        wafer = self.wafer_maps[idx].copy()
        meta  = self.meta_features[idx]
        label = self.labels[idx]
        if self.augment:
            wafer = self._augment_wafer(wafer)
        return (
            torch.tensor(wafer, dtype=torch.float32),
            torch.tensor(meta,  dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# ==================================================
# Mixup Augmentation
# ==================================================
def mixup_batch(wafers, meta, labels, num_classes, alpha=0.3):
    lam        = np.random.beta(alpha, alpha)
    batch_size = wafers.size(0)
    index      = torch.randperm(batch_size, device=wafers.device)
    mixed_wafers = lam * wafers + (1 - lam) * wafers[index]
    mixed_meta   = lam * meta   + (1 - lam) * meta[index]
    y_a          = F.one_hot(labels,        num_classes=num_classes).float()
    y_b          = F.one_hot(labels[index], num_classes=num_classes).float()
    mixed_labels = lam * y_a + (1 - lam) * y_b
    return mixed_wafers, mixed_meta, mixed_labels


# ==================================================
# Imbalance Handling
# ==================================================
def compute_class_balanced_alpha(y, num_classes, beta=0.9999):
    counts        = np.bincount(y, minlength=num_classes).astype(np.float32)
    effective_num = 1.0 - np.power(beta, counts)
    effective_num = np.clip(effective_num, 1e-8, None)
    alpha         = (1.0 - beta) / effective_num
    alpha         = alpha / alpha.sum() * num_classes
    return alpha.astype(np.float32), counts.astype(np.int64)


"""def make_weighted_sampler(y, num_classes, boost_classes=BOOST_CLASSES, boost_factor=BOOST_FACTOR):
    counts             = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts             = np.clip(counts, 1.0, None)
    per_class_weights  = 1.0 / counts
    for cls in boost_classes:
        if 0 <= cls < num_classes:
            per_class_weights[cls] *= boost_factor
    sample_weights = per_class_weights[y]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(y),
        replacement=True
    )
    return sampler, counts"""


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def make_weighted_sampler(
    y,
    num_classes,
    boost_classes=BOOST_CLASSES,
    boost_factor=BOOST_FACTOR,
    none_class=NONE_CLASS,
    none_sample_mult=NONE_SAMPLE_MULT
):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)

    per_class_weights = 1.0 / counts

    for cls in boost_classes:
        if 0 <= cls < num_classes:
            per_class_weights[cls] *= boost_factor

    if 0 <= none_class < num_classes:
        per_class_weights[none_class] *= none_sample_mult

    sample_weights = per_class_weights[y]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(y),
        replacement=True
    )
    return sampler, counts
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ==================================================
# Focal Loss
# ==================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.reduction       = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        log_probs   = F.log_softmax(logits, dim=1)
        probs       = log_probs.exp()
        if targets.dtype == torch.long:
            if self.label_smoothing > 0.0:
                with torch.no_grad():
                    smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                    smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
                ce_loss = -(smooth_targets * log_probs).sum(dim=1)
            else:
                ce_loss = F.cross_entropy(logits, targets, reduction="none")
            pt      = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            alpha_t = None
            if self.alpha is not None:
                alpha   = self.alpha.to(logits.device)
                alpha_t = alpha.gather(0, targets)
        else:
            ce_loss = -(targets * log_probs).sum(dim=1)
            pt      = (targets * probs).sum(dim=1)
            alpha_t = None
        pt         = torch.clamp(pt, 1e-6, 1.0 - 1e-6)
        focal_term = (1.0 - pt) ** self.gamma
        loss       = focal_term * ce_loss
        if alpha_t is not None:
            loss = alpha_t * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""class ConfusionAwareFocalLoss(FocalLoss):
    def __init__(
        self,
        alpha=None,
        gamma=2.5,
        reduction="mean",
        label_smoothing=0.0,
        none_class=NONE_CLASS,
        focus_classes=FOCUS_CLASSES,
        none_penalty=0.35
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction=reduction, label_smoothing=label_smoothing)
        self.none_class = none_class
        self.focus_classes = focus_classes
        self.none_penalty = none_penalty

    def forward(self, logits, targets):
        base_loss = super().forward(logits, targets)

        probs = F.softmax(logits, dim=1)
        none_prob = probs[:, self.none_class]

        focus_mask = torch.zeros_like(targets, dtype=torch.float32)
        for cls in self.focus_classes:
            focus_mask += (targets == cls).float()

        penalty = (focus_mask * none_prob).mean()
        return base_loss + self.none_penalty * penalty"""



class ConfusionAwareFocalLoss(FocalLoss):
    def __init__(
        self,
        alpha=None,
        gamma=2.5,
        reduction="mean",
        label_smoothing=0.0,
        none_class=NONE_CLASS,
        focus_classes=FOCUS_CLASSES,
        none_penalty=0.35
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction=reduction, label_smoothing=label_smoothing)
        self.none_class    = none_class
        self.focus_classes = focus_classes
        self.none_penalty  = none_penalty

    def forward(self, logits, targets):
        base_loss = super().forward(logits, targets)

        probs     = F.softmax(logits, dim=1)
        none_prob = probs[:, self.none_class]   # shape: [batch_size]

        # ── Handle BOTH hard labels (1D long) and soft labels (2D float from Mixup) ──
        if targets.dtype == torch.long:
            # Hard labels: targets shape is [batch_size]
            focus_mask = torch.zeros(logits.size(0), dtype=torch.float32, device=logits.device)
            for cls in self.focus_classes:
                focus_mask += (targets == cls).float()
        else:
            # Soft labels from Mixup: targets shape is [batch_size, num_classes]
            # Use the summed probability mass over focus classes as a soft mask
            focus_mask = torch.zeros(logits.size(0), dtype=torch.float32, device=logits.device)
            for cls in self.focus_classes:
                focus_mask += targets[:, cls]   # shape: [batch_size]

        # Both focus_mask and none_prob are now [batch_size] — safe to multiply
        penalty = (focus_mask * none_prob).mean()
        return base_loss + self.none_penalty * penalty


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
# ==================================================
# Early Stopping
# ==================================================
class EarlyStopping:
    def __init__(self, patience=PATIENCE, delta=1e-4, save_path="best_model.pt"):
        self.patience   = patience
        self.delta      = delta
        self.save_path  = save_path
        self.best_score = -np.inf
        self.counter    = 0
        self.early_stop = False

    def step(self, score, model):
        if score > self.best_score + self.delta:
            self.best_score = score
            self.counter    = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"   New best model saved (Val Macro F1: {score:.4f})")
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ==================================================
# Metrics Helpers
#----------------------------------------------------
def compute_basic_metrics(y_true, y_pred):
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, macro_f1, weighted_f1



# ─────────────────────────────────────────────────
#  MODEL 1: Baseline CNN (wafer-map only, no meta)
# ─────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """Simple 4-block CNN — no attention, no meta-features."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Block 4
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

    def forward(self, wafer, meta=None):  # meta ignored
        x = self.features(wafer)
        return self.classifier(x)


# ==================================================
# ─────────────────────────────────────────────────
#  MODEL 2: ResNet18 (adapted for 1-channel input)
# ─────────────────────────────────────────────────
# ==================================================
class ResNet18Wafer(nn.Module):
    """
    Standard ResNet-18 adapted for:
      - 1-channel (grayscale) wafer maps
      - 64×64 input size
      - No meta-feature fusion
    """
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        base = models.resnet18(weights=None if not pretrained else "IMAGENET1K_V1")

        # Adapt first conv to accept 1 channel
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove the initial MaxPool (too aggressive for 64×64)
        base.maxpool = nn.Identity()

        # Replace FC head
        in_features    = base.fc.in_features
        base.fc        = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.resnet = base

    def forward(self, wafer, meta=None):  # meta ignored
        return self.resnet(wafer)


# ==================================================
# ─────────────────────────────────────────────────
#  MODEL 3: Proposed WaferMetaNet (SE + ResConv + Meta)
# ─────────────────────────────────────────────────
# ==================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
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
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) \
                        if in_ch != out_ch else nn.Identity()
        self.pool     = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out      = self.conv(x)
        out      = self.se(out)
        out      = self.relu(out + identity)
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


# ==================================================
# Per-class threshold optimizer
# ==================================================
@torch.no_grad()
def find_optimal_thresholds(model, loader, device, num_classes, n_trials=40):
    model.eval()
    all_probs, all_labels = [], []
    for wafers, meta, labels in loader:
        wafers = wafers.to(device); meta = meta.to(device)
        logits = model(wafers, meta)
        probs  = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    best_thresholds = np.ones(num_classes)
    best_f1         = 0.0
    candidates = np.linspace(0.2, 2.0, n_trials)
    for cls in range(num_classes):
        for t in candidates:
            thresholds       = best_thresholds.copy()
            thresholds[cls]  = t
            adjusted         = all_probs / thresholds
            adjusted        /= adjusted.sum(axis=1, keepdims=True)
            preds            = np.argmax(adjusted, axis=1)
            score            = f1_score(all_labels, preds, average="macro", zero_division=0)
            if score > best_f1:
                best_f1              = score
                best_thresholds[cls] = t
    return best_thresholds, best_f1


def apply_thresholds(logits, thresholds, device):
    probs    = F.softmax(logits, dim=1)
    t_tensor = torch.tensor(thresholds, dtype=torch.float32, device=device)
    adjusted = probs / t_tensor.unsqueeze(0)
    adjusted /= adjusted.sum(dim=1, keepdim=True)
    return torch.argmax(adjusted, dim=1), adjusted


# ==================================================
# Train / Evaluate Functions
# ==================================================
def train_one_epoch(model, loader, criterion, optimizer, device,
                    num_classes, scaler=None, use_mixup=True, mixup_alpha=0.3):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="[TRAIN]", leave=False)
    for wafers, meta, labels in pbar:
        wafers = wafers.to(device, non_blocking=True)
        meta   = meta.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        apply_mix = use_mixup and random.random() < 0.5
        if apply_mix:
            wafers, meta, soft_labels = mixup_batch(wafers, meta, labels, num_classes, alpha=mixup_alpha)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(wafers, meta)
                loss   = criterion(logits, soft_labels if apply_mix else labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(wafers, meta)
            loss   = criterion(logits, soft_labels if apply_mix else labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        hard_labels = labels if not apply_mix else torch.argmax(soft_labels, dim=1)
        all_labels.extend(hard_labels.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    epoch_loss         = running_loss / len(loader.dataset)
    acc, macro_f1, wf1 = compute_basic_metrics(all_labels, all_preds)
    return {"loss": epoch_loss, "acc": acc, "macro_f1": macro_f1, "weighted_f1": wf1}


@torch.no_grad()
def evaluate(model, loader, criterion, device, thresholds=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for wafers, meta, labels in tqdm(loader, desc="[EVAL]", leave=False):
        wafers = wafers.to(device); meta = meta.to(device); labels = labels.to(device)
        logits = model(wafers, meta)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        probs = F.softmax(logits, dim=1)
        all_probs.extend(probs.cpu().numpy())
        if thresholds is not None:
            preds, _ = apply_thresholds(logits, thresholds, device)
        else:
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss           = total_loss / len(loader.dataset)
    acc, macro_f1, wf1 = compute_basic_metrics(all_labels, all_preds)
    return {
        "loss": avg_loss, "acc": acc, "macro_f1": macro_f1, "weighted_f1": wf1,
        "y_true": all_labels, "y_pred": all_preds, "y_probs": np.array(all_probs)
    }


# ==================================================
# Full Training Pipeline (generic for all models)
# ==================================================
def run_training(
    model_name, model, train_loader, val_loader, test_loader,
    criterion, num_classes, save_path, use_mixup=True
):
    optimizer  = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=MIN_LR)
    scaler     = torch.cuda.amp.GradScaler() if (torch.cuda.is_available() and USE_AMP) else None
    early_stop = EarlyStopping(patience=PATIENCE, save_path=save_path)

    history    = {k: [] for k in ["train_loss", "val_loss",
                                   "train_macro_f1", "val_macro_f1",
                                   "train_acc", "val_acc"]}
    best_f1    = -np.inf

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    t_start = time.time()

    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE,
                                  num_classes, scaler, use_mixup=use_mixup)
        val_m   = evaluate(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_macro_f1"].append(train_m["macro_f1"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        history["train_acc"].append(train_m["acc"])
        history["val_acc"].append(val_m["acc"])

        print(f"  Train → Loss: {train_m['loss']:.4f} | Acc: {train_m['acc']*100:.2f}% | Macro F1: {train_m['macro_f1']*100:.2f}%")
        print(f"  Val   → Loss: {val_m['loss']:.4f}   | Acc: {val_m['acc']*100:.2f}% | Macro F1: {val_m['macro_f1']*100:.2f}%")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            torch.save(model.state_dict(), save_path)
            print(f"   Best model updated (F1: {best_f1:.4f})")

        scheduler.step(epoch + val_m["macro_f1"] / EPOCHS)
        early_stop.step(val_m["macro_f1"], model)
        if early_stop.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    training_time = time.time() - t_start
    print(f"\n  Training time for {model_name}: {training_time:.1f}s ({training_time/60:.1f} min)")

    # Reload best and evaluate on test
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    # Use threshold optimization only for the proposed model
    if "Proposed" in model_name:
        best_thresholds, _ = find_optimal_thresholds(model, val_loader, DEVICE, num_classes)
    else:
        best_thresholds = None

    test_m = evaluate(model, test_loader, criterion, DEVICE, thresholds=best_thresholds)
    return history, test_m, training_time


# ==================================================
# Plotting: Individual Training Curves
# ==================================================
def plot_training_curves(history, model_name, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"Training History — {model_name}", fontsize=15, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", color="steelblue", lw=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="tomato",    lw=2)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Macro F1
    axes[1].plot(epochs, history["train_macro_f1"], label="Train", color="steelblue", lw=2)
    axes[1].plot(epochs, history["val_macro_f1"],   label="Val",   color="tomato",    lw=2)
    axes[1].set_title("Macro F1"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Accuracy
    axes[2].plot(epochs, history["train_acc"], label="Train", color="steelblue", lw=2)
    axes[2].plot(epochs, history["val_acc"],   label="Val",   color="tomato",    lw=2)
    axes[2].set_title("Accuracy"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"training_curves_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ==================================================
# Plotting: Confusion Matrix (High-Res)
# ==================================================
def plot_confusion_matrix(y_true, y_pred, class_names, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, axes = plt.subplots(1, 2, figsize=(26, 10))
    titles = ["Raw Count Confusion Matrix", "Normalized Confusion Matrix"]
    cms    = [cm, cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)]
    fmts   = ["d", ".2f"]
    cmaps  = ["Blues", "YlOrRd"]

    for ax, title, c, fmt, cmap in zip(axes, titles, cms, fmts, cmaps):
        sns.heatmap(c, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, linecolor="gray",
                    annot_kws={"size": 9})
        ax.set_title(f"{model_name} — {title}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle(f"Confusion Matrices — {model_name}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ==================================================
# Plotting: ROC-AUC (High-Res, per class + macro)
# ==================================================
def plot_roc_auc(y_true, y_probs, class_names, model_name, out_dir):
    num_classes = len(class_names)
    y_bin       = label_binarize(y_true, classes=list(range(num_classes)))
    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(num_classes):
        fpr_d[i], tpr_d[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc_d[i]               = auc(fpr_d[i], tpr_d[i])
    # Macro
    all_fpr  = np.unique(np.concatenate([fpr_d[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_d[i], tpr_d[i])
    mean_tpr  /= num_classes
    macro_auc  = auc(all_fpr, mean_tpr)

    palette    = plt.cm.get_cmap("tab10", num_classes)
    fig, axes  = plt.subplots(3, 3, figsize=(20, 16))
    axes       = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_classes:
            ax.plot(fpr_d[i], tpr_d[i], color=palette(i), lw=2,
                    label=f"AUC = {auc_d[i]:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax.fill_between(fpr_d[i], tpr_d[i], alpha=0.15, color=palette(i))
            ax.set_title(f"Class {i}: {class_names[i]}", fontsize=11, fontweight="bold")
            ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
        else:
            ax.plot(all_fpr, mean_tpr, color="darkorange", lw=2.5,
                    label=f"Macro AUC = {macro_auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            for i2 in range(num_classes):
                ax.plot(fpr_d[i2], tpr_d[i2], lw=0.8, alpha=0.5, color=palette(i2),
                        label=f"{class_names[i2]} ({auc_d[i2]:.3f})")
            ax.fill_between(all_fpr, mean_tpr, alpha=0.1, color="darkorange")
            ax.set_title("Macro-Average ROC", fontsize=11, fontweight="bold")
            ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
            ax.legend(loc="lower right", fontsize=6)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"ROC-AUC — {model_name}\nMacro AUC: {macro_auc:.4f}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"roc_auc_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    return macro_auc


# ==================================================
# Plotting: Overlay Training Curves (all 3 models)
# ==================================================
def plot_overlay_curves(all_histories, model_names, out_dir):
    colors  = ["steelblue", "darkorange", "green"]
    metrics = [
        ("train_loss",     "val_loss",     "Loss",     "Loss"),
        ("train_macro_f1", "val_macro_f1", "Macro F1", "Macro F1"),
        ("train_acc",      "val_acc",      "Accuracy", "Accuracy"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle("Training Curves — All Models Comparison", fontsize=16, fontweight="bold")

    for col, (train_key, val_key, title, ylabel) in enumerate(metrics):
        ax_train = axes[0][col]
        ax_val   = axes[1][col]
        for h, name, c in zip(all_histories, model_names, colors):
            epochs = range(1, len(h[train_key]) + 1)
            ax_train.plot(epochs, h[train_key], label=name, color=c, lw=2)
            ax_val.plot(  epochs, h[val_key],   label=name, color=c, lw=2)
        ax_train.set_title(f"Train {title}"); ax_train.set_ylabel(ylabel)
        ax_train.legend(); ax_train.grid(True, alpha=0.3)
        ax_val.set_title(f"Val {title}");   ax_val.set_ylabel(ylabel)
        ax_val.set_xlabel("Epoch")
        ax_val.legend(); ax_val.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, "overlay_training_curves_all_models.png")
    plt.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved overlay curves: {fname}")


# ==================================================
# Plotting: Final Comparison Bar Chart
# ==================================================
def plot_comparison_bar(summary, out_dir):
    model_names = [s["model"] for s in summary]
    metrics     = ["Accuracy", "Macro F1", "Weighted F1", "Macro AUC"]
    values      = {m: [s[m] for s in summary] for m in metrics}
    colors      = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    fig.suptitle("Model Comparison — All Metrics", fontsize=16, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(model_names, values[metric], color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim([max(0, min(values[metric]) - 0.05), 1.02])
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        for bar, val in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(out_dir, "comparison_bar_chart.png")
    plt.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved bar chart: {fname}")


# ==================================================
# Plotting: Training Time Bar Chart
# ==================================================
def plot_training_time(summary, out_dir):
    model_names = [s["model"] for s in summary]
    times       = [s["Training Time (min)"] for s in summary]
    colors      = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, times, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("Training Time Comparison (minutes)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Minutes")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.3,
                f"{t:.1f} min", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, "training_time_comparison.png")
    plt.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved training time chart: {fname}")


# ==================================================
# Save Summary CSV
# ==================================================
def save_summary_csv(summary, out_dir):
    fname = os.path.join(out_dir, "model_comparison_summary.csv")
    keys  = list(summary[0].keys())
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    print(f"  Saved summary CSV: {fname}")


# ---------------------------------------------------
# Print Distribution
#-----------------------------------------------------
def print_class_distribution(y, class_names, title):
    print(f"\n{title}")
    counts = np.bincount(y, minlength=len(class_names))
    total  = counts.sum()
    for i, name in enumerate(class_names):
        pct = 100.0 * counts[i] / total if total > 0 else 0.0
        print(f"  Class {i:>2} | {name:<15}: {counts[i]:>7} ({pct:>6.2f}%)")


#---------------------------------------------------------------
# MAIN
#----------------------------------------------------------------

def main():
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # ── Load & split data ──
    (wafer_train_raw, wafer_test_raw,
     X_train_meta, X_test_meta,
     y_train_full, y_test,
     class_names, num_classes) = load_preprocessed_data(PREPROCESSED_PICKLE)

    all_wafer  = np.concatenate([wafer_train_raw, wafer_test_raw], axis=0)
    all_meta   = np.concatenate([X_train_meta,    X_test_meta],    axis=0)
    all_labels = np.concatenate([y_train_full,    y_test],         axis=0)

    X_wafer_tr_raw, X_wafer_te_raw, \
    X_meta_tr_full, X_meta_te,      \
    y_tr_full, y_te = train_test_split(
        all_wafer, all_meta, all_labels,
        test_size=0.2, random_state=RANDOM_STATE, stratify=all_labels
    )

    X_wafer_tr_full = prepare_wafer_arrays(X_wafer_tr_raw)
    X_wafer_te      = prepare_wafer_arrays(X_wafer_te_raw)

    (X_wafer_tr, X_wafer_val,
     X_meta_tr,  X_meta_val,
     y_tr, y_val) = train_test_split(
        X_wafer_tr_full, X_meta_tr_full, y_tr_full,
        test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_tr_full
    )

    print_class_distribution(y_tr,  class_names, "Training")
    print_class_distribution(y_val, class_names, "Validation")
    print_class_distribution(y_te,  class_names, "Test")

    # ── Datasets ──
    train_ds = WaferDefectDataset(X_wafer_tr,  X_meta_tr,  y_tr,  augment=True)
    val_ds   = WaferDefectDataset(X_wafer_val, X_meta_val, y_val, augment=False)
    test_ds  = WaferDefectDataset(X_wafer_te,  X_meta_te,  y_te,  augment=False)

    sampler, _ = make_weighted_sampler(y_tr, num_classes)
    largs      = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
                  "pin_memory": torch.cuda.is_available()}
    train_ldr  = DataLoader(train_ds, sampler=sampler, shuffle=False, **largs)
    val_ldr    = DataLoader(val_ds,   shuffle=False, **largs)
    test_ldr   = DataLoader(test_ds,  shuffle=False, **largs)

    # ── Shared criterion ──
    alpha_np, _   = compute_class_balanced_alpha(y_tr, num_classes)
    alpha_tensor  = torch.tensor(alpha_np, dtype=torch.float32, device=DEVICE)
    criterion = ConfusionAwareFocalLoss(
    alpha=alpha_tensor,
    gamma=FOCAL_GAMMA,
    reduction="mean",
    label_smoothing=LABEL_SMOOTHING,
    none_class=NONE_CLASS,
    focus_classes=FOCUS_CLASSES,
    none_penalty=0.35
    )

    # ─────────────────────────────────────────────
    # Define all three models
    # ─────────────────────────────────────────────
    num_meta = X_meta_tr.shape[1]

    model_configs = [
        {
            "name":      "Baseline CNN",
            "model":     BaselineCNN(num_classes=num_classes).to(DEVICE),
            "save_path": os.path.join(RESULTS_SAVE_PATH, "best_baseline_cnn.pt"),
            "use_mixup": False,   # Plain CNN — no mixup for cleaner baseline
        },
        {
            "name":      "ResNet18",
            "model":     ResNet18Wafer(num_classes=num_classes).to(DEVICE),
            "save_path": os.path.join(RESULTS_SAVE_PATH, "best_resnet18.pt"),
            "use_mixup": True,
        },
        {
            "name":      "Proposed WaferMetaNet",
            "model":     WaferMetaNet(num_meta_features=num_meta,
                                      num_classes=num_classes,
                                      dropout=DROPOUT_RATE).to(DEVICE),
            "save_path": os.path.join(RESULTS_SAVE_PATH, "best_wafermetanet.pt"),
            "use_mixup": True,
        },
    ]

    # Print param counts
    for cfg in model_configs:
        total     = sum(p.numel() for p in cfg["model"].parameters())
        trainable = sum(p.numel() for p in cfg["model"].parameters() if p.requires_grad)
        print(f"\n{cfg['name']}: {total:,} total | {trainable:,} trainable params")

    # ─────────────────────────────────────────────
    # TRAIN all models
    # ─────────────────────────────────────────────
    all_histories  = []
    all_test_mets  = []
    all_train_times = []

    for cfg in model_configs:
        set_seed()   # reset seed before each model for fair comparison
        hist, test_m, t_time = run_training(
            model_name   = cfg["name"],
            model        = cfg["model"],
            train_loader = train_ldr,
            val_loader   = val_ldr,
            test_loader  = test_ldr,
            criterion    = criterion,
            num_classes  = num_classes,
            save_path    = cfg["save_path"],
            use_mixup    = cfg["use_mixup"]
        )
        all_histories.append(hist)
        all_test_mets.append(test_m)
        all_train_times.append(t_time)

        # Individual plots
        plot_training_curves(hist, cfg["name"], RESULTS_SAVE_PATH)
        plot_confusion_matrix(test_m["y_true"], test_m["y_pred"],
                              class_names, cfg["name"], RESULTS_SAVE_PATH)

    # ─────────────────────────────────────────────
    # ROC AUC + collect macro AUC per model
    # ─────────────────────────────────────────────
    model_names_list = [cfg["name"] for cfg in model_configs]
    macro_aucs       = []

    for cfg, test_m in zip(model_configs, all_test_mets):
        m_auc = plot_roc_auc(test_m["y_true"], test_m["y_probs"],
                             class_names, cfg["name"], RESULTS_SAVE_PATH)
        macro_aucs.append(m_auc)

    # ─────────────────────────────────────────────
    # Overlay comparison curves
    # ─────────────────────────────────────────────
    plot_overlay_curves(all_histories, model_names_list, RESULTS_SAVE_PATH)

    # ─────────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────────
    summary = []
    for cfg, test_m, t_time, m_auc in zip(
        model_configs, all_test_mets, all_train_times, macro_aucs
    ):
        summary.append({
            "model":                   cfg["name"],
            "Accuracy":                round(test_m["acc"], 6),
            "Macro F1":                round(test_m["macro_f1"], 6),
            "Weighted F1":             round(test_m["weighted_f1"], 6),
            "Macro AUC":               round(m_auc, 6),
            "Training Time (s)":       round(t_time, 1),
            "Training Time (min)":     round(t_time / 60, 2),
        })

    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    header = f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Wtd F1':>10} {'AUC':>10} {'Time(min)':>11}"
    print(header)
    print("-" * len(header))
    for s in summary:
        print(
            f"{s['model']:<25} "
            f"{s['Accuracy']:>10.4f} "
            f"{s['Macro F1']:>10.4f} "
            f"{s['Weighted F1']:>10.4f} "
            f"{s['Macro AUC']:>10.4f} "
            f"{s['Training Time (min)']:>10.1f}m"
        )

    plot_comparison_bar(summary, RESULTS_SAVE_PATH)
    plot_training_time(summary, RESULTS_SAVE_PATH)
    save_summary_csv(summary, RESULTS_SAVE_PATH)

    print("\n All done!")
    print(f"   Results saved to → {RESULTS_SAVE_PATH}")


if __name__ == "__main__":
    main()
