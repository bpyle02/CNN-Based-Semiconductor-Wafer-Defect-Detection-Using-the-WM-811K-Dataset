#!/usr/bin/env python3
"""
Interactive Streamlit dashboard for model interpretation and analysis.

Features:
- Real-time model evaluation on test set
- Per-class performance metrics
- GradCAM visualizations
- Prediction analysis and error inspection
- Model comparison across architectures

Usage:
    streamlit run dashboard.py
"""

import sys
from pathlib import Path
from typing import Dict, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)
try:
    import streamlit as st
except ImportError:
    logger.warning("Error: streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import (
    load_dataset, preprocess_wafer_maps, get_image_transforms,
    get_imagenet_normalize, WaferMapDataset, seed_worker, KNOWN_CLASSES
)
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.analysis import evaluate_model, count_params, count_trainable
from src.config import load_config


@st.cache_resource
def load_config_cached() -> Any:
    """Load config with caching."""
    return load_config("config.yaml")


@st.cache_data
def load_data_cached(dataset_path: str) -> tuple:
    """
    Load and preprocess data with caching.

    Args:
        dataset_path: Path to WM-811K dataset

    Returns:
        Tuple of (train_maps, test_maps, y_train, y_test, label_encoder)
    """
    df = load_dataset(Path(dataset_path))
    labeled_mask = df['failureClass'].isin(KNOWN_CLASSES)
    df_clean = df[labeled_mask].reset_index(drop=True)

    le = LabelEncoder()
    df_clean['label_encoded'] = le.fit_transform(df_clean['failureClass'])

    wafer_maps = df_clean['waferMap'].values
    labels = df_clean['label_encoded'].values

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        np.arange(len(labels)), labels, test_size=0.30, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Preprocess
    train_maps = preprocess_wafer_maps([wafer_maps[i] for i in X_train])
    test_maps = preprocess_wafer_maps([wafer_maps[i] for i in X_test])

    return (train_maps, test_maps, y_train, y_test, le)


def create_model(model_type: str, num_classes: int, device: str = 'cpu') -> nn.Module:
    """Create model based on type."""
    if model_type == 'Custom CNN':
        return WaferCNN(num_classes=num_classes).to(device)
    elif model_type == 'ResNet-18':
        return get_resnet18(num_classes=num_classes).to(device)
    else:  # EfficientNet-B0
        return get_efficientnet_b0(num_classes=num_classes).to(device)


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(page_title="Wafer Defect Detection Dashboard", layout="wide")
    st.title("🔬 Semiconductor Wafer Defect Detection")
    st.markdown("**Interactive Model Analysis & Interpretation Dashboard**")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    config = load_config_cached()

    # Device selection
    device_option = st.sidebar.selectbox("Device", ['cpu', 'cuda'], index=0)
    device = torch.device(device_option)

    # Model selection
    model_options = ['Custom CNN', 'ResNet-18', 'EfficientNet-B0']
    selected_model = st.sidebar.selectbox("Model Architecture", model_options)

    # Checkpoint path
    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path (optional)",
        value="checkpoints/best_model.pth"
    )

    # Dataset path
    dataset_path = st.sidebar.text_input(
        "Dataset Path",
        value=str(config.data.dataset_path)
    )

    # Load data
    try:
        train_maps, test_maps, y_train, y_test, le = load_data_cached(dataset_path)
        class_names = le.classes_.tolist()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Create main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Analysis",
        "📈 Performance Metrics",
        "🎯 Per-Class Analysis",
        "🖼️ Sample Predictions"
    ])

    with tab1:
        st.header("Model Architecture & Performance")

        # Create model
        model = create_model(selected_model, len(class_names), str(device))

        # Load checkpoint if exists
        if Path(checkpoint_path).exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict)
                st.success(f"✅ Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                st.warning(f"⚠️ Could not load checkpoint: {e}")
        else:
            st.info("ℹ️ Using untrained model weights")

        # Model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", f"{count_params(model):,}")
        with col2:
            st.metric("Trainable Parameters", f"{count_trainable(model):,}")
        with col3:
            st.metric("Model", selected_model)

        # Evaluate on test set
        if st.button("📈 Evaluate Model on Test Set"):
            with st.spinner("Evaluating..."):
                # Create test dataset
                test_transform = None if selected_model == 'Custom CNN' else get_imagenet_normalize()
                test_dataset = WaferMapDataset(test_maps, y_test, transform=test_transform)
                g = torch.Generator().manual_seed(42)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g)

                # Evaluate
                preds, labels, metrics = evaluate_model(
                    model, test_loader, class_names, selected_model, str(device)
                )

                # Display metrics
                st.subheader("Overall Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
                with col3:
                    st.metric("Weighted F1", f"{metrics['weighted_f1']:.4f}")

                # Confusion matrix
                st.subheader("Confusion Matrix")
                conf_mat = confusion_matrix(labels, preds)
                conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(conf_mat_norm, annot=True, fmt='.2%', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_title(f"Confusion Matrix - {selected_model}")
                ax.set_ylabel("True Label")
                ax.set_xlabel("Predicted Label")
                st.pyplot(fig)

    with tab2:
        st.header("Detailed Performance Metrics")

        if st.button("📊 Compute Performance Metrics"):
            with st.spinner("Computing metrics..."):
                test_transform = None if selected_model == 'Custom CNN' else get_imagenet_normalize()
                test_dataset = WaferMapDataset(test_maps, y_test, transform=test_transform)
                g = torch.Generator().manual_seed(42)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g)

                preds, labels, metrics = evaluate_model(
                    model, test_loader, class_names, selected_model, str(device)
                )

                # Compute per-class metrics
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, preds, average=None, zero_division=0
                )

                # Display table
                metrics_df = {
                    'Class': class_names,
                    'Precision': [f"{p:.4f}" for p in precision],
                    'Recall': [f"{r:.4f}" for r in recall],
                    'F1-Score': [f"{f:.4f}" for f in f1],
                    'Support': support
                }

                st.dataframe(metrics_df, use_container_width=True)

                # Plot metrics
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                axes[0].bar(class_names, precision)
                axes[0].set_title("Precision by Class")
                axes[0].set_ylabel("Precision")
                axes[0].tick_params(axis='x', rotation=45)

                axes[1].bar(class_names, recall)
                axes[1].set_title("Recall by Class")
                axes[1].set_ylabel("Recall")
                axes[1].tick_params(axis='x', rotation=45)

                axes[2].bar(class_names, f1)
                axes[2].set_title("F1-Score by Class")
                axes[2].set_ylabel("F1-Score")
                axes[2].tick_params(axis='x', rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

    with tab3:
        st.header("Per-Class Analysis")

        st.info("Analyze performance across different defect classes")

        test_transform = None if selected_model == 'Custom CNN' else get_imagenet_normalize()
        test_dataset = WaferMapDataset(test_maps, y_test, transform=test_transform)
        g = torch.Generator().manual_seed(42)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g)

        preds, labels, _ = evaluate_model(
            model, test_loader, class_names, selected_model, str(device)
        )

        # Class selector
        selected_class = st.selectbox("Select Class", class_names)
        class_idx = class_names.index(selected_class)

        # Class-specific metrics
        class_mask = labels == class_idx
        if class_mask.sum() > 0:
            class_acc = (preds[class_mask] == labels[class_mask]).mean()
            class_support = class_mask.sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Class Accuracy", f"{class_acc:.4f}")
            with col2:
                st.metric("Samples in Test Set", int(class_support))
            with col3:
                st.metric("Class Label Index", class_idx)

            # Show some predictions for this class
            st.subheader(f"Sample Predictions for '{selected_class}'")
            indices = np.where(labels == class_idx)[0][:10]
            for idx in indices:
                pred_class = class_names[preds[idx]]
                true_class = class_names[labels[idx]]
                correct = "✅" if preds[idx] == labels[idx] else "❌"
                st.write(f"{correct} True: {true_class}, Predicted: {pred_class}")

    with tab4:
        st.header("Sample Predictions & Visualization")

        st.info("Visualize predictions on individual wafer maps")

        # Sample index selector
        sample_idx = st.slider("Select Sample", 0, len(y_test) - 1, 0)

        # Get sample
        sample_map = test_maps[sample_idx]
        sample_label = y_test[sample_idx]

        # Forward pass
        model.eval()
        with torch.no_grad():
            sample_transform = (
                None if selected_model == 'Custom CNN' else get_imagenet_normalize()
            )
            sample_dataset = WaferMapDataset(
                [sample_map],
                np.array([sample_label]),
                transform=sample_transform,
            )
            sample_tensor, _ = sample_dataset[0]
            sample_tensor = sample_tensor.unsqueeze(0).to(device)
            logits = model(sample_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()

        # Display
        col1, col2 = st.columns(2)

        with col1:
            # Wafer map visualization
            st.subheader("Wafer Map")
            fig, ax = plt.subplots()
            # Handle both 3-channel and single-channel
            if sample_map.shape[0] == 3:
                display_map = sample_map[0]  # Use first channel
            else:
                display_map = sample_map
            ax.imshow(display_map, cmap='viridis')
            ax.set_title("Preprocessed Wafer Map")
            st.pyplot(fig)

        with col2:
            st.subheader("Prediction Probabilities")
            # Probability bar chart
            fig, ax = plt.subplots()
            probs_np = probs[0].cpu().numpy()
            colors = ['green' if i == pred_idx else 'lightgray' for i in range(len(class_names))]
            ax.barh(class_names, probs_np, color=colors)
            ax.set_xlabel("Probability")
            ax.set_title("Model Confidence by Class")
            st.pyplot(fig)

        # Ground truth and prediction
        st.subheader("Prediction Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("True Class", class_names[sample_label])
        with col2:
            st.metric("Predicted Class", class_names[pred_idx])
        with col3:
            confidence = probs[0, pred_idx].item()
            st.metric("Confidence", f"{confidence:.4f}")

        if sample_label == pred_idx:
            st.success("✅ Prediction Correct")
        else:
            st.error("❌ Prediction Incorrect")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "🔬 Wafer Defect Detection Dashboard\n\n"
        "This dashboard provides interactive analysis and interpretation "
        "of semiconductor wafer defect detection models trained on the WM-811K dataset."
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
