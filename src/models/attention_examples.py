"""
Examples demonstrating how to use SE and CBAM attention mechanisms.

This module provides practical examples of:
1. Creating standalone attention modules
2. Injecting attention into existing models
3. Evaluating performance improvements
4. Training models with attention mechanisms
"""

import torch
import torch.nn as nn
from src.models.attention import (
    SEBlock,
    SpatialAttention,
    CBAMBlock,
    add_se_to_model,
    add_cbam_to_model,
    attention_summary,
)
import logging

logger = logging.getLogger(__name__)
from src.models.cnn import WaferCNN
from src.models.pretrained import get_resnet18, get_efficientnet_b0


def example_1_standalone_se_block() -> None:
    """
    Example 1: Create and use a standalone SE block.

    SE blocks are lightweight (1-2% parameter increase) and can be easily
    added to any convolutional layer output.
    """
    logger.info("=" * 70)
    logger.info("EXAMPLE 1: Standalone SE Block")
    logger.info("=" * 70)

    # Create a simple feature extraction pipeline
    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        SEBlock(channels=64, reduction=16),  # Add SE after conv
    )

    # Forward pass
    x = torch.randn(4, 3, 96, 96)
    y = feature_extractor(x)

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {y.shape}")
    logger.info(f"Total parameters: {sum(p.numel() for p in feature_extractor.parameters()):,}")
    logger.info()


def example_2_cbam_block() -> None:
    """
    Example 2: Create and use a CBAM block (channel + spatial attention).

    CBAM provides both channel and spatial attention, offering better
    expressiveness at the cost of slightly higher computation.
    """
    logger.info("=" * 70)
    logger.info("EXAMPLE 2: CBAM Block (Channel + Spatial Attention)")
    logger.info("=" * 70)

    # Create feature extraction with CBAM
    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        CBAMBlock(channels=64, reduction=16, kernel_size=7),  # Add CBAM
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        CBAMBlock(channels=128, reduction=16, kernel_size=7),
    )

    x = torch.randn(4, 3, 96, 96)
    y = feature_extractor(x)

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {y.shape}")
    logger.info(f"Total parameters: {sum(p.numel() for p in feature_extractor.parameters()):,}")

    # Check attention modules
    summary = attention_summary(feature_extractor)
    logger.info(f"CBAM blocks: {summary['cbam_blocks']}")
    logger.info()


def example_3_inject_se_into_custom_cnn() -> None:
    """
    Example 3: Inject SE attention into the custom WaferCNN model.

    This is the most common use case: take a trained CNN and enhance it
    with attention mechanisms post-hoc.
    """
    logger.info("=" * 70)
    logger.info("EXAMPLE 3: Inject SE into Custom WaferCNN")
    logger.info("=" * 70)

    # Create baseline model
    model_baseline = WaferCNN(num_classes=9)
    baseline_params = sum(p.numel() for p in model_baseline.parameters())

    # Inject SE blocks
    model_with_se = add_se_to_model(model_baseline, reduction=16)
    se_params = sum(p.numel() for p in model_with_se.parameters())

    # Test forward pass
    x = torch.randn(1, 3, 96, 96)
    y_baseline = model_baseline(x)
    y_with_se = model_with_se(x)

    summary = attention_summary(model_with_se)

    logger.info(f"Baseline WaferCNN:")
    logger.info(f"  Total parameters: {baseline_params:,}")
    logger.info(f"Output shape: {y_baseline.shape}")
    logger.info()
    logger.info(f"WaferCNN with SE blocks:")
    logger.info(f"  Total parameters: {se_params:,}")
    logger.info(f"  Added parameters: {se_params - baseline_params:,}")
    logger.info(f"  Parameter increase: {(se_params - baseline_params) / baseline_params * 100:.2f}%")
    logger.info(f"  SE blocks injected: {summary['se_blocks']}")
    logger.info(f"Output shape: {y_with_se.shape}")
    logger.info()
    logger.info("Note: SE blocks add minimal parameters but can improve accuracy")
    logger.info()


def example_4_inject_cbam_into_resnet() -> None:
    """
    Example 4: Inject CBAM into ResNet-18, targeting only final layers.

    This strategy freezes early layers and adds attention only to the
    final task-specific layers, balancing expressiveness and efficiency.
    """
    logger.info("=" * 70)
    logger.info("EXAMPLE 4: Inject CBAM into ResNet-18 (Final Layers Only)")
    logger.info("=" * 70)

    # Create baseline ResNet
    model_baseline = get_resnet18(num_classes=9)
    baseline_params = sum(p.numel() for p in model_baseline.parameters())

    # Inject CBAM only into layer4 (final residual block)
    model_with_cbam = add_cbam_to_model(
        model_baseline,
        target_modules=['layer4'],
        reduction=16,
        kernel_size=7,
        use_batchnorm=False,
    )
    cbam_params = sum(p.numel() for p in model_with_cbam.parameters())

    # Test forward pass
    x = torch.randn(1, 3, 96, 96)
    y_baseline = model_baseline(x)
    y_with_cbam = model_with_cbam(x)

    summary = attention_summary(model_with_cbam)

    logger.info(f"Baseline ResNet-18:")
    logger.info(f"  Total parameters: {baseline_params:,}")
    logger.info(f"Output shape: {y_baseline.shape}")
    logger.info()
    logger.info(f"ResNet-18 with CBAM in layer4:")
    logger.info(f"  Total parameters: {cbam_params:,}")
    logger.info(f"  Added parameters: {cbam_params - baseline_params:,}")
    logger.info(f"  Parameter increase: {(cbam_params - baseline_params) / baseline_params * 100:.2f}%")
    logger.info(f"  CBAM blocks injected: {summary['cbam_blocks']}")
    logger.info(f"Output shape: {y_with_cbam.shape}")
    logger.info()
    logger.info("Strategy: CBAM only in final layers improves expressiveness while")
    logger.info("keeping computation low and preserving pretrained ImageNet features.")
    logger.info()


def example_5_inject_cbam_into_efficientnet() -> None:
    """
    Example 5: Inject CBAM into EfficientNet-B0, targeting final blocks.

    EfficientNet uses a different layer naming scheme. This shows how to
    target specific blocks (features.7, features.8).
    """
    logger.info("=" * 70)
    logger.info("EXAMPLE 5: Inject CBAM into EfficientNet-B0 (Final Blocks)")
    logger.info("=" * 70)

    # Create baseline EfficientNet
    model_baseline = get_efficientnet_b0(num_classes=9)
    baseline_params = sum(p.numel() for p in model_baseline.parameters())

    # Inject CBAM into final MBConv blocks (features.7, features.8)
    model_with_cbam = add_cbam_to_model(
        model_baseline,
        target_modules=['features.7', 'features.8'],
        reduction=16,
        kernel_size=7,
        use_batchnorm=False,
    )
    cbam_params = sum(p.numel() for p in model_with_cbam.parameters())

    # Test forward pass
    x = torch.randn(1, 3, 96, 96)
    y_baseline = model_baseline(x)
    y_with_cbam = model_with_cbam(x)

    summary = attention_summary(model_with_cbam)

    logger.info(f"Baseline EfficientNet-B0:")
    logger.info(f"  Total parameters: {baseline_params:,}")
    logger.info(f"Output shape: {y_baseline.shape}")
    logger.info()
    logger.info(f"EfficientNet-B0 with CBAM in features.7-8:")
    logger.info(f"  Total parameters: {cbam_params:,}")
    logger.info(f"  Added parameters: {cbam_params - baseline_params:,}")
    logger.info(f"  Parameter increase: {(cbam_params - baseline_params) / baseline_params * 100:.2f}%")
    logger.info(f"  CBAM blocks injected: {summary['cbam_blocks']}")
    logger.info(f"Output shape: {y_with_cbam.shape}")
    logger.info()
    logger.info("EfficientNet compounds multiple small blocks, so targeting final")
    logger.info("blocks (features.7-8) enables task-specific adaptation efficiently.")
    logger.info()


def example_6_comparison_all_models() -> None:
    """
    Example 6: Compare parameter counts and inference for all model variants.

    This shows the computational trade-offs of different attention strategies.
    """
    logger.info("=" * 70)
    logger.info("EXAMPLE 6: Model Comparison (Parameter Efficiency)")
    logger.info("=" * 70)

    # Models to compare
    models = {
        'WaferCNN': WaferCNN(9),
        'WaferCNN + SE': add_se_to_model(WaferCNN(9), reduction=16),
        'ResNet-18': get_resnet18(9),
        'ResNet-18 + CBAM': add_cbam_to_model(get_resnet18(9), target_modules=['layer4']),
        'EfficientNet-B0': get_efficientnet_b0(9),
        'EfficientNet-B0 + CBAM': add_cbam_to_model(
            get_efficientnet_b0(9),
            target_modules=['features.7', 'features.8'],
        ),
    }

    x = torch.randn(1, 3, 96, 96)

    logger.info(f"{'Model':<30} {'Params':>15} {'Increase':>15} {'Output':>15}")
    logger.info("-" * 75)

    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            y = model(x)

        # Estimate increase (compare to baseline)
        if 'SE' in name or 'CBAM' in name:
            baseline_name = name.split(' + ')[0]
            baseline_model = models[baseline_name]
            baseline_params = sum(p.numel() for p in baseline_model.parameters())
            increase_pct = (total_params - baseline_params) / baseline_params * 100
            increase_str = f"{increase_pct:+.2f}%"
        else:
            increase_str = "baseline"

        logger.info(f"{name:<30} {total_params:>15,} {increase_str:>15} {str(y.shape):>15}")

    logger.info()


def example_7_training_with_attention() -> None:
    """
    Example 7: Pseudocode for training models with attention.

    Shows the recommended workflow for fine-tuning with attention mechanisms.
    """
    logger.info("=" * 70)
    logger.info("EXAMPLE 7: Training Workflow with Attention")
    logger.info("=" * 70)

    code = """
    # Step 1: Load data and create baseline model
    train_loader, val_loader = load_data()
    model = WaferCNN(num_classes=9)

    # Step 2: Option A - Train baseline then add attention
    train_model(model, train_loader, val_loader, epochs=5)

    # Step 3: Inject attention (fine-tune with attention)
    model = add_se_to_model(model, reduction=16)

    # Step 4: Fine-tune with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    train_model(model, train_loader, val_loader, epochs=3)

    # ---- OR Option B - Start fresh with attention ----

    # Step 1: Create model with attention from the start
    model = WaferCNN(num_classes=9)
    model = add_cbam_to_model(model, reduction=16)

    # Step 2: Train end-to-end
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, train_loader, val_loader, epochs=5)

    # Recommended: Option A (inject after pretraining) often works better
    # because it combines pretrained features with learned attention patterns
    """
    logger.info(code)
    logger.info()


if __name__ == "__main__":
    example_1_standalone_se_block()
    example_2_cbam_block()
    example_3_inject_se_into_custom_cnn()
    example_4_inject_cbam_into_resnet()
    example_5_inject_cbam_into_efficientnet()
    example_6_comparison_all_models()
    example_7_training_with_attention()

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("""
SE Block (Squeeze-and-Excitation):
  - Adds 1-2% parameters
  - Channel attention only
  - Lightweight, suitable for all models
  - Use: add_se_to_model(model, reduction=16)

CBAM Block (Convolutional Block Attention Module):
  - Adds 2-5% parameters (higher than SE)
  - Channel + Spatial attention
  - More expressive, slightly more computation
  - Use: add_cbam_to_model(model, target_modules=['layer4'], reduction=16)

Key Recommendations:
  1. For custom CNNs: Use SE for low overhead, CBAM for better accuracy
  2. For ResNets: Target final layers (layer4) with CBAM
  3. For EfficientNets: Target final blocks (features.7-8) with CBAM
  4. Fine-tune with lower LR (1e-4 to 1e-5) after injecting attention
  5. Spatial attention may need use_batchnorm=False for small batches

References:
  - SENet: https://arxiv.org/abs/1709.01507
  - CBAM: https://arxiv.org/abs/1807.06521
    """)
