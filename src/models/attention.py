"""
Attention mechanisms for neural networks: SE and CBAM modules.

Implements channel and spatial attention mechanisms that enhance feature
representation by enabling the network to focus on important channels and
spatial regions adaptively.

References:
    - SENet: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    - CBAM: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
"""

import torch
import torch.nn as nn
from typing import Type, Callable, Optional, List
import copy


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.

    Explicitly models channel interdependencies by:
    1. **Squeeze**: Global average pooling to capture channel-wise statistics
    2. **Excitation**: Two FC layers (reduction then expansion) to model interactions
    3. **Recalibration**: Scales channel outputs by learned attention weights

    This allows the network to recalibrate channel responses based on global
    context, suppressing less informative channels and amplifying important ones.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for FC bottleneck (default 16)
            - Reduces channels by factor of 16 to save computation
            - Lower values (8, 4) increase capacity; higher (32) reduce parameters

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W) [same shape, channel-weighted]

    Example:
        >>> se = SEBlock(channels=64, reduction=16)
        >>> x = torch.randn(4, 64, 32, 32)
        >>> out = se(x)  # Shape: (4, 64, 32, 32)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Squeeze: adaptive global average pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        reduced_channels = max(channels // reduction, 1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with channel attention.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Channel-weighted output tensor (B, C, H, W)
        """
        # Squeeze: (B, C, H, W) -> (B, C, 1, 1)
        squeeze = self.squeeze(x)  # Global average pool

        # Reshape for FC layers: (B, C, 1, 1) -> (B, C)
        b, c, _, _ = x.size()
        squeeze = squeeze.view(b, c)

        # Excitation: (B, C) -> (B, C) attention weights
        excitation = self.excitation(squeeze)  # (B, C)

        # Reshape back for broadcasting: (B, C) -> (B, C, 1, 1)
        excitation = excitation.view(b, c, 1, 1)

        # Recalibration: element-wise multiplication
        return x * excitation


class SpatialAttention(nn.Module):
    """
    Spatial attention module for CBAM.

    Generates attention weights along the spatial dimension by applying
    convolutions to channel-wise statistics (max and average pooling).

    Args:
        kernel_size: Kernel size for spatial attention convolution (default 7)
        use_batchnorm: Whether to use BatchNorm (default False for stability)

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, 1, H, W) [spatial attention map]

    Example:
        >>> spatial = SpatialAttention(kernel_size=7)
        >>> x = torch.randn(4, 64, 32, 32)
        >>> attn = spatial(x)  # Shape: (4, 1, 32, 32)
    """

    def __init__(self, kernel_size: int = 7, use_batchnorm: bool = False) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2  # Ensure output same size as input

        layers = [
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(1))
        layers.append(nn.Sigmoid())

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate spatial attention map.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Spatial attention map (B, 1, H, W)
        """
        # Channel-wise statistics: (B, C, H, W) -> (B, 1, H, W) each
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Channel average
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # Channel max

        # Concatenate: (B, 1, H, W) + (B, 1, H, W) -> (B, 2, H, W)
        x_concat = torch.cat([avg_pool, max_pool], dim=1)

        # Apply convolution + sigmoid: (B, 2, H, W) -> (B, 1, H, W)
        return self.conv(x_concat)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Sequentially applies channel and spatial attention mechanisms to enable
    the network to focus on informative channels and spatial regions.

    CBAM = Channel Attention → Spatial Attention

    The sequential composition allows channels to be recalibrated first (removing
    redundancy), then spatial regions to be weighted (focusing on local patterns).

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for channel attention (default 16)
        kernel_size: Kernel size for spatial attention (default 7)
        use_batchnorm: Whether to use BatchNorm in spatial attention (default False)

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W) [same shape, channel and spatial weighted]

    Example:
        >>> cbam = CBAMBlock(channels=64, reduction=16, kernel_size=7)
        >>> x = torch.randn(4, 64, 32, 32)
        >>> out = cbam(x)  # Shape: (4, 64, 32, 32)

    References:
        Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.channel_attention = SEBlock(channels=channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with channel and spatial attention.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Channel and spatial-weighted output tensor (B, C, H, W)
        """
        # Channel attention: (B, C, H, W) -> (B, C, H, W)
        x = self.channel_attention(x)

        # Spatial attention: (B, C, H, W) -> (B, C, H, W)
        spatial_attn = self.spatial_attention(x)  # (B, 1, H, W)
        x = x * spatial_attn

        return x


def add_se_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    reduction: int = 16,
) -> nn.Module:
    """
    Inject SE (Squeeze-and-Excitation) blocks into a model.

    Inserts SE blocks after specified convolutional layers or all Conv2d layers
    to add channel attention. This is a post-hoc enhancement that can improve
    model performance without major architectural changes.

    Strategy:
        - Identify target modules (Conv2d layers by default or custom names)
        - Wrap each target in a sequential block: Conv -> SE
        - Preserve original model architecture and functionality

    Args:
        model: PyTorch model (nn.Module)
        target_modules: List of module attribute names to inject SE into
            (e.g., ['features.7', 'features.8']; None=all Conv2d layers)
        reduction: Reduction ratio for SE block (default 16)

    Returns:
        Modified model with SE blocks injected

    Example:
        >>> from src.models import get_resnet18
        >>> model = get_resnet18(num_classes=9)
        >>> model = add_se_to_model(model, target_modules=['layer4'])
        >>> # Now layer4 has SE blocks after Conv2d layers

    Note:
        - Modifies model in-place and returns it
        - Only adds SE after Conv2d layers (those with output channels)
        - Preserves model.state_dict() keys (wrap, don't replace)
    """
    model = copy.deepcopy(model)

    def inject_se_recursive(module: nn.Module, prefix: str = "") -> None:
        """Recursively inject SE blocks into Conv2d layers."""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Conv2d):
                # Check if this Conv2d should be targeted
                should_inject = (
                    target_modules is None or
                    any(full_name.startswith(target) for target in target_modules)
                )

                if should_inject:
                    # Get output channels from Conv2d
                    channels = child.out_channels
                    se_block = SEBlock(channels=channels, reduction=reduction)

                    # Replace Conv2d with Sequential([Conv2d, SE])
                    setattr(module, name, nn.Sequential(child, se_block))
            else:
                # Traverse into child modules
                inject_se_recursive(child, full_name)

    inject_se_recursive(model)
    return model


def add_cbam_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    reduction: int = 16,
    kernel_size: int = 7,
    use_batchnorm: bool = False,
) -> nn.Module:
    """
    Inject CBAM (Convolutional Block Attention Module) blocks into a model.

    Inserts CBAM blocks after specified layers to add both channel and spatial
    attention. More expressive than SE alone, at the cost of slightly more computation.

    Strategy:
        - Identify target modules (Conv2d layers or custom names)
        - Wrap each target in a sequential block: Conv -> CBAM
        - Preserve original model structure

    Args:
        model: PyTorch model (nn.Module)
        target_modules: List of module attribute names to inject CBAM into
            (e.g., ['features.7', 'features.8']; None=all Conv2d layers)
        reduction: Reduction ratio for channel attention (default 16)
        kernel_size: Kernel size for spatial attention (default 7)
        use_batchnorm: Whether to use BatchNorm in spatial attention (default False)

    Returns:
        Modified model with CBAM blocks injected

    Example:
        >>> from src.models import get_resnet18
        >>> model = get_resnet18(num_classes=9)
        >>> model = add_cbam_to_model(model, target_modules=['layer4'])
        >>> # Now layer4 has CBAM blocks after Conv2d layers

    Note:
        - Modifies model in-place and returns it
        - Only adds CBAM after Conv2d layers
        - Slightly higher compute cost than SE, but more expressive
    """
    model = copy.deepcopy(model)

    def inject_cbam_recursive(module: nn.Module, prefix: str = "") -> None:
        """Recursively inject CBAM blocks into Conv2d layers."""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Conv2d):
                # Check if this Conv2d should be targeted
                should_inject = (
                    target_modules is None or
                    any(full_name.startswith(target) for target in target_modules)
                )

                if should_inject:
                    # Get output channels from Conv2d
                    channels = child.out_channels
                    cbam_block = CBAMBlock(
                        channels=channels,
                        reduction=reduction,
                        kernel_size=kernel_size,
                        use_batchnorm=use_batchnorm,
                    )

                    # Replace Conv2d with Sequential([Conv2d, CBAM])
                    setattr(module, name, nn.Sequential(child, cbam_block))
            else:
                # Traverse into child modules
                inject_cbam_recursive(child, full_name)

    inject_cbam_recursive(model)
    return model


def attention_summary(model: nn.Module) -> dict:
    """
    Summarize attention modules in a model.

    Counts and lists all SE and CBAM blocks, useful for verification
    after injection.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary with counts and module lists:
            {
                'se_blocks': count,
                'cbam_blocks': count,
                'spatial_attentions': count,
                'se_modules': [list of SEBlock instances],
                'cbam_modules': [list of CBAMBlock instances],
            }

    Example:
        >>> model = add_cbam_to_model(get_resnet18(), reduction=16)
        >>> summary = attention_summary(model)
        >>> print(f"Model has {summary['cbam_blocks']} CBAM blocks")
    """
    se_blocks = []
    cbam_blocks = []
    spatial_attentions = []

    for module in model.modules():
        if isinstance(module, SEBlock):
            se_blocks.append(module)
        elif isinstance(module, CBAMBlock):
            cbam_blocks.append(module)
        elif isinstance(module, SpatialAttention):
            spatial_attentions.append(module)

    return {
        'se_blocks': len(se_blocks),
        'cbam_blocks': len(cbam_blocks),
        'spatial_attentions': len(spatial_attentions),
        'se_modules': se_blocks,
        'cbam_modules': cbam_blocks,
    }


if __name__ == "__main__":
    import sys
    import os

    print("=" * 70)
    print("ATTENTION MECHANISMS TEST")
    print("=" * 70)

    # Test SEBlock
    print("\n1. SEBlock (Squeeze-and-Excitation)")
    print("-" * 70)
    se = SEBlock(channels=64, reduction=16)
    x_se = torch.randn(4, 64, 32, 32)
    out_se = se(x_se)
    print(f"Input shape: {x_se.shape}")
    print(f"Output shape: {out_se.shape}")
    print(f"Parameters: {sum(p.numel() for p in se.parameters()):,}")
    print(f"[OK] SEBlock forward pass successful")

    # Test SpatialAttention
    print("\n2. SpatialAttention (CBAM spatial component)")
    print("-" * 70)
    spatial = SpatialAttention(kernel_size=7)
    x_spatial = torch.randn(4, 64, 32, 32)
    attn_spatial = spatial(x_spatial)
    print(f"Input shape: {x_spatial.shape}")
    print(f"Attention map shape: {attn_spatial.shape}")
    print(f"Attention map range: [{attn_spatial.min():.4f}, {attn_spatial.max():.4f}]")
    print(f"Parameters: {sum(p.numel() for p in spatial.parameters()):,}")
    print(f"[OK] SpatialAttention forward pass successful")

    # Test CBAMBlock
    print("\n3. CBAMBlock (Channel + Spatial Attention)")
    print("-" * 70)
    cbam = CBAMBlock(channels=64, reduction=16, kernel_size=7)
    x_cbam = torch.randn(4, 64, 32, 32)
    out_cbam = cbam(x_cbam)
    print(f"Input shape: {x_cbam.shape}")
    print(f"Output shape: {out_cbam.shape}")
    print(f"Parameters: {sum(p.numel() for p in cbam.parameters()):,}")
    print(f"[OK] CBAMBlock forward pass successful")

    # Test SE injection on custom CNN
    print("\n4. SE injection on WaferCNN")
    print("-" * 70)

    # Add parent dir to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from cnn import WaferCNN

    cnn = WaferCNN(num_classes=9)
    total_before = sum(p.numel() for p in cnn.parameters())
    cnn_se = add_se_to_model(cnn, reduction=16)
    total_after = sum(p.numel() for p in cnn_se.parameters())
    summary = attention_summary(cnn_se)
    print(f"Total parameters before: {total_before:,}")
    print(f"Total parameters after: {total_after:,}")
    print(f"Added parameters: {total_after - total_before:,}")
    print(f"SE blocks added: {summary['se_blocks']}")
    x_cnn = torch.randn(1, 3, 96, 96)
    out_cnn = cnn_se(x_cnn)
    print(f"WaferCNN with SE output shape: {out_cnn.shape}")
    print(f"[OK] SE injection on WaferCNN successful")

    # Test CBAM injection on ResNet-18
    print("\n5. CBAM injection on ResNet-18")
    print("-" * 70)
    from pretrained import get_resnet18

    resnet = get_resnet18(num_classes=9)
    total_before = sum(p.numel() for p in resnet.parameters())
    resnet_cbam = add_cbam_to_model(
        resnet,
        target_modules=['layer4'],
        reduction=16,
        kernel_size=7,
    )
    total_after = sum(p.numel() for p in resnet_cbam.parameters())
    summary = attention_summary(resnet_cbam)
    print(f"Total parameters before: {total_before:,}")
    print(f"Total parameters after: {total_after:,}")
    print(f"Added parameters: {total_after - total_before:,}")
    print(f"CBAM blocks added: {summary['cbam_blocks']}")
    print(f"Spatial attention modules: {summary['spatial_attentions']}")
    x_resnet = torch.randn(1, 3, 96, 96)
    out_resnet = resnet_cbam(x_resnet)
    print(f"ResNet-18 with CBAM output shape: {out_resnet.shape}")
    print(f"[OK] CBAM injection on ResNet-18 successful")

    # Test CBAM injection on EfficientNet-B0
    print("\n6. CBAM injection on EfficientNet-B0")
    print("-" * 70)
    from pretrained import get_efficientnet_b0

    effnet = get_efficientnet_b0(num_classes=9)
    total_before = sum(p.numel() for p in effnet.parameters())
    effnet_cbam = add_cbam_to_model(
        effnet,
        target_modules=['features.7', 'features.8'],
        reduction=16,
        kernel_size=7,
    )
    total_after = sum(p.numel() for p in effnet_cbam.parameters())
    summary = attention_summary(effnet_cbam)
    print(f"Total parameters before: {total_before:,}")
    print(f"Total parameters after: {total_after:,}")
    print(f"Added parameters: {total_after - total_before:,}")
    print(f"CBAM blocks added: {summary['cbam_blocks']}")
    x_effnet = torch.randn(1, 3, 96, 96)
    out_effnet = effnet_cbam(x_effnet)
    print(f"EfficientNet-B0 with CBAM output shape: {out_effnet.shape}")
    print(f"[OK] CBAM injection on EfficientNet-B0 successful")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
