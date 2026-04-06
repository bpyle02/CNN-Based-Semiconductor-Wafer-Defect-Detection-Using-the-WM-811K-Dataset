"""
Vision Transformer (ViT) architecture for wafer defect classification.

Adapts the Vision Transformer from Dosovitskiy et al. to work with 96x96 wafer maps.
Includes patch embedding, transformer encoder, and classification head.

References:
    [3] Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words". arXiv:2010.11929
    [23] Vaswani et al. (2017). "Attention Is All You Need". arXiv:1706.03762
    [102] Touvron et al. (2021). "DeiT: Data-Efficient Image Transformers". arXiv:2012.12877
    [103] Liu et al. (2021). "Swin Transformer". arXiv:2103.14030
    [105] Yuan et al. (2021). "Tokens-to-Token ViT". arXiv:2101.11986
    [107] Wu et al. (2021). "CvT: Convolutional Vision Transformer". arXiv:2103.15808
    [108] Dai et al. (2021). "CoAtNet". arXiv:2106.04803
    [109] Mehta & Rastegari (2022). "MobileViT". arXiv:2110.02178
    [110] Graham et al. (2021). "LeViT: Vision Transformer in ConvNet's Clothing". arXiv:2104.01136
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


# Ref [3]: Dosovitskiy et al. — patch embedding via Conv2d projection
class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.

    Splits input image into non-overlapping patches and linearly embeds them.
    Positional encodings are added by the ViT class after CLS token prepend.

    Args:
        image_size: Height/width of input image (assumed square)
        patch_size: Height/width of each patch
        in_channels: Number of input channels (e.g., 3 for RGB)
        embed_dim: Dimension of patch embeddings
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        # Project patches: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)

        # Flatten patches: (B, embed_dim, h, w) -> (B, embed_dim, num_patches)
        x = x.flatten(2)

        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


# Ref [23]: Vaswani et al. — multi-head self-attention + FFN
class TransformerEncoder(nn.Module):
    """
    Transformer encoder block with multi-head attention and feed-forward.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_dim: Dimension of feed-forward network
        dropout: Dropout rate
        attention_dropout: Dropout rate in attention
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, num_patches, embed_dim)

        Returns:
            Output tensor of shape (B, num_patches, embed_dim)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feed-forward with residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Adapts ViT from Dosovitskii et al. (An Image is Worth 16x16 Words, ICLR 2021)
    to work with 96x96 wafer maps.

    Architecture:
        - Patch embedding (96x96 -> 12x12 = 144 patches of 8x8)
        - Class token prepended
        - Positional embeddings added
        - Transformer encoder stack
        - Classification head (MLP on [CLS] token)

    Args:
        image_size: Input image size (96)
        patch_size: Patch size (16)
        in_channels: Input channels (3)
        num_classes: Number of classes (9)
        embed_dim: Embedding dimension (384 for ViT-small)
        num_heads: Number of attention heads (6 for ViT-small)
        num_layers: Number of transformer layers (12)
        mlp_dim: Feed-forward dimension
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 9,
        embed_dim: int = 384,  # ViT-small variant
        num_heads: int = 6,
        num_layers: int = 12,
        mlp_dim: int = 1536,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Ref [3]: Learnable [CLS] token for classification (Dosovitskiy et al.)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Joint positional embedding for CLS + all patches (standard ViT)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder stack
        self.encoder_layers = nn.Sequential(
            *[
                TransformerEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, embed_dim)

        # Add joint positional embedding (CLS + patches in shared space)
        x = x + self.pos_embed

        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder_layers(x)

        # Layer norm
        x = self.norm(x)

        # Classification (use [CLS] token)
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.cls_head(cls_output)  # (B, num_classes)

        return logits


def get_vit_small(num_classes: int = 9) -> ViT:
    """
    Create a ViT-small model for wafer defect classification.

    ViT-small configuration: embed_dim=384, num_heads=6, num_layers=12

    Args:
        num_classes: Number of output classes

    Returns:
        Initialized ViT model
    """
    return ViT(
        image_size=96,
        patch_size=8,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        num_heads=6,
        num_layers=12,
        mlp_dim=1536,
        dropout=0.1,
        attention_dropout=0.0,
    )


def get_vit_tiny(num_classes: int = 9) -> ViT:
    """
    Create a ViT-tiny model (smaller, faster) for wafer defect classification.

    ViT-tiny configuration: embed_dim=192, num_heads=3, num_layers=12

    Args:
        num_classes: Number of output classes

    Returns:
        Initialized ViT model
    """
    return ViT(
        image_size=96,
        patch_size=8,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        num_heads=3,
        num_layers=12,
        mlp_dim=768,
        dropout=0.1,
        attention_dropout=0.0,
    )
