"""
Swin Transformer architecture for wafer defect classification.

Hierarchical vision transformer using shifted windows for efficient local attention.
Designed for 96x96 wafer maps with Swin-Tiny scale (~28M params).

References:
    [101] Liu et al. (2021). "Swin Transformer: Hierarchical Vision Transformer
          using Shifted Windows". arXiv:2103.14030
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """Stochastic depth: randomly drop entire residual branch during training.

    Per-sample binary mask applied to the batch dimension so that each sample
    independently survives or is zeroed out, then rescaled.

    Args:
        x: Input tensor of any shape (B, ...).
        drop_prob: Probability of dropping the path.
        training: Whether model is in training mode.

    Returns:
        Tensor of same shape, with paths randomly dropped during training.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Binary mask: shape (B, 1, 1, ...) to broadcast over all dims except batch
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = torch.floor(mask + keep_prob)
    return x / keep_prob * mask


class DropPath(nn.Module):
    """Drop path (stochastic depth) as a module wrapper.

    Args:
        drop_prob: Probability of dropping the path during training.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size: Size of each square window.

    Returns:
        Windows tensor of shape (B * num_windows, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    num_h = H // window_size
    num_w = W // window_size
    # (B, num_h, window_size, num_w, window_size, C)
    x = x.view(B, num_h, window_size, num_w, window_size, C)
    # (B, num_h, num_w, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # (B * num_windows, window_size, window_size, C)
    x = x.view(-1, window_size, window_size, C)
    return x


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """Reverse window partition back to feature map.

    Args:
        windows: Windows tensor of shape (B * num_windows, window_size, window_size, C).
        window_size: Size of each square window.
        H: Height of the original feature map.
        W: Width of the original feature map.

    Returns:
        Feature map tensor of shape (B, H, W, C).
    """
    num_h = H // window_size
    num_w = W // window_size
    B = windows.shape[0] // (num_h * num_w)
    C = windows.shape[-1]
    # (B, num_h, num_w, window_size, window_size, C)
    x = windows.view(B, num_h, num_w, window_size, window_size, C)
    # (B, num_h, window_size, num_w, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # (B, H, W, C)
    x = x.view(B, H, W, C)
    return x


class PatchMerging(nn.Module):
    """Patch merging layer that reduces spatial resolution by 2x.

    Concatenates 2x2 neighboring patches and projects to 2*dim, halving the
    spatial dimensions while doubling the channel dimension.

    Ref [101]: Section 3.1 -- hierarchical feature maps via patch merging.

    Args:
        dim: Number of input channels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge 2x2 patches and project.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Merged tensor of shape (B, H//2, W//2, 2*C).
        """
        B, H, W, C = x.shape
        # Extract four sub-patches from each 2x2 region
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C) -- top-left
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C) -- bottom-left
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C) -- top-right
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C) -- bottom-right

        # Concatenate along channel dim: (B, H/2, W/2, 4C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    Computes attention within local windows of size window_size x window_size.
    A learnable relative position bias table is indexed by the relative
    position between each pair of tokens in the window.

    Ref [101]: Section 3.2 -- shifted window based self-attention.

    Args:
        dim: Number of input channels.
        window_size: Size of the attention window.
        num_heads: Number of attention heads.
        attention_dropout: Dropout rate for attention weights.
        proj_dropout: Dropout rate for the output projection.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2*Wh-1) * (2*Ww-1) entries, one per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index for each token pair in the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        # (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.view(2, -1)  # (2, Wh*Ww)

        # Pairwise relative coordinates: (2, Wh*Ww, Wh*Ww)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)

        # Shift to start from 0
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        # Flatten 2D relative position to 1D index
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute window attention with relative position bias.

        Args:
            x: Input tensor of shape (num_windows * B, N, C) where N = window_size^2.
            mask: Optional attention mask of shape (num_windows, N, N) for shifted
                  window attention. Values of -100.0 block attention; 0.0 allows it.

        Returns:
            Output tensor of shape (num_windows * B, N, C).
        """
        B_w, N, C = x.shape  # B_w = num_windows * batch_size

        # Compute Q, K, V: (B_w, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B_w, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_w, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # each: (B_w, num_heads, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_w, num_heads, N, N)

        # Add relative position bias
        # Index into bias table: relative_position_index is (N, N), table is (num_entries, num_heads)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads)  # (N, N, num_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, N, N)
        attn = attn + bias

        # Apply attention mask for shifted windows
        if mask is not None:
            num_windows = mask.shape[0]
            # Reshape attn to (B, num_windows, num_heads, N, N)
            attn = attn.view(B_w // num_windows, num_windows, self.num_heads, N, N)
            # mask: (1, num_windows, 1, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum: (B_w, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B_w, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window attention and optional shifted window attention.

    Alternates between regular and shifted window partitioning. Even-indexed blocks
    use regular windows; odd-indexed blocks shift by window_size // 2.

    Ref [101]: Section 3.2 -- successive blocks alternate between W-MSA and SW-MSA.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size for local attention.
        shift_size: Shift amount for shifted window attention (0 = regular).
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        dropout: Dropout rate for MLP.
        attention_dropout: Dropout rate for attention weights.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            proj_dropout=dropout,
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def _compute_attention_mask(
        self, H: int, W: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted window attention.

        Creates a mask that prevents attention between tokens that belong to
        different spatial regions after shifting. Each region gets a unique ID,
        and pairs from different regions receive a large negative bias (-100).

        Args:
            H: Feature map height.
            W: Feature map width.
            device: Device to place the mask on.

        Returns:
            Attention mask of shape (num_windows, N, N) or None if shift_size is 0.
        """
        if self.shift_size == 0:
            return None

        # Build region ID map: assign unique IDs to regions created by the shift
        img_mask = torch.zeros((1, H, W, 1), device=device)
        # Slice boundaries for the shifted regions
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        region_id = 0
        for h_s in h_slices:
            for w_s in w_slices:
                img_mask[:, h_s, w_s, :] = region_id
                region_id += 1

        # Partition into windows: (num_windows, Ws, Ws, 1)
        mask_windows = window_partition(img_mask, self.window_size)
        # (num_windows, Ws*Ws)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        # Create pairwise mask: same region = 0, different = -100
        # (num_windows, N, 1) - (num_windows, 1, N) -> (num_windows, N, N)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swin Transformer block.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Output tensor of shape (B, H, W, C).
        """
        B, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)

        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # Partition into windows: (B * num_windows, Ws, Ws, C)
        x_windows = window_partition(shifted_x, self.window_size)
        # Flatten spatial dims: (B * num_windows, Ws*Ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Compute attention mask
        attn_mask = self._compute_attention_mask(H, W, x.device)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows back: (B * num_windows, Ws, Ws, C)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        # Residual + drop path
        x = shortcut + self.drop_path(x)

        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerStage(nn.Module):
    """A stage of Swin Transformer blocks followed by optional patch merging.

    Each stage contains a sequence of Swin blocks that alternate between
    regular and shifted window attention. After the blocks, an optional
    patch merging layer halves the spatial resolution.

    Args:
        dim: Number of input channels.
        depth: Number of Swin Transformer blocks in this stage.
        num_heads: Number of attention heads.
        window_size: Window size for local attention.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        dropout: Dropout rate.
        attention_dropout: Attention dropout rate.
        drop_path_rates: Stochastic depth rates for each block in this stage.
        downsample: Whether to apply patch merging after this stage.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rates: Optional[list] = None,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=drop_path_rates[i],
                )
            )

        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stage blocks and optional downsampling.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Output tensor. If downsample: (B, H//2, W//2, 2*C).
            Otherwise: (B, H, W, C).
        """
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SwinTransformer(nn.Module):
    """Swin Transformer for image classification.

    Hierarchical architecture with shifted windows for efficient local attention.
    Designed for 96x96 wafer maps.

    Architecture for 96x96 input (Swin-Tiny defaults):
        - Patch embed: 96x96 -> 24x24 patches of dim 96
        - Stage 1: 2 blocks at 24x24, dim=96, heads=3, window=6
        - Merge: 24x24 -> 12x12, dim=192
        - Stage 2: 2 blocks at 12x12, dim=192, heads=6, window=6
        - Merge: 12x12 -> 6x6, dim=384
        - Stage 3: 6 blocks at 6x6, dim=384, heads=12, window=6
        - Merge: 6x6 -> 3x3, dim=768
        - Stage 4: 2 blocks at 3x3, dim=768, heads=24, window=3
        - Global avg pool -> classifier

    Reference: [101] Liu et al. (2021). arXiv:2103.14030

    Args:
        image_size: Input image size (assumed square).
        patch_size: Patch size for initial embedding.
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        embed_dim: Base embedding dimension (doubled at each stage).
        depths: Number of Swin blocks per stage.
        num_heads: Number of attention heads per stage.
        window_size: Window size for local attention.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        dropout: Dropout rate.
        attention_dropout: Attention dropout rate.
        drop_path: Maximum stochastic depth rate (linearly increases across blocks).
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 9,
        embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim

        # Patch embedding: Conv2d projects patches to embed_dim
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(dropout)

        patches_resolution = image_size // patch_size  # e.g. 96/4 = 24

        # Linearly increasing drop path rates across all blocks
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path, total_blocks)]

        # Build stages
        self.stages = nn.ModuleList()
        block_idx = 0
        for i_stage in range(self.num_stages):
            stage_dim = embed_dim * (2 ** i_stage)
            stage_resolution = patches_resolution // (2 ** i_stage)

            # For the last stage, adapt window_size if resolution < window_size
            stage_window_size = min(window_size, stage_resolution)

            stage = SwinTransformerStage(
                dim=stage_dim,
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=stage_window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path_rates=dpr[block_idx : block_idx + depths[i_stage]],
                downsample=(i_stage < self.num_stages - 1),
            )
            self.stages.append(stage)
            block_idx += depths[i_stage]

        # Final norm and classification head
        final_dim = embed_dim * (2 ** (self.num_stages - 1))
        self.norm = nn.LayerNorm(final_dim)
        self.head = nn.Linear(final_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following Swin Transformer conventions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H/ps, W/ps)
        x = self.patch_embed(x)
        # Rearrange to (B, H', W', C) for window-based attention
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.patch_norm(x)
        x = self.pos_drop(x)

        # Apply stages
        for stage in self.stages:
            x = stage(x)

        # Global average pooling: (B, H', W', C) -> (B, C)
        x = self.norm(x)
        x = x.mean(dim=[1, 2])

        # Classification
        x = self.head(x)
        return x


def get_swin_tiny(num_classes: int = 9) -> SwinTransformer:
    """Create a Swin-Tiny model for 96x96 wafer maps.

    Configuration: embed_dim=96, depths=(2,2,6,2), heads=(3,6,12,24), ~28M params.

    Args:
        num_classes: Number of output classes.

    Returns:
        Initialized SwinTransformer model.
    """
    return SwinTransformer(
        image_size=96,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=6,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path=0.1,
    )


def get_swin_micro(num_classes: int = 9) -> SwinTransformer:
    """Create a smaller Swin model for faster training.

    Configuration: embed_dim=48, depths=(2,2,2,2), ~3.5M params.

    Args:
        num_classes: Number of output classes.

    Returns:
        Initialized SwinTransformer model.
    """
    return SwinTransformer(
        image_size=96,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=6,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path=0.05,
    )
