"""
Synthetic data augmentation using generative models (GAN) and rule-based generators.

Implements both GAN-based synthesis (primary for realistic patterns) and rule-based
generation (fallback for fast augmentation). Designed for addressing class imbalance
in wafer defect detection by generating synthetic samples.
"""

from typing import List, Tuple, Optional, Dict, Callable
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


class WaferMapGenerator:
    """
    Rule-based synthetic wafer map generator.

    Generates synthetic wafer defect patterns using geometric rules and random
    defect placement. Serves as a fast fallback when GAN training is unavailable.

    Attributes:
        image_size: Output image dimensions (default 96x96)
        class_patterns: Dictionary mapping class names to pattern generation functions
    """

    def __init__(self, image_size: int = 96) -> None:
        """
        Initialize the rule-based wafer map generator.

        Args:
            image_size: Size of output square image (default 96)
        """
        self.image_size = image_size
        self.class_patterns = {
            'Center': self._generate_center,
            'Donut': self._generate_donut,
            'Edge-Loc': self._generate_edge_loc,
            'Edge-Ring': self._generate_edge_ring,
            'Loc': self._generate_loc,
            'Near-full': self._generate_near_full,
            'Random': self._generate_random,
            'Scratch': self._generate_scratch,
            'none': self._generate_none,
        }

    def generate_sample(
        self,
        defect_class: str,
        intensity: Optional[float] = None,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """
        Generate a synthetic wafer map for a specific defect class.

        Args:
            defect_class: Class name (one of class_patterns keys)
            intensity: Defect intensity in [0, 1]. If None, random.
            noise_level: Gaussian noise standard deviation (0-1)

        Returns:
            Synthetic wafer map array (H, W) normalized to [0, 1]

        Raises:
            ValueError: If defect_class not recognized
        """
        if defect_class not in self.class_patterns:
            raise ValueError(
                f"Unknown class '{defect_class}'. "
                f"Valid: {list(self.class_patterns.keys())}"
            )

        if intensity is None:
            intensity = np.random.uniform(0.3, 0.9)
        else:
            intensity = np.clip(intensity, 0.0, 1.0)

        # Generate base pattern
        wafer = self.class_patterns[defect_class](intensity)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, wafer.shape)
        wafer = np.clip(wafer + noise, 0, 1)

        return wafer.astype(np.float32)

    def _generate_center(self, intensity: float) -> np.ndarray:
        """Generate center defect: circular region in center."""
        wafer = np.zeros((self.image_size, self.image_size))
        center = self.image_size // 2
        radius = int(0.25 * self.image_size)
        y, x = np.ogrid[:self.image_size, :self.image_size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        wafer[mask] = intensity
        return wafer

    def _generate_donut(self, intensity: float) -> np.ndarray:
        """Generate donut defect: ring with clear center hole."""
        wafer = np.zeros((self.image_size, self.image_size))
        center = self.image_size // 2
        outer_radius = int(0.35 * self.image_size)
        inner_radius = int(0.15 * self.image_size)
        y, x = np.ogrid[:self.image_size, :self.image_size]
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        mask = (dist_sq <= outer_radius ** 2) & (dist_sq >= inner_radius ** 2)
        wafer[mask] = intensity
        return wafer

    def _generate_edge_loc(self, intensity: float) -> np.ndarray:
        """Generate edge-loc defect: localized edge defect."""
        wafer = np.zeros((self.image_size, self.image_size))
        edge_width = self.image_size // 6
        # Random edge selection
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            wafer[:edge_width, :] = intensity
        elif edge == 'bottom':
            wafer[-edge_width:, :] = intensity
        elif edge == 'left':
            wafer[:, :edge_width] = intensity
        else:  # right
            wafer[:, -edge_width:] = intensity
        return wafer

    def _generate_edge_ring(self, intensity: float) -> np.ndarray:
        """Generate edge-ring defect: ring around entire edge."""
        wafer = np.zeros((self.image_size, self.image_size))
        edge_width = self.image_size // 8
        wafer[:edge_width, :] = intensity
        wafer[-edge_width:, :] = intensity
        wafer[:, :edge_width] = intensity
        wafer[:, -edge_width:] = intensity
        return wafer

    def _generate_loc(self, intensity: float) -> np.ndarray:
        """Generate loc defect: small localized spot."""
        wafer = np.zeros((self.image_size, self.image_size))
        num_spots = np.random.randint(1, 4)
        for _ in range(num_spots):
            cx = np.random.randint(10, self.image_size - 10)
            cy = np.random.randint(10, self.image_size - 10)
            radius = np.random.randint(3, 8)
            y, x = np.ogrid[:self.image_size, :self.image_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            wafer[mask] = intensity
        return wafer

    def _generate_near_full(self, intensity: float) -> np.ndarray:
        """Generate near-full defect: covers most of wafer except small regions."""
        wafer = np.ones((self.image_size, self.image_size)) * intensity
        # Clear small circular regions (non-defective areas)
        num_clear = np.random.randint(2, 4)
        for _ in range(num_clear):
            cx = np.random.randint(10, self.image_size - 10)
            cy = np.random.randint(10, self.image_size - 10)
            radius = np.random.randint(5, 15)
            y, x = np.ogrid[:self.image_size, :self.image_size]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            wafer[mask] = 0
        return wafer

    def _generate_random(self, intensity: float) -> np.ndarray:
        """Generate random defect: scattered random pattern."""
        wafer = np.random.rand(self.image_size, self.image_size) * intensity
        return wafer

    def _generate_scratch(self, intensity: float) -> np.ndarray:
        """Generate scratch defect: linear line(s) across wafer."""
        wafer = np.zeros((self.image_size, self.image_size))
        num_scratches = np.random.randint(1, 3)
        for _ in range(num_scratches):
            # Random line (y = mx + b)
            x1, y1 = np.random.randint(0, self.image_size, 2)
            x2, y2 = np.random.randint(0, self.image_size, 2)
            # Draw line with thickness
            thickness = np.random.randint(2, 6)
            rr, cc = self._draw_line(x1, y1, x2, y2, thickness)
            rr = np.clip(rr, 0, self.image_size - 1)
            cc = np.clip(cc, 0, self.image_size - 1)
            wafer[rr, cc] = intensity
        return wafer

    def _generate_none(self, _intensity: float) -> np.ndarray:
        """Generate 'none' defect: clean wafer with minimal noise."""
        wafer = np.zeros((self.image_size, self.image_size))
        return wafer

    @staticmethod
    def _draw_line(
        x0: int, y0: int, x1: int, y1: int, thickness: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bresenham line drawing with thickness.

        Returns:
            Tuple of (row_indices, col_indices) for line pixels
        """
        # Simple implementation using numpy
        steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
        xs = np.linspace(x0, x1, steps, dtype=np.int32)
        ys = np.linspace(y0, y1, steps, dtype=np.int32)
        rr, cc = [], []
        for x, y in zip(xs, ys):
            for dx in range(-thickness // 2, thickness // 2 + 1):
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    rr.append(y + dy)
                    cc.append(x + dx)
        return np.array(rr), np.array(cc)


class SimpleWaferGAN(nn.Module):
    """
    Simple GAN for generating synthetic wafer maps.

    Architecture:
        - Generator: Latent vector (100D) -> 96x96 wafer map via upsampling
        - Discriminator: 96x96 image -> real/fake classification
        - Loss: Binary cross-entropy with discriminator and generator losses

    Attributes:
        generator: Generator network
        discriminator: Discriminator network
    """

    class Generator(nn.Module):
        """
        Generator network: latent vector -> wafer map.

        Progressively upsamples from dense layer through transposed convolutions.
        """

        def __init__(self, latent_dim: int = 100, image_size: int = 96) -> None:
            """
            Initialize generator.

            Args:
                latent_dim: Dimension of latent vector (default 100)
                image_size: Output image size (default 96)
            """
            super().__init__()
            self.latent_dim = latent_dim
            self.image_size = image_size

            # Initial dense layer: latent_dim -> 256 * 6 * 6
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, 256 * 6 * 6),
                nn.ReLU(inplace=True),
            )

            # Transposed convolution blocks: (256, 6, 6) -> (1, 96, 96)
            self.conv_layers = nn.Sequential(
                # (256, 6, 6) -> (128, 12, 12)
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # (128, 12, 12) -> (64, 24, 24)
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # (64, 24, 24) -> (32, 48, 48)
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                # (32, 48, 48) -> (1, 96, 96)
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
                nn.Sigmoid(),  # Ensure output in [0, 1]
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """
            Generate synthetic wafer map.

            Args:
                z: Latent vector (B, latent_dim)

            Returns:
                Generated wafer map (B, 1, 96, 96)
            """
            x = self.fc(z)
            x = x.view(-1, 256, 6, 6)
            x = self.conv_layers(x)
            return x

    class Discriminator(nn.Module):
        """
        Discriminator network: wafer map -> real/fake probability.

        Progressively downsamples image through convolutions to binary output.
        """

        def __init__(self, image_size: int = 96) -> None:
            """
            Initialize discriminator.

            Args:
                image_size: Input image size (default 96)
            """
            super().__init__()
            self.image_size = image_size

            self.conv_layers = nn.Sequential(
                # (1, 96, 96) -> (32, 48, 48)
                nn.Conv2d(1, 32, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # (32, 48, 48) -> (64, 24, 24)
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # (64, 24, 24) -> (128, 12, 12)
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # (128, 12, 12) -> (256, 6, 6)
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
            )

            # Global average pool + FC for binary classification
            self.fc = nn.Sequential(
                nn.Linear(256 * 6 * 6, 1),
                nn.Sigmoid(),  # Output: real/fake probability
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Discriminate real vs. fake wafer map.

            Args:
                x: Wafer map (B, 1, 96, 96)

            Returns:
                Real/fake probability (B, 1)
            """
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    def __init__(self, latent_dim: int = 100, image_size: int = 96) -> None:
        """
        Initialize GAN with generator and discriminator.

        Args:
            latent_dim: Dimension of latent vector (default 100)
            image_size: Output image size (default 96)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.generator = self.Generator(latent_dim, image_size)
        self.discriminator = self.Discriminator(image_size)

    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate a batch of synthetic wafer maps.

        Args:
            batch_size: Number of samples to generate
            device: Device to generate on (cpu or cuda)

        Returns:
            Generated wafer maps (batch_size, 1, 96, 96)
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.generator(z)
        return samples


class SyntheticDataAugmenter:
    """
    High-level API for synthetic wafer data augmentation.

    Provides training and generation interfaces for both GAN-based and rule-based
    synthetic sample generation. Handles dataset balancing and augmentation.

    Attributes:
        generator_type: 'gan' or 'rule-based'
        gan: SimpleWaferGAN instance (if generator_type == 'gan')
        rule_gen: WaferMapGenerator instance (if generator_type == 'rule-based')
        device: PyTorch device
    """

    def __init__(
        self,
        generator_type: str = 'rule-based',
        latent_dim: int = 100,
        image_size: int = 96,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the augmenter.

        Args:
            generator_type: 'gan' or 'rule-based' (default 'rule-based')
            latent_dim: Latent dimension for GAN (default 100)
            image_size: Output image size (default 96)
            device: PyTorch device. If None, uses CUDA if available.

        Raises:
            ValueError: If generator_type not recognized
        """
        if generator_type not in ['gan', 'rule-based']:
            raise ValueError(
                f"generator_type must be 'gan' or 'rule-based', got '{generator_type}'"
            )

        self.generator_type = generator_type
        self.image_size = image_size
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )

        if generator_type == 'gan':
            self.gan = SimpleWaferGAN(latent_dim, image_size).to(self.device)
            self.rule_gen = None
        else:
            self.gan = None
            self.rule_gen = WaferMapGenerator(image_size)

        self.training_history: Dict[str, List[float]] = {
            'gen_loss': [],
            'disc_loss': [],
        }

    def train_generator(
        self,
        wafer_maps: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the GAN on unlabeled or labeled wafer maps.

        Only applicable for 'gan' generator type. Trains both generator and
        discriminator using alternating optimization.

        Args:
            wafer_maps: Array of wafer maps (N, H, W) normalized to [0, 1]
            epochs: Number of training epochs (default 10)
            batch_size: Batch size for training (default 32)
            learning_rate: Learning rate for Adam optimizer (default 0.0002)
            beta1: Beta1 for Adam optimizer (default 0.5)
            verbose: Print training progress (default True)

        Returns:
            Training history dict with 'gen_loss' and 'disc_loss' lists

        Raises:
            RuntimeError: If generator_type != 'gan'
        """
        if self.generator_type != 'gan':
            raise RuntimeError(
                f"train_generator() only works with 'gan' type, "
                f"current type is '{self.generator_type}'"
            )

        # Prepare data
        wafer_tensor = torch.tensor(wafer_maps, dtype=torch.float32)
        if len(wafer_tensor.shape) == 3:
            wafer_tensor = wafer_tensor.unsqueeze(1)  # Add channel dim
        dataset = TensorDataset(wafer_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize optimizers
        optimizer_g = optim.Adam(
            self.gan.generator.parameters(),
            lr=learning_rate, betas=(beta1, 0.999)
        )
        optimizer_d = optim.Adam(
            self.gan.discriminator.parameters(),
            lr=learning_rate, betas=(beta1, 0.999)
        )
        criterion = nn.BCELoss()

        # Training loop
        self.training_history['gen_loss'] = []
        self.training_history['disc_loss'] = []

        pbar = tqdm(range(epochs), desc="GAN Training", disable=not verbose)
        for epoch in pbar:
            gen_losses, disc_losses = [], []

            for batch in loader:
                real_data = batch[0].to(self.device)
                batch_sz = real_data.size(0)

                # Labels for real/fake
                real_labels = torch.ones(batch_sz, 1, device=self.device)
                fake_labels = torch.zeros(batch_sz, 1, device=self.device)

                # ---- Train Discriminator ----
                optimizer_d.zero_grad()

                # Real samples
                real_output = self.gan.discriminator(real_data)
                loss_real = criterion(real_output, real_labels)

                # Fake samples
                z = torch.randn(batch_sz, self.gan.latent_dim, device=self.device)
                fake_data = self.gan.generator(z)
                fake_output = self.gan.discriminator(fake_data.detach())
                loss_fake = criterion(fake_output, fake_labels)

                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()
                disc_losses.append(loss_d.item())

                # ---- Train Generator ----
                optimizer_g.zero_grad()

                z = torch.randn(batch_sz, self.gan.latent_dim, device=self.device)
                fake_data = self.gan.generator(z)
                fake_output = self.gan.discriminator(fake_data)
                loss_g = criterion(fake_output, real_labels)

                loss_g.backward()
                optimizer_g.step()
                gen_losses.append(loss_g.item())

            avg_gen_loss = np.mean(gen_losses)
            avg_disc_loss = np.mean(disc_losses)
            self.training_history['gen_loss'].append(avg_gen_loss)
            self.training_history['disc_loss'].append(avg_disc_loss)

            if verbose:
                pbar.set_postfix({
                    'G_loss': f'{avg_gen_loss:.4f}',
                    'D_loss': f'{avg_disc_loss:.4f}'
                })

        return self.training_history

    def generate_samples(
        self,
        num_samples: int,
        class_label: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate synthetic wafer maps.

        For GAN: Generates num_samples random wafers (ignores class_label).
        For rule-based: Generates samples of specified class.

        Args:
            num_samples: Number of samples to generate
            class_label: Class for rule-based generator. Ignored for GAN.

        Returns:
            Synthetic wafer maps (num_samples, H, W) normalized to [0, 1]
        """
        if self.generator_type == 'gan':
            return self._generate_gan_samples(num_samples)
        else:
            if class_label is None:
                raise ValueError(
                    "class_label required for rule-based generator"
                )
            return self._generate_rule_samples(num_samples, class_label)

    def _generate_gan_samples(self, num_samples: int) -> np.ndarray:
        """Generate samples using trained GAN."""
        all_samples = []
        batch_size = 64
        for _ in range(0, num_samples, batch_size):
            curr_batch = min(batch_size, num_samples - len(all_samples))
            samples = self.gan.generate(curr_batch, self.device)
            samples = samples.squeeze(1).cpu().numpy()
            all_samples.append(samples)

        return np.concatenate(all_samples, axis=0)

    def _generate_rule_samples(self, num_samples: int, class_label: str) -> np.ndarray:
        """Generate samples using rule-based generator."""
        samples = []
        for _ in range(num_samples):
            sample = self.rule_gen.generate_sample(class_label)
            samples.append(sample)
        return np.array(samples)

    def augment_dataset(
        self,
        original_maps: np.ndarray,
        original_labels: np.ndarray,
        target_samples_per_class: int,
        class_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset by generating synthetic samples to balance classes.

        Generates synthetic samples for classes below target_samples_per_class
        until all classes reach the target.

        Args:
            original_maps: Original wafer maps (N, H, W)
            original_labels: Original labels (N,) with integer class indices
            target_samples_per_class: Target number of samples per class
            class_names: List of class names. If None, uses indices.

        Returns:
            Tuple of (augmented_maps, augmented_labels)
        """
        if class_names is None:
            class_names = [str(i) for i in range(int(original_labels.max()) + 1)]

        # Count samples per class
        class_counts = Counter(original_labels)
        augmented_maps = list(original_maps)
        augmented_labels = list(original_labels)

        print("\n--- Synthetic Data Augmentation ---")
        print(f"Generator type: {self.generator_type}")
        print(f"Target samples per class: {target_samples_per_class}")

        for class_idx in sorted(class_counts.keys()):
            class_name = class_names[int(class_idx)] if class_idx < len(class_names) else str(class_idx)
            current_count = class_counts[class_idx]
            deficit = max(0, target_samples_per_class - current_count)

            if deficit > 0:
                print(f"  {class_name}: {current_count} -> generating {deficit} synthetic samples")

                # Generate synthetic samples
                synthetic_maps = self.generate_samples(deficit, class_name)
                augmented_maps.extend(synthetic_maps)
                augmented_labels.extend([class_idx] * deficit)
            else:
                print(f"  {class_name}: {current_count} (no augmentation needed)")

        augmented_maps = np.array(augmented_maps)
        augmented_labels = np.array(augmented_labels)

        print(f"\nAugmentation complete:")
        print(f"  Original: {len(original_maps)} samples")
        print(f"  Augmented: {len(augmented_maps)} samples")
        print(f"  Added: {len(augmented_maps) - len(original_maps)} synthetic samples")

        return augmented_maps, augmented_labels

    def visualize_generated_samples(
        self,
        class_names: Optional[List[str]] = None,
        num_samples_per_class: int = 3,
        figsize: Tuple[int, int] = (15, 12)
    ) -> None:
        """
        Visualize generated synthetic samples.

        Args:
            class_names: List of class names. If None, uses numeric indices.
            num_samples_per_class: Number of samples to show per class (default 3)
            figsize: Figure size (default (15, 12))
        """
        if self.generator_type == 'gan':
            self._visualize_gan_samples(num_samples_per_class, figsize)
        else:
            self._visualize_rule_samples(
                class_names or [
                    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
                    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
                ],
                num_samples_per_class, figsize
            )

    def _visualize_gan_samples(
        self,
        num_samples: int,
        figsize: Tuple[int, int]
    ) -> None:
        """Visualize GAN-generated samples."""
        samples = self._generate_gan_samples(num_samples)

        fig, axes = plt.subplots(1, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.imshow(samples[i], cmap='viridis')
            ax.set_title(f'Generated Sample {i+1}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('gan_generated_samples.png', dpi=150, bbox_inches='tight')
        print("Saved GAN samples to 'gan_generated_samples.png'")
        plt.close()

    def _visualize_rule_samples(
        self,
        class_names: List[str],
        num_samples: int,
        figsize: Tuple[int, int]
    ) -> None:
        """Visualize rule-based generated samples."""
        num_classes = len(class_names)
        fig, axes = plt.subplots(num_classes, num_samples, figsize=figsize)

        for class_idx, class_name in enumerate(class_names):
            for sample_idx in range(num_samples):
                sample = self.rule_gen.generate_sample(class_name)
                ax = axes[class_idx, sample_idx] if num_classes > 1 else axes[sample_idx]
                ax.imshow(sample, cmap='viridis')
                if sample_idx == 0:
                    ax.set_ylabel(class_name, fontsize=10)
                if class_idx == 0:
                    ax.set_title(f'Sample {sample_idx+1}', fontsize=10)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('rule_based_generated_samples.png', dpi=150, bbox_inches='tight')
        print("Saved rule-based samples to 'rule_based_generated_samples.png'")
        plt.close()

    def save_gan(self, path: str) -> None:
        """
        Save trained GAN to disk.

        Args:
            path: File path for checkpoint

        Raises:
            RuntimeError: If generator_type != 'gan'
        """
        if self.generator_type != 'gan':
            raise RuntimeError("Can only save GAN models")

        checkpoint = {
            'generator': self.gan.generator.state_dict(),
            'discriminator': self.gan.discriminator.state_dict(),
            'training_history': self.training_history,
        }
        torch.save(checkpoint, path)
        print(f"GAN saved to {path}")

    def load_gan(self, path: str) -> None:
        """
        Load trained GAN from disk.

        Args:
            path: File path to checkpoint

        Raises:
            RuntimeError: If generator_type != 'gan'
        """
        if self.generator_type != 'gan':
            raise RuntimeError("Can only load GAN models")

        checkpoint = torch.load(path, map_location=self.device)
        self.gan.generator.load_state_dict(checkpoint['generator'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator'])
        self.training_history = checkpoint.get('training_history', {})
        print(f"GAN loaded from {path}")


def balance_dataset_with_synthetic(
    wafer_maps: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    generator_type: str = 'rule-based',
    strategy: str = 'oversample_to_max',
    oversample_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset using synthetic data augmentation.

    Helper function to augment a dataset until all classes have similar
    representation.

    Args:
        wafer_maps: Original wafer maps (N, H, W)
        labels: Original labels (N,) as integers
        class_names: List of class names. If None, uses numeric indices.
        generator_type: 'gan' or 'rule-based' (default 'rule-based')
        strategy: Balancing strategy:
            - 'oversample_to_max': Oversample minorities to match majority (default)
            - 'oversample_to_mean': Oversample to mean class count
            - 'oversample_to_custom': Oversample to custom target (set via oversample_ratio)
        oversample_ratio: Target ratio (0-1) relative to majority class (default 0.8)

    Returns:
        Tuple of (balanced_maps, balanced_labels)
    """
    class_counts = Counter(labels)
    max_count = max(class_counts.values())

    if strategy == 'oversample_to_max':
        target = max_count
    elif strategy == 'oversample_to_mean':
        target = int(np.mean(list(class_counts.values())))
    else:  # custom
        target = int(max_count * oversample_ratio)

    augmenter = SyntheticDataAugmenter(
        generator_type=generator_type,
        image_size=wafer_maps.shape[-1]
    )

    augmented_maps, augmented_labels = augmenter.augment_dataset(
        wafer_maps, labels, target, class_names
    )

    return augmented_maps, augmented_labels
