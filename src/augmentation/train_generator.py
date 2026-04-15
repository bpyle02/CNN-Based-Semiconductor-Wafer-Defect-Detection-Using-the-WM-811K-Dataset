"""Training script for GAN-based synthetic wafer map generation with FID evaluation."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm


def scipy_linalg_sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root using scipy if available, else eigenvalue decomposition fallback."""
    try:
        from scipy.linalg import sqrtm

        return sqrtm(matrix)
    except ImportError:
        evals, evecs = np.linalg.eigh(matrix)
        evals = np.maximum(evals, 0)
        return evecs @ np.diag(np.sqrt(evals)) @ np.linalg.inv(evecs)


def compute_fid_score(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    inception_model: nn.Module,
    device: str = "cuda",
) -> float:
    """
    Compute Frechet Inception Distance (FID) between real and fake images.
    Lower FID = better quality synthetic images.
    """
    inception_model.eval()

    if real_images.size(1) == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
    if fake_images.size(1) == 1:
        fake_images = fake_images.repeat(1, 3, 1, 1)

    resize = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False)
    real_images = resize(real_images)
    fake_images = resize(fake_images)

    with torch.no_grad():
        real_features = inception_model(real_images).cpu().numpy()
        fake_features = inception_model(fake_images).cpu().numpy()

    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    eps = 1e-6
    covmean = scipy_linalg_sqrtm(sigma_real.dot(sigma_fake) + np.eye(sigma_real.shape[0]) * eps)

    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


def train_generator(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    epochs: int = 50,
    device: str = "cuda",
    latent_dim: int = 100,
) -> Tuple[Dict[str, List[float]], float]:
    """Train GAN on wafer map data and track FID score."""

    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Load pre-trained InceptionV3 for FID
    # Replace final fully connected layer with Identity to get features
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception.fc = nn.Identity()
    inception = inception.to(device)

    history = {"d_loss": [], "g_loss": [], "fid": []}
    final_fid = float("inf")

    pbar = tqdm(range(epochs), desc="Training GAN")
    for epoch in pbar:
        epoch_d_loss = []
        epoch_g_loss = []

        for images, _ in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # Train discriminator
            d_optimizer.zero_grad()
            real_output = discriminator(images)
            real_loss = criterion(real_output, torch.ones(batch_size, 1).to(device))

            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            fake_loss = criterion(fake_output, torch.zeros(batch_size, 1).to(device))

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, torch.ones(batch_size, 1).to(device))

            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss.append(d_loss.item())
            epoch_g_loss.append(g_loss.item())

        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        history["d_loss"].append(avg_d_loss)
        history["g_loss"].append(avg_g_loss)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                noise = torch.randn(min(100, batch_size), latent_dim).to(device)
                fake_batch = generator(noise)
                real_batch_sample = images[: min(100, batch_size)]
                fid = compute_fid_score(real_batch_sample, fake_batch, inception, device)
                history["fid"].append(fid)
                final_fid = fid

            pbar.set_postfix(
                {
                    "D_Loss": f"{avg_d_loss:.4f}",
                    "G_Loss": f"{avg_g_loss:.4f}",
                    "FID": f"{final_fid:.2f}",
                }
            )
        else:
            pbar.set_postfix({"D_Loss": f"{avg_d_loss:.4f}", "G_Loss": f"{avg_g_loss:.4f}"})

    return history, final_fid
