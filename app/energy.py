"""A scalar image energy and its input-gradient Langevin sampler."""

import torch
from torch import nn

ENERGY_ARCHITECTURE = "cifar10-energy-v2"
STEP_SIZE = 12.0
NOISE_STD = 0.015
SMOOTHNESS_WEIGHT = 2.0


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.energy = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1),
        )

    def forward(self, images):
        # A quadratic pixel prior discourages high-frequency noise.
        horizontal = images[:, :, :, 1:] - images[:, :, :, :-1]
        vertical = images[:, :, 1:, :] - images[:, :, :-1, :]
        roughness = horizontal.square().mean((1, 2, 3)) + vertical.square().mean((1, 2, 3))
        return self.energy(images).squeeze(1) + SMOOTHNESS_WEIGHT * roughness


def langevin_sample(model, initial, steps, generator=None):
    """Update only the input pixels and return detached negative samples."""
    images = initial.detach()
    for _ in range(steps):
        with torch.enable_grad():
            images.requires_grad_(True)
            # Summing keeps each image's gradient independent of the batch size.
            (gradient,) = torch.autograd.grad(model(images).sum(), images)
        with torch.no_grad():
            noise = torch.randn(
                images.shape, device=images.device, dtype=images.dtype, generator=generator
            )
            images = (images - STEP_SIZE * gradient + NOISE_STD * noise).clamp(-1, 1)
    return images
