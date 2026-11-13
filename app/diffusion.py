"""An epsilon-predicting U-Net and the diffusion process it learns."""

import math

import torch
from torch import nn
from torch.nn import functional as F

DIFFUSION_STEPS = 1000
DIFFUSION_ARCHITECTURE = "cifar10-unet-v1"
IMAGE_SHAPE = (3, 32, 32)


def timestep_embedding(timesteps, dim, max_period=10_000):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    angles = timesteps[:, None].float() * frequencies
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(embedding, (0, dim % 2))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time = nn.Linear(time_dim, out_channels)
        self.output = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, image, time):
        hidden = self.input(image) + self.time(F.silu(time))[:, :, None, None]
        return self.skip(image) + self.output(hidden)


class DiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.input = nn.Conv2d(3, 32, 3, padding=1)
        self.encoder = nn.ModuleList(
            ResidualBlock(width, width, 128) for width in (32, 48, 64)
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(source, target, 3, stride=2, padding=1)
            for source, target in ((32, 48), (48, 64), (64, 96))
        )
        self.middle = ResidualBlock(96, 96, 128)
        self.decoder = nn.ModuleList(
            ResidualBlock(source + skip, skip, 128)
            for source, skip in ((96, 64), (64, 48), (48, 32))
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, image, timesteps):
        time = self.time(timestep_embedding(timesteps, 64))
        hidden = self.input(image)
        skips = []
        for block, downsample in zip(self.encoder, self.downsample):
            hidden = block(hidden, time)
            skips.append(hidden)
            hidden = downsample(hidden)
        hidden = self.middle(hidden, time)
        for block, skip in zip(self.decoder, reversed(skips)):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = block(torch.cat((hidden, skip), dim=1), time)
        return self.output(hidden)


class DiffusionProcess(nn.Module):
    def __init__(self):
        super().__init__()
        # Cosine schedule. Entries 0..999 correspond to mathematical t=1..1000.
        times = torch.linspace(0, 1, DIFFUSION_STEPS + 1, dtype=torch.float64)
        cosine = (((times + 0.008) / 1.008) * math.pi / 2).cos().square()
        beta = (1 - cosine[1:] / cosine[:-1]).clamp(max=0.999)
        self.register_buffer("alpha_bar", (1 - beta).cumprod(0).float())

    def training_loss(self, model, clean):
        timesteps = torch.randint(DIFFUSION_STEPS, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bar[timesteps, None, None, None]
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return F.mse_loss(model(noisy, timesteps), noise)

    @torch.no_grad()
    def sample(self, model, count, generator, steps=48):
        """Use deterministic DDIM updates after drawing the initial Gaussian noise."""
        images = torch.randn(
            count, *IMAGE_SHAPE, device=self.alpha_bar.device, generator=generator
        )
        indices = torch.linspace(DIFFUSION_STEPS - 1, 0, steps).round().int().tolist()
        for current, previous in zip(indices, indices[1:] + [-1]):
            alpha = self.alpha_bar[current]
            time = torch.full((count,), current, device=images.device, dtype=torch.long)
            noise = model(images, time)
            clean = ((images - (1 - alpha).sqrt() * noise) / alpha.sqrt()).clamp(-1, 1)
            if previous == -1:
                return clean
            # Recompute epsilon after clipping so it still agrees with x_t.
            noise = (images - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            next_alpha = self.alpha_bar[previous]
            images = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
