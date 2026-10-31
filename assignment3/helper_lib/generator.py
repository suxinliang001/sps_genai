from pathlib import Path
import math

import torch
from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import to_pil_image


def generate_samples(
    model, device="cpu", num_samples=10, *, seed=42, noise=None,
    output_path=None, show=False,
):
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if noise is None:
        random = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(num_samples, model.latent_dim, generator=random)
    if tuple(noise.shape) != (num_samples, model.latent_dim):
        raise ValueError("noise must have shape (num_samples, 100)")
    was_training = model.generator.training
    model.generator.eval()
    try:
        with torch.inference_mode():
            images = model.generator(noise.to(device)).cpu()
    finally:
        model.generator.train(was_training)
    columns = math.ceil(math.sqrt(num_samples))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(images, path, nrow=columns, normalize=True, value_range=(-1, 1))
    if show:
        grid = make_grid(images, nrow=columns, normalize=True, value_range=(-1, 1))
        to_pil_image(grid).show()
    return images
