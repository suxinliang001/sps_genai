"""CIFAR-10 image generation routes for the existing FastAPI app."""

import io
import math
from functools import lru_cache
from pathlib import Path
from pickle import UnpicklingError

import torch
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from torchvision.utils import save_image

from app.diffusion import (
    DIFFUSION_ARCHITECTURE,
    IMAGE_SHAPE,
    DiffusionProcess,
    DiffusionUNet,
)
from app.energy import ENERGY_ARCHITECTURE, EnergyModel, langevin_sample

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS = {
    "diffusion": (DIFFUSION_ARCHITECTURE, DiffusionUNet),
    "ebm": (ENERGY_ARCHITECTURE, EnergyModel),
}
PNG_RESPONSE = {200: {"content": {"image/png": {}}}}
router = APIRouter()


class GenerateImagesRequest(BaseModel):
    num_samples: int = Field(default=4, ge=1, le=16)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


@lru_cache(maxsize=2)
def load_model(kind):
    architecture, model_class = MODELS[kind]
    path = MODEL_DIR / f"cifar10_{kind}.pt"
    if not path.is_file():
        raise HTTPException(503, f"Checkpoint missing. Run python -m scripts.train_generators {kind}.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["architecture"] != architecture:
            raise ValueError("Checkpoint architecture mismatch")
        model = model_class()
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, UnpicklingError) as exc:
        raise HTTPException(503, f"Could not load the {kind} checkpoint.") from exc
    return model.eval().requires_grad_(False)


def png_grid(images):
    stream = io.BytesIO()
    save_image(
        (images + 1) / 2,
        stream,
        format="PNG",
        nrow=math.ceil(math.sqrt(len(images))),
    )
    return Response(stream.getvalue(), media_type="image/png")


@router.post("/diffusion/generate", tags=["Diffusion"], response_class=Response, responses=PNG_RESPONSE)
def generate_diffusion(request: GenerateImagesRequest):
    """Generate a grid of RGB images by denoising Gaussian noise."""
    model = load_model("diffusion")
    generator = torch.Generator().manual_seed(request.seed)
    images = DiffusionProcess().sample(model, request.num_samples, generator)
    return png_grid(images)


@router.post("/ebm/generate", tags=["Energy-based model"], response_class=Response, responses=PNG_RESPONSE)
def generate_ebm(request: GenerateImagesRequest):
    """Generate a grid of RGB images by applying Langevin updates to random pixels."""
    model = load_model("ebm")
    generator = torch.Generator().manual_seed(request.seed)
    initial = torch.rand(request.num_samples, *IMAGE_SHAPE, generator=generator) * 2 - 1
    images = langevin_sample(model, initial, steps=256, generator=generator)
    return png_grid(images)
