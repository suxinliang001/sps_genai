import io
import math
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from torchvision.utils import save_image

from .helper_lib import generate_samples, get_model
from .helper_lib.model import ARCHITECTURE_ID

MODEL_PATH = Path(__file__).resolve().parent / "models/gan.pt"
router = APIRouter(prefix="/gan", tags=["Assignment 3 - GAN"])


class GenerateRequest(BaseModel):
    num_samples: int = Field(default=25, ge=1, le=64)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


@lru_cache(maxsize=1)
def load_checkpoint():
    if not MODEL_PATH.is_file():
        raise HTTPException(503, "Model missing. Run python -m assignment3.train first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID:
        raise HTTPException(503, "Checkpoint architecture does not match Assignment 3.")
    model = get_model("GAN")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@router.get("/status")
def status():
    if not MODEL_PATH.is_file():
        return {"ready": False, "architecture_id": ARCHITECTURE_ID}
    _model, metadata = load_checkpoint()
    return {
        "ready": True, "architecture_id": ARCHITECTURE_ID,
        "epochs": metadata["epochs"], "training_images": metadata["training_images"],
    }


@router.post("/generate", response_class=Response, responses={200: {"content": {"image/png": {}}}})
def generate(request: GenerateRequest):
    model, _metadata = load_checkpoint()
    images = generate_samples(model, "cpu", request.num_samples, seed=request.seed)
    stream = io.BytesIO()
    save_image(images, stream, format="PNG", nrow=math.ceil(math.sqrt(request.num_samples)), normalize=True, value_range=(-1, 1))
    return Response(stream.getvalue(), media_type="image/png")


app = FastAPI(title="Assignment 3 - MNIST GAN")
app.include_router(router)
