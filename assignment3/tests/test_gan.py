import io
import math

import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from assignment3 import api
from assignment3.helper_lib import get_model, train_gan, generate_samples
from assignment3.helper_lib.model import ARCHITECTURE_ID


def test_architecture_matches_assignment_pdf():
    model = get_model("GAN")
    G, D = model.generator.network, model.discriminator.network
    assert [type(layer) for layer in G] == [
        nn.Linear, nn.Unflatten, nn.ConvTranspose2d, nn.BatchNorm2d,
        nn.ReLU, nn.ConvTranspose2d, nn.Tanh,
    ]
    assert (G[0].in_features, G[0].out_features) == (100, 6272)
    assert (G[2].in_channels, G[2].out_channels) == (128, 64)
    assert (G[5].in_channels, G[5].out_channels) == (64, 1)
    assert [type(layer) for layer in D] == [
        nn.Conv2d, nn.LeakyReLU, nn.Conv2d, nn.BatchNorm2d,
        nn.LeakyReLU, nn.Flatten, nn.Linear, nn.Sigmoid,
    ]
    assert (D[0].in_channels, D[0].out_channels) == (1, 64)
    assert (D[2].in_channels, D[2].out_channels) == (64, 128)
    assert (D[6].in_features, D[6].out_features) == (6272, 1)
    for layer in [G[2], G[5], D[0], D[2]]:
        assert (layer.kernel_size, layer.stride, layer.padding) == ((4, 4), (2, 2), (1, 1))
    assert D[1].negative_slope == D[4].negative_slope == 0.2
    images = model(torch.randn(3, 100))
    assert images.shape == (3, 1, 28, 28)
    assert images.min() >= -1 and images.max() <= 1
    probabilities = model.discriminator(images)
    assert probabilities.shape == (3, 1)
    assert probabilities.min() >= 0 and probabilities.max() <= 1


def test_training_updates_both_networks_and_freezes_D_batchnorm_for_G():
    torch.manual_seed(10)
    model = get_model("GAN")
    before_G = [p.detach().clone() for p in model.generator.parameters()]
    before_D = [p.detach().clone() for p in model.discriminator.parameters()]
    # Include an incomplete final batch.
    data = TensorDataset(torch.rand(7, 1, 28, 28) * 2 - 1, torch.zeros(7))
    loader = DataLoader(data, batch_size=4)
    optimizers = {
        "G": torch.optim.Adam(model.generator.parameters(), lr=0.0002),
        "D": torch.optim.Adam(model.discriminator.parameters(), lr=0.0002),
    }
    history = []
    train_gan(model, loader, nn.BCELoss(), optimizers, epochs=1, history=history)
    assert any(not torch.equal(a, b) for a, b in zip(before_G, model.generator.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_D, model.discriminator.parameters()))
    assert all(math.isfinite(history[0][name]) for name in ["loss_G", "loss_D"])
    # Only discriminator updates change the running statistics.
    assert model.discriminator.network[3].num_batches_tracked.item() == 4
    assert all(p.requires_grad for p in model.discriminator.parameters())
    image = generate_samples(model, num_samples=1)
    assert image.shape == (1, 1, 28, 28)
    assert model.generator.training


def test_API_returns_reproducible_png_and_validates_input(tmp_path, monkeypatch):
    checkpoint = tmp_path / "gan.pt"
    monkeypatch.setattr(api, "MODEL_PATH", checkpoint)
    api.load_checkpoint.cache_clear()
    client = TestClient(api.app)
    assert client.get("/gan/status").json()["ready"] is False
    assert client.post("/gan/generate", json={}).status_code == 503
    model = get_model("GAN")
    torch.save({
        "architecture_id": ARCHITECTURE_ID,
        "model_state_dict": model.state_dict(),
        "epochs": 1, "training_images": 7,
    }, checkpoint)
    assert client.get("/gan/status").json()["ready"] is True
    payload = {"num_samples": 9, "seed": 7}
    first = client.post("/gan/generate", json=payload)
    second = client.post("/gan/generate", json=payload)
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/png"
    assert first.content == second.content
    assert Image.open(io.BytesIO(first.content)).size == (92, 92)
    for payload in [{"num_samples": 0}, {"num_samples": 65}, {"seed": -1}]:
        assert client.post("/gan/generate", json=payload).status_code == 422
    api.load_checkpoint.cache_clear()


def test_combined_api_preserves_existing_routes():
    from assignment3.main import app
    paths = set(app.openapi()["paths"])
    assert {"/", "/generate", "/embedding", "/predict", "/gan/generate", "/gan/status"} <= paths
