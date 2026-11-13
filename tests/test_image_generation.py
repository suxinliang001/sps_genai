import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.energy import EnergyModel, langevin_sample
from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("path", ["/diffusion/generate", "/ebm/generate"])
def test_generation_returns_reproducible_rgb_png(path):
    payload = {"num_samples": 1, "seed": 42}
    response = client.post(path, json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(response.content)) as image:
        image.load()
        assert image.mode == "RGB"
        assert image.size == (32, 32)
    assert response.content == client.post(path, json=payload).content
    assert client.post(path, json={"num_samples": 17}).status_code == 422


def test_langevin_updates_only_the_inputs():
    model = EnergyModel()
    initial = torch.zeros(2, 3, 32, 32)
    with torch.no_grad():
        samples = langevin_sample(model, initial, steps=2)
    assert not torch.equal(initial, samples)
    assert not initial.requires_grad
    assert not samples.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
