import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.cnn import CIFAR10CNN, CLASSES
from app.main import app

client = TestClient(app)


def test_exact_architecture_shapes():
    model = CIFAR10CNN()
    x = torch.zeros(2, 3, 64, 64)
    expected = [(2, 16, 64, 64), (2, 16, 64, 64), (2, 16, 32, 32),
                (2, 32, 32, 32), (2, 32, 32, 32), (2, 32, 16, 16),
                (2, 8192), (2, 100), (2, 100), (2, 10)]
    for layer, shape in zip(model.layers, expected):
        x = layer(x)
        assert tuple(x.shape) == shape
    assert sum(p.numel() for p in model.parameters()) == 825398


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_uploaded_image(mode):
    buffer = io.BytesIO()
    Image.new(mode, (32, 32)).save(buffer, format="PNG")
    response = client.post("/predict", files={"file": ("image.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["class_name"] == CLASSES[result["class_id"]]
    assert set(result["probabilities"]) == set(CLASSES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_invalid_upload():
    assert client.post("/predict", files={"file": ("bad.png", b"not an image", "image/png")}).status_code == 400
    assert client.post("/predict").status_code == 422
    assert client.post("/predict", files={"file": ("big.png", b"x" * (10 * 1024 * 1024 + 1))}).status_code == 413


def test_original_endpoints():
    assert client.get("/").json() == {"Hello": "World"}
    assert client.post("/generate", json={"start_word": "bigram", "length": 3}).status_code == 200
    assert client.post("/embedding", json={"word": "king"}).json()["dimension"] == 300
