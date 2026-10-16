# Assignment 2: CIFAR-10 CNN and FastAPI

A PyTorch CIFAR-10 classifier using the exact assignment architecture, added to the Class Activity 1 FastAPI application. The six theory answers with derivations are in [submission/theory_answers.pdf](submission/theory_answers.pdf).

## Run with Docker

The trained model is included; retraining is optional. Start Docker or Colima first.

```bash
docker build -t sps-genai-assignment2 .
docker run --rm -p 8001:80 sps-genai-assignment2
```

Open http://127.0.0.1:8001/docs. Under **POST /predict**, click **Try it out**, upload `submission/example.png`, and click **Execute**. The example is the first CIFAR-10 test image, with ground-truth label **cat**. The submitted model predicts **cat**, with a softmax confidence of approximately **0.9637**.

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -F 'file=@submission/example.png'
```

The JSON response contains `class_id`, `class_name`, `confidence`, and probabilities for all ten classes. If port 8001 is occupied, use `-p 8002:80` and open http://127.0.0.1:8002/docs instead. Keep the terminal running; press Ctrl+C to stop the container.

## Architecture

Shapes below use PyTorch channel-first ordering and omit the batch dimension.

| Layer | Output shape |
|---|---|
| RGB input | 3 x 64 x 64 |
| Conv2d: 16 filters, 3 x 3, stride 1, padding 1 | 16 x 64 x 64 |
| ReLU | 16 x 64 x 64 |
| MaxPool2d: 2 x 2, stride 2 | 16 x 32 x 32 |
| Conv2d: 32 filters, 3 x 3, stride 1, padding 1 | 32 x 32 x 32 |
| ReLU | 32 x 32 x 32 |
| MaxPool2d: 2 x 2, stride 2 | 32 x 16 x 16 |
| Flatten | 8,192 |
| Fully connected + ReLU | 100 |
| Fully connected | 10 logits |

**Trainable parameters: 825,398.** CIFAR-10's original 32 x 32 images are resized to the required 64 x 64 input. Training and inference use the same RGB conversion, tensor conversion, and channel normalization (mean 0.5, standard deviation 0.5). Training uses logits with cross-entropy loss; inference applies softmax.

Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Training and results

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
uv run python -m scripts.train --epochs 10 --seed 42
```

| Setting | Submitted run |
|---|---|
| Training / test examples | 50,000 / 10,000 |
| Epochs / batch size | 10 / 128 |
| Optimizer / learning rate | Adam / 0.001 |
| Loss | CrossEntropyLoss |
| Random seed / device | 42 / Apple MPS |
| Final training accuracy | 87.31% |
| Held-out test accuracy | **65.58%** |

The saved checkpoint is the final epoch. Test images were not used for gradient updates or checkpoint selection. The training/test accuracy gap indicates overfitting. This small classifier recognizes only the ten CIFAR-10 classes, and softmax confidence is not calibrated certainty.

The script saves weights to `models/cifar10_cnn.pt` and the full training history and per-class test results to `models/metrics.json`. It downloads CIFAR-10 into the ignored `data/` folder. Automatic device selection prefers CUDA, then Apple MPS, then CPU; add `--device cpu` to force CPU. Exact numerical results can vary across devices.

During preparation, the original dataset server timed out. The unchanged archive was obtained from the [MIT Open Learning mirror](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python) and verified against torchvision's expected MD5, `c58f30108f718f92721af3b95e74349a`. The script uses torchvision's standard CIFAR10 loader.

## Run locally and test

To serve the included model without Docker:

```bash
uv sync --frozen
uv run fastapi dev app/main.py --port 8001
```

Run the automated checks:

```bash
uv run python -m pytest -q
```

The six tests cover every intermediate CNN shape, parameter count, RGB/grayscale/RGBA uploads, probability normalization, invalid/missing/oversize uploads, and the original API endpoints. The Docker API was also checked with a real image upload. Restart the local API after retraining; rebuild the image before starting a Docker container with new weights.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Original health check |
| POST | `/predict` | Multipart image upload and CIFAR-10 prediction |
| POST | `/generate` | Original bigram text generation |
| POST | `/embedding` | Original spaCy 300-dimensional word embedding |
| GET | `/docs` | Interactive API documentation |

Uploads are limited to 10 MiB and 20 million pixels. Invalid images return 400, missing files 422, oversize images 413, and unavailable model weights 503.

## Repository contents

- `app/`: CNN, shared preprocessing, FastAPI routes, and original bigram model.
- `scripts/train.py`: CIFAR-10 training, evaluation, and checkpoint export.
- `models/`: trained weights and measured results.
- `submission/`: final theory answer PDF and one example image.
- `tests/test_classifier.py`: architecture and API checks.
- `Dockerfile`, `pyproject.toml`, `uv.lock`: deployment and locked dependencies.

Submit this repository URL and the theory answer PDF through the course submission form.

## References

- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch training/evaluation modes](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.train)
