# Assignment 2: CIFAR-10 CNN and FastAPI

This project extends the Class Activity 1 text-generation and word-embedding API with a trained PyTorch image classifier. The CNN matches Assignment 2 exactly. The six theory answers, including calculations, are in [submission/theory_answers.pdf](submission/theory_answers.pdf) ([editable version](submission/theory_answers.md)).

## Quick start: Docker

The trained weights are included in `models/cifar10_cnn.pt`; no dataset download or training is needed to serve predictions.

```bash
docker build -t sps-genai-assignment2 .
docker run --rm -p 8001:80 sps-genai-assignment2
```

Open http://127.0.0.1:8001/docs. Expand `POST /predict`, click **Try it out**, select an image, and click **Execute**. Port 8001 avoids conflicts with an earlier version on port 8000. `0.0.0.0` is a server binding address; use `127.0.0.1` and the published port in the browser.

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -F 'file=@submission/example.png'
```

The included example is the first held-out CIFAR-10 test image; its ground-truth label is in `submission/example_label.txt`. The API returns `class_id`, `class_name`, `confidence`, and probabilities for all ten classes. Confidence is a softmax score, not a calibrated guarantee of correctness. Input images are converted to RGB, resized to 64 x 64, converted to tensors, and normalized with mean/std 0.5 for each channel, exactly as during training. Invalid images return 400, missing uploads 422, and images exceeding 10 MiB or 20 million pixels 413. Missing weights return 503 rather than using random predictions.

## Architecture

PyTorch tensors use NCHW ordering. Shapes below omit the batch dimension.

| Layer | Output shape | Trainable parameters |
|---|---|---:|
| RGB input | 3 x 64 x 64 | 0 |
| Conv2d(3, 16, kernel=3, stride=1, padding=1) | 16 x 64 x 64 | 448 |
| ReLU | 16 x 64 x 64 | 0 |
| MaxPool2d(kernel=2, stride=2) | 16 x 32 x 32 | 0 |
| Conv2d(16, 32, kernel=3, stride=1, padding=1) | 32 x 32 x 32 | 4,640 |
| ReLU | 32 x 32 x 32 | 0 |
| MaxPool2d(kernel=2, stride=2) | 32 x 16 x 16 | 0 |
| Flatten | 8,192 | 0 |
| Linear(8192, 100) | 100 | 819,300 |
| ReLU | 100 | 0 |
| Linear(100, 10) | 10 logits | 1,010 |
| **Total** | | **825,398** |

The model returns logits for `CrossEntropyLoss`; softmax is applied only when serving probabilities. CIFAR-10 images are originally 32 x 32 and are resized to meet the assignment's 64 x 64 input requirement.

Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Local setup and training

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
uv run python -m scripts.train --epochs 10 --seed 42
uv run fastapi dev app/main.py --port 8001
```

The training command downloads CIFAR-10 into `data/` and uses all 50,000 training examples with Adam (learning rate 0.001), batch size 128, and cross-entropy loss. It evaluates the final model once on all 10,000 held-out test examples. Test data is not used for gradient updates or checkpoint selection. A fixed seed controls initialization and shuffle; exact numerical results can vary across devices and PyTorch backends. The automatic device choice is CUDA, then Apple MPS, then CPU; use `--device cpu` to force CPU.

Outputs:

- `models/cifar10_cnn.pt`: portable CPU state dictionary and class metadata.
- `models/metrics.json`: real training history, test accuracy, per-class accuracy, training settings, and duration.
- `submission/example.png`: an image for testing uploads.

See [submission/results.md](submission/results.md) for the submitted run's results. This small classifier supports only the ten CIFAR-10 classes; arbitrary images are still assigned one of those classes.

## Verify

```bash
uv run python -m pytest -q
```

Tests verify every intermediate architecture shape, parameter count, RGB/grayscale/RGBA uploads, probability normalization, invalid/missing/oversize inputs, and the existing endpoints. Model weights must be present before starting the API or running upload tests. Restart the API after retraining to load the new weights.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Original health check: `{"Hello": "World"}` |
| POST | `/predict` | Multipart image upload and CIFAR-10 class prediction |
| POST | `/generate` | Bigram text generation |
| POST | `/embedding` | spaCy 300-dimensional word embedding |
| GET | `/docs` | Interactive API documentation |

```bash
curl -X POST http://127.0.0.1:8001/generate \
  -H 'Content-Type: application/json' \
  -d '{"start_word":"bigram","length":5}'
curl -X POST http://127.0.0.1:8001/embedding \
  -H 'Content-Type: application/json' \
  -d '{"word":"king"}'
```

## Files

- `app/cnn.py`: exact CNN and shared preprocessing.
- `scripts/train.py`: training, evaluation, checkpoint and metric export.
- `app/main.py`: prior endpoints plus uploaded-image inference.
- `tests/test_classifier.py`: architecture and API checks.
- `Dockerfile`, `pyproject.toml`, `uv.lock`: container and locked dependencies.
- `models/`: submitted trained weights and measured results.
- `submission/`: theory answers, results, and example request/response.

## Submission

Submit the GitHub repository URL and the theory answer PDF as required by your course submission form. Include the weights and metrics in Git so the grader can run the API without retraining. Exclude `.venv/`, `data/`, and temporary build files.

## References

- Assignment specification: `Assignment2.pdf` supplied with the coursework.
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch training/evaluation modes: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.train
