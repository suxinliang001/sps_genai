# Assignment 2: CIFAR-10 CNN

A PyTorch CNN trained on CIFAR-10, with an image classification endpoint added to the existing FastAPI app. The trained model is included. Test accuracy: 65.58%.

## Run

```bash
docker build -t sps-genai-assignment2 .
docker run --rm -p 8001:80 sps-genai-assignment2
```

Open http://127.0.0.1:8001/docs and use POST /predict to upload an image. Try `submission/example.png` (a cat). If port 8001 is occupied, use `-p 8002:80` and open port 8002 instead.

## Train and test

Requires Python 3.12+ and uv.

```bash
uv sync --frozen
uv run python -m scripts.train --epochs 10 --seed 42
uv run python -m pytest -q
```

Theory answers: [submission/theory_answers.txt](submission/theory_answers.txt).
