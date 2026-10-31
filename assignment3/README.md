# Assignment 3: MNIST GAN

A PyTorch GAN for generating handwritten digits. The included model was trained
on all 60,000 MNIST training images for 5 epochs.

## Run

From the repository root:

```bash
docker build -f assignment3/Dockerfile -t sps-genai-assignment3 .
docker run --rm -p 127.0.0.1:8003:80 sps-genai-assignment3
```

Open http://127.0.0.1:8003/docs and use **POST /gan/generate**:

```json
{"num_samples": 25, "seed": 42}
```

## Train

```bash
uv sync --frozen
uv run python -m assignment3.train --epochs 20 --device cpu
```

## Test

```bash
uv run python -m pytest assignment3/tests tests -q
```

Theory answers: [theory_answers.txt](theory_answers.txt).
