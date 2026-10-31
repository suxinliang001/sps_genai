# Assignment 3: MNIST image generation

A PyTorch GAN trained on MNIST, with generation endpoints added to the existing
FastAPI application through `assignment3.main`.

## Architecture

Generator: `(B,100)` -> Linear(100,6272) -> reshape `(B,128,7,7)` ->
ConvTranspose2d(128,64,4,2,1) -> BatchNorm2d(64) -> ReLU ->
ConvTranspose2d(64,1,4,2,1) -> Tanh -> `(B,1,28,28)`.

Discriminator: `(B,1,28,28)` -> Conv2d(1,64,4,2,1) -> LeakyReLU(0.2) ->
Conv2d(64,128,4,2,1) -> BatchNorm2d(128) -> LeakyReLU(0.2) ->
Flatten -> Linear(6272,1) -> Sigmoid.

The PDF calls the final output a real/fake probability but does not explicitly
list its activation. This implementation adds Sigmoid and uses BCELoss.
Adam, learning rate 0.0002 and betas (0.5,0.999) are implementation choices;
the assignment does not prescribe them. MNIST is normalized to [-1,1] to match
the generator's Tanh output. This assignment does not require classroom WGAN
weight clipping or a Wasserstein objective.

## Run in VS Code

Open the repository root in VS Code and select `.venv/bin/python`.
Open **Terminal > New Terminal** and run commands from the repository root:

```bash
source .venv/bin/activate
python -m uvicorn assignment3.main:app --host 127.0.0.1 --port 8003
```

Open http://127.0.0.1:8003/docs. The combined app retains `/generate`,
`/embedding` and `/predict`, and adds:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /gan/status | Check model availability and training metadata |
| POST | /gan/generate | Return a PNG grid of generated digits |

Example request body:

```json
{"num_samples": 25, "seed": 42}
```

To save a response from a second terminal:

```bash
curl -X POST http://127.0.0.1:8003/gan/generate \
  -H 'Content-Type: application/json' \
  -d '{"num_samples":25,"seed":42}' \
  --output assignment3/results/api-generated.png
```

The endpoint accepts 1 to 64 images. It returns 503 if the trained checkpoint is
missing. Use a different seed for another grid. This is an unconditional GAN:
the request does not select which digit to generate. Restart the API after
retraining so it loads the new checkpoint.

## Train

On another machine, first install the repository dependencies with
`uv sync --frozen`. No new package is needed for ordinary training or inference.

```bash
# Short workflow check, not a finished model:
python -m assignment3.train --epochs 1 --limit 1024 --device cpu \
  --checkpoint assignment3/results/smoke.pt --results-dir assignment3/results/smoke

# Full MNIST training; choose the number of epochs based on generated samples:
python -m assignment3.train --epochs 20 --device cpu
```

`models/gan.pt` is saved after each completed epoch. `results/losses.csv`
records losses and duration; `results/epoch_*.png` uses the same noise every
epoch for comparison; `results/generated.png` is the final sample grid.
Inspect both recognizable shapes and variety; GAN losses alone do not establish
image quality. Training metadata is stored in the checkpoint.

## Docker deployment

Build from the repository root, using the new Dockerfile:

```bash
docker build -f assignment3/Dockerfile -t sps-genai-assignment3 .
docker run --rm --name sps-genai-assignment3 -p 127.0.0.1:8003:80 sps-genai-assignment3
```

Open http://127.0.0.1:8003/docs and call POST `/gan/generate`.
The image includes the GAN checkpoint and the earlier API's source and model.
Training or dataset downloads are not needed when starting the container.
The build uses the repository's existing frozen dependency lock.

## Tests

```bash
python -m pytest assignment3/tests tests -q
```

Tests cover the specified layer types/dimensions, forward-pass shapes, both
networks updating, BatchNorm state during a generator update, checkpoint
loading, PNG responses, repeatable seeds, invalid requests, missing models,
and preservation of the earlier API routes.

## Submission

- Theory and calculation work: `assignment3/theory_answers.pdf` (formatted)
  and `assignment3/theory_answers.txt` (editable).
- Implementation: `assignment3/helper_lib/`, `train.py`, `api.py`, `main.py`.
- Deployment: `assignment3/Dockerfile` and the repository dependency files.
- Trained weights: `assignment3/models/gan.pt`.
- Training evidence: `assignment3/results/` and `assignment3/TRAINING_REPORT.txt`.

Clone the complete repository and follow the Docker commands above. The combined
entry point uses the existing `app/`, `models/`, `pyproject.toml` and `uv.lock`.
