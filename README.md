# SPS Generative AI coursework

## Assignment 4 · CIFAR-10 image generators

The existing FastAPI app serves two unconditional image generators. Both checkpoints
are included and use all 50,000 CIFAR-10 training images at 32 × 32 RGB.

| Endpoint | Training | Sampling |
| --- | --- | --- |
| `POST /diffusion/generate` | 5 epochs, batch size 64, noise-prediction MSE | 48 DDIM steps from Gaussian noise |
| `POST /ebm/generate` | 8 epochs, batch size 128, 12 Langevin steps per training batch | 256 Langevin steps from uniform noise |

Start the server using the commands below. Try either endpoint at
http://127.0.0.1:8001/docs or save the response from the terminal.

```bash
curl -fsS http://127.0.0.1:8001/diffusion/generate \
  -H 'Content-Type: application/json' \
  -d '{"num_samples": 4, "seed": 42}' -o diffusion.png

curl -fsS http://127.0.0.1:8001/ebm/generate \
  -H 'Content-Type: application/json' \
  -d '{"num_samples": 4, "seed": 42}' -o ebm.png
```

Both endpoints accept 1–16 samples and return an RGB PNG grid. A seed fixes
the samples for the same runtime and request. A missing checkpoint returns
HTTP 503.

To reproduce the training from the repository root, run the following
commands. The trainer selects CUDA, MPS, or CPU based on availability.
Use `--device cpu` to override it.

```bash
uv sync --frozen
uv run python -m scripts.train_generators diffusion --epochs 5 --batch-size 64
uv run python -m scripts.train_generators ebm --epochs 8 --batch-size 128 --langevin-steps 12
```

The diffusion model uses a timestep-conditioned U-Net, a cosine variance
schedule, and an exponential moving average of the weights. Its training
noise MSE fell from 0.13147 in epoch 1 to 0.07056 in epoch 5.
The EBM combines a learned CNN energy with a quadratic pixel smoothness
term. Its sampler calls `torch.autograd.grad` on the input pixels and
detaches the negative samples before the weight update. The training
history and run settings are stored in each checkpoint.

These compact models generate coarse 32 × 32 images. EBM samples can be
blurry or abstract.

See [the generated grids](assignment4/README.md) and [theory answers](assignment4/theory_answers.md).

## Assignment 2 · CIFAR-10 CNN

A PyTorch CNN trained on CIFAR-10, with an image classification endpoint added to the existing FastAPI app. The trained model is included. Test accuracy: 65.58%.

### Run the API

```bash
docker build -t sps-genai-assignment2 .
docker run --rm -p 8001:80 sps-genai-assignment2
```

Open http://127.0.0.1:8001/docs and use POST /predict to upload an image. Try `submission/example.png` (a cat). If port 8001 is occupied, use `-p 8002:80` and open port 8002 instead.

### Train and test the classifier

Requires Python 3.12+ and uv.

```bash
uv sync --frozen
uv run python -m scripts.train --epochs 10 --seed 42
uv run python -m pytest -q
```

Theory answers: [submission/theory_answers.txt](submission/theory_answers.txt).
