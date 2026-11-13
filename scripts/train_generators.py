"""Train the Assignment 4 generators on the CIFAR-10 training split."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR10

from app.diffusion import DIFFUSION_ARCHITECTURE, IMAGE_SHAPE, DiffusionProcess, DiffusionUNet
from app.energy import ENERGY_ARCHITECTURE, EnergyModel, langevin_sample

ROOT = Path(__file__).resolve().parents[1]
REPLAY_SIZE = 8192
EMA_DECAY = 0.995
ENERGY_REGULARIZATION = 0.1


def train_diffusion(model, process, optimizer, ema, images):
    optimizer.zero_grad(set_to_none=True)
    loss = process.training_loss(model, images)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    with torch.no_grad():
        for averaged, current in zip(ema.parameters(), model.parameters()):
            averaged.lerp_(current, 1 - EMA_DECAY)
    return torch.stack((loss.detach(),))


def train_energy(model, optimizer, replay, images, steps):
    optimizer.zero_grad(set_to_none=True)
    count = len(images)
    device = images.device
    indices = torch.randperm(len(replay), device=device)[:count]
    reset = torch.rand(count, 1, 1, 1, device=device) < 0.05
    fresh = torch.rand_like(images) * 2 - 1
    initial = torch.where(reset, fresh, replay[indices])
    negative = langevin_sample(model, initial, steps)
    replay[indices] = negative

    # Input gradients were used only in the sampler. This pass updates weights.
    positive_energy = model((images + 0.01 * torch.randn_like(images)).clamp(-1, 1))
    negative_energy = model(negative)
    loss = (positive_energy - negative_energy).mean()
    loss += ENERGY_REGULARIZATION * (positive_energy.square() + negative_energy.square()).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return torch.stack((loss.detach(), positive_energy.mean().detach(), negative_energy.mean().detach()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("diffusion", "ebm"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--langevin-steps", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--max-images", type=int, help="Limit the training split for a smoke test.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.langevin_steps) < 1:
        parser.error("epochs, batch size, and Langevin steps must be positive")
    if args.max_images is not None and args.max_images < 1:
        parser.error("max images must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    cifar = CIFAR10(args.data_dir, train=True, download=True)
    pixels = torch.from_numpy(cifar.data).permute(0, 3, 1, 2)
    dataset = TensorDataset(pixels)
    if args.max_images is not None:
        indices = torch.randperm(len(dataset))[:args.max_images].tolist()
        dataset = Subset(dataset, indices)
    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    diffusion = args.model == "diffusion"
    if diffusion:
        model = DiffusionUNet().to(device)
        process = DiffusionProcess().to(device)
        ema = deepcopy(model).requires_grad_(False)
        architecture = DIFFUSION_ARCHITECTURE
        learning_rate = 3e-4
        metrics = ("noise_mse",)
    else:
        model = EnergyModel().to(device)
        replay = torch.rand(max(REPLAY_SIZE, args.batch_size), *IMAGE_SHAPE, device=device) * 2 - 1
        architecture = ENERGY_ARCHITECTURE
        learning_rate = 2e-4
        metrics = ("loss", "positive_energy", "negative_energy")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"cifar10_{args.model}.pt"
    started = perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        totals = torch.zeros(len(metrics), device=device)
        for index, (batch,) in enumerate(batches, start=1):
            images = batch.to(device=device, dtype=torch.float32) / 127.5 - 1
            if diffusion:
                values = train_diffusion(model, process, optimizer, ema, images)
            else:
                values = train_energy(model, optimizer, replay, images, args.langevin_steps)
            totals += values * len(images)
            if index % 50 == 0:
                print(f"{args.model} epoch {epoch}: batch {index}/{len(batches)}", flush=True)
        values = (totals / len(dataset)).tolist()
        if not all(map(math.isfinite, values)):
            raise RuntimeError(f"Non-finite training metrics at epoch {epoch}: {values}")
        record = {"epoch": epoch, **dict(zip(metrics, values))}
        history.append(record)
        print(json.dumps(record), flush=True)
        inference_model = ema if diffusion else model
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "architecture": architecture,
                "model_state_dict": {
                    name: tensor.cpu() for name, tensor in inference_model.state_dict().items()
                },
                "dataset": "CIFAR-10 train",
                "training_images": len(dataset),
                "epochs": epoch,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "device": device,
                "learning_rate": learning_rate,
                "langevin_steps": None if diffusion else args.langevin_steps,
                "elapsed_seconds": round(perf_counter() - started, 1),
                "history": history,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
    print(f"Saved {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
