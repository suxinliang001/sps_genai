"""Train the MNIST GAN."""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .helper_lib import generate_samples, get_model, train_gan
from .helper_lib.model import ARCHITECTURE_ID, initialize_weights

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 uses all 60000 training images")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/gan.pt")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.limit < 0:
        parser.error("epochs, batch-size and lr must be positive; limit must be nonnegative")

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}; architecture: {ARCHITECTURE_ID}", flush=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(str(args.data_dir), train=True, download=True, transform=transform)
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"MNIST training images: {len(dataset)}", flush=True)
    model = get_model("GAN").to(device)
    model.apply(initialize_weights)
    optimizers = {
        "G": torch.optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999)),
        "D": torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def save_epoch(trained, history):
        torch.save({
            "architecture_id": ARCHITECTURE_ID,
            "model_state_dict": {k: v.detach().cpu() for k, v in trained.state_dict().items()},
            "epochs": history[-1]["epoch"], "training_images": len(dataset),
            "seed": args.seed, "batch_size": args.batch_size, "lr": args.lr,
            "history": history,
        }, args.checkpoint)
        generate_samples(
            trained, device, 64, seed=2026,
            output_path=args.results_dir / f"epoch_{history[-1]['epoch']:03d}.png",
        )
        with (args.results_dir / "losses.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "loss_G", "loss_D", "seconds"])
            writer.writeheader()
            writer.writerows(history)

    history = []
    train_gan(model, loader, nn.BCELoss(), optimizers, device, args.epochs, history=history, on_epoch_end=save_epoch)
    generate_samples(model, device, 64, seed=2026, output_path=args.results_dir / "generated.png")
    print(f"Saved checkpoint: {args.checkpoint}", flush=True)
    print(f"Saved samples: {args.results_dir / 'generated.png'}", flush=True)


if __name__ == "__main__":
    main()
