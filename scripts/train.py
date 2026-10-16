"""Train on CIFAR-10 training data; evaluate once on its held-out test set."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from app.cnn import CIFAR10CNN, CLASSES, image_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train = CIFAR10("data", train=True, download=True, transform=image_transform())
    test = CIFAR10("data", train=False, download=True, transform=image_transform())
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = CIFAR10CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_total = correct = count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
        record = {"epoch": epoch + 1, "train_loss": loss_total / count, "train_accuracy": correct / count}
        history.append(record)
        print(json.dumps(record), flush=True)
    model.eval()
    correct = count = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.inference_mode():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += (predictions == labels).sum().item()
            count += len(labels)
            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                class_total[label] += 1
                class_correct[label] += int(label == prediction)
    Path("models").mkdir(exist_ok=True)
    torch.save({"model_state_dict": model.cpu().state_dict(), "classes": list(CLASSES), "input_size": 64}, "models/cifar10_cnn.pt")
    metrics = {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
               "device": device, "optimizer": "Adam", "learning_rate": 0.001,
               "train_samples": len(train), "test_samples": count,
               "test_accuracy": correct / count,
               "per_class_accuracy": {name: class_correct[i] / class_total[i] for i, name in enumerate(CLASSES)},
               "parameters": sum(p.numel() for p in model.parameters()),
               "duration_seconds": time.time() - started, "history": history}
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    Path("submission/example.png").parent.mkdir(exist_ok=True)
    raw_test = CIFAR10("data", train=False, download=False)
    raw_test[0][0].save("submission/example.png")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
