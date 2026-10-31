import time
import torch
from torch import nn


def train_gan(
    model, data_loader, criterion, optimizer, device="cpu", epochs=10,
    *, history=None, on_epoch_end=None,
):
    if epochs < 1 or len(data_loader) == 0:
        raise ValueError("Use positive epochs and a nonempty data loader.")
    model.to(device)
    G, D = model.generator, model.discriminator
    opt_G, opt_D = optimizer["G"], optimizer["D"]
    if history is None:
        history = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total_G = total_D = 0.0
        count = 0
        for real, _digit_labels in data_loader:
            real = real.to(device)
            batch_size = len(real)
            ones = torch.ones(batch_size, 1, device=device)
            zeros = torch.zeros(batch_size, 1, device=device)
            fake = G(torch.randn(batch_size, model.latent_dim, device=device))

            opt_D.zero_grad(set_to_none=True)
            loss_D = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
            loss_D.backward()
            opt_D.step()

            # Use batch statistics without updating running estimates.
            opt_G.zero_grad(set_to_none=True)
            batch_norms = [layer for layer in D.modules() if isinstance(layer, nn.BatchNorm2d)]
            tracking_flags = [layer.track_running_stats for layer in batch_norms]
            for layer in batch_norms:
                layer.track_running_stats = False
            flags = [p.requires_grad for p in D.parameters()]
            for parameter in D.parameters():
                parameter.requires_grad_(False)
            try:
                loss_G = criterion(D(fake), ones)
                loss_G.backward()
                opt_G.step()
            finally:
                for parameter, flag in zip(D.parameters(), flags):
                    parameter.requires_grad_(flag)
                for layer, flag in zip(batch_norms, tracking_flags):
                    layer.track_running_stats = flag

            total_G += float(loss_G.detach()) * batch_size
            total_D += float(loss_D.detach()) * batch_size
            count += batch_size
        row = {
            "epoch": epoch,
            "loss_G": total_G / count,
            "loss_D": total_D / count,
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | D={row['loss_D']:.4f} "
            f"G={row['loss_G']:.4f} | {row['seconds']:.1f}s", flush=True,
        )
        if on_epoch_end is not None:
            on_epoch_end(model, history)
    return model
