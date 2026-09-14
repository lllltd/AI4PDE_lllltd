"""DeepONet training utilities."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import DeepONet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def make_model(config: dict[str, Any]) -> DeepONet:
    model = config["model"]
    return DeepONet(
        n_sensors=int(config["pde"]["n_grid"]),
        branch_hidden=[int(v) for v in model["branch_hidden"]],
        trunk_hidden=[int(v) for v in model["trunk_hidden"]],
        latent_dim=int(model["latent_dim"]),
        activation=str(model["activation"]),
        output_bias=bool(model.get("output_bias", False)),
    )


def train_model(
    config: dict[str, Any],
    dataset: dict[str, np.ndarray],
    run_dir: Path,
) -> tuple[DeepONet, list[dict[str, float]], dict[str, float]]:
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    model = make_model(config).to(device)
    training = config["training"]

    # Cornell normalizes the scalar target, not the input source functions.
    u_mean = float(dataset["u_train"].mean())
    u_std = float(dataset["u_train"].std())
    if u_std < 1e-12:
        u_std = 1.0
    normalization = {"u_mean": u_mean, "u_std": u_std}

    f_train = torch.as_tensor(dataset["f_train"], dtype=torch.float32)
    u_train = torch.as_tensor(
        (dataset["u_train"] - u_mean) / u_std, dtype=torch.float32
    )
    has_validation = dataset["f_val"].shape[0] > 0
    if has_validation:
        f_val = torch.as_tensor(dataset["f_val"], dtype=torch.float32, device=device)
        u_val = torch.as_tensor(
            (dataset["u_val"] - u_mean) / u_std, dtype=torch.float32, device=device
        )
    x = torch.as_tensor(dataset["x"][:, None], dtype=torch.float32, device=device)

    generator = torch.Generator().manual_seed(seed)
    train_data = TensorDataset(f_train, u_train)
    loader = DataLoader(
        train_data, batch_size=int(training["batch_size"]), shuffle=True, generator=generator
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        if str(training.get("sampling", "full_epoch")) == "cornell_random_batch":
            indices = torch.randperm(len(train_data), generator=generator)[: int(training["batch_size"])]
            batches = [(f_train[indices], u_train[indices])]
        else:
            batches = loader
        for sources, targets in batches:
            sources = sources.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(sources, x)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * sources.shape[0]
            total_items += sources.shape[0]
        scheduler.step()
        train_loss = total_loss / total_items

        model.eval()
        if has_validation:
            with torch.no_grad():
                val_loss = float(loss_fn(model(f_val, x), u_val).item())
        else:
            val_loss = train_loss
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr}
        history.append(row)
        eval_every = int(training.get("eval_every", 100))
        if epoch == 1 or epoch == epochs or epoch % eval_every == 0:
            label = "val" if has_validation else "selection"
            print(
                f"epoch {epoch:5d}/{epochs}  train={train_loss:.6e}  "
                f"{label}={val_loss:.6e}  lr={current_lr:.3e}"
            )
        if has_validation and val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(run_dir / "best.pt", model, config, normalization, epoch, val_loss)

    _save_checkpoint(run_dir / "last.pt", model, config, normalization, epochs, history[-1]["val_loss"])
    _write_history(run_dir / "history.csv", history)
    if has_validation:
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    return model, history, normalization


def load_checkpoint(
    path: Path, config: dict[str, Any], device: torch.device
) -> tuple[DeepONet, dict[str, float]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint["normalization"]


def _save_checkpoint(
    path: Path,
    model: DeepONet,
    config: dict[str, Any],
    normalization: dict[str, float],
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "normalization": normalization,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history)
