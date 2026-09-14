"""Evaluation metrics and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import DeepONet
from .train import choose_device


def predict(
    model: DeepONet,
    sources: np.ndarray,
    x: np.ndarray,
    normalization: dict[str, float],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        source_tensor = torch.as_tensor(sources, dtype=torch.float32, device=device)
        x_tensor = torch.as_tensor(x[:, None], dtype=torch.float32, device=device)
        normalized = model(source_tensor, x_tensor).cpu().numpy()
    return normalized * normalization["u_std"] + normalization["u_mean"]


def evaluate_model(
    config: dict[str, Any],
    dataset: dict[str, np.ndarray],
    model: DeepONet,
    normalization: dict[str, float],
    run_dir: Path,
) -> dict[str, Any]:
    device = choose_device(str(config.get("device", "auto")))
    model = model.to(device)
    truth = dataset["u_test"]
    predictions = predict(model, dataset["f_test"], dataset["x"], normalization, device)
    error = predictions - truth
    per_sample_relative_l2 = np.linalg.norm(error, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1), 1e-12
    )
    metrics: dict[str, Any] = {
        "test_mse": float(np.mean(error**2)),
        "test_rmse": float(np.sqrt(np.mean(error**2))),
        "test_mae": float(np.mean(np.abs(error))),
        "global_relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(truth), 1e-12)),
        "per_sample_relative_l2_mean": float(np.mean(per_sample_relative_l2)),
        "per_sample_relative_l2_median": float(np.median(per_sample_relative_l2)),
        "per_sample_relative_l2_max": float(np.max(per_sample_relative_l2)),
        "n_test": int(truth.shape[0]),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    np.savez_compressed(
        run_dir / "predictions.npz",
        x=dataset["x"],
        f_test=dataset["f_test"],
        u_true=truth,
        u_pred=predictions,
        relative_l2=per_sample_relative_l2,
    )
    _plot_test_samples(config, dataset, predictions, run_dir)
    return metrics


def plot_history(history: list[dict[str, float]], run_dir: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.semilogy(epochs, [row["train_loss"] for row in history], label="train")
    axis.semilogy(epochs, [row["val_loss"] for row in history], label="validation/selection")
    axis.set(xlabel="Epoch", ylabel="Normalized MSE", title="DeepONet training loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


def _plot_test_samples(
    config: dict[str, Any],
    dataset: dict[str, np.ndarray],
    predictions: np.ndarray,
    run_dir: Path,
) -> None:
    x = dataset["x"]
    n_plots = min(int(config["plots"]["n_test_samples"]), predictions.shape[0])
    for index in range(n_plots):
        truth = dataset["u_test"][index]
        prediction = predictions[index]
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
        axes[0].plot(x, dataset["f_test"][index], color="tab:blue")
        axes[0].set(title="Source f(x)", xlabel="x", ylabel="f")
        axes[1].plot(x, truth, label="u true", color="black", linewidth=2)
        axes[1].plot(x, prediction, "--", label="u predicted", color="tab:orange")
        axes[1].set(title="Darcy solution", xlabel="x", ylabel="u")
        axes[1].legend()
        axes[2].plot(x, np.abs(prediction - truth), color="tab:red")
        axes[2].set(title="Absolute error", xlabel="x", ylabel="|error|")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        relative = np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-12)
        fig.suptitle(f"Test sample {index} — relative L2 = {relative:.3e}")
        fig.tight_layout()
        fig.savefig(run_dir / f"prediction_sample_{index:03d}.png", dpi=160)
        plt.close(fig)
