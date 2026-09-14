"""End-to-end orchestration shared by the command-line entry points."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data import generate_dataset, load_dataset, save_dataset
from .evaluate import evaluate_model, plot_history
from .train import choose_device, load_checkpoint, train_model


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)
    return config


def prepare_run_dir(config: dict[str, Any], output_dir: Path | None) -> Path:
    run_dir = output_dir or Path("runs") / str(config["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return run_dir


def generate_stage(config: dict[str, Any], run_dir: Path) -> dict[str, np.ndarray]:
    started = time.perf_counter()
    generated = generate_dataset(config)
    save_dataset(generated, run_dir / "data.npz")
    info = generated.pop("solver_info")
    iterations = np.asarray([item["iterations"] for item in info], dtype=np.float64)
    residuals = np.asarray([item["relative_residual"] for item in info], dtype=np.float64)
    stats = {
        "n_samples": len(info),
        "n_converged": int(sum(bool(item["converged"]) for item in info)),
        "iterations_mean": float(iterations.mean()),
        "iterations_max": int(iterations.max()),
        "relative_residual_max": float(residuals.max()),
        "generation_seconds": float(time.perf_counter() - started),
    }
    with (run_dir / "solver_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return {key: value for key, value in generated.items() if isinstance(value, np.ndarray)}


def run_all(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    dataset = generate_stage(config, run_dir)
    model, history, normalization = train_model(config, dataset, run_dir)
    plot_history(history, run_dir)
    return evaluate_model(config, dataset, model, normalization, run_dir)


def train_stage(config: dict[str, Any], run_dir: Path) -> None:
    dataset = load_dataset(run_dir / "data.npz")
    _, history, _ = train_model(config, dataset, run_dir)
    plot_history(history, run_dir)


def evaluate_stage(config: dict[str, Any], run_dir: Path, checkpoint: Path | None) -> dict[str, Any]:
    dataset = load_dataset(run_dir / "data.npz")
    if checkpoint is None:
        checkpoint = run_dir / ("best.pt" if (run_dir / "best.pt").exists() else "last.pt")
    device = choose_device(str(config.get("device", "auto")))
    model, normalization = load_checkpoint(checkpoint, config, device)
    return evaluate_model(config, dataset, model, normalization, run_dir)


def _validate_config(config: dict[str, Any]) -> None:
    required = {"run_name", "seed", "pde", "data", "model", "training", "plots"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"configuration is missing keys: {sorted(missing)}")
    if int(config["pde"]["n_grid"]) < 3:
        raise ValueError("pde.n_grid must be at least 3")
    if int(config["data"]["n_train"]) < 1 or int(config["data"]["n_test"]) < 1:
        raise ValueError("data.n_train and data.n_test must be positive")
