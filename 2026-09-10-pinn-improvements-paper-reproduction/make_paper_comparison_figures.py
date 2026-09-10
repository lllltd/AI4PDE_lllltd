"""Create direct original-PINN versus improved-method comparison figures.

This script only reads completed checkpoints; it does not retrain the models.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch

from burgers_reference import exact_burgers
from run_paper_adaptive_weight import PaperMLP
from run_paper_rar import pde


ROOT = Path("paper_reproductions")


def latest_checkpoint(patterns: tuple[str, ...]) -> Path:
    """Find a DeepXDE checkpoint while remaining compatible with older runs."""
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matches: {patterns}")
    return max(matches, key=lambda path: int(path.stem.rsplit("-", 1)[-1]))


def build_rar_model(checkpoint: Path) -> dde.Model:
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(
        geomtime,
        lambda x: -np.sin(np.pi * x[:, 0:1]),
        lambda _, on_initial: on_initial,
    )
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic], num_domain=2500, num_boundary=100, num_initial=160
    )
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    # Both checkpoints were saved after the official L-BFGS phase.
    model.compile("L-BFGS")
    model.restore(str(checkpoint), verbose=0)
    return model


def burgers_grid(t_max: float = 0.99) -> tuple[np.ndarray, ...]:
    x = np.linspace(-1.0, 1.0, 401)
    times = np.linspace(0.0, t_max, int(round(100 * t_max)) + 1)
    xx, tt = np.meshgrid(x, times)
    points = np.column_stack((xx.ravel(), tt.ravel()))
    exact = np.stack([exact_burgers(x, float(time)) for time in times])
    return x, times, points, exact


def plot_rar_comparison() -> None:
    baseline_checkpoint = latest_checkpoint(
        ("rar/original_pinn_model-*.pt", "rar_baseline/deepxde_rar_model-*.pt")
    )
    rar_checkpoint = latest_checkpoint(("rar/deepxde_rar_model-*.pt",))
    baseline = build_rar_model(baseline_checkpoint)
    rar = build_rar_model(rar_checkpoint)
    x, times, points, exact = burgers_grid()
    pred_baseline = baseline.predict(points).reshape(exact.shape)
    pred_rar = rar.predict(points).reshape(exact.shape)

    with (ROOT / "rar/rar_cycles.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    l2_values = [float(rows[0]["relative_l2"]), float(rows[-1]["relative_l2"])]
    residual_values = [
        float(rows[0]["mean_candidate_residual"]),
        float(rows[-1]["mean_candidate_residual"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), layout="constrained")
    colors = {"baseline": "#e67e22", "rar": "#2878b5"}
    for axis, target_time in zip(axes[0, :2], (0.50, 0.99)):
        index = int(np.argmin(np.abs(times - target_time)))
        axis.plot(x, exact[index], "k--", linewidth=2.45, label="Exact")
        axis.plot(x, pred_baseline[index], color=colors["baseline"], linewidth=2, label="Original PINN")
        axis.plot(x, pred_rar[index], color=colors["rar"], linewidth=2, label="PINN + RAR")
        axis.set(title=f"Solution slice at t={times[index]:.2f}", xlabel="x", ylabel="u")
        axis.grid(alpha=0.25)
        axis.legend()

    bars = axes[0, 2].bar(
        ["Original PINN", "PINN + RAR"], l2_values,
        color=[colors["baseline"], colors["rar"]],
    )
    axes[0, 2].bar_label(bars, labels=[f"{value:.4f}" for value in l2_values], padding=3)
    axes[0, 2].set(title="Relative L2 error (lower is better)", ylabel="Relative L2")
    axes[0, 2].set_ylim(0, max(l2_values) * 1.25)

    vmax = max(np.abs(pred_baseline - exact).max(), np.abs(pred_rar - exact).max())
    for axis, error, title in (
        (axes[1, 0], np.abs(pred_baseline - exact), "Original PINN absolute error"),
        (axes[1, 1], np.abs(pred_rar - exact), "PINN + RAR absolute error"),
    ):
        image = axis.imshow(
            error, origin="lower", aspect="auto", extent=[-1, 1, 0, 0.99],
            cmap="magma", vmin=0, vmax=vmax,
        )
        axis.set(title=title, xlabel="x", ylabel="t")
    fig.colorbar(image, ax=axes[1, :2].tolist(), label="Absolute error", shrink=0.88, pad=0.025)

    bars = axes[1, 2].bar(
        ["Original PINN", "PINN + RAR"], residual_values,
        color=[colors["baseline"], colors["rar"]],
    )
    axes[1, 2].bar_label(bars, labels=[f"{value:.4f}" for value in residual_values], padding=3)
    axes[1, 2].set(title="Mean candidate residual", ylabel="Mean |PDE residual|")
    axes[1, 2].set_ylim(0, max(residual_values) * 1.25)
    fig.suptitle("Residual-based Adaptive Refinement: direct comparison", fontsize=16)
    fig.savefig(ROOT / "rar/rar_vs_original_pinn.png", dpi=190)
    plt.close(fig)


def load_weight_model(problem: str, mode: str) -> PaperMLP:
    np.random.seed(1234)
    sample = -1.0 + 2.0 * np.random.rand(100000, 2)
    if problem == "burgers":
        sample[:, 1] = 0.5 * (sample[:, 1] + 1.0)
    model = PaperMLP([2, 50, 50, 50, 1], sample.mean(0), sample.std(0))
    state = torch.load(
        ROOT / f"adaptive_weight/{problem}/{mode}/model.pt",
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: PaperMLP, points: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(points, dtype=torch.float32)).numpy().ravel()


def plot_weight_comparison() -> None:
    helm_fixed = load_weight_model("helmholtz", "M1_fixed")
    helm_adaptive = load_weight_model("helmholtz", "M2_adaptive")
    burg_fixed = load_weight_model("burgers", "M1_fixed")
    burg_adaptive = load_weight_model("burgers", "M2_adaptive")

    axis_values = np.linspace(-1.0, 1.0, 160)
    hx, hy = np.meshgrid(axis_values, axis_values)
    helm_points = np.column_stack((hx.ravel(), hy.ravel()))
    helm_exact = np.sin(math.pi * helm_points[:, 0]) * np.sin(4.0 * math.pi * helm_points[:, 1])
    helm_errors = [
        np.abs(predict(helm_fixed, helm_points) - helm_exact).reshape(hx.shape),
        np.abs(predict(helm_adaptive, helm_points) - helm_exact).reshape(hx.shape),
    ]

    x, times, burgers_points, burgers_exact = burgers_grid(t_max=1.0)
    burg_predictions = [
        predict(burg_fixed, burgers_points).reshape(burgers_exact.shape),
        predict(burg_adaptive, burgers_points).reshape(burgers_exact.shape),
    ]
    helm_l2 = [0.1677821856, 0.0194178549]
    burg_l2 = [0.1101230914, 0.0598412814]
    colors = ["#e67e22", "#2878b5"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), layout="constrained")
    vmax = max(error.max() for error in helm_errors)
    for axis, error, title in zip(
        axes[0, :2], helm_errors,
        ("Helmholtz: original PINN error", "Helmholtz: adaptive-weight error"),
    ):
        image = axis.imshow(
            error, origin="lower", extent=[-1, 1, -1, 1], aspect="auto",
            cmap="magma", vmin=0, vmax=vmax,
        )
        axis.set(title=title, xlabel="x", ylabel="y")
    fig.colorbar(image, ax=axes[0, :2].tolist(), label="Absolute error", shrink=0.88, pad=0.025)
    bars = axes[0, 2].bar(["Original PINN", "Adaptive weight"], helm_l2, color=colors)
    axes[0, 2].bar_label(bars, labels=[f"{v:.4f}" for v in helm_l2], padding=3)
    axes[0, 2].set(title="Helmholtz relative L2", ylabel="Relative L2")
    axes[0, 2].set_ylim(0, max(helm_l2) * 1.25)

    for axis, target_time in zip(axes[1, :2], (0.5, 1.0)):
        index = int(np.argmin(np.abs(times - target_time)))
        axis.plot(x, burgers_exact[index], "k--", linewidth=2.5, label="Exact")
        axis.plot(x, burg_predictions[0][index], color=colors[0], linewidth=2, label="Original PINN")
        axis.plot(x, burg_predictions[1][index], color=colors[1], linewidth=2, label="Adaptive weight")
        axis.set(title=f"Burgers transfer at t={times[index]:.2f}", xlabel="x", ylabel="u")
        axis.grid(alpha=0.25)
        axis.legend()
    bars = axes[1, 2].bar(["Original PINN", "Adaptive weight"], burg_l2, color=colors)
    axes[1, 2].bar_label(bars, labels=[f"{v:.4f}" for v in burg_l2], padding=3)
    axes[1, 2].set(title="Burgers relative L2", ylabel="Relative L2")
    axes[1, 2].set_ylim(0, max(burg_l2) * 1.25)
    fig.suptitle("Gradient-based adaptive loss weighting: direct comparison", fontsize=16)
    fig.savefig(ROOT / "adaptive_weight/adaptive_weight_vs_original_pinn.png", dpi=190)
    plt.close(fig)


def main() -> None:
    plot_rar_comparison()
    plot_weight_comparison()
    print("Saved direct comparison figures.")


if __name__ == "__main__":
    main()
