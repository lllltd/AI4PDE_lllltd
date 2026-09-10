"""Paper-faithful DeepXDE RAR experiment for the Burgers equation.

Adapted from the official DeepXDE example:
https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Burgers_RAR.py

The PDE, point counts, network, optimizers, one-point anchor update, and stopping
criterion follow the upstream example. Extra code only saves metrics and figures.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

from burgers_reference import NU, exact_burgers


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - NU * dy_xx


def evaluate(model: dde.Model) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 401)
    times = np.linspace(0.0, 0.99, 100)
    xx, tt = np.meshgrid(x, times)
    points = np.column_stack((xx.ravel(), tt.ravel()))
    prediction = model.predict(points).reshape(xx.shape)
    exact = np.stack([exact_burgers(x, float(time)) for time in times])
    relative_l2 = float(np.linalg.norm(prediction - exact) / np.linalg.norm(exact))
    return relative_l2, prediction, exact, times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_reproductions/rar"))
    parser.add_argument("--adam-iterations", type=int, default=10000)
    parser.add_argument("--rar-adam-iterations", type=int, default=10000)
    parser.add_argument("--max-rar-cycles", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=100000)
    parser.add_argument("--mean-residual-target", type=float, default=0.005)
    parser.add_argument("--skip-lbfgs", action="store_true", help="Smoke-test only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dde.config.set_random_seed(42)
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
        geomtime,
        pde,
        [bc, ic],
        num_domain=2500,
        num_boundary=100,
        num_initial=160,
    )
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    records: list[dict[str, float]] = []
    model.compile("adam", lr=1e-3)
    model.train(iterations=args.adam_iterations, display_every=1000)
    if not args.skip_lbfgs:
        model.compile("L-BFGS")
        model.train(display_every=1000)

    # Keep the matched original-PINN checkpoint for a direct before/after plot.
    model.save(str(args.output_dir / "original_pinn_model"), protocol="backend")

    candidates = geomtime.random_points(args.candidate_count)
    for cycle in range(args.max_rar_cycles + 1):
        residual = np.abs(model.predict(candidates, operator=pde)).ravel()
        mean_residual = float(np.mean(residual))
        relative_l2, _, _, _ = evaluate(model)
        records.append(
            {
                "cycle": cycle,
                "anchors_added": cycle,
                "mean_candidate_residual": mean_residual,
                "max_candidate_residual": float(np.max(residual)),
                "relative_l2": relative_l2,
            }
        )
        print(
            f"RAR cycle {cycle}: mean residual={mean_residual:.4e}, "
            f"max residual={np.max(residual):.4e}, relative L2={relative_l2:.4e}"
        )
        if mean_residual <= args.mean_residual_target or cycle == args.max_rar_cycles:
            break

        index = int(np.argmax(residual))
        anchor = candidates[index : index + 1]
        print(f"Adding one official-style anchor: {anchor[0].tolist()}")
        data.add_anchors(anchor)
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
        model.compile("adam", lr=1e-3)
        model.train(
            iterations=args.rar_adam_iterations,
            disregard_previous_best=True,
            callbacks=[early_stopping],
            display_every=1000,
        )
        if not args.skip_lbfgs:
            model.compile("L-BFGS")
            model.train(display_every=1000)

    relative_l2, prediction, exact, times = evaluate(model)
    with (args.output_dir / "rar_cycles.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    model.save(str(args.output_dir / "deepxde_rar_model"), protocol="backend")

    x = np.linspace(-1.0, 1.0, prediction.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for time in (0.25, 0.5, 0.75, 0.99):
        index = int(np.argmin(np.abs(times - time)))
        axes[0].plot(x, exact[index], "k--", alpha=0.65)
        axes[0].plot(x, prediction[index], label=f"t={times[index]:.2f}")
    axes[0].set(title=f"Official-style RAR, relative L2={relative_l2:.3e}", xlabel="x", ylabel="u")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].semilogy(
        [record["anchors_added"] for record in records],
        [record["mean_candidate_residual"] for record in records],
        marker="o",
        label="mean residual",
    )
    axes[1].axhline(args.mean_residual_target, color="black", linestyle="--", label="paper stop target")
    axes[1].set(xlabel="Added anchors", ylabel="Candidate residual", title="RAR refinement history")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "rar_result.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
