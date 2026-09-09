"""Run controlled experiments for Burgers PINN limitations."""

from __future__ import annotations

import argparse
import csv
import math
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_burgers_pinn import (
    BurgersPINN,
    pde_residual,
    save_loss_history,
    save_solution_plot,
    set_seed,
    train,
)


def make_args(cli: argparse.Namespace, nu: float, n_f: int, out: Path) -> Namespace:
    """All controls are identical except nu or N_f."""
    return Namespace(
        epochs=cli.epochs,
        n_f=n_f,
        n_ic=256,
        n_bc=256,
        hidden_width=64,
        hidden_layers=4,
        learning_rate=1e-3,
        nu=nu,
        device=cli.device,
        seed=42,
        print_every=cli.print_every,
        output_dir=out,
    )


def predict_grid(
    model: BurgersPINN, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 401)
    t = np.linspace(0.0, 1.0, 201)
    xx, tt = np.meshgrid(x, t)
    with torch.no_grad():
        xt = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=device)
        tt_tensor = torch.tensor(tt.reshape(-1, 1), dtype=torch.float32, device=device)
        u = model(xt, tt_tensor).cpu().numpy().reshape(xx.shape)
    return x, t, u


def validate(model: BurgersPINN, nu: float, device: torch.device) -> dict[str, float]:
    """Measure every run on the same fixed points, not its random training batch."""
    generator = torch.Generator(device=device).manual_seed(20260909)
    x_f = -1.0 + 2.0 * torch.rand(5000, 1, generator=generator, device=device)
    t_f = torch.rand(5000, 1, generator=generator, device=device)
    x_f.requires_grad_(True)
    t_f.requires_grad_(True)
    residual = pde_residual(model, x_f, t_f, nu)
    pde_rmse = torch.sqrt(torch.mean(residual.square())).item()

    with torch.no_grad():
        x_ic = torch.linspace(-1.0, 1.0, 1001, device=device).reshape(-1, 1)
        ic_error = model(x_ic, torch.zeros_like(x_ic)) + torch.sin(math.pi * x_ic)
        ic_rmse = torch.sqrt(torch.mean(ic_error.square())).item()
        t_bc = torch.linspace(0.0, 1.0, 1001, device=device).reshape(-1, 1)
        left = model(-torch.ones_like(t_bc), t_bc).abs().max().item()
        right = model(torch.ones_like(t_bc), t_bc).abs().max().item()

    x_slope = torch.linspace(-1.0, 1.0, 2001, device=device).reshape(-1, 1)
    x_slope.requires_grad_(True)
    u_slope = model(x_slope, torch.full_like(x_slope, 0.5))
    u_x = torch.autograd.grad(u_slope, x_slope, torch.ones_like(u_slope))[0]
    return {
        "pde_rmse_fixed": pde_rmse,
        "ic_rmse_fixed": ic_rmse,
        "bc_max_abs_fixed": max(left, right),
        "max_abs_ux_t0.5": u_x.abs().max().item(),
    }


def run_one(
    name: str, nu: float, n_f: int, cli: argparse.Namespace
) -> dict[str, object]:
    out = cli.output_dir / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    args = make_args(cli, nu, n_f, out)
    set_seed(args.seed)
    print("\n" + "=" * 68)
    print(f"Run {name}: nu={nu:.8g}, N_f={n_f}")
    print("=" * 68)
    model, history = train(args)
    device = next(model.parameters()).device
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_width": args.hidden_width,
            "hidden_layers": args.hidden_layers,
            "nu": nu,
            "n_f": n_f,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        out / "burgers_pinn.pt",
    )
    save_loss_history(history, out)
    save_solution_plot(model, device, out)
    x, t, u = predict_grid(model, device)
    return {
        "name": name,
        "nu": nu,
        "n_f": n_f,
        "history": history,
        "metrics": validate(model, nu, device),
        "x": x,
        "t": t,
        "u": u,
    }


def comparison_slices(
    results: list[dict[str, object]],
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    """Plot identical time slices so only the controlled variable changes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for result, label in zip(results, labels):
        x = np.asarray(result["x"])
        times = np.asarray(result["t"])
        u = np.asarray(result["u"])
        for axis, time_value in zip(axes, (0.5, 1.0)):
            row = int(np.argmin(np.abs(times - time_value)))
            axis.plot(x, u[row], label=label)
            axis.set_title(f"t = {time_value}")
            axis.set_xlabel("x")
            axis.grid(alpha=0.25)
    axes[0].set_ylabel("PINN prediction u(x,t)")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def viscosity_heatmaps(results: list[dict[str, object]], path: Path) -> None:
    labels = [r"$\nu=0.1/\pi$", r"$\nu=0.01/\pi$", r"$\nu=0.001/\pi$"]
    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for axis, result, label in zip(axes, results, labels):
        image = axis.imshow(
            np.asarray(result["u"]),
            extent=[-1.0, 1.0, 0.0, 1.0],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.set_title(label)
        axis.set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(image, ax=axes, label="u", shrink=0.82, pad=0.02)
    fig.suptitle("Viscosity experiment")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_summary(results: list[dict[str, object]], out: Path) -> None:
    fields = [
        "name", "nu", "n_f", "final_total", "final_pde", "final_ic", "final_bc",
        "pde_rmse_fixed", "ic_rmse_fixed", "bc_max_abs_fixed", "max_abs_ux_t0.5",
    ]
    rows = []
    for result in results:
        history = result["history"]
        row = {
            "name": result["name"],
            "nu": result["nu"],
            "n_f": result["n_f"],
            "final_total": history["total"][-1],
            "final_pde": history["pde"][-1],
            "final_ic": history["ic"][-1],
            "final_bc": history["bc"][-1],
        }
        row.update(result["metrics"])
        rows.append(row)
    with (out / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    positions = np.arange(len(results))
    width = 0.24
    fig, axis = plt.subplots(figsize=(12, 5))
    for offset, key, label in (
        (-width, "pde", "PDE"), (0.0, "ic", "IC"), (width, "bc", "BC")
    ):
        values = [result["history"][key][-1] for result in results]
        axis.bar(positions + offset, values, width, label=label)
    axis.set_yscale("log")
    axis.set_ylabel("Final training loss")
    axis.set_xticks(positions, [str(result["name"]) for result in results])
    axis.tick_params(axis="x", rotation=18)
    axis.set_title("Loss imbalance across controlled experiments")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_balance_comparison.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_nu = 0.01 / math.pi
    configurations = [
        ("nu_high", 0.1 / math.pi, 2000),
        ("baseline", baseline_nu, 2000),
        ("nu_low", 0.001 / math.pi, 2000),
        ("nf_500", baseline_nu, 500),
        ("nf_10000", baseline_nu, 10000),
    ]
    results = [run_one(name, nu, n_f, cli) for name, nu, n_f in configurations]
    nu_results = results[:3]
    nf_results = [results[3], results[1], results[4]]
    viscosity_heatmaps(nu_results, cli.output_dir / "nu_comparison_heatmaps.png")
    comparison_slices(
        nu_results,
        [r"$\nu=0.1/\pi$", r"$\nu=0.01/\pi$", r"$\nu=0.001/\pi$"],
        "Viscosity experiment: lower viscosity should be harder",
        cli.output_dir / "nu_comparison_slices.png",
    )
    comparison_slices(
        nf_results,
        [r"$N_f=500$", r"$N_f=2000$", r"$N_f=10000$"],
        r"Collocation experiment at $\nu=0.01/\pi$",
        cli.output_dir / "nf_comparison_slices.png",
    )
    save_summary(results, cli.output_dir)
    print(f"\nAll experiment outputs saved to {cli.output_dir.resolve()}")


if __name__ == "__main__":
    main()
