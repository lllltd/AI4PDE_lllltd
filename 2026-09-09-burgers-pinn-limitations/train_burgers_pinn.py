"""Train a Physics-Informed Neural Network for the 1D viscous Burgers equation.

The problem is

    u_t + u*u_x - nu*u_xx = 0,              x in [-1, 1], t in [0, 1]
    u(x, 0) = -sin(pi*x),
    u(-1, t) = u(1, t) = 0.

No labelled interior solution data are used during training.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class BurgersPINN(nn.Module):
    """A fully connected tanh network mapping (x, t) to u(x, t)."""

    def __init__(self, hidden_width: int = 64, hidden_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((x, t), dim=1))


def set_seed(seed: int) -> None:
    """Make repeated runs as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    """Use CUDA when requested and available; otherwise use the CPU."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def uniform_column(
    count: int,
    low: float,
    high: float,
    device: torch.device,
    requires_grad: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a column of uniform random samples on [low, high]."""
    values = low + (high - low) * torch.rand(
        count, 1, device=device, generator=generator
    )
    values.requires_grad_(requires_grad)
    return values


def pde_residual(
    model: BurgersPINN,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """Compute r = u_t + u*u_x - nu*u_xx with automatic differentiation."""
    u = model(x, t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    return u_t + u * u_x - nu * u_xx


def compute_losses(
    model: BurgersPINN,
    n_f: int,
    n_ic: int,
    n_bc: int,
    nu: float,
    device: torch.device,
    generators: tuple[torch.Generator, torch.Generator, torch.Generator] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly sample points and compute PDE, initial, and boundary losses."""
    mse = nn.MSELoss()

    generator_f, generator_ic, generator_bc = generators or (None, None, None)
    x_f = uniform_column(
        n_f, -1.0, 1.0, device, requires_grad=True, generator=generator_f
    )
    t_f = uniform_column(
        n_f, 0.0, 1.0, device, requires_grad=True, generator=generator_f
    )
    residual = pde_residual(model, x_f, t_f, nu)
    loss_pde = mse(residual, torch.zeros_like(residual))

    x_ic = uniform_column(n_ic, -1.0, 1.0, device, generator=generator_ic)
    t_ic = torch.zeros_like(x_ic)
    u_ic_target = -torch.sin(math.pi * x_ic)
    loss_ic = mse(model(x_ic, t_ic), u_ic_target)

    t_bc = uniform_column(n_bc, 0.0, 1.0, device, generator=generator_bc)
    x_left = -torch.ones_like(t_bc)
    x_right = torch.ones_like(t_bc)
    u_left = model(x_left, t_bc)
    u_right = model(x_right, t_bc)
    loss_bc = mse(u_left, torch.zeros_like(u_left)) + mse(
        u_right, torch.zeros_like(u_right)
    )

    loss_total = loss_pde + loss_ic + loss_bc
    return loss_total, loss_pde, loss_ic, loss_bc


def train(args: argparse.Namespace) -> tuple[BurgersPINN, dict[str, list[float]]]:
    """Optimize the PINN and return the model and loss history."""
    device = choose_device(args.device)
    model = BurgersPINN(args.hidden_width, args.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Independent streams keep IC/BC samples unchanged when only N_f changes.
    generators = (
        torch.Generator(device=device).manual_seed(args.seed + 101),
        torch.Generator(device=device).manual_seed(args.seed + 202),
        torch.Generator(device=device).manual_seed(args.seed + 303),
    )
    history: dict[str, list[float]] = {
        "total": [], "pde": [], "ic": [], "bc": []
    }

    print(f"Training on {device} with nu={args.nu:.8f}")
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss_total, loss_pde, loss_ic, loss_bc = compute_losses(
            model, args.n_f, args.n_ic, args.n_bc, args.nu, device, generators
        )
        loss_total.backward()
        optimizer.step()

        history["total"].append(loss_total.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:5d}/{args.epochs} | total {loss_total.item():.3e} | "
                f"PDE {loss_pde.item():.3e} | IC {loss_ic.item():.3e} | "
                f"BC {loss_bc.item():.3e}"
            )
    return model, history


def save_loss_history(history: dict[str, list[float]], output_dir: Path) -> None:
    """Save numeric losses and their log-scale plot."""
    with (output_dir / "loss_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss_total", "loss_pde", "loss_ic", "loss_bc"])
        for index, values in enumerate(
            zip(history["total"], history["pde"], history["ic"], history["bc"]),
            start=1,
        ):
            writer.writerow([index, *values])

    epochs = np.arange(1, len(history["total"]) + 1)
    plt.figure(figsize=(8, 5))
    for name, label in (
        ("total", "Total"),
        ("pde", "PDE residual"),
        ("ic", "Initial condition"),
        ("bc", "Boundary condition"),
    ):
        plt.semilogy(epochs, history[name], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared loss")
    plt.title("Burgers PINN training losses")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close()


@torch.no_grad()
def save_solution_plot(
    model: BurgersPINN, device: torch.device, output_dir: Path
) -> None:
    """Visualize u(x,t) as a heat map and as several time slices."""
    model.eval()
    x_values = np.linspace(-1.0, 1.0, 401)
    t_values = np.linspace(0.0, 1.0, 201)
    x_grid, t_grid = np.meshgrid(x_values, t_values)
    x_tensor = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    u_grid = model(x_tensor, t_tensor).cpu().numpy().reshape(t_grid.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    image = axes[0].imshow(
        u_grid,
        extent=[-1.0, 1.0, 0.0, 1.0],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title("Predicted u(x, t)")
    fig.colorbar(image, ax=axes[0], label="u")

    for time_value in (0.0, 0.25, 0.5, 0.75, 1.0):
        row = int(round(time_value * (len(t_values) - 1)))
        axes[1].plot(x_values, u_grid[row], label=f"t = {time_value:.2f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x, t)")
    axes[1].set_title("Solution at different times")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "solution.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--n-f", type=int, default=2000, help="Interior points per epoch")
    parser.add_argument("--n-ic", type=int, default=256, help="Initial points per epoch")
    parser.add_argument("--n-bc", type=int, default=256, help="Boundary times per epoch")
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--nu", type=float, default=0.01 / math.pi)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.n_f, args.n_ic, args.n_bc) < 1:
        raise ValueError("epochs and all sample counts must be positive")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, history = train(args)
    device = next(model.parameters()).device
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_width": args.hidden_width,
            "hidden_layers": args.hidden_layers,
            "nu": args.nu,
        },
        args.output_dir / "burgers_pinn.pt",
    )
    save_loss_history(history, args.output_dir)
    save_solution_plot(model, device, args.output_dir)
    print(f"Saved model, losses, and plots to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
