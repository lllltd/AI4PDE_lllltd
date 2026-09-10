"""PyTorch port of Wang-Teng-Perdikaris gradient annealing experiments.

The Helmholtz setup and adaptive-weight update follow the authors' TensorFlow 1
code: https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs
An additional Burgers transfer uses the same algorithm so it can be compared to
this project's earlier Burgers teaching experiment.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from burgers_reference import NU, exact_burgers


class PaperMLP(nn.Module):
    def __init__(self, widths: list[int], mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
        )
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        values = (points - self.mean) / self.std
        for layer in self.layers[:-1]:
            values = torch.tanh(layer(values))
        return self.layers[-1](values)


def derivatives_2d(model: nn.Module, points: torch.Tensor) -> tuple[torch.Tensor, ...]:
    output = model(points)
    gradient = torch.autograd.grad(output, points, torch.ones_like(output), create_graph=True)[0]
    first, second = gradient[:, :1], gradient[:, 1:2]
    first2 = torch.autograd.grad(first, points, torch.ones_like(first), create_graph=True)[0][:, :1]
    second2 = torch.autograd.grad(second, points, torch.ones_like(second), create_graph=True)[0][:, 1:2]
    return output, first, second, first2, second2


def layer_gradient_statistics(
    loss: torch.Tensor, model: PaperMLP, maximum: bool
) -> torch.Tensor:
    # The official code uses weight matrices only and averages layer-wise means.
    weights = tuple(layer.weight for layer in model.layers)
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=False)
    statistics = [gradient.detach().abs().max() if maximum else gradient.detach().abs().mean() for gradient in gradients]
    return torch.stack(statistics).max() if maximum else torch.stack(statistics).mean()


def sample_helmholtz(batch: int, generator: torch.Generator) -> tuple[torch.Tensor, list[torch.Tensor]]:
    residual = -1.0 + 2.0 * torch.rand(batch, 2, generator=generator)
    residual.requires_grad_(True)
    values = -1.0 + 2.0 * torch.rand(batch, generator=generator)
    boundaries = [
        torch.stack((values, -torch.ones_like(values)), dim=1),
        torch.stack((torch.ones_like(values), values), dim=1),
        torch.stack((values, torch.ones_like(values)), dim=1),
        torch.stack((-torch.ones_like(values), values), dim=1),
    ]
    return residual, boundaries


def helmholtz_exact(points: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * points[:, :1]) * torch.sin(4.0 * math.pi * points[:, 1:2])


def helmholtz_losses(model: PaperMLP, batch: int, generator: torch.Generator):
    residual_points, boundaries = sample_helmholtz(batch, generator)
    output, _, _, output_xx, output_yy = derivatives_2d(model, residual_points)
    target = -16.0 * math.pi**2 * helmholtz_exact(residual_points) - math.pi**2 * helmholtz_exact(residual_points) + helmholtz_exact(residual_points)
    loss_pde = torch.mean((output_xx + output_yy + output - target).square())
    loss_bc = sum(torch.mean((model(points) - helmholtz_exact(points)).square()) for points in boundaries)
    return loss_pde, loss_bc, None


def burgers_losses(model: PaperMLP, batch: int, generator: torch.Generator):
    points = torch.rand(batch, 2, generator=generator)
    points[:, 0] = -1.0 + 2.0 * points[:, 0]
    points.requires_grad_(True)
    output, output_x, output_t, output_xx, _ = derivatives_2d(model, points)
    loss_pde = torch.mean((output_t + output * output_x - NU * output_xx).square())
    x_ic = -1.0 + 2.0 * torch.rand(batch, 1, generator=generator)
    ic_points = torch.cat((x_ic, torch.zeros_like(x_ic)), dim=1)
    loss_ic = torch.mean((model(ic_points) + torch.sin(math.pi * x_ic)).square())
    times = torch.rand(batch, 1, generator=generator)
    left = torch.cat((-torch.ones_like(times), times), dim=1)
    right = torch.cat((torch.ones_like(times), times), dim=1)
    loss_bc = torch.mean(model(left).square()) + torch.mean(model(right).square())
    return loss_pde, loss_bc, loss_ic


def train(problem: str, adaptive: bool, iterations: int, output_dir: Path, reuse_existing: bool = False) -> dict[str, float]:
    torch.manual_seed(42)
    np.random.seed(1234)
    # Official implementation estimates normalization statistics with 100000 samples.
    normalization_sample = -1.0 + 2.0 * np.random.rand(100000, 2)
    if problem == "burgers":
        normalization_sample[:, 1] = 0.5 * (normalization_sample[:, 1] + 1.0)
    model = PaperMLP([2, 50, 50, 50, 1], normalization_sample.mean(0), normalization_sample.std(0))
    mode = "M2_adaptive" if adaptive else "M1_fixed"
    run_dir = output_dir / problem / mode
    model_path = run_dir / "model.pt"
    history_path = run_dir / "history.csv"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9 ** (1.0 / 1000.0))
    generator = torch.Generator().manual_seed(2024)
    lambda_bc, lambda_ic = 1.0, 1.0
    history: list[dict[str, float]] = []
    loss_function = helmholtz_losses if problem == "helmholtz" else burgers_losses

    start_iteration = 0
    if reuse_existing and model_path.exists() and history_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        history = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(history_path.open(encoding="utf-8"))
        ]
        lambda_bc = history[-1]["lambda_bc"]
        lambda_ic = history[-1]["lambda_ic"]
        start_iteration = iterations
        print(f"Reusing {problem} {mode} completed model")

    for iteration in range(start_iteration, iterations):
        optimizer.zero_grad(set_to_none=True)
        loss_pde, loss_bc, loss_ic = loss_function(model, 128, generator)
        objective = loss_pde + lambda_bc * loss_bc
        if loss_ic is not None:
            objective = objective + lambda_ic * loss_ic
        objective.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            loss_pde, loss_bc, loss_ic = loss_function(model, 128, generator)
            if adaptive:
                max_pde = layer_gradient_statistics(loss_pde, model, maximum=True)
                # Crucial official detail: denominator differentiates the CURRENT weighted loss.
                mean_bc = layer_gradient_statistics(lambda_bc * loss_bc, model, maximum=False)
                target_bc = float(max_pde / torch.clamp(mean_bc, min=1e-12))
                lambda_bc = 0.9 * lambda_bc + 0.1 * target_bc
                if loss_ic is not None:
                    mean_ic = layer_gradient_statistics(lambda_ic * loss_ic, model, maximum=False)
                    target_ic = float(max_pde / torch.clamp(mean_ic, min=1e-12))
                    lambda_ic = 0.9 * lambda_ic + 0.1 * target_ic
            history.append(
                {
                    "iteration": iteration,
                    "pde": loss_pde.detach().item(),
                    "bc": loss_bc.detach().item(),
                    "ic": 0.0 if loss_ic is None else loss_ic.detach().item(),
                    "lambda_bc": lambda_bc,
                    "lambda_ic": lambda_ic,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        if iteration == 0 or (iteration + 1) % 5000 == 0:
            latest = history[-1]
            print(
                f"{problem} {'M2' if adaptive else 'M1'} {iteration+1}/{iterations}: "
                f"PDE={latest['pde']:.3e}, BC={latest['bc']:.3e}, "
                f"IC={latest['ic']:.3e}, weights=({latest['lambda_bc']:.3g},{latest['lambda_ic']:.3g})"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    if problem == "helmholtz":
        axis = np.linspace(-1.0, 1.0, 100)
        xx, yy = np.meshgrid(axis, axis)
        points_np = np.column_stack((xx.ravel(), yy.ravel()))
        exact = np.sin(math.pi * points_np[:, 0]) * np.sin(4.0 * math.pi * points_np[:, 1])
    else:
        x = np.linspace(-1.0, 1.0, 401)
        times = np.linspace(0.0, 1.0, 101)
        xx, yy = np.meshgrid(x, times)
        points_np = np.column_stack((xx.ravel(), yy.ravel()))
        exact = np.concatenate([exact_burgers(x, float(time)) for time in times])
    with torch.no_grad():
        prediction = model(torch.tensor(points_np, dtype=torch.float32)).numpy().ravel()
    relative_l2 = float(np.linalg.norm(prediction - exact) / np.linalg.norm(exact))
    return {
        "problem": problem,
        "mode": mode,
        "relative_l2": relative_l2,
        "final_lambda_bc": lambda_bc,
        "final_lambda_ic": lambda_ic,
        "model": model,
        "history": history,
        "prediction": prediction,
        "exact": exact,
        "shape": xx.shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=40001)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_reproductions/adaptive_weight"))
    parser.add_argument("--problem", choices=("helmholtz", "burgers", "both"), default="both")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problems = ("helmholtz", "burgers") if args.problem == "both" else (args.problem,)
    results = [train(problem, adaptive, args.iterations, args.output_dir, args.reuse_existing) for problem in problems for adaptive in (False, True)]
    with (args.output_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["problem", "mode", "relative_l2", "final_lambda_bc", "final_lambda_ic"])
        for result in results:
            writer.writerow([result[key] for key in ("problem", "mode", "relative_l2", "final_lambda_bc", "final_lambda_ic")])

    fig, axes = plt.subplots(len(problems), 3, figsize=(13, 4.1 * len(problems)), squeeze=False)
    for row, problem in enumerate(problems):
        fixed = next(result for result in results if result["problem"] == problem and result["mode"] == "M1_fixed")
        adaptive = next(result for result in results if result["problem"] == problem and result["mode"] == "M2_adaptive")
        if problem == "helmholtz":
            for column, result in enumerate((fixed, adaptive)):
                error = np.abs(result["prediction"] - result["exact"]).reshape(result["shape"])
                image = axes[row, column].imshow(error, origin="lower", extent=[-1, 1, -1, 1], aspect="auto", cmap="magma")
                axes[row, column].set_title(f"{problem} {result['mode']}\nrelative L2={result['relative_l2']:.3e}")
                fig.colorbar(image, ax=axes[row, column])
        else:
            x = np.linspace(-1.0, 1.0, fixed["shape"][1])
            for column, result in enumerate((fixed, adaptive)):
                prediction = result["prediction"].reshape(result["shape"])
                exact = result["exact"].reshape(result["shape"])
                for time_value in (0.5, 1.0):
                    index = int(round(time_value * (result["shape"][0] - 1)))
                    axes[row, column].plot(x, exact[index], "k--", linewidth=2)
                    axes[row, column].plot(x, prediction[index], label=f"t={time_value:g}")
                axes[row, column].set_title(f"{problem} {result['mode']}\nrelative L2={result['relative_l2']:.3e}")
                axes[row, column].legend()
                axes[row, column].grid(alpha=0.25)
        history = adaptive["history"]
        axes[row, 2].semilogy([item["iteration"] for item in history], [item["lambda_bc"] for item in history], label=r"$\lambda_{BC}$")
        if problem == "burgers":
            axes[row, 2].semilogy([item["iteration"] for item in history], [item["lambda_ic"] for item in history], label=r"$\lambda_{IC}$")
        axes[row, 2].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[row, 2].set_title(f"{problem}: official gradient annealing")
        axes[row, 2].set_xlabel("Iteration")
        axes[row, 2].set_ylabel("Adaptive weight")
        axes[row, 2].grid(alpha=0.25)
        axes[row, 2].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "adaptive_weight_result.png", dpi=180)
    plt.close(fig)

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["problem", "mode", "relative_l2", "final_lambda_bc", "final_lambda_ic"])
        for result in results:
            writer.writerow([result[key] for key in ("problem", "mode", "relative_l2", "final_lambda_bc", "final_lambda_ic")])

    fig, axes = plt.subplots(len(problems), 3, figsize=(13, 4 * len(problems)), squeeze=False)
    for row, problem in enumerate(problems):
        pair = [result for result in results if result["problem"] == problem]
        for result in pair:
            label = result["mode"]
            history = result["history"]
            axes[row, 0].semilogy([x["iteration"] for x in history], [x["pde"] for x in history], label=label)
            axes[row, 1].semilogy([x["iteration"] for x in history], [x["lambda_bc"] for x in history], label=f"{label} BC")
            if problem == "burgers":
                axes[row, 1].semilogy([x["iteration"] for x in history], [x["lambda_ic"] for x in history], linestyle="--", label=f"{label} IC")
            error = np.abs(result["prediction"] - result["exact"]).reshape(result["shape"])
            if result["mode"] == "M2_adaptive":
                image = axes[row, 2].imshow(error, origin="lower", aspect="auto", cmap="magma")
                fig.colorbar(image, ax=axes[row, 2])
                axes[row, 2].set_title(f"{problem} M2 error, L2={result['relative_l2']:.2e}")
        axes[row, 0].set_title(f"{problem}: PDE loss")
        axes[row, 1].set_title(f"{problem}: adaptive weights")
        axes[row, 0].legend()
        axes[row, 1].legend(fontsize=8)
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "adaptive_weight_results.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
