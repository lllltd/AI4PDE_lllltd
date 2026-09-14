"""Random source functions and nonlinear Darcy dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import multivariate_normal

from .darcy import solve_nonlinear_darcy


def sample_gaussian_random_fields(
    x: np.ndarray,
    n_samples: int,
    *,
    length_scale: float = 0.2,
    amplitude: float = 1.0,
    jitter: float = 1e-10,
    rng: np.random.Generator | None = None,
    random_state: np.random.RandomState | None = None,
    method: str = "cornell_scipy",
) -> np.ndarray:
    """Sample zero-mean squared-exponential Gaussian random fields."""
    x = np.asarray(x, dtype=np.float64)
    if n_samples < 1 or length_scale <= 0 or amplitude <= 0 or jitter < 0:
        raise ValueError("invalid Gaussian random field parameters")
    distances = x[:, None] - x[None, :]
    covariance = np.exp(-0.5 * (distances / length_scale) ** 2)
    covariance = amplitude**2 * covariance + jitter * np.eye(x.size)
    if method == "cornell_scipy":
        return np.asarray(
            multivariate_normal.rvs(
                mean=np.zeros(x.size),
                cov=covariance,
                size=n_samples,
                random_state=random_state,
            ),
            dtype=np.float64,
        ).reshape(n_samples, x.size)
    if method == "cholesky":
        if rng is None:
            raise ValueError("rng is required for the cholesky sampling method")
        cholesky = np.linalg.cholesky(covariance)
        return rng.standard_normal((n_samples, x.size)) @ cholesky.T
    raise ValueError("method must be 'cornell_scipy' or 'cholesky'")


def generate_dataset(config: dict[str, Any]) -> dict[str, np.ndarray | list[dict[str, Any]]]:
    """Generate train/validation/test sources and corresponding PDE solutions."""
    pde = config["pde"]
    data = config["data"]
    n_grid = int(pde["n_grid"])
    x = np.linspace(0.0, 1.0, n_grid, dtype=np.float64)
    counts = [int(data["n_train"]), int(data["n_val"]), int(data["n_test"])]
    n_total = sum(counts)
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    sources = sample_gaussian_random_fields(
        x,
        n_total,
        length_scale=float(data["length_scale"]),
        amplitude=float(data["source_amplitude"]),
        jitter=float(data.get("jitter", 1e-10)),
        rng=rng,
        random_state=np.random.RandomState(seed),
        method=str(data.get("sampler", "cornell_scipy")),
    )
    solutions = np.empty_like(sources)
    solver_info: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        solution, info = solve_nonlinear_darcy(
            source,
            x,
            kappa0=float(pde["kappa0"]),
            max_iter=int(pde["max_iter"]),
            relaxation=float(pde["relaxation"]),
            update_tol=float(pde["update_tol"]),
            residual_tol=float(pde["residual_tol"]),
            face_scheme=str(pde.get("face_scheme", "cornell_left")),
            stop_on_convergence=bool(pde.get("stop_on_convergence", True)),
        )
        if not info.converged:
            raise RuntimeError(
                f"Darcy solver did not converge for sample {index}: {info.to_dict()}"
            )
        solutions[index] = solution
        solver_info.append(info.to_dict())

    n_train, n_val, _ = counts
    i_val, i_test = n_train, n_train + n_val
    return {
        "x": x,
        "f_train": sources[:i_val],
        "u_train": solutions[:i_val],
        "f_val": sources[i_val:i_test],
        "u_val": solutions[i_val:i_test],
        "f_test": sources[i_test:],
        "u_test": solutions[i_test:],
        "solver_info": solver_info,
    }


def save_dataset(dataset: dict[str, Any], path: Path) -> None:
    arrays = {key: value for key, value in dataset.items() if key != "solver_info"}
    np.savez_compressed(path, **arrays)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}
