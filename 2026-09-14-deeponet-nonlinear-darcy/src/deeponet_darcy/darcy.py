"""Finite-volume-style finite differences for the 1D nonlinear Darcy PDE."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class SolverInfo:
    converged: bool
    iterations: int
    relative_update: float
    relative_residual: float

    def to_dict(self) -> dict[str, bool | int | float]:
        return asdict(self)


def permeability(u: np.ndarray, kappa0: float = 0.2) -> np.ndarray:
    """Cornell tutorial permeability: kappa(u) = kappa0 + u**2."""
    return kappa0 + np.asarray(u, dtype=np.float64) ** 2


def assemble_operator(
    u: np.ndarray,
    h: float,
    kappa0: float = 0.2,
    face_scheme: str = "cornell_left",
) -> csc_matrix:
    """Assemble the conservative interior operator.

    ``cornell_left`` reproduces the tutorial's indexing: each face uses the
    permeability at its left node. ``arithmetic`` is a standard second-order
    alternative and is intentionally opt-in.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or u.size < 3:
        raise ValueError("u must be a one-dimensional array with at least 3 grid points")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    if not np.isfinite(kappa0) or kappa0 <= 0:
        raise ValueError("kappa0 must be positive and finite")

    kappa = permeability(u, kappa0)
    if face_scheme == "cornell_left":
        face_kappa = kappa[:-1]
    elif face_scheme == "arithmetic":
        face_kappa = 0.5 * (kappa[:-1] + kappa[1:])
    else:
        raise ValueError("face_scheme must be 'cornell_left' or 'arithmetic'")
    diagonal = (face_kappa[:-1] + face_kappa[1:]) / h**2
    off_diagonal = -face_kappa[1:-1] / h**2
    return diags(
        (off_diagonal, diagonal, off_diagonal),
        offsets=(-1, 0, 1),
        shape=(u.size - 2, u.size - 2),
        format="csc",
    )


def nonlinear_residual(
    u: np.ndarray,
    f: np.ndarray,
    x: np.ndarray,
    kappa0: float = 0.2,
    face_scheme: str = "cornell_left",
) -> np.ndarray:
    """Return the interior residual for -d/dx(kappa(u) du/dx) = f."""
    u, f, x = _validated_arrays(u, f, x)
    h = float(x[1] - x[0])
    kappa = permeability(u, kappa0)
    if face_scheme == "cornell_left":
        face_kappa = kappa[:-1]
    elif face_scheme == "arithmetic":
        face_kappa = 0.5 * (kappa[:-1] + kappa[1:])
    else:
        raise ValueError("face_scheme must be 'cornell_left' or 'arithmetic'")
    flux = -face_kappa * np.diff(u) / h
    return np.asarray(np.diff(flux) / h - f[1:-1], dtype=np.float64)


def solve_nonlinear_darcy(
    f: np.ndarray,
    x: np.ndarray,
    *,
    kappa0: float = 0.2,
    max_iter: int = 100,
    relaxation: float = 0.5,
    update_tol: float = 1e-9,
    residual_tol: float = 1e-8,
    face_scheme: str = "cornell_left",
    stop_on_convergence: bool = True,
) -> tuple[np.ndarray, SolverInfo]:
    """Solve the nonlinear boundary value problem by relaxed Picard iteration.

    The boundary conditions are u(0)=u(1)=0.  In each iteration kappa is
    frozen at the previous iterate, the sparse tridiagonal system is solved,
    and the result is under-relaxed.  Convergence requires both the relative
    iterate change and the true nonlinear residual to pass their tolerances.
    """
    f = np.asarray(f, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _validated_arrays(np.zeros_like(f), f, x)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not 0 < relaxation <= 1:
        raise ValueError("relaxation must be in (0, 1]")
    if update_tol < 0 or residual_tol < 0:
        raise ValueError("convergence tolerances must be nonnegative")

    h = float(x[1] - x[0])
    u = np.zeros_like(f)
    relative_update = np.inf
    relative_residual = np.inf
    f_scale = max(float(np.linalg.norm(f[1:-1])), 1.0)

    for iteration in range(1, max_iter + 1):
        matrix = assemble_operator(u, h, kappa0, face_scheme)
        candidate = np.zeros_like(u)
        candidate[1:-1] = spsolve(matrix, f[1:-1])
        if not np.all(np.isfinite(candidate)):
            raise FloatingPointError("sparse solve returned non-finite values")

        u_next = (1.0 - relaxation) * u + relaxation * candidate
        u_next[[0, -1]] = 0.0
        relative_update = float(
            np.linalg.norm(u_next - u) / max(np.linalg.norm(u_next), 1.0)
        )
        residual = nonlinear_residual(u_next, f, x, kappa0, face_scheme)
        relative_residual = float(np.linalg.norm(residual) / f_scale)
        u = u_next
        converged = relative_update <= update_tol and relative_residual <= residual_tol
        if converged and stop_on_convergence:
            return u, SolverInfo(True, iteration, relative_update, relative_residual)

    converged = relative_update <= update_tol and relative_residual <= residual_tol
    return u, SolverInfo(converged, max_iter, relative_update, relative_residual)


def _validated_arrays(
    u: np.ndarray, f: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if u.ndim != 1 or f.ndim != 1 or x.ndim != 1:
        raise ValueError("u, f, and x must all be one-dimensional")
    if not (u.shape == f.shape == x.shape) or x.size < 3:
        raise ValueError("u, f, and x must have the same length of at least 3")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(x)):
        raise ValueError("u, f, and x must contain only finite values")
    dx = np.diff(x)
    if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-10, atol=1e-12):
        raise ValueError("x must be a strictly increasing uniform grid")
    return u, f, x
