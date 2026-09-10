"""Shared constants and Cole-Hopf reference solution for Burgers experiments."""

from __future__ import annotations

import math

import numpy as np


NU = 0.01 / math.pi


def exact_burgers(
    x: np.ndarray, t: float, nu: float = NU, quadrature_order: int = 120
) -> np.ndarray:
    """Evaluate the viscous Burgers solution using Gauss-Hermite quadrature."""
    if t == 0.0:
        return -np.sin(np.pi * x)
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    shifted = x[:, None] - math.sqrt(4.0 * nu * t) * nodes[None, :]
    exponent = -np.cos(np.pi * shifted) / (2.0 * np.pi * nu)
    exponent -= exponent.max(axis=1, keepdims=True)
    weighted = weights[None, :] * np.exp(exponent)
    return -np.sum(weighted * np.sin(np.pi * shifted), axis=1) / np.sum(
        weighted, axis=1
    )
