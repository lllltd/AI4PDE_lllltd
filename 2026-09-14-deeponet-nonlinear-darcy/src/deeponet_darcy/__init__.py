"""Cornell-style DeepONet example for a 1D nonlinear Darcy equation."""

from .darcy import SolverInfo, nonlinear_residual, solve_nonlinear_darcy
from .model import DeepONet

__all__ = ["DeepONet", "SolverInfo", "nonlinear_residual", "solve_nonlinear_darcy"]
__version__ = "0.1.0"
