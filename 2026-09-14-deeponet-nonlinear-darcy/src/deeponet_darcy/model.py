"""Standard branch/trunk DeepONet."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported activation: {name}") from exc


def _mlp(
    input_dim: int,
    hidden: Sequence[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    act = _activation(activation)
    dimensions = [input_dim, *hidden, output_dim]
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dimensions) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """G(f)(x) = sum_k branch_k(f) trunk_k(x) + bias."""

    def __init__(
        self,
        n_sensors: int,
        branch_hidden: Sequence[int],
        trunk_hidden: Sequence[int],
        latent_dim: int,
        activation: str = "gelu",
        output_bias: bool = False,
    ) -> None:
        super().__init__()
        self.branch = _mlp(n_sensors, branch_hidden, latent_dim, activation)
        self.trunk = _mlp(1, trunk_hidden, latent_dim, activation)
        self.output_bias = nn.Parameter(torch.zeros(())) if output_bias else None

    def forward(self, sources: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        if query_x.ndim == 1:
            query_x = query_x[:, None]
        branch_features = self.branch(sources)
        trunk_features = self.trunk(query_x)
        output = torch.einsum("bp,qp->bq", branch_features, trunk_features)
        return output + self.output_bias if self.output_bias is not None else output
