"""Neural network modules for BrainBlock agents."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DQNNetwork(nn.Module):
    """Simple MLP mapping flattened observations to Q-values."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if output_dim < 1:
            raise ValueError("output_dim must be positive")

        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            if hidden_dim < 1:
                raise ValueError("hidden dimensions must be positive")
            layers.extend([nn.Linear(previous_dim, int(hidden_dim)), nn.ReLU()])
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, int(output_dim)))
        self.model = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action."""

        return self.model(observations.float())


__all__ = ["DQNNetwork"]
