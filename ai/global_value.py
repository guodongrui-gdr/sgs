"""
Global Value Function for Independent Multi-Agent Training

A single shared critic that takes the full game state and outputs value estimates.
All agents share this value function during training.
"""

from typing import Optional

import torch
import torch.nn as nn


class GlobalValueConfig:
    """Configuration for GlobalValueFunction."""

    def __init__(
        self,
        global_state_dim: int = 13350,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        self.global_state_dim = global_state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout


class GlobalValueFunction(nn.Module):
    """
    Global value function using full game state.

    All agents share this critic during independent training.
    Input: global_state (batch, global_state_dim)
    Output: value estimate (batch,)
    """

    def __init__(self, config: Optional[GlobalValueConfig] = None):
        super().__init__()
        self.config = config or GlobalValueConfig()
        c = self.config

        layers = []
        in_dim = c.global_state_dim

        for i in range(c.num_layers):
            layers.append(nn.Linear(in_dim, c.hidden_dim))
            if c.activation == "relu":
                layers.append(nn.ReLU())
            elif c.activation == "tanh":
                layers.append(nn.Tanh())
            if c.dropout > 0:
                layers.append(nn.Dropout(c.dropout))
            in_dim = c.hidden_dim

        layers.append(nn.Linear(c.hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate from global state.

        Args:
            global_state: (batch, global_state_dim)

        Returns:
            value: (batch,)
        """
        return self.network(global_state).squeeze(-1)

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(global_state)
