"""
MAPPO Actor - Decentralized Policy Network

Each agent has its own actor that only sees local observations during execution.
During training, actors share the centralized critic for value estimation.

Architecture:
- Input: local_obs (batch, 2670) + action_mask (batch, action_dim)
- local_encoder: MLP (2670 -> 256)
- action_head: Linear (256 -> action_dim)
- Output: action_logits (batch, action_dim) with mask applied

Key Features:
- Decentralized execution (local observation only)
- Action masking for valid action selection
- Policy entropy monitoring (prevent collapse)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class MAPPOActorConfig:
    """Configuration for MAPPO Actor"""

    local_state_dim: int = 2670
    action_dim: int = 20

    encoder_hidden_dim: int = 256
    encoder_output_dim: int = 256

    activation: str = "relu"


class MAPPOActor(nn.Module):
    """
    MAPPO Actor - Decentralized policy network

    Each agent uses this actor for action selection based on local observation.
    Action masks are applied to ensure only valid actions are selected.
    """

    def __init__(self, config: Optional[MAPPOActorConfig] = None):
        super().__init__()

        self.config = config or MAPPOActorConfig()
        c = self.config

        self.local_encoder = nn.Sequential(
            nn.Linear(c.local_state_dim, c.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(c.encoder_hidden_dim, c.encoder_output_dim),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(c.encoder_output_dim, c.action_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        local_obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoding = self.local_encoder(local_obs)
        logits = self.action_head(encoding)

        if action_mask is not None:
            logits = logits.clone()
            invalid_mask = action_mask == 0
            logits[invalid_mask] = float("-inf")

            if (action_mask.sum(dim=-1) == 0).any():
                valid_mask_sum = action_mask.sum(dim=-1, keepdim=True)
                fallback_mask = (valid_mask_sum == 0).float()
                logits = logits + fallback_mask * 100.0
                logits[0] = logits[0] + 50.0

        return logits

    def get_action(
        self,
        local_obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action with log probability

        Args:
            local_obs: (batch, local_state_dim)
            action_mask: (batch, action_dim) - 1 for valid, 0 for invalid
            deterministic: If True, select argmax; else sample

        Returns:
            action: (batch,) - selected action indices
            log_prob: (batch,) - log probability of selected action
            entropy: (batch,) - entropy of action distribution
        """
        logits = self.forward(local_obs, action_mask)

        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()

        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations (used during PPO update)

        Args:
            local_obs: (batch, local_state_dim)
            actions: (batch,) - actions to evaluate
            action_mask: (batch, action_dim)

        Returns:
            log_prob: (batch,) - log probability of actions
            entropy: (batch,) - entropy of distribution
        """
        logits = self.forward(local_obs, action_mask)

        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]

        dist = Categorical(probs)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy

    def get_entropy(
        self, local_obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.forward(local_obs, action_mask)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]

        dist = Categorical(probs)
        return dist.entropy()
