"""
Centralized Critic for MAPPO

The centralized critic sees the global state (all agents' observations concatenated)
and all agents' actions. This enables better value estimation by considering
the joint action space during training.

Architecture:
- Input: global_state (batch, 2670*5) + joint_actions (batch, 5)
- global_state_encoder: MLP (2670*5 -> 256 -> 256)
- action_encoder: Embedding (20 actions -> 64 dim, per agent)
- Output: value (batch,)

Key Features:
- Centralized value estimation (sees global state + joint actions)
- Shared across all agents (parameter sharing)
- Used during training only (actors use local obs during execution)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class CentralizedCriticConfig:
    """Configuration for Centralized Critic"""

    # State dimensions
    local_state_dim: int = 2670  # Per-agent observation dim
    num_agents: int = 5  # Number of agents (主公, 忠臣, 反贼×2, 内奸)
    global_state_dim: int = 2670 * 5  # Concatenated observations

    # Action dimensions
    action_dim: int = 20  # Number of discrete actions per agent
    action_embed_dim: int = 64  # Embedding dimension for actions

    # Network dimensions
    encoder_hidden_dim: int = 256
    encoder_output_dim: int = 256
    critic_hidden_dim: int = 128

    # Activation
    activation: str = "relu"


class CentralizedCritic(nn.Module):
    """
    Centralized Critic for MAPPO

    Input: global_state (batch, global_state_dim), joint_actions (batch, num_agents)
    Output: value (batch,)

    Architecture:
    1. Global state encoder: MLP (global_state_dim -> encoder_output_dim)
    2. Action encoder: Embedding + MLP for each agent's action
    3. Value head: MLP combining encoded state + actions -> value
    """

    def __init__(self, config: Optional[CentralizedCriticConfig] = None):
        super().__init__()

        self.config = config or CentralizedCriticConfig()
        c = self.config

        # Global state encoder: MLP (global_state_dim -> encoder_output_dim)
        self.global_state_encoder = nn.Sequential(
            nn.Linear(c.global_state_dim, c.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(c.encoder_hidden_dim, c.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(c.encoder_hidden_dim, c.encoder_output_dim),
        )

        # Action embedding: Embedding layer for discrete actions
        self.action_embedding = nn.Embedding(
            num_embeddings=c.action_dim, embedding_dim=c.action_embed_dim
        )

        # Joint action encoder: MLP to process concatenated action embeddings
        # Input: (batch, num_agents * action_embed_dim)
        joint_action_dim = c.num_agents * c.action_embed_dim
        self.joint_action_encoder = nn.Sequential(
            nn.Linear(joint_action_dim, c.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(c.critic_hidden_dim, c.critic_hidden_dim),
        )

        # Value head: combines state encoding + action encoding -> value
        combined_dim = c.encoder_output_dim + c.critic_hidden_dim
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, c.critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(c.critic_hidden_dim, 1),  # Single value output
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        global_state: torch.Tensor,
        joint_actions: torch.Tensor,
        is_alive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute value estimate

        Args:
            global_state: (batch, global_state_dim) - concatenated agent observations
            joint_actions: (batch, num_agents) - discrete action indices for each agent
            is_alive_mask: (batch, num_agents) - 1 for alive agents, 0 for dead agents

        Returns:
            value: (batch,) - centralized value estimate
        """
        batch_size = global_state.shape[0]

        # Encode global state
        state_encoding = self.global_state_encoder(
            global_state
        )  # (batch, encoder_output_dim)

        # Embed actions for each agent
        # joint_actions: (batch, num_agents) -> embed: (batch, num_agents, action_embed_dim)
        action_embeds = self.action_embedding(
            joint_actions
        )  # (batch, num_agents, action_embed_dim)

        # Apply death masking: zero out embeddings for dead agents
        if is_alive_mask is not None:
            # is_alive_mask: (batch, num_agents) -> expand to (batch, num_agents, action_embed_dim)
            alive_mask_expanded = is_alive_mask.unsqueeze(-1).expand_as(action_embeds)
            action_embeds = action_embeds * alive_mask_expanded

        # Flatten action embeddings
        action_embeds_flat = action_embeds.view(
            batch_size, -1
        )  # (batch, num_agents * action_embed_dim)

        # Encode joint actions
        action_encoding = self.joint_action_encoder(
            action_embeds_flat
        )  # (batch, critic_hidden_dim)

        # Combine state and action encodings
        combined = torch.cat(
            [state_encoding, action_encoding], dim=-1
        )  # (batch, combined_dim)

        # Compute value
        value = self.value_head(combined).squeeze(-1)  # (batch,)

        return value

    def get_value(
        self,
        global_state: torch.Tensor,
        joint_actions: torch.Tensor,
        is_alive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Alias for forward pass (for compatibility with SB3-style interface)

        Args:
            global_state: (batch, global_state_dim)
            joint_actions: (batch, num_agents)
            is_alive_mask: (batch, num_agents) - 1 for alive agents, 0 for dead agents

        Returns:
            value: (batch,)
        """
        return self.forward(global_state, joint_actions, is_alive_mask)

    def get_value_std(self, global_state: torch.Tensor) -> float:
        """
        Compute value estimate standard deviation (for monitoring)

        Args:
            global_state: (batch, global_state_dim)

        Returns:
            value_std: Standard deviation of value estimates across random actions
        """
        batch_size = global_state.shape[0]
        device = global_state.device

        # Sample random joint actions for variance estimation
        num_samples = 100
        random_actions = torch.randint(
            0,
            self.config.action_dim,
            (num_samples, batch_size, self.config.num_agents),
            device=device,
        )

        # Compute values for each action sample
        values = []
        with torch.no_grad():
            for i in range(num_samples):
                value = self.forward(global_state, random_actions[i])
                values.append(value)

        # Stack and compute std
        values_stack = torch.stack(values, dim=0)  # (num_samples, batch)
        value_std = values_stack.std().item()

        return value_std
