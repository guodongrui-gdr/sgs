"""
Agent Replay Buffer for Independent Multi-Agent Training

Each agent has its own replay buffer storing transitions.
Supports on-policy (PPO) and off-policy sampling.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class AgentReplayBuffer:
    """
    Replay buffer for a single agent.

    Stores transitions: (global_state, local_obs, action, log_prob, reward, done, mask)
    """

    def __init__(
        self,
        capacity: int = 10000,
        global_state_dim: int = 13350,
        local_state_dim: int = 2670,
        action_dim: int = 45,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.global_state_dim = global_state_dim
        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.device = device

        # Storage
        self.global_states = []
        self.local_observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.masks = []
        self.values = []

        self.position = 0
        self.size = 0

    def add(
        self,
        global_state: np.ndarray,
        local_obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        mask: np.ndarray,
        value: float = 0.0,
    ) -> None:
        """Add a transition to the buffer."""
        if self.size < self.capacity:
            self.global_states.append(global_state)
            self.local_observations.append(local_obs)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.dones.append(done)
            self.masks.append(mask)
            self.values.append(value)
            self.size += 1
        else:
            # Overwrite oldest
            idx = self.position % self.capacity
            self.global_states[idx] = global_state
            self.local_observations[idx] = local_obs
            self.actions[idx] = action
            self.log_probs[idx] = log_prob
            self.rewards[idx] = reward
            self.dones[idx] = done
            self.masks[idx] = mask
            self.values[idx] = value

        self.position += 1

    def sample(
        self, batch_size: int, recent_only: bool = False
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            recent_only: If True, only sample from the most recent data (on-policy)

        Returns:
            Dict of tensors, or None if buffer is empty
        """
        if self.size == 0:
            return None

        if recent_only:
            # Sample from the end (most recent on-policy data)
            end = self.size
            start = max(0, end - batch_size)
            indices = np.arange(start, end)
            if len(indices) < batch_size:
                return None
        else:
            # Random sampling (off-policy)
            indices = np.random.choice(
                self.size, min(batch_size, self.size), replace=False
            )

        batch = {
            "global_states": torch.from_numpy(
                np.stack([self.global_states[i] for i in indices])
            )
            .float()
            .to(self.device),
            "local_observations": torch.from_numpy(
                np.stack([self.local_observations[i] for i in indices])
            )
            .float()
            .to(self.device),
            "actions": torch.from_numpy(np.array([self.actions[i] for i in indices]))
            .long()
            .to(self.device),
            "old_log_probs": torch.from_numpy(
                np.array([self.log_probs[i] for i in indices])
            )
            .float()
            .to(self.device),
            "rewards": torch.from_numpy(np.array([self.rewards[i] for i in indices]))
            .float()
            .to(self.device),
            "dones": torch.from_numpy(np.array([self.dones[i] for i in indices]))
            .float()
            .to(self.device),
            "masks": torch.from_numpy(np.stack([self.masks[i] for i in indices]))
            .float()
            .to(self.device),
            "values": torch.from_numpy(np.array([self.values[i] for i in indices]))
            .float()
            .to(self.device),
        }

        return batch

    def get_recent(self, n: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get the most recent n transitions (for on-policy PPO update)."""
        if self.size == 0:
            return None

        start = max(0, self.size - n)
        indices = np.arange(start, self.size)

        batch = {
            "global_states": torch.from_numpy(
                np.stack([self.global_states[i] for i in indices])
            )
            .float()
            .to(self.device),
            "local_observations": torch.from_numpy(
                np.stack([self.local_observations[i] for i in indices])
            )
            .float()
            .to(self.device),
            "actions": torch.from_numpy(np.array([self.actions[i] for i in indices]))
            .long()
            .to(self.device),
            "old_log_probs": torch.from_numpy(
                np.array([self.log_probs[i] for i in indices])
            )
            .float()
            .to(self.device),
            "rewards": torch.from_numpy(np.array([self.rewards[i] for i in indices]))
            .float()
            .to(self.device),
            "dones": torch.from_numpy(np.array([self.dones[i] for i in indices]))
            .float()
            .to(self.device),
            "masks": torch.from_numpy(np.stack([self.masks[i] for i in indices]))
            .float()
            .to(self.device),
            "values": torch.from_numpy(np.array([self.values[i] for i in indices]))
            .float()
            .to(self.device),
        }

        return batch

    def clear(self) -> None:
        """Clear all transitions."""
        self.global_states.clear()
        self.local_observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.masks.clear()
        self.values.clear()
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size


class MultiAgentReplayBuffer:
    """
    Collection of per-agent replay buffers.
    """

    def __init__(
        self,
        num_agents: int = 5,
        capacity_per_agent: int = 10000,
        global_state_dim: int = 13350,
        local_state_dim: int = 2670,
        action_dim: int = 45,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.buffers = [
            AgentReplayBuffer(
                capacity=capacity_per_agent,
                global_state_dim=global_state_dim,
                local_state_dim=local_state_dim,
                action_dim=action_dim,
                device=device,
            )
            for _ in range(num_agents)
        ]

    def add(
        self,
        agent_idx: int,
        global_state: np.ndarray,
        local_obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        mask: np.ndarray,
        value: float = 0.0,
    ) -> None:
        """Add transition for a specific agent."""
        self.buffers[agent_idx].add(
            global_state, local_obs, action, log_prob, reward, done, mask, value
        )

    def sample(
        self, agent_idx: int, batch_size: int, recent_only: bool = False
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample from a specific agent's buffer."""
        return self.buffers[agent_idx].sample(batch_size, recent_only)

    def get_recent(self, agent_idx: int, n: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get recent transitions for a specific agent."""
        return self.buffers[agent_idx].get_recent(n)

    def clear(self, agent_idx: Optional[int] = None) -> None:
        """Clear buffer(s)."""
        if agent_idx is not None:
            self.buffers[agent_idx].clear()
        else:
            for buf in self.buffers:
                buf.clear()

    def __len__(self) -> int:
        return sum(len(buf) for buf in self.buffers)
