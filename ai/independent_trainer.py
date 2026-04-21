"""
Independent Multi-Agent Trainer

Each agent trains independently with:
- Its own policy network (MAPPOActor)
- Shared global value function
- Independent replay buffer
- Per-agent PPO updates
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

from ai.mappo.mappo_policy import MAPPOActor, MAPPOActorConfig
from ai.global_value import GlobalValueFunction, GlobalValueConfig
from ai.replay_buffer import MultiAgentReplayBuffer


class IndependentAgentConfig:
    """Configuration for independent multi-agent training."""

    def __init__(
        self,
        num_agents: int = 5,
        local_state_dim: int = 2670,
        action_dim: int = 20,
        global_state_dim: int = 13350,
        ppo_clip_range: float = 0.2,
        ppo_ent_coef: float = 0.01,
        ppo_vf_coef: float = 0.5,
        ppo_gamma: float = 0.99,
        ppo_gae_lambda: float = 0.95,
        learning_rate: float = 3e-4,
        lr_decay: float = 0.9995,
        max_grad_norm: float = 0.5,
        buffer_capacity: int = 10000,
        hidden_dim: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.ppo_clip_range = ppo_clip_range
        self.ppo_ent_coef = ppo_ent_coef
        self.ppo_vf_coef = ppo_vf_coef
        self.ppo_gamma = ppo_gamma
        self.ppo_gae_lambda = ppo_gae_lambda
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.max_grad_norm = max_grad_norm
        self.buffer_capacity = buffer_capacity
        self.hidden_dim = hidden_dim
        self.device = device


class IndependentAgentTrainer(nn.Module):
    """
    Trainer for independent multi-agent learning.

    Each agent has its own policy but shares a global value function.
    Data is stored per-agent in independent replay buffers.
    """

    def __init__(self, config: Optional[IndependentAgentConfig] = None):
        super().__init__()
        self.config = config or IndependentAgentConfig()
        c = self.config

        # Each agent has its own actor
        actor_config = MAPPOActorConfig(
            local_state_dim=c.local_state_dim,
            action_dim=c.action_dim,
            encoder_hidden_dim=c.hidden_dim,
            encoder_output_dim=c.hidden_dim,
        )
        self.actors = nn.ModuleList(
            [MAPPOActor(actor_config) for _ in range(c.num_agents)]
        )

        # Shared global value function
        value_config = GlobalValueConfig(
            global_state_dim=c.global_state_dim,
            hidden_dim=c.hidden_dim,
        )
        self.global_value_fn = GlobalValueFunction(value_config)

        # Per-agent optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=c.learning_rate) for actor in self.actors
        ]
        self.value_optimizer = optim.Adam(
            self.global_value_fn.parameters(), lr=c.learning_rate
        )

        # Per-agent replay buffers
        self.buffers = MultiAgentReplayBuffer(
            num_agents=c.num_agents,
            capacity_per_agent=c.buffer_capacity,
            global_state_dim=c.global_state_dim,
            local_state_dim=c.local_state_dim,
            action_dim=c.action_dim,
            device=c.device,
        )

        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self._update_count = 0

    def get_action(
        self,
        agent_idx: int,
        local_obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """
        Get action for a single agent.

        Args:
            agent_idx: Which agent to act
            local_obs: (local_state_dim,) numpy array
            mask: (action_dim,) numpy array
            deterministic: If True, use argmax

        Returns:
            action: int
            log_prob: float
        """
        obs_t = torch.from_numpy(local_obs).float().unsqueeze(0).to(self.config.device)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            action, log_prob, _ = self.actors[agent_idx].get_action(
                obs_t, mask_t, deterministic
            )

        return int(action.item()), float(log_prob.item())

    def get_value(self, global_state: np.ndarray) -> float:
        """Get value estimate from global state."""
        state_t = (
            torch.from_numpy(global_state).float().unsqueeze(0).to(self.config.device)
        )
        with torch.no_grad():
            value = self.global_value_fn(state_t)
        return float(value.item())

    def store_transition(
        self,
        agent_idx: int,
        global_state: np.ndarray,
        local_obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        mask: np.ndarray,
        value: float,
    ) -> None:
        """Store a transition for a specific agent."""
        self.buffers.add(
            agent_idx=agent_idx,
            global_state=global_state,
            local_obs=local_obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            mask=mask,
            value=value,
        )

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: (T,)
            values: (T,)
            dones: (T,)
            next_value: scalar

        Returns:
            advantages: (T,)
            returns: (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]

            delta = (
                rewards[t] + self.config.ppo_gamma * next_v * (1 - dones[t]) - values[t]
            )
            gae = (
                delta
                + self.config.ppo_gamma
                * self.config.ppo_gae_lambda
                * (1 - dones[t])
                * gae
            )
            advantages[t] = gae
            returns[t] = gae + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_agent(
        self,
        agent_idx: int,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Update a single agent using PPO.

        Args:
            agent_idx: Which agent to update
            n_epochs: Number of PPO epochs
            batch_size: Mini-batch size

        Returns:
            metrics dict with losses
        """
        buffer = self.buffers.buffers[agent_idx]
        if len(buffer) == 0:
            return {}

        # Get recent on-policy data
        batch = buffer.get_recent(min(batch_size * 4, len(buffer)))
        if batch is None:
            return {}

        rewards = batch["rewards"].cpu().numpy()
        values = batch["values"].cpu().numpy()
        dones = batch["dones"].cpu().numpy()

        # Compute next value (use last state)
        last_global_state = batch["global_states"][-1].cpu().numpy()
        next_value = self.get_value(last_global_state)

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        advantages_t = torch.from_numpy(advantages).float().to(self.config.device)
        returns_t = torch.from_numpy(returns).float().to(self.config.device)

        T = len(batch["actions"])
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(n_epochs):
            indices = np.random.permutation(T)

            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                mb_idx = indices[start:end]

                mb_obs = batch["local_observations"][mb_idx]
                mb_actions = batch["actions"][mb_idx]
                mb_old_log_probs = batch["old_log_probs"][mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_masks = batch["masks"][mb_idx]
                mb_global_states = batch["global_states"][mb_idx]

                # Policy loss
                log_prob, entropy = self.actors[agent_idx].evaluate_actions(
                    mb_obs, mb_actions, mb_masks
                )

                ratio = torch.exp(log_prob - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.ppo_clip_range,
                        1 + self.config.ppo_clip_range,
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values_pred = self.global_value_fn(mb_global_states)
                value_loss = (values_pred - mb_returns).pow(2).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.ppo_vf_coef * value_loss
                    - self.config.ppo_ent_coef * entropy.mean()
                )

                # Update
                self.actor_optimizers[agent_idx].zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actors[agent_idx].parameters(), self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.global_value_fn.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizers[agent_idx].step()
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        if num_updates > 0:
            self._update_count += 1

            # Learning rate decay
            for param_group in self.actor_optimizers[agent_idx].param_groups:
                param_group["lr"] *= self.config.lr_decay
            for param_group in self.value_optimizer.param_groups:
                param_group["lr"] *= self.config.lr_decay

            return {
                "policy_loss": total_policy_loss / num_updates,
                "value_loss": total_value_loss / num_updates,
                "entropy": total_entropy / num_updates,
                "agent_idx": agent_idx,
            }

        return {}

    def update_all_agents(
        self, n_epochs: int = 10, batch_size: int = 64
    ) -> Dict[str, float]:
        """Update all agents sequentially."""
        all_metrics = {}
        for agent_idx in range(self.config.num_agents):
            metrics = self.update_agent(agent_idx, n_epochs, batch_size)
            if metrics:
                for k, v in metrics.items():
                    all_metrics[f"agent_{agent_idx}_{k}"] = v
        return all_metrics

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "actors": [actor.state_dict() for actor in self.actors],
                "value_fn": self.global_value_fn.state_dict(),
                "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
                "value_optimizer": self.value_optimizer.state_dict(),
                "config": self.config,
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint["actors"][i])
        self.global_value_fn.load_state_dict(checkpoint["value_fn"])
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint["actor_optimizers"][i])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self._update_count = checkpoint.get("update_count", 0)
