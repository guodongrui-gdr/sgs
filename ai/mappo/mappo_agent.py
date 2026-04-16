"""
MAPPO Agent - Coordinator for Multi-Agent PPO

Coordinates 5 decentralized actors with 1 shared centralized critic.
Handles:
- Action collection from all agents
- PPO update with centralized value estimation
- Entropy monitoring (prevent policy collapse)

Architecture:
- 5 MAPPOActor instances (one per agent: 主公, 忠臣, 反贼×2, 内奸)
- 1 CentralizedCritic instance (shared across all agents)
- PPO update with centralized value targets
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .centralized_critic import CentralizedCritic, CentralizedCriticConfig
from .mappo_policy import MAPPOActor, MAPPOActorConfig


@dataclass
class MAPPOAgentConfig:
    """Configuration for MAPPO Agent"""

    num_agents: int = 5

    actor_config: Optional[MAPPOActorConfig] = None
    critic_config: Optional[CentralizedCriticConfig] = None

    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95

    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    entropy_threshold: float = 0.01
    value_std_threshold: float = 0.01


class MAPPOAgent(nn.Module):
    """
    MAPPO Agent coordinating multiple actors with centralized critic

    Handles:
    - Collecting actions from all 5 agents
    - Computing centralized value estimates
    - PPO update with centralized value targets
    - Entropy/value_std monitoring
    """

    def __init__(self, config: Optional[MAPPOAgentConfig] = None):
        super().__init__()

        self.config = config or MAPPOAgentConfig()
        c = self.config

        self.actor_config = c.actor_config or MAPPOActorConfig()
        self.critic_config = c.critic_config or CentralizedCriticConfig()

        self.actors = nn.ModuleList(
            [MAPPOActor(self.actor_config) for _ in range(c.num_agents)]
        )

        self.shared_critic = CentralizedCritic(self.critic_config)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=c.learning_rate,
        )

        self._entropy_history: List[float] = []
        self._value_std_history: List[float] = []

    def get_actions(
        self,
        local_observations: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        is_alive_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions for all agents

        Args:
            local_observations: (batch, num_agents, local_state_dim)
            action_masks: (batch, num_agents, action_dim)
            is_alive_mask: (batch, num_agents) - 1 for alive, 0 for dead
            deterministic: If True, select argmax

        Returns:
            joint_actions: (batch, num_agents) - action indices
            log_probs: (batch, num_agents) - log probabilities
            entropies: (batch, num_agents) - entropies per agent
            mean_entropy: scalar - mean entropy for monitoring
        """
        batch_size = local_observations.shape[0]
        num_agents = local_observations.shape[1]

        joint_actions = []
        log_probs = []
        entropies = []

        for agent_idx in range(num_agents):
            local_obs = local_observations[:, agent_idx, :]
            mask = action_masks[:, agent_idx, :] if action_masks is not None else None

            action, log_prob, entropy = self.actors[agent_idx].get_action(
                local_obs, mask, deterministic
            )

            joint_actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)

        joint_actions = torch.stack(joint_actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1)

        mean_entropy = entropies.mean()

        return joint_actions, log_probs, entropies, mean_entropy

    def get_centralized_value(
        self,
        global_state: torch.Tensor,
        joint_actions: torch.Tensor,
        is_alive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute centralized value estimate

        Args:
            global_state: (batch, global_state_dim)
            joint_actions: (batch, num_agents)
            is_alive_mask: (batch, num_agents) - 1 for alive, 0 for dead

        Returns:
            value: (batch,)
        """
        return self.shared_critic(global_state, joint_actions, is_alive_mask)

    def compute_gae(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation

        Args:
            values: (T,) - value estimates
            rewards: (T,) - rewards received
            dones: (T,) - episode termination flags
            next_values: (T,) - next value estimates

        Returns:
            advantages: (T,)
            returns: (T,)
        """
        gamma = self.config.ppo_gamma
        gae_lambda = self.config.ppo_gae_lambda

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                last_gae = 0
            else:
                next_value = next_values[t]

            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae

        returns = advantages + values

        return advantages, returns

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        PPO update with centralized value estimation

        Args:
            batch: Dict containing:
                - global_states: (T, global_state_dim)
                - local_observations: (T, num_agents, local_state_dim)
                - joint_actions: (T, num_agents)
                - old_log_probs: (T, num_agents)
                - advantages: (T,)
                - returns: (T,)
                - action_masks: (T, num_agents, action_dim)
                - is_alive_masks: (T, num_agents) - 1 for alive, 0 for dead
            n_epochs: Number of PPO epochs
            batch_size: Mini-batch size

        Returns:
            metrics: Dict with loss, entropy, value_loss, etc.
        """
        global_states = batch["global_states"]
        local_observations = batch["local_observations"]
        joint_actions = batch["joint_actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        action_masks = batch.get("action_masks")
        is_alive_masks = batch.get("is_alive_masks")

        T = global_states.shape[0]
        num_agents = local_observations.shape[1]

        if advantages.dim() > 1:
            advantages = advantages.flatten()
        if returns.dim() > 1:
            returns = returns.flatten()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8
        returns_normalized = (returns - returns_mean) / returns_std

        old_values = batch.get("old_values")
        if old_values is not None and old_values.dim() > 1:
            old_values = old_values.flatten()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0

        for epoch in range(n_epochs):
            indices = np.random.permutation(T)

            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                mb_indices = indices[start:end]

                mb_global_states = global_states[mb_indices]
                mb_local_obs = local_observations[mb_indices]
                mb_joint_actions = joint_actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns_normalized[mb_indices]
                mb_action_masks = (
                    action_masks[mb_indices] if action_masks is not None else None
                )
                mb_old_values = (
                    old_values[mb_indices] if old_values is not None else None
                )
                mb_is_alive_masks = (
                    is_alive_masks[mb_indices] if is_alive_masks is not None else None
                )

                mb_values = self.shared_critic(
                    mb_global_states, mb_joint_actions, mb_is_alive_masks
                )

                policy_loss = 0
                entropy_sum = 0
                kl_sum = 0
                alive_agent_count = 0

                for agent_idx in range(num_agents):
                    agent_alive = (
                        mb_is_alive_masks[:, agent_idx].mean() > 0.5
                        if mb_is_alive_masks is not None
                        else True
                    )

                    mb_log_prob, mb_entropy = self.actors[agent_idx].evaluate_actions(
                        mb_local_obs[:, agent_idx, :],
                        mb_joint_actions[:, agent_idx],
                        mb_action_masks[:, agent_idx, :]
                        if mb_action_masks is not None
                        else None,
                    )

                    ratio = torch.exp(mb_log_prob - mb_old_log_probs[:, agent_idx])

                    agent_advantages = mb_advantages
                    surr1 = ratio * agent_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1 - self.config.ppo_clip_range,
                            1 + self.config.ppo_clip_range,
                        )
                        * agent_advantages
                    )
                    policy_loss_agent = -torch.min(surr1, surr2).mean()

                    if agent_alive:
                        policy_loss += policy_loss_agent
                        entropy_sum += mb_entropy.mean()
                        kl_sum += (ratio - 1).abs().mean()
                        alive_agent_count += 1
                    else:
                        policy_loss += 0.0 * policy_loss_agent

                policy_loss = policy_loss / max(alive_agent_count, 1)
                entropy_mean = entropy_sum / max(alive_agent_count, 1)
                kl_mean = kl_sum / max(alive_agent_count, 1)

                if mb_old_values is not None:
                    values_clipped = mb_old_values + torch.clamp(
                        mb_values - mb_old_values,
                        -self.config.ppo_clip_range,
                        self.config.ppo_clip_range,
                    )
                    value_loss_unclipped = (mb_values - mb_returns).pow(2)
                    value_loss_clipped = (values_clipped - mb_returns).pow(2)
                    value_loss = torch.max(
                        value_loss_unclipped, value_loss_clipped
                    ).mean()
                else:
                    value_loss = nn.functional.mse_loss(mb_values, mb_returns)

                loss = (
                    policy_loss
                    - self.config.ppo_ent_coef * entropy_mean
                    + self.config.ppo_vf_coef * value_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)

                if mb_is_alive_masks is not None:
                    self._zero_dead_agent_gradients(mb_is_alive_masks, num_agents)

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_kl += kl_mean.item()
                num_updates += 1

        metrics = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_kl / num_updates,
        }

        self._entropy_history.append(metrics["entropy"])
        self._value_std_history.append(
            self.shared_critic.get_value_std(global_states[:1])
        )

        return metrics

    def _zero_dead_agent_gradients(
        self, is_alive_masks: torch.Tensor, num_agents: int
    ) -> None:
        """
        Zero gradients for dead agents' actor parameters

        Args:
            is_alive_masks: (batch, num_agents) - 1 for alive, 0 for dead
            num_agents: Number of agents
        """
        for agent_idx in range(num_agents):
            agent_alive = is_alive_masks[:, agent_idx].mean() > 0.5
            if not agent_alive:
                for param in self.actors[agent_idx].parameters():
                    if param.grad is not None:
                        param.grad.zero_()

    def check_training_health(self) -> Dict[str, bool]:
        """
        Check if training is healthy (entropy not collapsed, value_std reasonable)

        Returns:
            health: Dict with entropy_ok, value_std_ok, overall_ok
        """
        entropy_ok = True
        if self._entropy_history:
            recent_entropy = self._entropy_history[-1]
            entropy_ok = recent_entropy > self.config.entropy_threshold

        value_std_ok = True
        if self._value_std_history:
            recent_value_std = self._value_std_history[-1]
            value_std_ok = recent_value_std > self.config.value_std_threshold

        return {
            "entropy_ok": entropy_ok,
            "value_std_ok": value_std_ok,
            "overall_ok": entropy_ok and value_std_ok,
            "recent_entropy": self._entropy_history[-1]
            if self._entropy_history
            else None,
            "recent_value_std": self._value_std_history[-1]
            if self._value_std_history
            else None,
        }

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "config": self.config,
                "actors_state_dict": [actor.state_dict() for actor in self.actors],
                "critic_state_dict": self.shared_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "entropy_history": self._entropy_history,
                "value_std_history": self._value_std_history,
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, weights_only=False)

        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint["actors_state_dict"][i])

        self.shared_critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self._entropy_history = checkpoint.get("entropy_history", [])
        self._value_std_history = checkpoint.get("value_std_history", [])
