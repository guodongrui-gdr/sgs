"""
Tests for Death Masking in MAPPO

Verifies that dead agents don't contribute gradients during training.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.mappo import (
    CentralizedCritic,
    CentralizedCriticConfig,
    MAPPOActor,
    MAPPOActorConfig,
    MAPPOAgent,
    MAPPOAgentConfig,
)


class TestCentralizedCriticDeathMasking:
    def test_critic_accepts_alive_mask(self):
        config = CentralizedCriticConfig()
        critic = CentralizedCritic(config)
        batch_size = 4
        global_state = torch.randn(batch_size, config.global_state_dim)
        joint_actions = torch.randint(
            0, config.action_dim, (batch_size, config.num_agents)
        )
        is_alive_mask = torch.ones(batch_size, config.num_agents)
        values = critic(global_state, joint_actions, is_alive_mask)
        assert values.shape == (batch_size,)
        assert not torch.isnan(values).any()

    def test_critic_masks_dead_agent_embeddings(self):
        config = CentralizedCriticConfig()
        critic = CentralizedCritic(config)
        batch_size = 2
        global_state = torch.randn(batch_size, config.global_state_dim)
        joint_actions = torch.randint(
            0, config.action_dim, (batch_size, config.num_agents)
        )

        is_alive_mask = torch.ones(batch_size, config.num_agents)
        is_alive_mask[:, 0] = 0

        values_with_mask = critic(global_state, joint_actions, is_alive_mask)
        values_all_alive = critic(global_state, joint_actions, None)

        assert values_with_mask.shape == (batch_size,)
        assert not torch.isnan(values_with_mask).any()
        assert not torch.isinf(values_with_mask).any()

    def test_critic_all_dead_agents(self):
        config = CentralizedCriticConfig()
        critic = CentralizedCritic(config)
        batch_size = 2
        global_state = torch.randn(batch_size, config.global_state_dim)
        joint_actions = torch.randint(
            0, config.action_dim, (batch_size, config.num_agents)
        )
        is_alive_mask = torch.zeros(batch_size, config.num_agents)

        values = critic(global_state, joint_actions, is_alive_mask)
        assert values.shape == (batch_size,)
        assert not torch.isnan(values).any()

    def test_critic_mixed_alive_status(self):
        config = CentralizedCriticConfig()
        critic = CentralizedCritic(config)
        batch_size = 4
        global_state = torch.randn(batch_size, config.global_state_dim)
        joint_actions = torch.randint(
            0, config.action_dim, (batch_size, config.num_agents)
        )

        is_alive_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )

        values = critic(global_state, joint_actions, is_alive_mask)
        assert values.shape == (batch_size,)
        assert not torch.isnan(values).any()


class TestMAPPOAgentDeathMasking:
    def test_agent_update_with_alive_masks(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 32
        batch = {
            "global_states": torch.randn(T, agent.critic_config.global_state_dim),
            "local_observations": torch.randn(
                T, config.num_agents, agent.actor_config.local_state_dim
            ),
            "joint_actions": torch.randint(
                0, agent.critic_config.action_dim, (T, config.num_agents)
            ),
            "old_log_probs": torch.randn(T, config.num_agents),
            "advantages": torch.randn(T),
            "returns": torch.randn(T),
            "action_masks": torch.ones(
                T, config.num_agents, agent.actor_config.action_dim
            ),
            "is_alive_masks": torch.ones(T, config.num_agents),
        }
        metrics = agent.update(batch, n_epochs=1, batch_size=16)
        assert "policy_loss" in metrics
        assert "entropy" in metrics

    def test_dead_agent_zero_gradients(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 32

        is_alive_masks = torch.ones(T, config.num_agents)
        is_alive_masks[:, 0] = 0

        batch = {
            "global_states": torch.randn(T, agent.critic_config.global_state_dim),
            "local_observations": torch.randn(
                T, config.num_agents, agent.actor_config.local_state_dim
            ),
            "joint_actions": torch.randint(
                0, agent.critic_config.action_dim, (T, config.num_agents)
            ),
            "old_log_probs": torch.randn(T, config.num_agents),
            "advantages": torch.randn(T),
            "returns": torch.randn(T),
            "action_masks": torch.ones(
                T, config.num_agents, agent.actor_config.action_dim
            ),
            "is_alive_masks": is_alive_masks,
        }

        agent.optimizer.zero_grad()
        metrics = agent.update(batch, n_epochs=1, batch_size=32)

        for param in agent.actors[0].parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_alive_agent_nonzero_gradients(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 32

        is_alive_masks = torch.ones(T, config.num_agents)
        is_alive_masks[:, 0] = 0

        batch = {
            "global_states": torch.randn(T, agent.critic_config.global_state_dim),
            "local_observations": torch.randn(
                T, config.num_agents, agent.actor_config.local_state_dim
            ),
            "joint_actions": torch.randint(
                0, agent.critic_config.action_dim, (T, config.num_agents)
            ),
            "old_log_probs": torch.randn(T, config.num_agents),
            "advantages": torch.randn(T),
            "returns": torch.randn(T),
            "action_masks": torch.ones(
                T, config.num_agents, agent.actor_config.action_dim
            ),
            "is_alive_masks": is_alive_masks,
        }

        agent.optimizer.zero_grad()
        metrics = agent.update(batch, n_epochs=1, batch_size=32)

        for param in agent.actors[1].parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0

    def test_mixed_death_pattern(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 32

        is_alive_masks = torch.ones(T, config.num_agents)
        is_alive_masks[:, 0] = 0
        is_alive_masks[:, 2] = 0
        is_alive_masks[16:, 3] = 0

        batch = {
            "global_states": torch.randn(T, agent.critic_config.global_state_dim),
            "local_observations": torch.randn(
                T, config.num_agents, agent.actor_config.local_state_dim
            ),
            "joint_actions": torch.randint(
                0, agent.critic_config.action_dim, (T, config.num_agents)
            ),
            "old_log_probs": torch.randn(T, config.num_agents),
            "advantages": torch.randn(T),
            "returns": torch.randn(T),
            "action_masks": torch.ones(
                T, config.num_agents, agent.actor_config.action_dim
            ),
            "is_alive_masks": is_alive_masks,
        }

        metrics = agent.update(batch, n_epochs=1, batch_size=16)
        assert metrics["entropy"] > 0.0

    def test_tensor_shapes_preserved(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 32

        is_alive_masks = torch.zeros(T, config.num_agents)

        batch = {
            "global_states": torch.randn(T, agent.critic_config.global_state_dim),
            "local_observations": torch.randn(
                T, config.num_agents, agent.actor_config.local_state_dim
            ),
            "joint_actions": torch.randint(
                0, agent.critic_config.action_dim, (T, config.num_agents)
            ),
            "old_log_probs": torch.randn(T, config.num_agents),
            "advantages": torch.randn(T),
            "returns": torch.randn(T),
            "action_masks": torch.ones(
                T, config.num_agents, agent.actor_config.action_dim
            ),
            "is_alive_masks": is_alive_masks,
        }

        metrics = agent.update(batch, n_epochs=1, batch_size=16)

        assert len(agent.actors) == config.num_agents
        assert agent.shared_critic.config.num_agents == config.num_agents

    def test_get_centralized_value_with_mask(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        batch_size = 2
        global_state = torch.randn(batch_size, agent.critic_config.global_state_dim)
        joint_actions = torch.randint(
            0, agent.critic_config.action_dim, (batch_size, config.num_agents)
        )
        is_alive_mask = torch.ones(batch_size, config.num_agents)
        is_alive_mask[:, 0] = 0

        values = agent.get_centralized_value(global_state, joint_actions, is_alive_mask)
        assert values.shape == (batch_size,)
        assert not torch.isnan(values).any()


def test_death_masking_integration():
    config = MAPPOAgentConfig()
    agent = MAPPOAgent(config)
    batch_size = 2

    local_obs = torch.randn(
        batch_size, config.num_agents, agent.actor_config.local_state_dim
    )
    action_masks = torch.ones(
        batch_size, config.num_agents, agent.actor_config.action_dim
    )
    is_alive_mask = torch.ones(batch_size, config.num_agents)
    is_alive_mask[:, 0] = 0

    joint_actions, log_probs, entropies, mean_entropy = agent.get_actions(
        local_obs, action_masks, is_alive_mask
    )

    assert joint_actions.shape == (batch_size, config.num_agents)

    global_state = torch.randn(batch_size, agent.critic_config.global_state_dim)
    values = agent.get_centralized_value(global_state, joint_actions, is_alive_mask)
    assert values.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
