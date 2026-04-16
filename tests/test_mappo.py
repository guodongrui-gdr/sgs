"""
Tests for MAPPO Components

Verifies centralized critic, actor, and agent components.
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


class TestCentralizedCritic:
    def test_critic_config_defaults(self):
        config = CentralizedCriticConfig()
        assert config.local_state_dim == 2670
        assert config.num_agents == 5
        assert config.global_state_dim == 2670 * 5
        assert config.action_dim == 20

    def test_critic_forward_pass(self):
        config = CentralizedCriticConfig()
        critic = CentralizedCritic(config)
        batch_size = 4
        global_state = torch.randn(batch_size, config.global_state_dim)
        joint_actions = torch.randint(0, config.action_dim, (batch_size, config.num_agents))
        values = critic(global_state, joint_actions)
        assert values.shape == (batch_size,)
        assert not torch.isnan(values).any()
        assert not torch.isinf(values).any()

    def test_critic_value_range(self):
        config = CentralizedCriticConfig()
        critic = CentralizedCritic(config)
        global_state = torch.randn(10, config.global_state_dim)
        joint_actions = torch.randint(0, config.action_dim, (10, config.num_agents))
        values = critic(global_state, joint_actions)
        assert values.shape == (10,)
        value_std = values.std().item()
        assert value_std > 0.0

    def test_critic_get_value_std(self):
        config = CentralizedCriticConfig()
        critic = CentralizedCritic(config)
        global_state = torch.randn(1, config.global_state_dim)
        value_std = critic.get_value_std(global_state)
        assert isinstance(value_std, float)
        assert value_std >= 0.0


class TestMAPPOActor:
    def test_actor_config_defaults(self):
        config = MAPPOActorConfig()
        assert config.local_state_dim == 2670
        assert config.action_dim == 20

    def test_actor_forward_pass(self):
        config = MAPPOActorConfig()
        actor = MAPPOActor(config)
        batch_size = 4
        local_obs = torch.randn(batch_size, config.local_state_dim)
        logits = actor(local_obs)
        assert logits.shape == (batch_size, config.action_dim)
        assert not torch.isnan(logits).any()

    def test_actor_with_action_mask(self):
        config = MAPPOActorConfig()
        actor = MAPPOActor(config)
        batch_size = 4
        local_obs = torch.randn(batch_size, config.local_state_dim)
        action_mask = torch.zeros(batch_size, config.action_dim)
        action_mask[:, :5] = 1.0
        logits = actor(local_obs, action_mask)
        assert logits.shape == (batch_size, config.action_dim)
        assert logits[:, 5:].isinf().all()
        assert not logits[:, :5].isinf().any()

    def test_actor_get_action(self):
        config = MAPPOActorConfig()
        actor = MAPPOActor(config)
        local_obs = torch.randn(1, config.local_state_dim)
        action_mask = torch.ones(1, config.action_dim)
        action, log_prob, entropy = actor.get_action(local_obs, action_mask, deterministic=False)
        assert action.shape == (1,)
        assert action.item() >= 0 and action.item() < config.action_dim
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert entropy.item() > 0.0

    def test_actor_deterministic_action(self):
        config = MAPPOActorConfig()
        actor = MAPPOActor(config)
        local_obs = torch.randn(1, config.local_state_dim)
        action_mask = torch.ones(1, config.action_dim)
        action1, _, _ = actor.get_action(local_obs, action_mask, deterministic=True)
        action2, _, _ = actor.get_action(local_obs, action_mask, deterministic=True)
        assert action1.item() == action2.item()

    def test_actor_evaluate_actions(self):
        config = MAPPOActorConfig()
        actor = MAPPOActor(config)
        batch_size = 4
        local_obs = torch.randn(batch_size, config.local_state_dim)
        actions = torch.randint(0, config.action_dim, (batch_size,))
        log_prob, entropy = actor.evaluate_actions(local_obs, actions)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_actor_entropy_positive(self):
        config = MAPPOActorConfig()
        actor = MAPPOActor(config)
        local_obs = torch.randn(10, config.local_state_dim)
        entropy = actor.get_entropy(local_obs)
        assert entropy.shape == (10,)
        assert entropy.mean().item() > 0.0


class TestMAPPOAgent:
    def test_agent_config_defaults(self):
        config = MAPPOAgentConfig()
        assert config.num_agents == 5
        assert config.ppo_clip_range == 0.2
        assert config.entropy_threshold == 0.01

    def test_agent_creation(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        assert len(agent.actors) == config.num_agents
        assert agent.shared_critic is not None
        assert agent.optimizer is not None

    def test_agent_get_actions(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        batch_size = 2
        local_obs = torch.randn(batch_size, config.num_agents, agent.actor_config.local_state_dim)
        action_masks = torch.ones(batch_size, config.num_agents, agent.actor_config.action_dim)
        joint_actions, log_probs, entropies, mean_entropy = agent.get_actions(local_obs, action_masks)
        assert joint_actions.shape == (batch_size, config.num_agents)
        assert log_probs.shape == (batch_size, config.num_agents)
        assert entropies.shape == (batch_size, config.num_agents)
        assert isinstance(mean_entropy, torch.Tensor)
        assert mean_entropy.item() > 0.0

    def test_agent_centralized_value(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        batch_size = 2
        global_state = torch.randn(batch_size, agent.critic_config.global_state_dim)
        joint_actions = torch.randint(0, agent.critic_config.action_dim, (batch_size, config.num_agents))
        values = agent.get_centralized_value(global_state, joint_actions)
        assert values.shape == (batch_size,)

    def test_agent_compute_gae(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 10
        values = torch.randn(T)
        rewards = torch.randn(T)
        dones = torch.zeros(T)
        dones[5] = 1.0
        next_values = torch.randn(T)
        advantages, returns = agent.compute_gae(values, rewards, dones, next_values)
        assert advantages.shape == (T,)
        assert returns.shape == (T,)

    def test_agent_update(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 32
        batch = {
            "global_states": torch.randn(T, agent.critic_config.global_state_dim),
            "local_observations": torch.randn(T, config.num_agents, agent.actor_config.local_state_dim),
            "joint_actions": torch.randint(0, agent.critic_config.action_dim, (T, config.num_agents)),
            "old_log_probs": torch.randn(T, config.num_agents),
            "advantages": torch.randn(T),
            "returns": torch.randn(T),
            "action_masks": torch.ones(T, config.num_agents, agent.actor_config.action_dim),
        }
        metrics = agent.update(batch, n_epochs=1, batch_size=16)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert metrics["entropy"] > 0.0

    def test_agent_health_check(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 32
        batch = {
            "global_states": torch.randn(T, agent.critic_config.global_state_dim),
            "local_observations": torch.randn(T, config.num_agents, agent.actor_config.local_state_dim),
            "joint_actions": torch.randint(0, agent.critic_config.action_dim, (T, config.num_agents)),
            "old_log_probs": torch.randn(T, config.num_agents),
            "advantages": torch.randn(T),
            "returns": torch.randn(T),
            "action_masks": torch.ones(T, config.num_agents, agent.actor_config.action_dim),
        }
        agent.update(batch, n_epochs=1, batch_size=16)
        health = agent.check_training_health()
        assert "entropy_ok" in health
        assert "value_std_ok" in health
        assert "overall_ok" in health
        assert "recent_entropy" in health

    def test_agent_save_load(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        save_path = "/tmp/test_mappo_agent.pt"
        agent.save(save_path)
        agent2 = MAPPOAgent(config)
        agent2.load(save_path)
        assert len(agent2.actors) == len(agent.actors)


class TestMAPPOIntegration:
    def test_full_forward_pass(self):
        actor_config = MAPPOActorConfig()
        critic_config = CentralizedCriticConfig()
        agent_config = MAPPOAgentConfig(actor_config=actor_config, critic_config=critic_config)
        agent = MAPPOAgent(agent_config)
        batch_size = 2
        local_obs = torch.randn(batch_size, agent_config.num_agents, actor_config.local_state_dim)
        action_masks = torch.ones(batch_size, agent_config.num_agents, actor_config.action_dim)
        joint_actions, log_probs, entropies, mean_entropy = agent.get_actions(local_obs, action_masks)
        global_state = torch.randn(batch_size, critic_config.global_state_dim)
        values = agent.get_centralized_value(global_state, joint_actions)
        assert values.shape == (batch_size,)
        assert mean_entropy.item() > 0.0

    def test_training_step_simulation(self):
        config = MAPPOAgentConfig()
        agent = MAPPOAgent(config)
        T = 64
        batch = {
            "global_states": torch.randn(T, agent.critic_config.global_state_dim),
            "local_observations": torch.randn(T, config.num_agents, agent.actor_config.local_state_dim),
            "joint_actions": torch.randint(0, agent.critic_config.action_dim, (T, config.num_agents)),
            "old_log_probs": torch.randn(T, config.num_agents),
            "advantages": torch.randn(T),
            "returns": torch.randn(T),
            "action_masks": torch.ones(T, config.num_agents, agent.actor_config.action_dim),
        }
        metrics = agent.update(batch, n_epochs=3, batch_size=32)
        assert metrics["entropy"] > 0.01
        health = agent.check_training_health()
        assert health["entropy_ok"]


def test_entropy_threshold():
    config = MAPPOAgentConfig(entropy_threshold=0.01)
    agent = MAPPOAgent(config)
    T = 64
    batch = {
        "global_states": torch.randn(T, agent.critic_config.global_state_dim),
        "local_observations": torch.randn(T, config.num_agents, agent.actor_config.local_state_dim),
        "joint_actions": torch.randint(0, agent.critic_config.action_dim, (T, config.num_agents)),
        "old_log_probs": torch.randn(T, config.num_agents),
        "advantages": torch.randn(T),
        "returns": torch.randn(T),
        "action_masks": torch.ones(T, config.num_agents, agent.actor_config.action_dim),
    }
    for _ in range(5):
        agent.update(batch, n_epochs=1, batch_size=32)
    health = agent.check_training_health()
    assert health["recent_entropy"] > config.entropy_threshold


def test_value_std_threshold():
    config = MAPPOAgentConfig(value_std_threshold=0.01)
    agent = MAPPOAgent(config)
    T = 64
    batch = {
        "global_states": torch.randn(T, agent.critic_config.global_state_dim),
        "local_observations": torch.randn(T, config.num_agents, agent.actor_config.local_state_dim),
        "joint_actions": torch.randint(0, agent.critic_config.action_dim, (T, config.num_agents)),
        "old_log_probs": torch.randn(T, config.num_agents),
        "advantages": torch.randn(T),
        "returns": torch.randn(T),
        "action_masks": torch.ones(T, config.num_agents, agent.actor_config.action_dim),
    }
    agent.update(batch, n_epochs=1, batch_size=32)
    health = agent.check_training_health()
    assert health.get("recent_value_std", 0) > config.value_std_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
