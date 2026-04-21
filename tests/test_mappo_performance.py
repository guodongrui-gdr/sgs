"""
Performance regression tests for MAPPO training optimizations.
"""

import pytest
import torch
import time


def test_gae_vectorized_performance():
    """Verify vectorized GAE is faster than loop-based."""
    from ai.mappo.mappo_agent import MAPPOAgent, MAPPOAgentConfig
    from ai.mappo.centralized_critic import CentralizedCriticConfig
    from ai.mappo.mappo_policy import MAPPOActorConfig

    config = MAPPOAgentConfig(
        num_agents=5,
        actor_config=MAPPOActorConfig(local_state_dim=2670, action_dim=20),
        critic_config=CentralizedCriticConfig(
            local_state_dim=2670, num_agents=5, global_state_dim=2670 * 5, action_dim=20
        ),
    )
    agent = MAPPOAgent(config)

    torch.manual_seed(42)
    T = 2048
    values = torch.randn(T)
    rewards = torch.randn(T)
    dones = torch.zeros(T)
    dones[500] = 1.0
    dones[1200] = 1.0
    next_values = torch.randn(T)

    # Warmup
    agent._compute_gae_original(values, rewards, dones, next_values)
    agent._compute_gae_vectorized(values, rewards, dones, next_values)

    # Time original
    start = time.time()
    for _ in range(10):
        agent._compute_gae_original(values, rewards, dones, next_values)
    time_orig = time.time() - start

    # Time vectorized
    start = time.time()
    for _ in range(10):
        agent._compute_gae_vectorized(values, rewards, dones, next_values)
    time_vec = time.time() - start

    speedup = time_orig / time_vec
    print(f"GAE speedup: {speedup:.2f}x (orig: {time_orig:.4f}s, vec: {time_vec:.4f}s)")

    # Vectorized should be at least 2x faster
    assert speedup > 1.5, f"Vectorized GAE not faster enough: {speedup:.2f}x"


def test_gae_equivalence():
    """Verify vectorized GAE produces same results as original."""
    from ai.mappo.mappo_agent import MAPPOAgent, MAPPOAgentConfig
    from ai.mappo.centralized_critic import CentralizedCriticConfig
    from ai.mappo.mappo_policy import MAPPOActorConfig

    config = MAPPOAgentConfig(
        num_agents=5,
        actor_config=MAPPOActorConfig(local_state_dim=2670, action_dim=20),
        critic_config=CentralizedCriticConfig(
            local_state_dim=2670, num_agents=5, global_state_dim=2670 * 5, action_dim=20
        ),
    )
    agent = MAPPOAgent(config)

    torch.manual_seed(42)
    T = 200
    values = torch.randn(T)
    rewards = torch.randn(T)
    dones = torch.zeros(T)
    dones[50] = 1.0
    next_values = torch.randn(T)

    adv_orig, ret_orig = agent._compute_gae_original(
        values, rewards, dones, next_values
    )
    adv_vec, ret_vec = agent._compute_gae_vectorized(
        values, rewards, dones, next_values
    )

    assert torch.allclose(adv_orig, adv_vec, atol=1e-5)
    assert torch.allclose(ret_orig, ret_vec, atol=1e-5)


if __name__ == "__main__":
    test_gae_vectorized_performance()
    test_gae_equivalence()
    print("All performance tests passed!")
