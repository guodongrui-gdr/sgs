"""
Tests for IPPO Global State Environment - Verification Tests

Tests:
1. GlobalStateWrapper initialization
2. Global state dimension (3066 * 5 = 15330)
3. Observation structure (global_state, action_mask, agent_id)
4. Reset returns valid global state
5. Step returns valid global state
6. Action masking per agent
7. All agents receive same global state
"""

import sys
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_global_state_wrapper_init():
    """Test GlobalStateWrapper initialization"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    config = GlobalStateConfig(player_num=5)
    env = GlobalStateWrapper(config)

    assert env.player_num == 5
    assert env.num_agents == 5
    assert env.single_state_dim > 0
    assert env.global_state_dim == env.single_state_dim * 5
    print(
        f"[PASS] GlobalStateWrapper init: player_num={env.player_num}, global_dim={env.global_state_dim}"
    )


def test_state_dimensions():
    """Test state dimension calculation"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig
    from ai.state_encoder import StateEncoder

    encoder = StateEncoder()
    single_dim = encoder.get_state_dim(5)

    config = GlobalStateConfig(player_num=5)
    env = GlobalStateWrapper(config)

    assert env.single_state_dim == single_dim
    assert env.global_state_dim == single_dim * 5
    print(
        f"[PASS] State dimensions: single={single_dim}, global={env.global_state_dim}"
    )


def test_observation_space():
    """Test observation space structure"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    env = GlobalStateWrapper(GlobalStateConfig(player_num=5))

    assert "global_state" in env.observation_space.spaces
    assert "action_mask" in env.observation_space.spaces
    assert "agent_id" in env.observation_space.spaces

    global_shape = env.observation_space["global_state"].shape
    mask_shape = env.observation_space["action_mask"].shape

    assert global_shape[0] == env.global_state_dim
    print(
        f"[PASS] Observation space: global_shape={global_shape}, mask_shape={mask_shape}"
    )


def test_reset():
    """Test reset returns valid global state"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    env = GlobalStateWrapper(GlobalStateConfig(player_num=5))
    obs, info = env.reset(seed=42)

    assert "global_state" in obs
    assert "action_mask" in obs
    assert "agent_id" in obs

    global_state = obs["global_state"]
    assert global_state.shape == (env.global_state_dim,)
    assert global_state.dtype == np.float32

    assert info["agent_id"] == env.current_agent_idx
    assert info["global_state_dim"] == env.global_state_dim

    print(
        f"[PASS] Reset: global_state_shape={global_state.shape}, agent_id={obs['agent_id']}"
    )


def test_get_global_state():
    """Test get_global_state returns concatenated states"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    env = GlobalStateWrapper(GlobalStateConfig(player_num=5))
    env.reset(seed=42)

    global_state = env.get_global_state()

    assert global_state.shape == (env.global_state_dim,)
    assert env.global_state_dim == env.single_state_dim * 5

    print(
        f"[PASS] get_global_state: shape={global_state.shape}, dim={env.global_state_dim}"
    )


def test_get_local_obs_for_agent():
    """Test all agents receive same global state"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    env = GlobalStateWrapper(GlobalStateConfig(player_num=5))
    env.reset(seed=42)

    observations = env.get_all_agent_obs()

    global_states = [obs["global_state"] for obs in observations.values()]

    for i in range(1, len(global_states)):
        assert np.allclose(global_states[0], global_states[i]), (
            f"Agent 0 and {i} have different global states"
        )

    print(f"[PASS] All agents receive same global state")


def test_action_masks():
    """Test action masking per agent"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    env = GlobalStateWrapper(GlobalStateConfig(player_num=5))
    env.reset(seed=42)

    mask = env.action_masks()

    assert mask.shape == (env.action_space.n,)
    assert mask.dtype == np.float32

    current_mask = env._cached_action_masks[env.current_agent_idx]
    assert np.allclose(mask, current_mask)

    other_agent = (env.current_agent_idx + 1) % 5
    other_mask = env._cached_action_masks.get(other_agent, np.zeros(env.action_space.n))

    print(
        f"[PASS] Action masks: current_agent={env.current_agent_idx}, mask_shape={mask.shape}"
    )


def test_step():
    """Test step returns valid global state"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    env = GlobalStateWrapper(GlobalStateConfig(player_num=5))
    obs, _ = env.reset(seed=42)

    mask = obs["action_mask"]
    valid_actions = np.where(mask > 0)[0]

    if len(valid_actions) > 0:
        action = valid_actions[0]
    else:
        action = 0

    obs, reward, done, truncated, info = env.step(action)

    assert "global_state" in obs
    assert obs["global_state"].shape == (env.global_state_dim,)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    print(
        f"[PASS] Step: reward={reward:.2f}, done={done}, agent_id={info.get('agent_id')}"
    )


def test_multiple_steps():
    """Test multiple consecutive steps"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    env = GlobalStateWrapper(GlobalStateConfig(player_num=5))
    obs, _ = env.reset(seed=42)

    step_count = 0
    total_reward = 0.0

    for _ in range(50):
        mask = obs["action_mask"]
        valid_actions = np.where(mask > 0)[0]

        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = 0

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if done or truncated:
            break

    print(f"[PASS] Multiple steps: count={step_count}, total_reward={total_reward:.2f}")


def test_global_state_consistency():
    """Test global state remains consistent shape"""
    from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig

    env = GlobalStateWrapper(GlobalStateConfig(player_num=5))
    obs, _ = env.reset(seed=42)

    expected_dim = env.global_state_dim

    for _ in range(10):
        mask = obs["action_mask"]
        valid_actions = np.where(mask > 0)[0]

        if len(valid_actions) > 0:
            action = valid_actions[0]
        else:
            action = 0

        obs, reward, done, truncated, info = env.step(action)

        assert obs["global_state"].shape == (expected_dim,)

        if done:
            break

    print(f"[PASS] Global state consistency: dim={expected_dim}")


def run_all_tests():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print("IPPO Global State Environment Tests")
    print("=" * 60 + "\n")

    tests = [
        test_global_state_wrapper_init,
        test_state_dimensions,
        test_observation_space,
        test_reset,
        test_get_global_state,
        test_get_local_obs_for_agent,
        test_action_masks,
        test_step,
        test_multiple_steps,
        test_global_state_consistency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
