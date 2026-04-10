"""
SelfPlayWrapper测试 - TDD RED Phase

测试目标:
- 验证SelfPlayWrapper继承SGSEnv并保持相同接口
- 验证reset()时从AgentPoolManager采样对手
- 验证step()时controlled_player_idx循环
- 验证对手观察注入功能
- 验证observation_space和action_space与SGSEnv一致

Expected: All tests FAIL (SelfPlayWrapper尚未实现)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ai.gym_wrapper import SGSEnv, SGSConfig
from gymnasium import spaces


class TestSelfPlayWrapperInheritance(unittest.TestCase):
    """测试SelfPlayWrapper继承自SGSEnv"""

    def test_self_play_wrapper_inherits_sgs_env(self):
        """SelfPlayWrapper必须继承自SGSEnv"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        self.assertIsInstance(env, SGSEnv)

    def test_self_play_wrapper_has_agent_pool_manager(self):
        """SelfPlayWrapper必须有agent_pool_manager属性"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        self.assertTrue(hasattr(env, "agent_pool_manager"))
        self.assertIsNotNone(env.agent_pool_manager)


class TestSelfPlayWrapperReset(unittest.TestCase):
    """测试reset()时的对手采样行为"""

    def test_reset_samples_opponent_from_pool(self):
        """reset()必须从AgentPoolManager采样对手策略"""
        from ai.self_play_wrapper import SelfPlayWrapper

        config = SGSConfig()
        env = SelfPlayWrapper(config)

        # Mock the pool manager to track sampling
        original_sample = env.agent_pool_manager.sample_opponents
        sample_called = False
        sampled_opponents = None

        def mock_sample(*args, **kwargs):
            nonlocal sample_called, sampled_opponents
            sample_called = True
            sampled_opponents = original_sample(*args, **kwargs)
            return sampled_opponents

        env.agent_pool_manager.sample_opponents = mock_sample

        obs, info = env.reset(seed=42)

        self.assertTrue(
            sample_called, "sample_opponents should be called during reset()"
        )
        self.assertIsNotNone(sampled_opponents)
        self.assertEqual(len(sampled_opponents), config.player_num - 1)

    def test_reset_stores_opponent_versions_in_info(self):
        """reset()必须在info中存储对手版本信息"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        obs, info = env.reset(seed=42)

        self.assertIn("opponent_versions", info)
        self.assertIsInstance(info["opponent_versions"], list)
        self.assertEqual(len(info["opponent_versions"]), env.config.player_num - 1)

    def test_reset_opponent_versions_not_none(self):
        """对手版本不能为None"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        obs, info = env.reset(seed=42)

        for version in info["opponent_versions"]:
            self.assertIsNotNone(version)


class TestSelfPlayWrapperControlledPlayerIdx(unittest.TestCase):
    """测试controlled_player_idx循环行为"""

    def test_initial_controlled_player_idx_is_zero(self):
        """初始controlled_player_idx必须为0"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        self.assertEqual(env.controlled_player_idx, 0)

    def test_step_advances_controlled_player_idx(self):
        """step()必须推进controlled_player_idx"""
        from ai.self_play_wrapper import SelfPlayWrapper

        config = SGSConfig()
        env = SelfPlayWrapper(config)
        env.reset(seed=42)

        initial_idx = env.controlled_player_idx

        # Take a valid action (END_TURN = 1)
        obs, reward, done, truncated, info = env.step(1)

        # controlled_player_idx should cycle to next alive player
        self.assertNotEqual(env.controlled_player_idx, initial_idx)

    def test_controlled_player_idx_cycles_through_players(self):
        """controlled_player_idx必须在所有玩家间循环"""
        from ai.self_play_wrapper import SelfPlayWrapper

        config = SGSConfig(player_num=5)
        env = SelfPlayWrapper(config)
        env.reset(seed=42)

        seen_indices = {env.controlled_player_idx}

        for _ in range(config.player_num * 2):  # Cycle twice
            obs, reward, done, truncated, info = env.step(1)  # END_TURN
            seen_indices.add(env.controlled_player_idx)

            if done:
                break

        # Should see at least 2 different player indices
        self.assertGreaterEqual(len(seen_indices), 2)

    def test_controlled_player_idx_skips_dead_players(self):
        """controlled_player_idx必须跳过死亡玩家"""
        from ai.self_play_wrapper import SelfPlayWrapper

        config = SGSConfig(player_num=5)
        env = SelfPlayWrapper(config)
        env.reset(seed=42)

        # Simulate player death
        env.players[1].is_alive = False

        initial_idx = env.controlled_player_idx
        obs, reward, done, truncated, info = env.step(1)

        # Next player should not be dead player (index 1)
        if env.controlled_player_idx != initial_idx:
            self.assertTrue(env.players[env.controlled_player_idx].is_alive)


class TestSelfPlayWrapperObservationSpace(unittest.TestCase):
    """测试观察空间与SGSEnv一致"""

    def test_observation_space_matches_sgs_env(self):
        """observation_space必须与SGSEnv完全一致"""
        from ai.self_play_wrapper import SelfPlayWrapper

        config = SGSConfig()
        sgs_env = SGSEnv(config)
        wrapper_env = SelfPlayWrapper(config)

        self.assertEqual(wrapper_env.observation_space, sgs_env.observation_space)

    def test_observation_space_state_dim(self):
        """state维度约3000"""
        from ai.self_play_wrapper import SelfPlayWrapper

        config = SGSConfig(player_num=5)
        env = SelfPlayWrapper(config)

        state_space = env.observation_space["state"]
        self.assertEqual(state_space.shape[0], 2845)  # ~3000 dims

    def test_observation_space_contains_expected_keys(self):
        """观察空间必须包含所有预期键"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        expected_keys = {
            "state",
            "action_mask_type",
            "action_mask_card",
            "action_mask_target",
            "current_step",
            "skill_decision_type",
            "skill_decision_mask",
        }

        actual_keys = set(env.observation_space.spaces.keys())
        self.assertEqual(actual_keys, expected_keys)


class TestSelfPlayWrapperActionSpace(unittest.TestCase):
    """测试动作空间与SGSEnv一致"""

    def test_action_space_matches_sgs_env(self):
        """action_space必须与SGSEnv完全一致"""
        from ai.self_play_wrapper import SelfPlayWrapper

        config = SGSConfig()
        sgs_env = SGSEnv(config)
        wrapper_env = SelfPlayWrapper(config)

        self.assertEqual(wrapper_env.action_space, sgs_env.action_space)

    def test_action_space_is_discrete(self):
        """action_space必须是Discrete类型"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        self.assertIsInstance(env.action_space, spaces.Discrete)

    def test_action_space_dimension(self):
        """action_space维度必须为20"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        self.assertEqual(env.action_space.n, 20)


class TestSelfPlayWrapperOpponentObservation(unittest.TestCase):
    """测试对手观察注入"""

    def test_observation_includes_opponent_info(self):
        """观察必须包含对手信息"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        env.reset(seed=42)

        obs = env._get_observation()

        # Should have opponent-related fields or structure
        self.assertIsInstance(obs, dict)
        self.assertIn("state", obs)

    def test_opponent_observations_injected(self):
        """对手观察应该被注入到观察中"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        env.reset(seed=42)

        # Check if opponent observations are being managed
        self.assertTrue(hasattr(env, "_opponent_observations"))


class TestSelfPlayWrapperInterfaceCompatibility(unittest.TestCase):
    """测试与SGSEnv接口兼容性"""

    def test_has_reset_method(self):
        """必须有reset方法"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        self.assertTrue(hasattr(env, "reset"))
        self.assertTrue(callable(getattr(env, "reset")))

    def test_has_step_method(self):
        """必须有step方法"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        self.assertTrue(hasattr(env, "step"))
        self.assertTrue(callable(getattr(env, "step")))

    def test_reset_returns_tuple(self):
        """reset必须返回(obs, info)元组"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        result = env.reset(seed=42)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        obs, info = result
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(info, dict)

    def test_step_returns_tuple(self):
        """step必须返回(obs, reward, done, truncated, info)元组"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        env.reset(seed=42)

        result = env.step(1)  # END_TURN action

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

        obs, reward, done, truncated, info = result
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


class TestSelfPlayWrapperAgentPoolIntegration(unittest.TestCase):
    """测试与AgentPoolManager的集成"""

    def test_agent_pool_manager_is_callable(self):
        """AgentPoolManager必须有可调用方法"""
        from ai.self_play_wrapper import SelfPlayWrapper

        env = SelfPlayWrapper(SGSConfig())
        self.assertTrue(hasattr(env.agent_pool_manager, "sample_opponents"))
        self.assertTrue(callable(env.agent_pool_manager.sample_opponents))

    def test_sample_opponents_returns_correct_count(self):
        """sample_opponents返回正确数量的对手"""
        from ai.self_play_wrapper import SelfPlayWrapper

        config = SGSConfig(player_num=5)
        env = SelfPlayWrapper(config)

        opponents = env.agent_pool_manager.sample_opponents(config.player_num - 1)
        self.assertEqual(len(opponents), config.player_num - 1)


if __name__ == "__main__":
    unittest.main()
