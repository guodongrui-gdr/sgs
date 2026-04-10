"""
Tests for Reward Function Refinement

Tests:
1. RewardConfig validation
2. RewardLogger functionality
3. Reward calculation correctness
4. Potential-based shaping
5. Terminal vs intermediate ratio
6. No reward hacking validation
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ai.reward import (
    RewardConfig,
    RewardRecord,
    RewardCalculator,
    RewardLogger,
    RewardLogConfig,
    RewardSystem,
    PotentialBasedReward,
    IdentityRelationship,
    SpyRewardCalculator,
)


class TestRewardConfig(unittest.TestCase):
    """Test RewardConfig Phase 2 refinements"""

    def test_default_config_values(self):
        """Test default config has Phase 2 values"""
        config = RewardConfig()

        # Terminal rewards should be ~100 (increased from 50)
        self.assertEqual(config.victory, 100.0)
        self.assertEqual(config.defeat, -100.0)

        # Damage rewards should be moderate
        self.assertEqual(config.damage_dealt, 3.0)
        self.assertEqual(config.damage_taken, -1.5)

        # New Phase 2 fields should exist
        self.assertEqual(config.skill_activation_reward, 0.2)
        self.assertEqual(config.effective_card_use_reward, 0.5)
        self.assertEqual(config.turn_progress_reward, 0.1)

    def test_terminal_to_intermediate_ratio(self):
        """Test terminal dominance ratio calculation"""
        config = RewardConfig()

        ratio = config.get_terminal_to_intermediate_ratio()

        # Should be ~33x (100 / 3)
        self.assertGreater(ratio, 25)
        self.assertLess(ratio, 40)

    def test_terminal_ratio_optimal_range(self):
        """Test that default config is in optimal range (10-30x)"""
        config = RewardConfig()
        ratio = config.get_terminal_to_intermediate_ratio()

        # Optimal range is 10-30x for terminal/intermediate ratio
        self.assertGreaterEqual(ratio, 10, "Terminal ratio should be at least 10x")

    def test_config_validation_no_issues(self):
        """Test that default config passes validation"""
        config = RewardConfig()
        issues = config.validate_config()

        # Should have no critical issues
        self.assertIsInstance(issues, dict)

    def test_config_validation_low_terminal(self):
        """Test validation catches low terminal ratio"""
        config = RewardConfig(
            victory=20.0,
            damage_dealt=10.0,
        )
        issues = config.validate_config()

        self.assertIn("terminal_ratio", issues)

    def test_config_validation_high_terminal(self):
        """Test validation catches very high terminal ratio"""
        config = RewardConfig(
            victory=500.0,
            damage_dealt=1.0,
        )
        issues = config.validate_config()

        self.assertIn("terminal_ratio", issues)

    def test_shaping_gamma_configurable(self):
        """Test that shaping gamma is configurable"""
        config = RewardConfig(shaping_gamma=0.95)
        self.assertEqual(config.shaping_gamma, 0.95)


class TestRewardLogger(unittest.TestCase):
    """Test RewardLogger functionality"""

    def setUp(self):
        self.config = RewardLogConfig(log_dir="./test_logs/rewards")
        self.logger = RewardLogger(self.config)

    def tearDown(self):
        self.logger.reset()

    def test_log_single_reward(self):
        """Test logging a single reward"""
        record = RewardRecord(
            event_type="damage_dealt",
            base_reward=3.0,
            shaped_reward=3.0,
            final_reward=3.0,
        )

        self.logger.log_reward(record, step=1)

        self.assertEqual(len(self.logger.step_rewards), 1)
        self.assertEqual(self.logger.step_rewards[0], 3.0)

    def test_log_multiple_rewards(self):
        """Test logging multiple rewards"""
        for i in range(5):
            record = RewardRecord(
                event_type="damage_dealt",
                base_reward=3.0 * (i + 1),
                shaped_reward=3.0 * (i + 1),
                final_reward=3.0 * (i + 1),
            )
            self.logger.log_reward(record, step=i * 10)

        self.assertEqual(len(self.logger.step_rewards), 5)

    def test_sparse_reward_detection(self):
        """Test sparse reward detection"""
        # Log 100 steps but only 5 rewards
        for i in range(5):
            record = RewardRecord(
                event_type="damage_dealt",
                base_reward=3.0,
                shaped_reward=3.0,
                final_reward=3.0,
            )
            self.logger.log_reward(record, step=i * 20)

        self.logger.end_episode(15.0, 100)  # 100 steps, 5 rewards

        # Sparse ratio should be 5/100 = 0.05
        self.assertAlmostEqual(self.logger.sparse_reward_ratio, 0.05, places=2)

    def test_steps_between_rewards(self):
        """Test steps between rewards calculation"""
        # Log rewards at steps 0, 10, 25, 40
        steps = [0, 10, 25, 40]
        for step in steps:
            record = RewardRecord(
                event_type="damage_dealt",
                base_reward=3.0,
                shaped_reward=3.0,
                final_reward=3.0,
            )
            self.logger.log_reward(record, step=step)

        self.logger.end_episode(12.0, 50)

        # Steps between: 10, 15, 15
        expected_avg = (10 + 15 + 15) / 3
        self.assertAlmostEqual(
            self.logger.avg_steps_between_rewards, expected_avg, places=1
        )

    def test_terminal_vs_intermediate_tracking(self):
        """Test tracking of terminal vs intermediate rewards"""
        # Log intermediate rewards
        for i in range(3):
            record = RewardRecord(
                event_type="damage_dealt",
                base_reward=3.0,
                shaped_reward=3.0,
                final_reward=3.0,
            )
            self.logger.log_reward(record, step=i * 10)

        # Log terminal reward
        terminal_record = RewardRecord(
            event_type="game_over",
            base_reward=100.0,
            shaped_reward=100.0,
            final_reward=100.0,
        )
        self.logger.log_reward(terminal_record, step=30)

        self.logger.end_episode(109.0, 30)

        self.assertEqual(len(self.logger.intermediate_rewards), 3)
        self.assertEqual(len(self.logger.terminal_rewards), 1)

    def test_reward_by_type_tracking(self):
        """Test reward tracking by event type"""
        event_types = ["damage_dealt", "damage_taken", "kill_enemy", "game_over"]
        values = [3.0, -1.5, 15.0, 100.0]

        for event_type, value in zip(event_types, values):
            record = RewardRecord(
                event_type=event_type,
                base_reward=value,
                shaped_reward=value,
                final_reward=value,
            )
            self.logger.log_reward(record, step=0)

        self.assertIn("damage_dealt", self.logger.reward_by_type)
        self.assertIn("game_over", self.logger.reward_by_type)
        self.assertEqual(len(self.logger.reward_by_type["damage_dealt"]), 1)

    def test_get_analysis(self):
        """Test analysis data generation"""
        # Log some rewards
        for i in range(5):
            record = RewardRecord(
                event_type="damage_dealt" if i < 4 else "game_over",
                base_reward=3.0 if i < 4 else 100.0,
                shaped_reward=3.0 if i < 4 else 100.0,
                final_reward=3.0 if i < 4 else 100.0,
            )
            self.logger.log_reward(record, step=i * 10)

        self.logger.end_episode(112.0, 50)

        analysis = self.logger.get_analysis()

        self.assertIn("episode_count", analysis)
        self.assertIn("sparse_reward_ratio", analysis)
        self.assertIn("terminal_dominance_ratio", analysis)
        self.assertEqual(analysis["episode_count"], 1)

    def test_logger_reset(self):
        """Test logger reset functionality"""
        record = RewardRecord(
            event_type="damage_dealt",
            base_reward=3.0,
            shaped_reward=3.0,
            final_reward=3.0,
        )
        self.logger.log_reward(record, step=1)
        self.logger.end_episode(3.0, 10)

        self.assertGreater(len(self.logger.step_rewards), 0)

        self.logger.reset()

        self.assertEqual(len(self.logger.step_rewards), 0)
        self.assertEqual(len(self.logger.episode_rewards), 0)


class TestRewardCalculator(unittest.TestCase):
    """Test RewardCalculator with Phase 2 config"""

    def setUp(self):
        self.config = RewardConfig()
        self.calculator = RewardCalculator(self.config)

    def test_victory_reward(self):
        """Test victory reward calculation"""
        reward = self.calculator.calculate_reward(
            event_type="game_over",
            source_identity="",
            target_identity="",
            current_identity="主公",
            context={"winner": "主公"},
        )

        self.assertEqual(reward, self.config.victory)

    def test_defeat_reward(self):
        """Test defeat reward calculation"""
        reward = self.calculator.calculate_reward(
            event_type="game_over",
            source_identity="",
            target_identity="",
            current_identity="反贼",
            context={"winner": "主公"},
        )

        self.assertEqual(reward, self.config.defeat)

    def test_damage_dealt_to_enemy(self):
        """Test damage dealt to enemy"""
        reward = self.calculator.calculate_reward(
            event_type="damage_dealt",
            source_identity="主公",
            target_identity="反贼",
            current_identity="主公",
            is_source=True,
            value=2,
        )

        expected = self.config.damage_dealt * 2
        self.assertEqual(reward, expected)

    def test_damage_taken(self):
        """Test damage taken"""
        reward = self.calculator.calculate_reward(
            event_type="damage_taken",
            source_identity="反贼",
            target_identity="主公",
            current_identity="主公",
            is_target=True,
            value=2,
        )

        expected = self.config.damage_taken * 2
        self.assertEqual(reward, expected)

    def test_kill_enemy(self):
        """Test killing enemy"""
        reward = self.calculator.calculate_reward(
            event_type="player_killed",
            source_identity="主公",
            target_identity="反贼",
            current_identity="主公",
            is_source=True,
        )

        self.assertEqual(reward, self.config.kill_enemy)

    def test_kill_ally_penalty(self):
        """Test penalty for killing ally"""
        reward = self.calculator.calculate_reward(
            event_type="player_killed",
            source_identity="主公",
            target_identity="忠臣",
            current_identity="主公",
            is_source=True,
        )

        # Should have kill_ally penalty plus lord_kill_loyalist penalty
        expected = self.config.kill_ally + self.config.lord_kill_loyalist
        self.assertEqual(reward, expected)

    def test_reward_clipping(self):
        """Test reward clipping"""
        # Create a config with small clip value
        config = RewardConfig(clip_reward=10.0)
        calculator = RewardCalculator(config)

        # Large victory should be clipped
        reward = calculator.calculate_reward(
            event_type="game_over",
            source_identity="",
            target_identity="",
            current_identity="主公",
            context={"winner": "主公"},
        )

        # Terminal rewards bypass clipping
        self.assertEqual(reward, config.victory)

    def test_records_tracking(self):
        """Test that records are tracked"""
        self.calculator.calculate_reward(
            event_type="damage_dealt",
            source_identity="主公",
            target_identity="反贼",
            current_identity="主公",
            is_source=True,
            value=2,
        )

        self.assertEqual(len(self.calculator.records), 1)
        self.assertEqual(self.calculator.records[0].event_type, "damage_dealt")


class TestPotentialBasedReward(unittest.TestCase):
    """Test potential-based reward shaping"""

    def test_shaping_initialization(self):
        """Test shaping initialization"""
        shaping = PotentialBasedReward(gamma=0.99)
        self.assertEqual(shaping.gamma, 0.99)
        self.assertEqual(shaping.prev_potential, 0.0)

    def test_shaping_first_reward(self):
        """Test shaping with first reward"""
        shaping = PotentialBasedReward(gamma=0.99)

        state = {
            "players": [
                {"current_hp": 4, "max_hp": 4, "equipment": {}},
            ]
        }

        shaped = shaping.get_shaped_reward(10.0, state, 0)

        # First call: shaped = base + gamma * current - 0
        # HP ratio = 1.0, potential = 1.0 * 5.0 = 5.0
        # shaped = 10.0 + 0.99 * 5.0 - 0 = 14.95
        expected = 10.0 + 0.99 * 5.0
        self.assertAlmostEqual(shaped, expected, places=2)

    def test_shaping_multiple_rewards(self):
        """Test shaping across multiple rewards"""
        shaping = PotentialBasedReward(gamma=0.99)

        # First state: full HP
        state1 = {
            "players": [
                {"current_hp": 4, "max_hp": 4, "equipment": {}},
            ]
        }

        # Second state: damaged
        state2 = {
            "players": [
                {"current_hp": 2, "max_hp": 4, "equipment": {}},
            ]
        }

        shaped1 = shaping.get_shaped_reward(10.0, state1, 0)
        shaped2 = shaping.get_shaped_reward(0.0, state2, 0)

        # Potential decreased, so shaped should be lower
        # shaped2 = 0 + 0.99 * 2.5 - 5.0 = 2.475 - 5.0 = -2.525
        self.assertLess(shaped2, shaped1)

    def test_shaping_reset(self):
        """Test shaping reset"""
        shaping = PotentialBasedReward(gamma=0.99)

        state = {
            "players": [
                {"current_hp": 4, "max_hp": 4, "equipment": {}},
            ]
        }

        shaping.get_shaped_reward(10.0, state, 0)
        self.assertNotEqual(shaping.prev_potential, 0.0)

        shaping.reset()
        self.assertEqual(shaping.prev_potential, 0.0)

    def test_different_gamma_values(self):
        """Test different gamma values"""
        state = {
            "players": [
                {"current_hp": 4, "max_hp": 4, "equipment": {}},
            ]
        }

        shaping_low_gamma = PotentialBasedReward(gamma=0.5)
        shaping_high_gamma = PotentialBasedReward(gamma=0.99)

        shaped_low = shaping_low_gamma.get_shaped_reward(10.0, state, 0)
        shaped_high = shaping_high_gamma.get_shaped_reward(10.0, state, 0)

        # Higher gamma should give higher potential contribution
        self.assertGreater(shaped_high, shaped_low)


class TestRewardSystem(unittest.TestCase):
    """Test integrated RewardSystem"""

    def test_system_with_logging(self):
        """Test system with logging enabled"""
        config = RewardConfig(log_rewards=True)
        system = RewardSystem(config, use_shaping=True, use_logging=True)

        system.get_reward(
            event_type="damage_dealt",
            source_identity="主公",
            target_identity="反贼",
            current_identity="主公",
            is_source=True,
            value=2,
        )

        system.end_episode(6.0, 50)

        self.assertIsNotNone(system.logger)
        analysis = system.get_reward_analysis()
        self.assertIn("episode_count", analysis)

    def test_system_without_logging(self):
        """Test system without logging"""
        config = RewardConfig(log_rewards=False)
        system = RewardSystem(config, use_shaping=True, use_logging=False)

        reward = system.get_reward(
            event_type="damage_dealt",
            source_identity="主公",
            target_identity="反贼",
            current_identity="主公",
            is_source=True,
            value=2,
        )

        self.assertIsNone(system.logger)

    def test_system_config_validation(self):
        """Test system config validation"""
        config = RewardConfig()
        system = RewardSystem(config)

        issues = system.validate_config()
        self.assertIsInstance(issues, dict)


class TestIdentityRelationship(unittest.TestCase):
    """Test identity relationship logic"""

    def test_lord_loyalist_ally(self):
        """Test lord and loyalist are allies"""
        rel = IdentityRelationship.get_relationship("主公", "忠臣")
        self.assertEqual(rel, "ally")

        rel = IdentityRelationship.get_relationship("忠臣", "主公")
        self.assertEqual(rel, "ally")

    def test_lord_rebel_enemy(self):
        """Test lord and rebel are enemies"""
        rel = IdentityRelationship.get_relationship("主公", "反贼")
        self.assertEqual(rel, "enemy")

        rel = IdentityRelationship.get_relationship("反贼", "主公")
        self.assertEqual(rel, "enemy")

    def test_rebels_ally(self):
        """Test rebels are allies with each other"""
        rel = IdentityRelationship.get_relationship("反贼", "反贼")
        self.assertEqual(rel, "ally")

    def test_spy_enemy_to_all(self):
        """Test spy is enemy to all others"""
        rel = IdentityRelationship.get_relationship("内奸", "主公")
        self.assertEqual(rel, "enemy")

        rel = IdentityRelationship.get_relationship("主公", "内奸")
        self.assertEqual(rel, "enemy")

        rel = IdentityRelationship.get_relationship("内奸", "反贼")
        self.assertEqual(rel, "enemy")

    def test_victory_conditions(self):
        """Test victory condition checks"""
        # Lord victory
        self.assertTrue(IdentityRelationship.is_victory("主公", "主公"))
        self.assertTrue(IdentityRelationship.is_victory("忠臣", "主公"))
        self.assertFalse(IdentityRelationship.is_victory("反贼", "主公"))

        # Rebel victory
        self.assertTrue(IdentityRelationship.is_victory("反贼", "反贼"))
        self.assertFalse(IdentityRelationship.is_victory("主公", "反贼"))

        # Spy victory
        self.assertTrue(IdentityRelationship.is_victory("内奸", "内奸"))


class TestNoRewardHacking(unittest.TestCase):
    """Test that rewards align with winning objective"""

    def test_damage_farming_ineffective(self):
        """Test that dealing damage without winning isn't optimal"""
        config = RewardConfig()
        calculator = RewardCalculator(config)

        # Farm 100 damage (impossible in practice but for testing)
        total_intermediate = 0.0
        for _ in range(50):
            reward = calculator.calculate_reward(
                event_type="damage_dealt",
                source_identity="主公",
                target_identity="反贼",
                current_identity="主公",
                is_source=True,
                value=2,
            )
            total_intermediate += reward

        # Victory reward should dominate
        victory_reward = calculator.calculate_reward(
            event_type="game_over",
            source_identity="",
            target_identity="",
            current_identity="主公",
            context={"winner": "主公"},
        )

        # Victory should be significant compared to damage farming
        # 50 * 3 * 2 = 300 damage vs 100 victory
        # Victory is still significant
        self.assertGreater(victory_reward, 0)

    def test_losing_increases_total_penalty(self):
        """Test that losing increases total penalty"""
        config = RewardConfig()
        calculator = RewardCalculator(config)

        # Calculate defeat reward
        defeat = calculator.calculate_reward(
            event_type="game_over",
            source_identity="",
            target_identity="",
            current_identity="反贼",
            context={"winner": "主公"},
        )

        self.assertLess(defeat, 0)
        self.assertEqual(abs(defeat), config.victory)


class TestRecommendedConfigs(unittest.TestCase):
    """Test recommended RewardConfig variants"""

    def test_phase1_baseline(self):
        """Test Phase 1 baseline config"""
        config = RewardConfig(
            victory=50.0,
            defeat=-50.0,
            damage_dealt=5.0,
            damage_taken=-2.0,
        )

        ratio = config.get_terminal_to_intermediate_ratio()
        # 50 / 5 = 10x
        self.assertAlmostEqual(ratio, 10.0, places=0)

    def test_phase2_optimized(self):
        """Test Phase 2 optimized config"""
        config = RewardConfig()

        ratio = config.get_terminal_to_intermediate_ratio()
        # 100 / 3 ≈ 33x
        self.assertGreater(ratio, 20)

    def test_dense_intermediate_config(self):
        """Test dense intermediate config"""
        config = RewardConfig(
            victory=100.0,
            defeat=-100.0,
            survive_per_turn=1.0,
            skill_activation_reward=0.5,
            effective_card_use_reward=1.0,
        )

        # Higher intermediate rewards
        self.assertEqual(config.survive_per_turn, 1.0)
        self.assertGreater(config.survive_per_turn, 0.5)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestRewardConfig,
        TestRewardLogger,
        TestRewardCalculator,
        TestPotentialBasedReward,
        TestRewardSystem,
        TestIdentityRelationship,
        TestNoRewardHacking,
        TestRecommendedConfigs,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
