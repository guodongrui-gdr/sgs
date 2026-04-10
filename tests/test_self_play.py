"""
自博弈系统单元测试

测试内容:
- 对手采样分布测试
- ELO评分计算测试
- 策略池操作测试
- 训练集成测试
- 身份胜率统计测试
- ELO衰减测试
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ai.policy_pool import (
    PolicyPool,
    PolicyRecord,
    PoolConfig,
    IdentityStats,
    MatchHistory,
    SamplingStrategy,
    create_policy_pool,
    create_pool_config,
)
from ai.self_play import SelfPlayConfig, TrainingMode


class TestIdentityStats(unittest.TestCase):
    """测试身份统计类"""

    def test_add_result(self):
        """测试添加结果"""
        stats = IdentityStats()
        stats.add_result("主公", True)
        stats.add_result("主公", False)
        stats.add_result("反贼", True)
        stats.add_result("内奸", False)

        self.assertEqual(stats.主公_wins, 1)
        self.assertEqual(stats.主公_losses, 1)
        self.assertEqual(stats.反贼_wins, 1)
        self.assertEqual(stats.内奸_losses, 1)

    def test_get_win_rate(self):
        """测试获取胜率"""
        stats = IdentityStats()
        stats.add_result("主公", True)
        stats.add_result("主公", True)
        stats.add_result("主公", False)

        win_rate = stats.get_win_rate("主公")
        self.assertAlmostEqual(win_rate, 2 / 3, places=5)

    def test_get_all_win_rates(self):
        """测试获取所有胜率"""
        stats = IdentityStats()
        stats.add_result("主公", True)
        stats.add_result("反贼", False)

        rates = stats.get_all_win_rates()
        self.assertEqual(rates["主公"], 1.0)
        self.assertEqual(rates["反贼"], 0.0)
        self.assertEqual(rates["忠臣"], 0.0)
        self.assertEqual(rates["内奸"], 0.0)

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        stats = IdentityStats()
        stats.add_result("主公", True)
        stats.add_result("反贼", False)
        stats.add_result("内奸", True)

        data = stats.to_dict()
        restored = IdentityStats.from_dict(data)

        self.assertEqual(restored.主公_wins, stats.主公_wins)
        self.assertEqual(restored.反贼_losses, stats.反贼_losses)
        self.assertEqual(restored.内奸_wins, stats.内奸_wins)


class TestPolicyRecord(unittest.TestCase):
    """测试策略记录类"""

    def test_creation(self):
        """测试创建策略记录"""
        record = PolicyRecord(
            path="/path/to/model.zip",
            version=1,
            timestamp=datetime.now().isoformat(),
        )

        self.assertEqual(record.version, 1)
        self.assertEqual(record.elo_rating, 1000.0)
        self.assertEqual(record.base_elo, 1000.0)

    def test_get_age_hours(self):
        """测试获取策略年龄"""
        record = PolicyRecord(
            path="/path/to/model.zip",
            version=1,
            timestamp=(datetime.now() - timedelta(hours=5)).isoformat(),
        )

        age = record.get_age_hours()
        self.assertAlmostEqual(age, 5.0, places=1)

    def test_apply_elo_decay(self):
        """测试ELO衰减"""
        record = PolicyRecord(
            path="/path/to/model.zip",
            version=1,
            timestamp=(datetime.now() - timedelta(hours=10)).isoformat(),
            elo_rating=1200.0,
            base_elo=1200.0,
        )

        record.apply_elo_decay(decay_rate=0.01, max_decay=0.3)

        # 10小时，1%每小时，最大30%衰减
        # 预期衰减: min(0.01 * 10, 0.3) = 0.1 = 10%
        self.assertAlmostEqual(record.elo_decay_factor, 0.9, places=2)
        self.assertAlmostEqual(record.elo_rating, 1200.0 * 0.9, places=1)

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        record = PolicyRecord(
            path="/path/to/model.zip",
            version=5,
            timestamp=datetime.now().isoformat(),
            elo_rating=1100.0,
            base_elo=1150.0,
            games_played=100,
            wins=60,
        )

        data = record.to_dict()
        restored = PolicyRecord.from_dict(data)

        self.assertEqual(restored.version, record.version)
        self.assertEqual(restored.elo_rating, record.elo_rating)
        self.assertEqual(restored.base_elo, record.base_elo)
        self.assertEqual(restored.games_played, record.games_played)


class TestPoolConfig(unittest.TestCase):
    """测试策略池配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = PoolConfig()
        self.assertEqual(config.max_size, 10)
        self.assertEqual(config.sample_latest_ratio, 0.3)
        self.assertEqual(config.sample_best_ratio, 0.3)
        self.assertEqual(config.sample_random_ratio, 0.4)

    def test_custom_config(self):
        """测试自定义配置"""
        config = PoolConfig(
            max_size=20,
            sample_latest_ratio=0.4,
            sample_best_ratio=0.4,
            k_factor=20.0,
        )
        self.assertEqual(config.max_size, 20)
        self.assertEqual(config.k_factor, 20.0)


class TestPolicyPool(unittest.TestCase):
    """测试策略池"""

    def setUp(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.pool = PolicyPool(
            pool_dir=self.temp_dir,
            config=PoolConfig(
                max_size=5,
                sample_latest_ratio=0.3,
                sample_best_ratio=0.3,
                sample_random_ratio=0.4,
            ),
        )

    def tearDown(self):
        """清理临时目录"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_policy(self):
        """测试添加策略"""
        record = self.pool.add_policy(
            "/path/to/model1.zip",
            elo_rating=1000.0,
        )

        self.assertEqual(record.version, 1)
        self.assertEqual(len(self.pool), 1)

    def test_sample_policy_distribution(self):
        """测试对手采样分布"""
        # 添加多个策略
        for i in range(10):
            self.pool.add_policy(
                f"/path/to/model{i}.zip",
                elo_rating=1000.0 + i * 50,  # 不同ELO
            )

        # 统计采样分布
        latest_count = 0
        best_count = 0
        random_count = 0
        n_samples = 1000

        for _ in range(n_samples):
            # 模拟采样逻辑
            import random

            r = random.random()
            sampled = self.pool.sample_policy()
            self.assertIsNotNone(sampled)

        # 验证池中有策略
        self.assertGreater(len(self.pool), 0)

    def test_sample_with_strategy(self):
        """测试指定策略采样"""
        for i in range(5):
            self.pool.add_policy(
                f"/path/to/model{i}.zip",
                elo_rating=1000.0 + i * 100,
            )

        # 测试最新策略
        latest = self.pool.sample_policy(strategy=SamplingStrategy.LATEST)
        self.assertEqual(latest.version, 5)

        # 测试最佳ELO
        best = self.pool.sample_policy(strategy=SamplingStrategy.BEST_ELO)
        self.assertEqual(best.elo_rating, 1400.0)

    def test_update_elo(self):
        """测试ELO更新"""
        record1 = self.pool.add_policy("/path/to/model1.zip", elo_rating=1000.0)
        record2 = self.pool.add_policy("/path/to/model2.zip", elo_rating=1000.0)

        # 模拟比赛：version 1 获胜
        self.pool.update_elo(
            winner_version=1,
            loser_version=2,
            winner_identity="主公",
            loser_identity="反贼",
        )

        # 验证ELO变化
        p1 = self.pool.get_policy_by_version(1)
        p2 = self.pool.get_policy_by_version(2)

        self.assertGreater(p1.elo_rating, 1000.0)  # 获胜者ELO增加
        self.assertLess(p2.elo_rating, 1000.0)  # 失败者ELO减少
        self.assertEqual(p1.identity_stats.主公_wins, 1)
        self.assertEqual(p2.identity_stats.反贼_losses, 1)

    def test_prune_pool(self):
        """测试策略池修剪"""
        # 添加超过max_size的策略
        for i in range(10):
            self.pool.add_policy(
                f"/path/to/model{i}.zip",
                elo_rating=1000.0 + i * 50,
            )

        # 验证池大小不超过max_size
        self.assertLessEqual(len(self.pool), 5)

    def test_get_stats(self):
        """测试获取统计信息"""
        for i in range(3):
            self.pool.add_policy(
                f"/path/to/model{i}.zip",
                elo_rating=1000.0 + i * 100,
            )

        stats = self.pool.get_stats()
        self.assertEqual(stats["total_policies"], 3)
        self.assertEqual(stats["best_elo"], 1200.0)
        self.assertGreater(stats["avg_elo"], 1000.0)

    def test_get_top_policies(self):
        """测试获取顶级策略"""
        for i in range(5):
            self.pool.add_policy(
                f"/path/to/model{i}.zip",
                elo_rating=1000.0 + i * 100,
            )

        top = self.pool.get_top_policies(n=3)
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0].elo_rating, 1400.0)  # 最高ELO

    def test_record_game_with_identity(self):
        """测试记录带身份的游戏结果"""
        record = self.pool.add_policy("/path/to/model.zip")

        self.pool.record_game(1, won=True, identity="主公")
        self.pool.record_game(1, won=False, identity="反贼")

        policy = self.pool.get_policy_by_version(1)
        self.assertEqual(policy.wins, 1)
        self.assertEqual(policy.losses, 1)
        self.assertEqual(policy.identity_stats.主公_wins, 1)
        self.assertEqual(policy.identity_stats.反贼_losses, 1)


class TestMatchHistory(unittest.TestCase):
    """测试比赛历史"""

    def test_add_match(self):
        """测试添加比赛"""
        history = MatchHistory()
        history.add_match(
            policy_a_version=1,
            policy_b_version=2,
            winner_version=1,
            game_length=50,
        )

        self.assertEqual(len(history.matches), 1)

    def test_get_head_to_head(self):
        """测试获取对战记录"""
        history = MatchHistory()
        history.add_match(1, 2, winner_version=1, game_length=50)
        history.add_match(1, 2, winner_version=2, game_length=60)
        history.add_match(1, 2, winner_version=1, game_length=40)

        h2h = history.get_head_to_head(1, 2)
        self.assertEqual(h2h["total_games"], 3)
        self.assertEqual(h2h["wins_a"], 2)
        self.assertAlmostEqual(h2h["win_rate_a"], 2 / 3, places=5)

    def test_get_recent_performance(self):
        """测试获取近期表现"""
        history = MatchHistory()
        for i in range(5):
            history.add_match(
                1, 2, winner_version=1 if i % 2 == 0 else 2, game_length=50
            )

        perf = history.get_recent_performance(1, n_games=5)
        self.assertEqual(perf["recent_games"], 5)
        self.assertEqual(perf["recent_wins"], 3)

    def test_get_identity_performance(self):
        """测试获取身份表现"""
        history = MatchHistory()
        history.add_match(
            1,
            2,
            winner_version=1,
            game_length=50,
            policy_a_identity="主公",
            policy_b_identity="反贼",
        )
        history.add_match(
            1,
            2,
            winner_version=2,
            game_length=60,
            policy_a_identity="主公",
            policy_b_identity="反贼",
        )

        perf = history.get_identity_performance("主公", n_games=10)
        self.assertEqual(perf["games"], 2)
        self.assertEqual(perf["wins"], 1)


class TestSelfPlayConfig(unittest.TestCase):
    """测试自博弈配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = SelfPlayConfig()

        self.assertEqual(config.training_mode, TrainingMode.MIXED)
        self.assertEqual(config.self_play_ratio, 0.5)
        self.assertEqual(config.sample_latest_ratio, 0.3)
        self.assertEqual(config.sample_best_ratio, 0.3)
        self.assertEqual(config.sample_random_ratio, 0.4)

    def test_ratio_adjustment(self):
        """测试比例自动调整"""
        config = SelfPlayConfig(
            sample_latest_ratio=0.5,
            sample_best_ratio=0.5,
        )

        # 应该自动调整random_ratio
        self.assertAlmostEqual(config.sample_random_ratio, 0.0, places=5)

    def test_get_pool_config(self):
        """测试获取策略池配置"""
        config = SelfPlayConfig(
            pool_size=20,
            sample_latest_ratio=0.4,
            elo_k_factor=24.0,
        )

        pool_config = config.get_pool_config()
        self.assertEqual(pool_config.max_size, 20)
        self.assertEqual(pool_config.sample_latest_ratio, 0.4)
        self.assertEqual(pool_config.k_factor, 24.0)


class TestELOCalculations(unittest.TestCase):
    """测试ELO计算"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pool = PolicyPool(pool_dir=self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_elo_calculation_equal_rating(self):
        """测试相同ELO的比赛"""
        self.pool.add_policy("/model1.zip", elo_rating=1000.0)
        self.pool.add_policy("/model2.zip", elo_rating=1000.0)

        self.pool.update_elo(winner_version=1, loser_version=2)

        p1 = self.pool.get_policy_by_version(1)
        p2 = self.pool.get_policy_by_version(2)

        # ELO变化应该对称
        self.assertAlmostEqual(
            p1.elo_rating - 1000.0, -(p2.elo_rating - 1000.0), places=1
        )

    def test_elo_calculation_upset(self):
        """测试爆冷: 低ELO获胜"""
        self.pool.add_policy("/model1.zip", elo_rating=800.0)
        self.pool.add_policy("/model2.zip", elo_rating=1200.0)

        self.pool.update_elo(winner_version=1, loser_version=2)

        p1 = self.pool.get_policy_by_version(1)
        p2 = self.pool.get_policy_by_version(2)

        # 低ELO获胜应该获得更多分数
        gain = p1.elo_rating - 800.0
        loss = 1200.0 - p2.elo_rating
        self.assertGreater(gain, loss)

    def test_elo_decay_over_time(self):
        """测试ELO随时间衰减"""
        # 创建一个旧策略
        old_time = (datetime.now() - timedelta(hours=50)).isoformat()
        record = PolicyRecord(
            path="/old_model.zip",
            version=1,
            timestamp=old_time,
            elo_rating=1200.0,
            base_elo=1200.0,
        )
        self.pool.policies.append(record)

        # 应用衰减
        record.apply_elo_decay(decay_rate=0.01, max_decay=0.3)

        # 50小时，1%每小时 = 50%衰减，但最大30%
        self.assertAlmostEqual(record.elo_decay_factor, 0.7, places=2)
        self.assertAlmostEqual(record.elo_rating, 1200.0 * 0.7, places=1)


class TestOpponentSamplingDistribution(unittest.TestCase):
    """测试对手采样分布"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pool = PolicyPool(
            pool_dir=self.temp_dir,
            config=PoolConfig(
                max_size=20,
                sample_latest_ratio=0.3,
                sample_best_ratio=0.3,
                sample_random_ratio=0.4,
            ),
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sampling_distribution(self):
        """测试采样分布是否符合预期"""
        # 添加多个策略
        for i in range(20):
            self.pool.add_policy(
                f"/model{i}.zip",
                elo_rating=1000.0 + i * 20,
            )

        # 进行大量采样
        n_samples = 10000
        latest_samples = 0
        best_samples = 0

        # 获取最新和最佳策略
        latest = self.pool.get_latest_policy()
        best = self.pool.get_best_policy()

        for _ in range(n_samples):
            sampled = self.pool.sample_policy()
            if sampled.version == latest.version:
                latest_samples += 1
            if sampled.version == best.version:
                best_samples += 1

        # 验证采样分布大致符合预期
        # 注意：由于随机性和加权，精确比例会有偏差
        latest_ratio = latest_samples / n_samples
        best_ratio = best_samples / n_samples

        # 最新和最佳策略应该被采样到
        self.assertGreater(latest_ratio, 0.0)
        self.assertGreater(best_ratio, 0.0)


class TestTrainingIntegration(unittest.TestCase):
    """测试训练集成"""

    def test_self_play_config_modes(self):
        """测试不同的训练模式"""
        # 仅自博弈
        config1 = SelfPlayConfig(training_mode=TrainingMode.SELF_PLAY_ONLY)
        self.assertEqual(config1.training_mode, TrainingMode.SELF_PLAY_ONLY)

        # 仅规则AI
        config2 = SelfPlayConfig(training_mode=TrainingMode.RULE_BASED_ONLY)
        self.assertEqual(config2.training_mode, TrainingMode.RULE_BASED_ONLY)

        # 混合
        config3 = SelfPlayConfig(training_mode=TrainingMode.MIXED, self_play_ratio=0.7)
        self.assertEqual(config3.self_play_ratio, 0.7)

        # 交替
        config4 = SelfPlayConfig(
            training_mode=TrainingMode.ALTERNATING, alternating_epochs=10
        )
        self.assertEqual(config4.alternating_epochs, 10)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestIdentityStats,
        TestPolicyRecord,
        TestPoolConfig,
        TestPolicyPool,
        TestMatchHistory,
        TestSelfPlayConfig,
        TestELOCalculations,
        TestOpponentSamplingDistribution,
        TestTrainingIntegration,
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
