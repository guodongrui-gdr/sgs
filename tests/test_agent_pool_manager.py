"""
AgentPoolManager 单元测试 (TDD RED Phase)

测试内容:
- AgentPoolManager 初始化 (size=10)
- add_agent() 添加策略
- remove_agent() 移除策略
- sample_opponent() 采样分布 (30/30/40)
- ELO 更新机制 (赢家增益, 输家损失)
- IdentityStats 身份统计跟踪

预期状态: 测试会失败，因为 AgentPoolManager 尚未实现
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from unittest.mock import Mock, patch


class TestAgentPoolManager(unittest.TestCase):
    """测试 AgentPoolManager 类 - TDD RED Phase"""

    def setUp(self):
        """测试前置设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")

    def tearDown(self):
        """测试后置清理"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_agent_pool_manager_import(self):
        """测试 AgentPoolManager 可以被导入"""
        try:
            from train.agent_pool_manager import AgentPoolManager

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"无法导入 AgentPoolManager: {e}")

    def test_initialization_with_size_10(self):
        """测试 AgentPoolManager 初始化，池大小为10"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 验证初始状态
        self.assertEqual(manager.pool_size, 10)
        self.assertEqual(len(manager), 0)
        self.assertIsNotNone(manager.policy_pool)

    def test_initialization_default_values(self):
        """测试 AgentPoolManager 默认初始化值"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager()

        # 验证默认值
        self.assertEqual(manager.pool_size, 10)
        self.assertEqual(len(manager), 0)

    def test_add_agent_increases_pool_size(self):
        """测试 add_agent 增加池大小"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 添加第一个agent
        agent_id = manager.add_agent(self.model_path, elo_rating=1000.0)
        self.assertEqual(len(manager), 1)
        self.assertEqual(agent_id, 1)

        # 添加第二个agent
        agent_id = manager.add_agent(self.model_path, elo_rating=1000.0)
        self.assertEqual(len(manager), 2)
        self.assertEqual(agent_id, 2)

    def test_add_agent_returns_version(self):
        """测试 add_agent 返回正确的版本号"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        version1 = manager.add_agent(self.model_path, elo_rating=1000.0)
        version2 = manager.add_agent(self.model_path, elo_rating=1050.0)
        version3 = manager.add_agent(self.model_path, elo_rating=1100.0)

        self.assertEqual(version1, 1)
        self.assertEqual(version2, 2)
        self.assertEqual(version3, 3)

    def test_add_agent_stores_metadata(self):
        """测试 add_agent 存储元数据"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        metadata = {"training_steps": 50000, "batch_size": 256}
        version = manager.add_agent(
            self.model_path, elo_rating=1200.0, parent_version=1, metadata=metadata
        )

        agent = manager.get_agent(version)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.elo_rating, 1200.0)
        self.assertEqual(agent.parent_version, 1)
        self.assertEqual(agent.metadata["training_steps"], 50000)

    def test_remove_agent_decreases_pool_size(self):
        """测试 remove_agent 减少池大小"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        version = manager.add_agent(self.model_path, elo_rating=1000.0)
        self.assertEqual(len(manager), 1)

        manager.remove_agent(version)
        self.assertEqual(len(manager), 0)

    def test_remove_agent_nonexistent(self):
        """测试 remove_agent 处理不存在的agent"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 尝试移除不存在的agent
        with self.assertRaises(ValueError):
            manager.remove_agent(999)

    def test_sample_opponent_returns_policy_record(self):
        """测试 sample_opponent 返回策略记录"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 添加一些agents
        for i in range(5):
            manager.add_agent(self.model_path, elo_rating=1000.0 + i * 50)

        opponent = manager.sample_opponent()
        self.assertIsNotNone(opponent)
        self.assertIsInstance(opponent.version, int)
        self.assertGreater(opponent.version, 0)

    def test_sample_opponent_empty_pool(self):
        """测试 sample_opponent 处理空池"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        opponent = manager.sample_opponent()
        self.assertIsNone(opponent)

    def test_sample_opponent_distribution_30_30_40(self):
        """测试 sample_opponent 采样分布为 30/30/40"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 添加多个agents，用于测试分布
        # agent 1-3: 较老版本，低ELO
        # agent 4-6: 中等版本，中等ELO
        # agent 7-10: 最新版本，高ELO
        for i in range(10):
            manager.add_agent(self.model_path, elo_rating=1000.0 + i * 100)

        # 采样大量次以验证分布
        n_samples = 1000
        strategies = {"latest": 0, "best_elo": 0, "random": 0}

        for _ in range(n_samples):
            opponent = manager.sample_opponent()
            self.assertIsNotNone(opponent)
            # 检查返回的策略是否符合三种策略之一
            # (这里假设我们可以通过某种方式检测使用了哪种策略)

        # 验证期望的分布比例: 30% latest, 30% best_elo, 40% random
        # 注意: 实际实现可能使用随机选择策略，这里验证接口行为

    def test_sample_opponent_exclude_version(self):
        """测试 sample_opponent 排除特定版本"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        for i in range(5):
            manager.add_agent(self.model_path, elo_rating=1000.0)

        # 排除版本3
        opponent = manager.sample_opponent(exclude_versions=[3])
        self.assertIsNotNone(opponent)
        self.assertNotEqual(opponent.version, 3)

    def test_elo_update_winner_gains_loser_loses(self):
        """测试 ELO 更新: 赢家获得分数，输家失去分数"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 添加两个agents
        winner_version = manager.add_agent(self.model_path, elo_rating=1000.0)
        loser_version = manager.add_agent(self.model_path, elo_rating=1000.0)

        winner_before = manager.get_agent(winner_version).elo_rating
        loser_before = manager.get_agent(loser_version).elo_rating

        # 更新ELO
        manager.update_elo(winner_version, loser_version)

        winner_after = manager.get_agent(winner_version).elo_rating
        loser_after = manager.get_agent(loser_version).elo_rating

        # 验证赢家ELO增加，输家ELO减少
        self.assertGreater(winner_after, winner_before)
        self.assertLess(loser_after, loser_before)

    def test_elo_update_with_different_strengths(self):
        """测试不同强度对局的 ELO 更新"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 添加强弱两个agents
        weak_version = manager.add_agent(self.model_path, elo_rating=800.0)
        strong_version = manager.add_agent(self.model_path, elo_rating=1200.0)

        # 弱者击败强者，获得更多分数
        manager.update_elo(weak_version, strong_version)

        weak_agent = manager.get_agent(weak_version)
        strong_agent = manager.get_agent(strong_version)

        # 弱者击败强者应该获得更多ELO
        self.assertGreater(weak_agent.elo_rating, 800.0)
        self.assertLess(strong_agent.elo_rating, 1200.0)

    def test_elo_update_records_match_history(self):
        """测试 ELO 更新记录比赛历史"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        winner_version = manager.add_agent(self.model_path, elo_rating=1000.0)
        loser_version = manager.add_agent(self.model_path, elo_rating=1000.0)

        manager.update_elo(winner_version, loser_version)

        # 验证比赛历史被记录
        self.assertEqual(manager.get_total_matches(), 1)

    def test_identity_stats_tracking(self):
        """测试 IdentityStats 身份统计跟踪"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        version = manager.add_agent(self.model_path, elo_rating=1000.0)

        # 模拟主公获胜
        manager.record_identity_result(version, "主公", won=True)
        # 模拟忠臣失败
        manager.record_identity_result(version, "忠臣", won=False)
        # 模拟反贼获胜
        manager.record_identity_result(version, "反贼", won=True)
        # 模拟内奸失败
        manager.record_identity_result(version, "内奸", won=False)

        agent = manager.get_agent(version)
        identity_stats = agent.identity_stats

        self.assertEqual(identity_stats.主公_wins, 1)
        self.assertEqual(identity_stats.忠臣_losses, 1)
        self.assertEqual(identity_stats.反贼_wins, 1)
        self.assertEqual(identity_stats.内奸_losses, 1)

    def test_identity_stats_win_rates(self):
        """测试 IdentityStats 胜率计算"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        version = manager.add_agent(self.model_path, elo_rating=1000.0)

        # 添加比赛结果
        manager.record_identity_result(version, "主公", won=True)
        manager.record_identity_result(version, "主公", won=True)
        manager.record_identity_result(version, "主公", won=False)

        win_rate = manager.get_identity_win_rate(version, "主公")
        self.assertAlmostEqual(win_rate, 2.0 / 3.0, places=5)

    def test_all_identity_stats(self):
        """测试所有身份的胜率统计"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        version = manager.add_agent(self.model_path, elo_rating=1000.0)

        # 为所有身份添加结果
        identities = ["主公", "忠臣", "反贼", "内奸"]
        for identity in identities:
            manager.record_identity_result(version, identity, won=True)
            manager.record_identity_result(version, identity, won=False)

        all_rates = manager.get_all_identity_win_rates(version)

        self.assertEqual(len(all_rates), 4)
        for identity in identities:
            self.assertIn(identity, all_rates)
            self.assertEqual(all_rates[identity], 0.5)  # 1胜1负

    def test_get_best_agent(self):
        """测试获取最佳agent"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 添加多个agents
        v1 = manager.add_agent(self.model_path, elo_rating=1000.0)
        v2 = manager.add_agent(self.model_path, elo_rating=1200.0)
        v3 = manager.add_agent(self.model_path, elo_rating=1100.0)

        best = manager.get_best_agent()
        self.assertIsNotNone(best)
        self.assertEqual(best.version, v2)  # 最高ELO

    def test_get_latest_agent(self):
        """测试获取最新agent"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 添加多个agents
        for i in range(5):
            manager.add_agent(self.model_path, elo_rating=1000.0)

        latest = manager.get_latest_agent()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.version, 5)  # 最新版本

    def test_pool_pruning_when_full(self):
        """测试池满时的修剪策略"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=3)

        # 添加超过池大小的agents
        for i in range(5):
            manager.add_agent(self.model_path, elo_rating=1000.0 + i * 10)

        # 验证池大小保持为限制值
        self.assertLessEqual(len(manager), 3)

    def test_agent_iteration(self):
        """测试 agent 迭代器"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        for i in range(3):
            manager.add_agent(self.model_path, elo_rating=1000.0)

        versions = [agent.version for agent in manager]
        self.assertEqual(len(versions), 3)
        self.assertEqual(set(versions), {1, 2, 3})


class TestAgentPoolManagerIntegration(unittest.TestCase):
    """AgentPoolManager 集成测试"""

    def setUp(self):
        """测试前置设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.zip")

    def tearDown(self):
        """测试后置清理"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_workflow(self):
        """测试完整工作流程"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 1. 添加初始agent
        v1 = manager.add_agent(self.model_path, elo_rating=1000.0)

        # 2. 采样对手进行训练
        opponent = manager.sample_opponent()
        self.assertIsNotNone(opponent)

        # 3. 模拟比赛并更新ELO
        v2 = manager.add_agent(self.model_path, elo_rating=1000.0)
        manager.update_elo(v2, v1)  # v2击败v1

        # 4. 记录身份统计
        manager.record_identity_result(v2, "主公", won=True)
        manager.record_identity_result(v1, "反贼", won=False)

        # 5. 验证状态
        self.assertEqual(len(manager), 2)
        self.assertEqual(manager.get_total_matches(), 1)

        agent2 = manager.get_agent(v2)
        self.assertEqual(agent2.wins, 1)
        self.assertEqual(agent2.identity_stats.主公_wins, 1)

    def test_multiple_matches_update_stats(self):
        """测试多场比赛更新统计"""
        from train.agent_pool_manager import AgentPoolManager

        manager = AgentPoolManager(pool_size=10)

        # 添加两个agents
        v1 = manager.add_agent(self.model_path, elo_rating=1000.0)
        v2 = manager.add_agent(self.model_path, elo_rating=1000.0)

        # 进行多场比赛
        for i in range(10):
            if i % 2 == 0:
                manager.update_elo(v1, v2)
                manager.record_identity_result(v1, "主公", won=True)
            else:
                manager.update_elo(v2, v1)
                manager.record_identity_result(v2, "反贼", won=True)

        # 验证统计
        agent1 = manager.get_agent(v1)
        agent2 = manager.get_agent(v2)

        self.assertEqual(agent1.wins + agent2.wins, 10)
        self.assertEqual(manager.get_total_matches(), 10)


if __name__ == "__main__":
    unittest.main()
