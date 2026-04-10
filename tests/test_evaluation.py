"""
评估框架单元测试

测试内容:
- 置信区间计算
- 统计函数
- 身份追踪
- 基线比较
- 评估结果类
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from train.statistics import (
    wilson_score_interval,
    clopper_pearson_interval,
    calculate_standard_error,
    calculate_z_score,
    calculate_p_value,
    compare_to_baseline,
    format_confidence_interval,
    required_sample_size,
    cohen_h,
    WinRateStats,
    IdentityTracker,
    get_baseline_win_rate,
)


class TestWilsonScoreInterval(unittest.TestCase):
    """测试Wilson score interval"""

    def test_basic_interval(self):
        """测试基本置信区间计算"""
        # 50/100的胜率
        ci_low, ci_high = wilson_score_interval(50, 100, confidence=0.95)
        self.assertGreater(ci_low, 0)
        self.assertLess(ci_high, 1)
        self.assertLess(ci_low, 0.5)
        self.assertGreater(ci_high, 0.5)

    def test_extreme_zero_wins(self):
        """测试0%胜率边界情况"""
        ci_low, ci_high = wilson_score_interval(0, 100)
        self.assertAlmostEqual(ci_low, 0.0, places=10)  # 应该接近0
        self.assertGreater(ci_high, 0)
        self.assertLess(ci_high, 0.1)  # 应该相对较小

    def test_extreme_all_wins(self):
        """测试100%胜率边界情况"""
        ci_low, ci_high = wilson_score_interval(100, 100)
        self.assertGreater(ci_low, 0.9)  # 应该相对较大
        self.assertEqual(ci_high, 1.0)

    def test_small_sample(self):
        """测试小样本"""
        ci_low, ci_high = wilson_score_interval(5, 10)
        self.assertGreater(ci_low, 0)
        self.assertLess(ci_high, 1)
        # Wilson区间应该比简单正态近似更准确

    def test_confidence_levels(self):
        """测试不同置信水平"""
        ci_95 = wilson_score_interval(50, 100, confidence=0.95)
        ci_99 = wilson_score_interval(50, 100, confidence=0.99)

        # 99% CI应该比95% CI更宽 (检查上下限距离)
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        self.assertLess(width_95, width_99)  # 99%区间更宽

    def test_zero_total(self):
        """测试总次数为0的情况"""
        ci_low, ci_high = wilson_score_interval(0, 0)
        self.assertEqual(ci_low, 0.0)
        self.assertEqual(ci_high, 0.0)

    def test_invalid_input(self):
        """测试无效输入"""
        with self.assertRaises(ValueError):
            wilson_score_interval(10, 5)  # wins > total
        with self.assertRaises(ValueError):
            wilson_score_interval(-1, 10)  # negative wins


class TestClopperPearsonInterval(unittest.TestCase):
    """测试Clopper-Pearson区间"""

    def test_conservative_interval(self):
        """测试CP区间比Wilson更保守"""
        wilson = wilson_score_interval(5, 10, confidence=0.95)
        cp = clopper_pearson_interval(5, 10, confidence=0.95)

        # CP区间应该更宽
        self.assertLessEqual(cp[0], wilson[0])
        self.assertGreaterEqual(cp[1], wilson[1])

    def test_extreme_cases(self):
        """测试极端情况"""
        # 0/10
        ci_low, ci_high = clopper_pearson_interval(0, 10)
        self.assertEqual(ci_low, 0.0)
        self.assertGreater(ci_high, 0)

        # 10/10
        ci_low, ci_high = clopper_pearson_interval(10, 10)
        self.assertLess(ci_low, 1.0)
        self.assertEqual(ci_high, 1.0)


class TestStandardError(unittest.TestCase):
    """测试标准误差计算"""

    def test_basic_se(self):
        """测试基本标准误差"""
        se = calculate_standard_error(50, 100)
        expected = np.sqrt(0.5 * 0.5 / 100)
        self.assertAlmostEqual(se, expected, places=5)

    def test_zero_se_at_extremes(self):
        """测试极端值时SE趋近于0"""
        se_0 = calculate_standard_error(0, 100)
        se_100 = calculate_standard_error(100, 100)
        self.assertAlmostEqual(se_0, 0.0, places=5)
        self.assertAlmostEqual(se_100, 0.0, places=5)

    def test_zero_total(self):
        """测试总次数为0"""
        se = calculate_standard_error(0, 0)
        self.assertEqual(se, 0.0)


class TestZScoreAndPValue(unittest.TestCase):
    """测试Z-score和P-value计算"""

    def test_z_score_positive(self):
        """测试正Z-score"""
        z = calculate_z_score(0.6, 0.5, 100)
        self.assertGreater(z, 0)

    def test_z_score_negative(self):
        """测试负Z-score"""
        z = calculate_z_score(0.4, 0.5, 100)
        self.assertLess(z, 0)

    def test_z_score_zero(self):
        """测试Z-score为0"""
        z = calculate_z_score(0.5, 0.5, 100)
        self.assertAlmostEqual(z, 0.0, places=5)

    def test_p_value_range(self):
        """测试P-value范围"""
        z = 1.96  # 约对应95%置信水平的双尾检验
        p_two = calculate_p_value(z, two_tailed=True)
        p_one = calculate_p_value(z, two_tailed=False)

        self.assertGreater(p_two, 0)
        self.assertLess(p_two, 1)
        self.assertLess(p_one, p_two)  # 单尾p值应该更小

    def test_symmetric_p_value(self):
        """测试P-value对称性"""
        p1 = calculate_p_value(1.0, two_tailed=True)
        p2 = calculate_p_value(-1.0, two_tailed=True)
        self.assertAlmostEqual(p1, p2, places=5)


class TestBaselineComparison(unittest.TestCase):
    """测试基线比较功能"""

    def test_significant_better(self):
        """测试显著优于基线"""
        result = compare_to_baseline(60, 100, baseline_rate=0.5, confidence=0.95)

        self.assertTrue(result["better_than_baseline"])
        self.assertEqual(result["observed_rate"], 0.6)
        self.assertEqual(result["baseline_rate"], 0.5)
        self.assertGreater(result["z_score"], 0)
        # p值应该小于0.05才能显著

    def test_not_significant(self):
        """测试不显著的情况"""
        # 21/100 vs 20%基线,差距很小
        result = compare_to_baseline(21, 100, baseline_rate=0.2)

        self.assertTrue(result["better_than_baseline"])
        self.assertFalse(result["significant"])  # 应该不显著

    def test_worse_than_baseline(self):
        """测试比基线差"""
        result = compare_to_baseline(30, 100, baseline_rate=0.5)

        self.assertFalse(result["better_than_baseline"])
        self.assertLess(result["z_score"], 0)

    def test_relative_improvement(self):
        """测试相对提升计算"""
        result = compare_to_baseline(60, 100, baseline_rate=0.5)
        # 从50%提升到60%,相对提升20%
        self.assertAlmostEqual(result["relative_improvement"], 20.0, places=5)

    def test_zero_total(self):
        """测试零总次数"""
        result = compare_to_baseline(0, 0, baseline_rate=0.5)

        self.assertEqual(result["observed_rate"], 0.0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertFalse(result["significant"])


class TestWinRateStats(unittest.TestCase):
    """测试WinRateStats类"""

    def test_initialization(self):
        """测试初始化"""
        stats = WinRateStats("Test")
        self.assertEqual(stats.name, "Test")
        self.assertEqual(stats.wins, 0)
        self.assertEqual(stats.total, 0)

    def test_add_result(self):
        """测试添加结果"""
        stats = WinRateStats("Test")
        stats.add_result(True)
        stats.add_result(False)

        self.assertEqual(stats.wins, 1)
        self.assertEqual(stats.total, 2)
        self.assertEqual(stats.win_rate, 0.5)

    def test_confidence_interval(self):
        """测试置信区间"""
        stats = WinRateStats("Test")
        for _ in range(50):
            stats.add_result(True)
        for _ in range(50):
            stats.add_result(False)

        ci_low, ci_high = stats.get_confidence_interval()
        self.assertGreater(ci_low, 0)
        self.assertLess(ci_high, 1)
        self.assertLess(ci_low, 0.5)
        self.assertGreater(ci_high, 0.5)

    def test_compare_to_baseline(self):
        """测试与基线比较"""
        stats = WinRateStats("Test")
        for _ in range(60):
            stats.add_result(True)
        for _ in range(40):
            stats.add_result(False)

        comparison = stats.compare_to(0.5)
        self.assertTrue(comparison["better_than_baseline"])

    def test_to_dict(self):
        """测试转换为字典"""
        stats = WinRateStats("Test")
        stats.add_result(True)
        stats.add_result(True)
        stats.add_result(False)

        data = stats.to_dict()
        self.assertEqual(data["name"], "Test")
        self.assertEqual(data["wins"], 2)
        self.assertEqual(data["total"], 3)
        self.assertAlmostEqual(data["win_rate"], 2 / 3, places=5)


class TestIdentityTracker(unittest.TestCase):
    """测试IdentityTracker类"""

    def test_add_result(self):
        """测试添加结果"""
        tracker = IdentityTracker()
        tracker.add_result("主公", True)
        tracker.add_result("主公", False)
        tracker.add_result("反贼", True)

        stats = tracker.get_stats("主公")
        self.assertEqual(stats.win_rate, 0.5)

    def test_get_all_stats(self):
        """测试获取所有统计"""
        tracker = IdentityTracker()
        tracker.add_result("主公", True)
        tracker.add_result("反贼", True)

        all_stats = tracker.get_all_stats()
        self.assertIn("主公", all_stats)
        self.assertIn("反贼", all_stats)
        self.assertNotIn("内奸", all_stats)  # 没有数据的身份

    def test_invalid_identity(self):
        """测试无效身份"""
        tracker = IdentityTracker()
        tracker.add_result("无效身份", True)  # 应该被忽略

        all_stats = tracker.get_all_stats()
        self.assertEqual(len(all_stats), 0)

    def test_formatted_report(self):
        """测试格式化报告"""
        tracker = IdentityTracker()
        tracker.add_result("主公", True)
        tracker.add_result("主公", True)
        tracker.add_result("主公", False)

        report = tracker.get_formatted_report()
        self.assertIn("主公", report)
        self.assertIn("胜率", report or "win_rate")  # 报告中应该包含胜率信息


class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def test_format_confidence_interval(self):
        """测试CI格式化"""
        result = format_confidence_interval(0.5, 0.4, 0.6)
        self.assertIn("50.0%", result)
        self.assertIn("40.0%", result)
        self.assertIn("60.0%", result)

    def test_format_as_decimal(self):
        """测试小数格式化"""
        result = format_confidence_interval(
            0.5, 0.4, 0.6, as_percentage=False, decimals=2
        )
        self.assertIn("0.50", result)

    def test_required_sample_size(self):
        """测试样本量计算"""
        n = required_sample_size(0.5, margin_of_error=0.05, confidence=0.95)
        # 标准正态分布95%对应的样本量应该约为384
        self.assertGreater(n, 300)
        self.assertLess(n, 500)

    def test_cohen_h(self):
        """测试Cohen's h"""
        h = cohen_h(0.6, 0.5)
        self.assertGreater(h, 0)  # 0.6 > 0.5, h应该为正

        h_symmetric = cohen_h(0.5, 0.6)
        self.assertAlmostEqual(h, -h_symmetric, places=5)  # 应该对称

    def test_get_baseline_win_rate(self):
        """测试获取基线胜率"""
        # 随机策略
        self.assertAlmostEqual(get_baseline_win_rate(5, "random"), 0.20, places=2)
        self.assertAlmostEqual(get_baseline_win_rate(8, "random"), 0.125, places=3)

        # 规则策略
        self.assertEqual(get_baseline_win_rate(5, "rule"), 0.35)

        # 默认回退
        self.assertAlmostEqual(get_baseline_win_rate(6, "random"), 1 / 6, places=3)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_single_episode(self):
        """测试单局游戏"""
        ci_low, ci_high = wilson_score_interval(1, 1)
        self.assertGreater(ci_low, 0)
        self.assertEqual(ci_high, 1.0)

    def test_very_large_sample(self):
        """测试超大样本"""
        ci_low, ci_high = wilson_score_interval(5000, 10000)
        # 大样本下CI应该较窄
        self.assertLess(ci_high - ci_low, 0.05)

    def test_very_small_p_value(self):
        """测试极小P值"""
        z = 5.0  # 很大的Z-score
        p = calculate_p_value(z, two_tailed=True)
        self.assertLess(p, 0.0001)

    def test_perfect_win_rate_stats(self):
        """测试100%胜率统计"""
        stats = WinRateStats("Perfect")
        for _ in range(100):
            stats.add_result(True)

        ci_low, ci_high = stats.get_confidence_interval()
        self.assertEqual(ci_high, 1.0)
        self.assertLess(ci_low, 1.0)  # 下限应该小于1


class TestStatisticalProperties(unittest.TestCase):
    """测试统计性质"""

    def test_coverage_probability_simulation(self):
        """测试覆盖率概率(模拟)"""
        np.random.seed(42)
        true_rate = 0.5
        n_trials = 100
        n_simulations = 1000

        covered = 0
        for _ in range(n_simulations):
            successes = np.random.binomial(n_trials, true_rate)
            ci_low, ci_high = wilson_score_interval(successes, n_trials)
            if ci_low <= true_rate <= ci_high:
                covered += 1

        coverage = covered / n_simulations
        # 95% CI应该覆盖约95%的情况(允许一定误差)
        self.assertGreater(coverage, 0.93)
        self.assertLess(coverage, 0.97)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    test_classes = [
        TestWilsonScoreInterval,
        TestClopperPearsonInterval,
        TestStandardError,
        TestZScoreAndPValue,
        TestBaselineComparison,
        TestWinRateStats,
        TestIdentityTracker,
        TestUtilityFunctions,
        TestEdgeCases,
        TestStatisticalProperties,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
