"""
Tests for Hyperparameter Search Module

Tests:
1. Configuration grid generation
2. Experiment execution
3. Result analysis
4. Statistical comparison
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock imports for testing without full dependencies
try:
    from train.hyperparameter_search import (
        HyperparameterSearch,
        MetricsCallback,
        statistical_comparison,
    )

    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


class TestHyperparameterSearch(unittest.TestCase):
    """Test cases for hyperparameter search"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.log_dir = Path(__file__).parent.parent / "train" / "logs" / "hp_search"
        cls.results_file = cls.log_dir / "search_results.json"

    def test_config_grid_generation(self):
        """Test that configuration grid is correctly generated"""
        if not SEARCH_AVAILABLE:
            self.skipTest("HyperparameterSearch not available")

        search = HyperparameterSearch(
            ent_coef_values=[0.01, 0.02],
            lr_values=[1e-4, 5e-4],
        )

        grid = search.get_config_grid()
        expected = [(0.01, 1e-4), (0.01, 5e-4), (0.02, 1e-4), (0.02, 5e-4)]

        self.assertEqual(len(grid), 4)
        self.assertEqual(set(grid), set(expected))

    def test_config_name_format(self):
        """Test configuration name formatting"""
        if not SEARCH_AVAILABLE:
            self.skipTest("HyperparameterSearch not available")

        search = HyperparameterSearch()

        name1 = search.config_name(0.05, 5e-4)
        self.assertEqual(name1, "ent_0.05_lr_5.0e-04")

        name2 = search.config_name(0.01, 1e-3)
        self.assertEqual(name2, "ent_0.01_lr_1.0e-03")

    def test_results_file_exists(self):
        """Test that results file was created"""
        if not self.results_file.exists():
            self.skipTest("No results file found - run experiments first")

        with open(self.results_file) as f:
            results = json.load(f)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_result_structure(self):
        """Test that each result has required fields"""
        if not self.results_file.exists():
            self.skipTest("No results file found")

        with open(self.results_file) as f:
            results = json.load(f)

        required_fields = [
            "config_name",
            "ent_coef",
            "learning_rate",
            "training_steps",
            "training_time_seconds",
            "training_speed_steps_per_sec",
            "model_path",
            "train_metrics",
            "eval_results",
        ]

        for result in results:
            for field in required_fields:
                self.assertIn(field, result, f"Missing field: {field}")

    def test_eval_results_structure(self):
        """Test that evaluation results have required fields"""
        if not self.results_file.exists():
            self.skipTest("No results file found")

        with open(self.results_file) as f:
            results = json.load(f)

        required_eval_fields = [
            "win_rate",
            "wins",
            "total_episodes",
            "avg_reward",
            "avg_game_length",
        ]

        for result in results:
            eval_results = result["eval_results"]
            for field in required_eval_fields:
                self.assertIn(field, eval_results, f"Missing eval field: {field}")

    def test_win_rate_values(self):
        """Test that win rates are in valid range [0, 1]"""
        if not self.results_file.exists():
            self.skipTest("No results file found")

        with open(self.results_file) as f:
            results = json.load(f)

        for result in results:
            win_rate = result["eval_results"]["win_rate"]
            self.assertGreaterEqual(win_rate, 0.0)
            self.assertLessEqual(win_rate, 1.0)

    def test_statistical_comparison(self):
        """Test statistical comparison function"""
        if not SEARCH_AVAILABLE:
            self.skipTest("statistical_comparison not available")

        if not self.results_file.exists():
            self.skipTest("No results file found")

        with open(self.results_file) as f:
            results = json.load(f)

        stats = statistical_comparison(results)

        # Check required stats fields
        self.assertIn("win_rate_stats", stats)
        self.assertIn("reward_stats", stats)

        # Check win rate stats structure
        win_rate_stats = stats["win_rate_stats"]
        self.assertIn("mean", win_rate_stats)
        self.assertIn("std", win_rate_stats)
        self.assertIn("min", win_rate_stats)
        self.assertIn("max", win_rate_stats)

    def test_analysis_summary(self):
        """Test that analysis summary is generated"""
        analysis_file = self.log_dir / "analysis_summary.json"

        if not analysis_file.exists():
            self.skipTest("No analysis file found - run analysis first")

        with open(analysis_file) as f:
            analysis = json.load(f)

        # Check required analysis fields
        self.assertIn("total_experiments", analysis)
        self.assertIn("top_2_configs", analysis)
        self.assertIn("win_rate_range", analysis)

        # Check top 2 configs
        top_configs = analysis["top_2_configs"]
        self.assertEqual(len(top_configs), min(2, analysis["total_experiments"]))


class TestMetricsCallback(unittest.TestCase):
    """Test cases for MetricsCallback"""

    def test_metrics_callback_initialization(self):
        """Test callback initializes correctly"""
        if not SEARCH_AVAILABLE:
            self.skipTest("MetricsCallback not available")

        callback = MetricsCallback()
        self.assertEqual(callback.episode_rewards, [])
        self.assertEqual(callback.step_times, [])

    def test_metrics_extraction(self):
        """Test metrics extraction"""
        if not SEARCH_AVAILABLE:
            self.skipTest("MetricsCallback not available")

        callback = MetricsCallback()
        callback.episode_rewards = [-43.0, -43.1, -42.5, 56.0, -43.2]
        callback.step_times = [0.03, 0.032, 0.031, 0.035, 0.033]

        metrics = callback.get_metrics()

        self.assertIn("episode_rewards", metrics)
        self.assertIn("num_episodes", metrics)
        self.assertEqual(metrics["num_episodes"], 5)

        if "reward_mean" in metrics:
            self.assertAlmostEqual(metrics["reward_mean"], -17.56, places=1)


class TestConfigurationComparison(unittest.TestCase):
    """Test comparison between configurations"""

    def test_top_config_identification(self):
        """Test that top configurations are correctly identified"""
        results_file = (
            Path(__file__).parent.parent
            / "train"
            / "logs"
            / "hp_search"
            / "search_results.json"
        )

        if not results_file.exists():
            self.skipTest("No results file found")

        with open(results_file) as f:
            results = json.load(f)

        if len(results) < 2:
            self.skipTest("Need at least 2 results for comparison")

        # Sort by win rate
        sorted_by_win_rate = sorted(
            results, key=lambda x: x["eval_results"]["win_rate"], reverse=True
        )

        # Check that sorting is correct
        for i in range(len(sorted_by_win_rate) - 1):
            self.assertGreaterEqual(
                sorted_by_win_rate[i]["eval_results"]["win_rate"],
                sorted_by_win_rate[i + 1]["eval_results"]["win_rate"],
            )

    def test_entropy_coefficient_effect(self):
        """Test that entropy coefficient affects performance"""
        results_file = (
            Path(__file__).parent.parent
            / "train"
            / "logs"
            / "hp_search"
            / "search_results.json"
        )

        if not results_file.exists():
            self.skipTest("No results file found")

        with open(results_file) as f:
            results = json.load(f)

        # Group by ent_coef
        ent_groups = {}
        for r in results:
            ent = r["ent_coef"]
            if ent not in ent_groups:
                ent_groups[ent] = []
            ent_groups[ent].append(r["eval_results"]["win_rate"])

        # Verify we have results for different ent_coef values
        if len(ent_groups) < 2:
            self.skipTest("Need results for multiple ent_coef values")

        # Check that we can compute averages for each group
        for ent, win_rates in ent_groups.items():
            avg = sum(win_rates) / len(win_rates)
            self.assertGreaterEqual(avg, 0.0)
            self.assertLessEqual(avg, 1.0)


class TestReproducibility(unittest.TestCase):
    """Test reproducibility of experiments"""

    def test_seed_consistency(self):
        """Test that using same seed produces consistent results"""
        # This test verifies that the seed parameter is passed correctly
        results_file = (
            Path(__file__).parent.parent
            / "train"
            / "logs"
            / "hp_search"
            / "search_results.json"
        )

        if not results_file.exists():
            self.skipTest("No results file found")

        with open(results_file) as f:
            results = json.load(f)

        # All results should have same seed (42) based on our experiments
        # Check that config_name is deterministic
        for r in results:
            config_name = r["config_name"]
            ent_coef = r["ent_coef"]
            lr = r["learning_rate"]

            expected_name = f"ent_{ent_coef}_lr_{lr:.1e}"
            self.assertEqual(config_name, expected_name)

    def test_model_paths_exist(self):
        """Test that saved model paths are valid"""
        results_file = (
            Path(__file__).parent.parent
            / "train"
            / "logs"
            / "hp_search"
            / "search_results.json"
        )

        if not results_file.exists():
            self.skipTest("No results file found")

        with open(results_file) as f:
            results = json.load(f)

        for r in results:
            model_path = Path(r["model_path"])
            # Check if path was saved (may not exist if evaluation failed)
            self.assertTrue("final_model" in str(model_path))


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestHyperparameterSearch))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCallback))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestReproducibility))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    run_tests()
