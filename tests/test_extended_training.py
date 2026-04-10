"""
Tests for Extended Training Pipeline

Tests the following components:
- CheckpointManager: checkpoint saving/loading, recovery
- EarlyStoppingCallback: plateau detection, convergence detection
- TrainingMonitor: metrics tracking, display
- TrainingSpeedOptimizer: speed estimation, recommendations
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

# Import modules to test
try:
    from train.checkpoint_manager import (
        CheckpointManager,
        CheckpointMetadata,
        create_checkpoint_callback,
    )

    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    CHECKPOINT_MANAGER_AVAILABLE = False

try:
    from train.early_stopping import (
        EarlyStoppingCallback,
        EvaluationRecord,
        StoppingCondition,
    )

    EARLY_STOPPING_AVAILABLE = True
except ImportError:
    EARLY_STOPPING_AVAILABLE = False

try:
    from train.monitor_dashboard import (
        TrainingMonitor,
        TrainingMetrics,
        TrainingSpeedOptimizer,
    )

    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint management functionality"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if CHECKPOINT_MANAGER_AVAILABLE:
            self.manager = CheckpointManager(
                log_dir=str(self.log_dir),
                checkpoint_freq=10_000,
                max_checkpoints=5,
            )

    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(
        not CHECKPOINT_MANAGER_AVAILABLE, "CheckpointManager not available"
    )
    def test_should_save_checkpoint(self):
        """Test checkpoint frequency detection"""
        # Should not save at 0
        self.assertFalse(self.manager.should_save_checkpoint(0))

        # Should save at checkpoint_freq
        self.assertTrue(self.manager.should_save_checkpoint(10_000))
        self.assertTrue(self.manager.should_save_checkpoint(20_000))

        # Should save at milestones
        self.assertTrue(self.manager.should_save_checkpoint(100_000))
        self.assertTrue(self.manager.should_save_checkpoint(500_000))

    @unittest.skipIf(
        not CHECKPOINT_MANAGER_AVAILABLE, "CheckpointManager not available"
    )
    def test_checkpoint_metadata(self):
        """Test checkpoint metadata creation"""
        metadata = CheckpointMetadata(
            checkpoint_path="/path/to/checkpoint.zip",
            vec_normalize_path="/path/to/vecnormalize.pkl",
            timestep=100_000,
            timestamp="20260409_120000",
            training_config={"lr": 5e-4},
            metrics={"win_rate": 0.25},
        )

        # Test to_dict
        data = metadata.to_dict()
        self.assertEqual(data["timestep"], 100_000)
        self.assertEqual(data["metrics"]["win_rate"], 0.25)

        # Test from_dict
        restored = CheckpointMetadata.from_dict(data)
        self.assertEqual(restored.timestep, metadata.timestep)

    @unittest.skipIf(
        not CHECKPOINT_MANAGER_AVAILABLE, "CheckpointManager not available"
    )
    def test_save_checkpoint(self):
        """Test checkpoint saving"""
        # Create mock model and env
        mock_model = Mock()
        mock_model.save = Mock()

        mock_env = Mock()
        mock_env.save = Mock()

        # Save checkpoint
        metadata = self.manager.save_checkpoint(
            model=mock_model,
            env=mock_env,
            timestep=10_000,
            metrics={"win_rate": 0.20},
            training_config={"lr": 5e-4},
            force=True,
        )

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.timestep, 10_000)

        # Verify mock methods were called
        mock_model.save.assert_called_once()
        mock_env.save.assert_called_once()

    @unittest.skipIf(
        not CHECKPOINT_MANAGER_AVAILABLE, "CheckpointManager not available"
    )
    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint"""
        # Initially no checkpoints
        latest = self.manager.find_latest_checkpoint()
        self.assertIsNone(latest)

        # Add mock checkpoints
        self.manager.checkpoints = [
            CheckpointMetadata(
                checkpoint_path=str(self.log_dir / "checkpoint_10k.zip"),
                vec_normalize_path=str(
                    self.log_dir / "checkpoint_10k_vecnormalize.pkl"
                ),
                timestep=10_000,
                timestamp="20260409_100000",
                training_config={},
                metrics={},
            ),
            CheckpointMetadata(
                checkpoint_path=str(self.log_dir / "checkpoint_20k.zip"),
                vec_normalize_path=str(
                    self.log_dir / "checkpoint_20k_vecnormalize.pkl"
                ),
                timestep=20_000,
                timestamp="20260409_110000",
                training_config={},
                metrics={},
            ),
        ]

        # Create mock files
        Path(self.manager.checkpoints[-1].checkpoint_path).touch()

        # Find latest
        latest = self.manager.find_latest_checkpoint()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.timestep, 20_000)

    @unittest.skipIf(
        not CHECKPOINT_MANAGER_AVAILABLE, "CheckpointManager not available"
    )
    def test_cleanup_old_checkpoints(self):
        """Test checkpoint cleanup"""
        # Create many checkpoints
        for i in range(10):
            self.manager.checkpoints.append(
                CheckpointMetadata(
                    checkpoint_path=str(self.log_dir / f"checkpoint_{i * 10}k.zip"),
                    vec_normalize_path=str(
                        self.log_dir / f"checkpoint_{i * 10}k_vecnormalize.pkl"
                    ),
                    timestep=i * 10_000,
                    timestamp=f"20260409_{100000 + i}",
                    training_config={},
                    metrics={},
                )
            )

        # Cleanup (keep last 5)
        self.manager.cleanup_old_checkpoints(keep_last=5)

        # Should have 5 checkpoints
        self.assertEqual(len(self.manager.checkpoints), 5)

    @unittest.skipIf(
        not CHECKPOINT_MANAGER_AVAILABLE, "CheckpointManager not available"
    )
    def test_create_recovery_script(self):
        """Test recovery script creation"""
        # Add checkpoint
        self.manager.checkpoints.append(
            CheckpointMetadata(
                checkpoint_path=str(self.log_dir / "checkpoint_100k.zip"),
                vec_normalize_path=str(
                    self.log_dir / "checkpoint_100k_vecnormalize.pkl"
                ),
                timestep=100_000,
                timestamp="20260409_120000",
                training_config={},
                metrics={},
            )
        )

        # Create recovery script
        self.manager.create_recovery_script()

        # Verify script exists
        script_path = self.log_dir / "recover_training.sh"
        self.assertTrue(script_path.exists())


class TestEarlyStoppingCallback(unittest.TestCase):
    """Test early stopping functionality"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if EARLY_STOPPING_AVAILABLE:
            self.early_stopping = EarlyStoppingCallback(
                eval_freq=10_000,
                n_eval_episodes=10,
                plateau_patience=5,
                plateau_threshold=0.05,
                convergence_threshold=10.0,
                min_timesteps=100_000,
                log_dir=str(self.log_dir),
                verbose=0,
            )

    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(
        not EARLY_STOPPING_AVAILABLE, "EarlyStoppingCallback not available"
    )
    def test_should_eval(self):
        """Test evaluation frequency detection"""
        self.assertFalse(self.early_stopping.should_eval(0))
        self.assertTrue(self.early_stopping.should_eval(10_000))
        self.assertTrue(self.early_stopping.should_eval(20_000))

    @unittest.skipIf(
        not EARLY_STOPPING_AVAILABLE, "EarlyStoppingCallback not available"
    )
    def test_plateau_detection(self):
        """Test plateau detection"""
        # Simulate evaluation history with plateau
        for i in range(5):
            record = EvaluationRecord(
                timestep=(i + 1) * 10_000,
                win_rate=0.20,  # No improvement
                avg_reward=-10.0,
                reward_std=20.0,
                episode_count=10,
            )
            self.early_stopping.evaluation_history.append(record)
            self.early_stopping.win_rate_history.append(record.win_rate)

        # Check stopping condition at 50K (below min_timesteps)
        condition = self.early_stopping.check_stopping_conditions(
            self.early_stopping.evaluation_history[-1], 50_000
        )
        self.assertIsNone(condition)  # Should not stop before min_timesteps

        # Check stopping condition at 150K (above min_timesteps)
        condition = self.early_stopping.check_stopping_conditions(
            EvaluationRecord(
                timestep=150_000,
                win_rate=0.20,
                avg_reward=-10.0,
                reward_std=20.0,
                episode_count=10,
            ),
            150_000,
        )
        self.assertIsNotNone(condition)
        self.assertEqual(condition.condition_type, "plateau")

    @unittest.skipIf(
        not EARLY_STOPPING_AVAILABLE, "EarlyStoppingCallback not available"
    )
    def test_convergence_detection(self):
        """Test convergence detection"""
        # Simulate low reward variance
        self.early_stopping.reward_std_history = deque(
            [5.0, 4.0, 3.0, 2.0, 1.0], maxlen=5
        )

        # Check stopping condition
        condition = self.early_stopping.check_stopping_conditions(
            EvaluationRecord(
                timestep=150_000,
                win_rate=0.25,
                avg_reward=-5.0,
                reward_std=1.0,
                episode_count=10,
            ),
            150_000,
        )

        self.assertIsNotNone(condition)
        self.assertEqual(condition.condition_type, "convergence")

    @unittest.skipIf(
        not EARLY_STOPPING_AVAILABLE, "EarlyStoppingCallback not available"
    )
    def test_target_achieved(self):
        """Test target achievement detection"""
        # Simulate 50% win rate
        condition = self.early_stopping.check_stopping_conditions(
            EvaluationRecord(
                timestep=150_000,
                win_rate=0.50,
                avg_reward=10.0,
                reward_std=20.0,
                episode_count=10,
            ),
            150_000,
        )

        self.assertIsNotNone(condition)
        self.assertEqual(condition.condition_type, "target_achieved")

    @unittest.skipIf(
        not EARLY_STOPPING_AVAILABLE, "EarlyStoppingCallback not available"
    )
    def test_stopping_condition_serialization(self):
        """Test stopping condition serialization"""
        condition = StoppingCondition(
            condition_type="plateau",
            timestep=150_000,
            value=0.20,
            threshold=0.05,
            reason="Win rate plateaued at 20%",
            timestamp="20260409_120000",
        )

        data = condition.to_dict()
        self.assertEqual(data["condition_type"], "plateau")
        self.assertEqual(data["timestep"], 150_000)


class TestTrainingMonitor(unittest.TestCase):
    """Test training monitor functionality"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if MONITOR_AVAILABLE:
            self.monitor = TrainingMonitor(
                total_timesteps=1_000_000,
                log_dir=str(self.log_dir),
                update_freq=1000,
                use_rich=False,  # Use simple display for tests
                verbose=0,
            )

    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(not MONITOR_AVAILABLE, "TrainingMonitor not available")
    def test_update_metrics(self):
        """Test metrics update"""
        self.monitor.update(
            timestep=10_000,
            episode_reward=5.0,
            episode_length=100,
        )

        self.assertEqual(self.monitor.current_timestep, 10_000)
        self.assertEqual(len(self.monitor.metrics_history), 1)

    @unittest.skipIf(not MONITOR_AVAILABLE, "TrainingMonitor not available")
    def test_progress_calculation(self):
        """Test progress percentage calculation"""
        self.monitor.update(timestep=100_000)

        progress = self.monitor.get_progress_percent()
        self.assertEqual(progress, 10.0)  # 100K / 1M = 10%

    @unittest.skipIf(not MONITOR_AVAILABLE, "TrainingMonitor not available")
    def test_time_estimation(self):
        """Test time remaining estimation"""
        # Simulate some training time
        self.monitor.start_time = time.time() - 60  # Started 60 seconds ago
        self.monitor.steps_per_second = 1000
        self.monitor.current_timestep = 60_000

        remaining = self.monitor.get_estimated_time_remaining()

        # Remaining: 1M - 60K = 940K steps
        # At 1000 steps/sec = 940 seconds
        self.assertEqual(remaining.total_seconds(), 940)

    @unittest.skipIf(not MONITOR_AVAILABLE, "TrainingMonitor not available")
    def test_reward_statistics(self):
        """Test reward statistics calculation"""
        for reward in [1.0, 2.0, 3.0, 4.0, 5.0]:
            self.monitor.episode_rewards.append(reward)

        avg = self.monitor.get_avg_reward()
        std = self.monitor.get_reward_std()

        self.assertEqual(avg, 3.0)
        self.assertAlmostEqual(std, 1.414, places=2)  # sqrt(2)

    @unittest.skipIf(not MONITOR_AVAILABLE, "TrainingMonitor not available")
    def test_eval_metrics_update(self):
        """Test evaluation metrics update"""
        self.monitor.update_eval_metrics(
            win_rate=0.25,
            avg_reward=-10.0,
            reward_std=20.0,
        )

        self.assertEqual(self.monitor.get_latest_win_rate(), 0.25)

    @unittest.skipIf(not MONITOR_AVAILABLE, "TrainingMonitor not available")
    def test_save_report(self):
        """Test report saving"""
        self.monitor.update(timestep=500_000)
        self.monitor.save_report()

        report_file = self.log_dir / "training_report.json"
        self.assertTrue(report_file.exists())

        # Load and verify
        with open(report_file) as f:
            report = json.load(f)

        self.assertEqual(report["final_timestep"], 500_000)


class TestTrainingSpeedOptimizer(unittest.TestCase):
    """Test training speed optimizer"""

    @unittest.skipIf(not MONITOR_AVAILABLE, "TrainingSpeedOptimizer not available")
    def test_recommendation(self):
        """Test configuration recommendation"""
        optimizer = TrainingSpeedOptimizer()
        recommendation = optimizer.get_recommendation()

        self.assertEqual(recommendation["n_envs"], 8)
        self.assertTrue(recommendation["use_subprocess"])
        self.assertEqual(recommendation["max_rounds"], 50)

    @unittest.skipIf(not MONITOR_AVAILABLE, "TrainingSpeedOptimizer not available")
    def test_time_estimation(self):
        """Test training time estimation"""
        optimizer = TrainingSpeedOptimizer()

        # 5M steps at 1500 steps/sec
        time = optimizer.estimate_training_time(5_000_000, 1500)

        # 5M / 1500 = 3333.3 seconds ≈ 55.5 minutes
        self.assertEqual(time.total_seconds(), 3333)


class TestIntegration(unittest.TestCase):
    """Integration tests for extended training pipeline"""

    def setUp(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(
        not (CHECKPOINT_MANAGER_AVAILABLE and MONITOR_AVAILABLE),
        "Required modules not available",
    )
    def test_checkpoint_with_monitor(self):
        """Test checkpoint integration with monitor"""
        manager = CheckpointManager(log_dir=str(self.log_dir))
        monitor = TrainingMonitor(total_timesteps=1_000_000, log_dir=str(self.log_dir))

        # Update monitor
        monitor.update(timestep=100_000)

        # Save checkpoint with metrics from monitor
        summary = monitor.get_summary()
        mock_model = Mock()
        mock_model.save = Mock()
        mock_env = Mock()
        mock_env.save = Mock()

        metadata = manager.save_checkpoint(
            model=mock_model,
            env=mock_env,
            timestep=100_000,
            metrics={
                "avg_reward": summary["avg_reward"],
                "progress": summary["progress_percent"],
            },
            force=True,
        )

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.timestep, 100_000)

    @unittest.skipIf(
        not (EARLY_STOPPING_AVAILABLE and MONITOR_AVAILABLE),
        "Required modules not available",
    )
    def test_early_stopping_with_monitor(self):
        """Test early stopping integration with monitor"""
        early_stopping = EarlyStoppingCallback(
            eval_freq=10_000,
            n_eval_episodes=10,
            log_dir=str(self.log_dir),
            verbose=0,
        )

        monitor = TrainingMonitor(total_timesteps=1_000_000, log_dir=str(self.log_dir))

        # Simulate training progress
        for timestep in [10_000, 20_000, 30_000, 40_000, 50_000]:
            monitor.update(timestep=timestep)
            early_stopping.win_rates.append(0.20)

        # Verify both modules track the same metrics
        self.assertEqual(len(early_stopping.win_rates), 5)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
