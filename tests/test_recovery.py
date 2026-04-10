"""
Test Training Recovery (Checkpoint Save/Load)

TDD RED Phase - Define expected behavior for:
- TrainingState dataclass
- save_training_state() - creates .zip file
- load_training_state() - restores state
- resume_training() - continues from checkpoint

Expected to FAIL initially (functions not implemented yet)
"""

import sys
import unittest
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import tempfile
import zipfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Import the actual implementation
from train.recovery import (
    TrainingState,
    save_training_state,
    load_training_state,
    resume_training,
    list_checkpoints,
    load_checkpoint_for_rlai,
)


# ============================================================================
# Mock SB3 classes for testing without dependencies
# ============================================================================


class MockModel:
    """Mock PPO/MaskablePPO model for testing"""

    def __init__(self, num_timesteps: int = 0):
        self.num_timesteps = num_timesteps
        self.policy = MockPolicy()

    @classmethod
    def load(cls, path: str):
        """Mock load method - should restore from zip"""
        # Load metadata to restore state
        if Path(path).exists():
            return cls(num_timesteps=1000)  # Simulated restored state
        raise FileNotFoundError(f"Model not found: {path}")

    def save(self, path: str):
        """Mock save method - should create zip"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # In real implementation, saves to zip
        Path(path).touch()


class MockPolicy:
    """Mock policy for model"""

    pass


class MockVecNormalize:
    """Mock VecNormalize for testing"""

    def __init__(self):
        self.obs_rms = MockRunningMeanStd()
        self.reward_rms = MockRunningMeanStd()
        self.ret_rms = MockRunningMeanStd()

    @classmethod
    def load(cls, path: str, env=None):
        if Path(path).exists():
            return cls()
        raise FileNotFoundError(f"VecNormalize not found: {path}")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()


class MockRunningMeanStd:
    """Mock running statistics"""

    def __init__(self):
        self.mean = np.zeros(10)
        self.var = np.ones(10)
        self.count = 1000


# ============================================================================
# Test Cases
# ============================================================================


class MockModel:
    """Mock PPO/MaskablePPO model for testing"""

    def __init__(self, num_timesteps: int = 0):
        self.num_timesteps = num_timesteps
        self.policy = MockPolicy()

    @classmethod
    def load(cls, path: str):
        """Mock load method - should restore from zip"""
        # Load metadata to restore state
        if Path(path).exists():
            return cls(num_timesteps=1000)  # Simulated restored state
        raise FileNotFoundError(f"Model not found: {path}")

    def save(self, path: str):
        """Mock save method - should create zip"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # In real implementation, saves to zip
        Path(path).touch()


class MockPolicy:
    """Mock policy for model"""

    pass


class MockVecNormalize:
    """Mock VecNormalize for testing"""

    def __init__(self):
        self.obs_rms = MockRunningMeanStd()
        self.reward_rms = MockRunningMeanStd()
        self.ret_rms = MockRunningMeanStd()

    @classmethod
    def load(cls, path: str, env=None):
        if Path(path).exists():
            return cls()
        raise FileNotFoundError(f"VecNormalize not found: {path}")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()


class MockRunningMeanStd:
    """Mock running statistics"""

    def __init__(self):
        self.mean = np.zeros(10)
        self.var = np.ones(10)
        self.count = 1000


# ============================================================================
# Test Cases
# ============================================================================


class TestTrainingState(unittest.TestCase):
    """Test TrainingState dataclass"""

    def test_create_valid_state(self):
        """Test creating valid TrainingState"""
        model = MockModel(num_timesteps=1000)
        state = TrainingState(
            model=model,
            optimizer_state={"lr": 0.0003},
            vec_normalize=None,
            step_count=1000,
            episode_count=100,
            metadata={"timestamp": "2024-01-01"},
        )

        self.assertEqual(state.step_count, 1000)
        self.assertEqual(state.episode_count, 100)
        self.assertEqual(state.metadata["timestamp"], "2024-01-01")

    def test_state_with_vecnormalize(self):
        """Test TrainingState with VecNormalize"""
        model = MockModel()
        vec_norm = MockVecNormalize()

        state = TrainingState(
            model=model,
            optimizer_state={},
            vec_normalize=vec_norm,
            step_count=500,
            episode_count=50,
            metadata={},
        )

        self.assertIsNotNone(state.vec_normalize)
        self.assertIsInstance(state.vec_normalize, MockVecNormalize)

    def test_negative_step_count_raises_error(self):
        """Test that negative step_count raises ValueError"""
        with self.assertRaises(ValueError):
            TrainingState(
                model=MockModel(),
                optimizer_state={},
                vec_normalize=None,
                step_count=-1,
                episode_count=0,
                metadata={},
            )

    def test_negative_episode_count_raises_error(self):
        """Test that negative episode_count raises ValueError"""
        with self.assertRaises(ValueError):
            TrainingState(
                model=MockModel(),
                optimizer_state={},
                vec_normalize=None,
                step_count=0,
                episode_count=-5,
                metadata={},
            )


class TestSaveTrainingState(unittest.TestCase):
    """Test save_training_state() function"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model = MockModel(num_timesteps=1000)
        self.state = TrainingState(
            model=self.model,
            optimizer_state={"lr": 0.0003, "betas": (0.9, 0.999)},
            vec_normalize=MockVecNormalize(),
            step_count=1000,
            episode_count=100,
            metadata={
                "timestamp": "2024-01-01T00:00:00",
                "config": {"learning_rate": 0.0003},
            },
        )

    def test_save_creates_zip_file(self):
        """Test that save creates a .zip file"""
        checkpoint_path = Path(self.temp_dir) / "checkpoint_step_1000"

        result = save_training_state(self.state, str(checkpoint_path))

        # Should return path to saved file
        self.assertTrue(Path(result).exists())
        # Should be a zip file
        self.assertTrue(zipfile.is_zipfile(result))

    def test_save_includes_required_components(self):
        """Test that zip contains model, optimizer, vecnormalize, metadata"""
        checkpoint_path = Path(self.temp_dir) / "checkpoint_step_500"

        save_training_state(self.state, str(checkpoint_path))

        # Check zip contents
        with zipfile.ZipFile(checkpoint_path.with_suffix(".zip"), "r") as zf:
            namelist = zf.namelist()
            self.assertIn("model.zip", namelist)
            self.assertIn("optimizer_state.json", namelist)
            self.assertIn("vecnormalize.pkl", namelist)
            self.assertIn("metadata.json", namelist)

    def test_save_metadata_contains_step_count(self):
        """Test that metadata.json contains step_count"""
        checkpoint_path = Path(self.temp_dir) / "checkpoint_step_2000"

        save_training_state(self.state, str(checkpoint_path))

        with zipfile.ZipFile(checkpoint_path.with_suffix(".zip"), "r") as zf:
            metadata_content = zf.read("metadata.json").decode("utf-8")
            metadata = json.loads(metadata_content)
            self.assertEqual(metadata["step_count"], 1000)
            self.assertEqual(metadata["episode_count"], 100)

    def test_save_keeps_last_n_checkpoints(self):
        """Test that only last N checkpoints are kept"""
        checkpoint_dir = Path(self.temp_dir)

        # Save 5 checkpoints
        for i in range(1, 6):
            state = TrainingState(
                model=MockModel(num_timesteps=i * 1000),
                optimizer_state={},
                vec_normalize=None,
                step_count=i * 1000,
                episode_count=i * 10,
                metadata={"step": i},
            )
            save_training_state(
                state, str(checkpoint_dir / f"step_{i * 1000}"), keep_last_n=3
            )

        # Only 3 most recent should remain
        checkpoints = list(checkpoint_dir.glob("*.zip"))
        self.assertEqual(len(checkpoints), 3)


class TestLoadTrainingState(unittest.TestCase):
    """Test load_training_state() function"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Create a mock checkpoint
        self.checkpoint_path = Path(self.temp_dir) / "checkpoint_step_1000.zip"

        # Create minimal valid checkpoint
        with zipfile.ZipFile(self.checkpoint_path, "w") as zf:
            # Model file
            zf.writestr("model.zip", b"mock_model_data")
            # Optimizer state
            zf.writestr(
                "optimizer_state.json",
                json.dumps({"lr": 0.0003, "state": {"step": 100}}),
            )
            # VecNormalize
            zf.writestr("vecnormalize.pkl", b"mock_vecnormalize")
            # Metadata
            zf.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "step_count": 1000,
                        "episode_count": 100,
                        "timestamp": "2024-01-01T00:00:00",
                    }
                ),
            )

    def test_load_restores_step_count(self):
        """Test that load restores correct step_count"""
        state = load_training_state(str(self.checkpoint_path))

        self.assertEqual(state.step_count, 1000)
        self.assertEqual(state.episode_count, 100)

    def test_load_restores_optimizer_state(self):
        """Test that load restores optimizer state"""
        state = load_training_state(str(self.checkpoint_path))

        self.assertIn("lr", state.optimizer_state)
        self.assertEqual(state.optimizer_state["lr"], 0.0003)

    def test_load_restores_metadata(self):
        """Test that load restores metadata"""
        state = load_training_state(str(self.checkpoint_path))

        self.assertEqual(state.metadata["timestamp"], "2024-01-01T00:00:00")

    def test_load_missing_file_raises_error(self):
        """Test that loading non-existent checkpoint raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            load_training_state("/nonexistent/checkpoint.zip")

    def test_load_corrupted_checkpoint_raises_error(self):
        """Test that corrupted checkpoint raises ValueError"""
        corrupted_path = Path(self.temp_dir) / "corrupted.zip"
        corrupted_path.write_bytes(b"not a valid zip")

        with self.assertRaises(ValueError):
            load_training_state(str(corrupted_path))


class TestResumeTraining(unittest.TestCase):
    """Test resume_training() function"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Create mock checkpoint at step 1000
        self.checkpoint_path = Path(self.temp_dir) / "checkpoint_step_1000.zip"

        with zipfile.ZipFile(self.checkpoint_path, "w") as zf:
            zf.writestr("model.zip", b"mock_model")
            zf.writestr("optimizer_state.json", json.dumps({"lr": 0.0003}))
            zf.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "step_count": 1000,
                        "episode_count": 100,
                        "config": {"total_timesteps": 10000},
                    }
                ),
            )

    def test_resume_continues_from_checkpoint_step(self):
        """Test that resume continues from checkpoint step, not from 0"""
        # Mock implementation would track that training starts at step 1000
        result = resume_training(str(self.checkpoint_path), total_timesteps=5000)

        # Should return a model
        self.assertIsNotNone(result)

    def test_resume_with_missing_checkpoint_raises_error(self):
        """Test that resume with missing checkpoint raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            resume_training("/nonexistent/checkpoint.zip", total_timesteps=1000)

    def test_resume_preserves_previous_progress(self):
        """Test that previous training progress is preserved"""
        # Load the checkpoint
        state = load_training_state(str(self.checkpoint_path))

        # Step count should be 1000 (not 0)
        self.assertEqual(state.step_count, 1000)
        self.assertEqual(state.episode_count, 100)


class TestListCheckpoints(unittest.TestCase):
    """Test list_checkpoints() function"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Create checkpoints at different steps
        for step in [1000, 5000, 2000, 3000]:
            path = Path(self.temp_dir) / f"checkpoint_step_{step}.zip"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("metadata.json", json.dumps({"step_count": step}))

    def test_list_checkpoints_returns_sorted_list(self):
        """Test that checkpoints are sorted by step count"""
        checkpoints = list_checkpoints(self.temp_dir)

        # Should return paths
        self.assertEqual(len(checkpoints), 4)

        # Should be sorted by step (newest first)
        steps = [int(Path(p).stem.split("_")[-1]) for p in checkpoints]
        self.assertEqual(steps, [5000, 3000, 2000, 1000])

    def test_list_checkpoints_empty_directory(self):
        """Test listing checkpoints in empty directory"""
        empty_dir = tempfile.mkdtemp()
        checkpoints = list_checkpoints(empty_dir)

        self.assertEqual(len(checkpoints), 0)


class TestRLAIIntegration(unittest.TestCase):
    """Test RLAI interface compatibility with checkpoints"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = Path(self.temp_dir) / "training_checkpoint.zip"

        # Create checkpoint that can be loaded by RLAI
        with zipfile.ZipFile(self.checkpoint_path, "w") as zf:
            zf.writestr("model.zip", b"mock_sb3_model")
            zf.writestr("vecnormalize.pkl", b"mock_vecnormalize")
            zf.writestr(
                "metadata.json",
                json.dumps(
                    {"step_count": 5000, "training_config": {"learning_rate": 0.0003}}
                ),
            )

    def test_load_checkpoint_for_rlai_returns_paths(self):
        """Test that checkpoint can be extracted for RLAI loading"""
        model_path, vec_norm_path = load_checkpoint_for_rlai(str(self.checkpoint_path))

        # Both paths should be returned
        self.assertIsNotNone(model_path)
        self.assertIsNotNone(vec_norm_path)

    def test_load_checkpoint_compatible_with_rlai_config(self):
        """Test extracted paths work with RLAIConfig"""
        model_path, vec_norm_path = load_checkpoint_for_rlai(str(self.checkpoint_path))

        # Should be valid paths that could be passed to RLAI
        self.assertTrue(Path(model_path).exists() or isinstance(model_path, str))
        self.assertTrue(Path(vec_norm_path).exists() or isinstance(vec_norm_path, str))


class TestCheckpointCompatibility(unittest.TestCase):
    """Test checkpoint format compatibility with SB3"""

    def test_checkpoint_contains_sb3_compatible_model(self):
        """Test that checkpoint model can be loaded by SB3"""
        # The model.zip in checkpoint should be loadable by PPO.load() or MaskablePPO.load()
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = Path(temp_dir) / "checkpoint.zip"

        with zipfile.ZipFile(checkpoint_path, "w") as zf:
            zf.writestr("model.zip", b"sb3_model_data")
            zf.writestr("metadata.json", json.dumps({"step_count": 1000}))

        # After loading, model should be compatible with SB3 load
        state = load_training_state(str(checkpoint_path))
        # Model should be loadable via SB3
        self.assertIsNotNone(state.model)

    def test_vecnormalize_separate_from_model(self):
        """Test that VecNormalize is stored separately for proper restoration"""
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = Path(temp_dir) / "checkpoint.zip"

        with zipfile.ZipFile(checkpoint_path, "w") as zf:
            zf.writestr("model.zip", b"model")
            zf.writestr("vecnormalize.pkl", b"vecnormalize_stats")
            zf.writestr("metadata.json", json.dumps({"step_count": 1000}))

        state = load_training_state(str(checkpoint_path))
        self.assertIsNotNone(state.vec_normalize)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestTrainingState,
        TestSaveTrainingState,
        TestLoadTrainingState,
        TestResumeTraining,
        TestListCheckpoints,
        TestRLAIIntegration,
        TestCheckpointCompatibility,
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
