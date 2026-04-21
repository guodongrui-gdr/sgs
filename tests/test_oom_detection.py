"""
Unit tests for OOM Detection Module.
"""

import logging
import os
import unittest
from unittest.mock import Mock, patch

import torch

from train.memo_check import (
    OOMConfig,
    OOMDetector,
    check_memory_before_rollout,
    get_detector_from_env,
)


class TestOOMConfig(unittest.TestCase):
    """Test OOMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OOMConfig()
        self.assertEqual(config.warn_threshold, 0.85)
        self.assertEqual(config.reduce_threshold, 0.90)
        self.assertEqual(config.critical_threshold, 0.95)
        self.assertEqual(config.min_batch_size, 1)
        self.assertTrue(config.enabled)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OOMConfig(
            warn_threshold=0.80,
            reduce_threshold=0.85,
            critical_threshold=0.90,
            min_batch_size=4,
            enabled=False,
        )
        self.assertEqual(config.warn_threshold, 0.80)
        self.assertEqual(config.reduce_threshold, 0.85)
        self.assertEqual(config.critical_threshold, 0.90)
        self.assertEqual(config.min_batch_size, 4)
        self.assertFalse(config.enabled)


class TestOOMDetectorInit(unittest.TestCase):
    """Test OOMDetector initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        detector = OOMDetector()
        self.assertIsNotNone(detector.config)
        self.assertEqual(detector._check_count, 0)
        self.assertEqual(detector._warning_count, 0)
        self.assertEqual(detector._reduction_count, 0)
        self.assertEqual(detector._last_utilization, 0.0)

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = OOMConfig(warn_threshold=0.75)
        detector = OOMDetector(config)
        self.assertEqual(detector.config.warn_threshold, 0.75)

    def test_cuda_available_attribute(self):
        """Test cuda_available attribute is set."""
        detector = OOMDetector()
        self.assertIsInstance(detector.cuda_available, bool)


class TestOOMDetectorMemoryStats(unittest.TestCase):
    """Test OOMDetector memory statistics methods."""

    def test_get_memory_stats_without_cuda(self):
        """Test memory stats when CUDA is not available."""
        detector = OOMDetector()
        detector.cuda_available = False

        stats = detector._get_memory_stats()

        self.assertEqual(stats["allocated"], 0)
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["reserved"], 0)
        self.assertEqual(stats["free"], 0)
        self.assertEqual(stats["utilization"], 0.0)
        self.assertEqual(stats["allocated_gb"], 0.0)
        self.assertEqual(stats["total_gb"], 0.0)

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.get_device_properties")
    def test_get_memory_stats_with_cuda(self, mock_props, mock_allocated):
        """Test memory stats when CUDA is available."""
        mock_props.return_value = Mock(total_memory=16 * (1024**3))
        mock_allocated.return_value = 8 * (1024**3)

        detector = OOMDetector()
        detector.cuda_available = True

        stats = detector._get_memory_stats()

        self.assertEqual(stats["allocated"], 8 * (1024**3))
        self.assertEqual(stats["total"], 16 * (1024**3))
        self.assertEqual(stats["reserved"], 0)
        self.assertEqual(stats["free"], 8 * (1024**3))
        self.assertAlmostEqual(stats["utilization"], 0.5, places=2)
        self.assertAlmostEqual(stats["allocated_gb"], 8.0, places=1)
        self.assertAlmostEqual(stats["total_gb"], 16.0, places=1)


class TestOOMDetectorCheck(unittest.TestCase):
    """Test OOMDetector check method."""

    def test_check_when_disabled(self):
        """Test check returns correct values when disabled."""
        config = OOMConfig(enabled=False)
        detector = OOMDetector(config)

        result = detector.check()

        self.assertFalse(result["enabled"])
        self.assertFalse(result["should_warn"])
        self.assertFalse(result["should_reduce"])
        self.assertFalse(result["should_critical"])
        self.assertEqual(result["utilization"], 0.0)

    def test_check_increments_counter(self):
        """Test check increments check counter."""
        detector = OOMDetector()
        detector.cuda_available = False

        detector.check()
        self.assertEqual(detector._check_count, 1)

        detector.check()
        self.assertEqual(detector._check_count, 2)

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_check_triggers_warning(self, mock_stats):
        """Test check triggers warning at threshold."""
        mock_stats.return_value = {
            "allocated": 86,
            "total": 100,
            "free": 14,
            "utilization": 0.86,
            "allocated_gb": 8.6,
            "total_gb": 10.0,
        }

        config = OOMConfig(warn_threshold=0.85)
        detector = OOMDetector(config)
        detector.cuda_available = True

        result = detector.check()

        self.assertTrue(result["should_warn"])
        self.assertEqual(detector._warning_count, 1)

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_check_triggers_reduce(self, mock_stats):
        """Test check triggers reduce at threshold."""
        mock_stats.return_value = {
            "allocated": 92,
            "total": 100,
            "free": 8,
            "utilization": 0.92,
            "allocated_gb": 9.2,
            "total_gb": 10.0,
        }

        config = OOMConfig(warn_threshold=0.85, reduce_threshold=0.90)
        detector = OOMDetector(config)
        detector.cuda_available = True

        result = detector.check()

        self.assertTrue(result["should_reduce"])
        self.assertEqual(detector._reduction_count, 1)


class TestOOMDetectorThresholdChecks(unittest.TestCase):
    """Test OOMDetector threshold check methods."""

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_should_warn_true(self, mock_stats):
        """Test should_warn returns True when above threshold."""
        mock_stats.return_value = {
            "allocated": 86,
            "total": 100,
            "free": 14,
            "utilization": 0.86,
            "allocated_gb": 8.6,
            "total_gb": 10.0,
        }

        config = OOMConfig(warn_threshold=0.85)
        detector = OOMDetector(config)
        detector.cuda_available = True

        self.assertTrue(detector.should_warn())

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_should_warn_false(self, mock_stats):
        """Test should_warn returns False when below threshold."""
        mock_stats.return_value = {
            "allocated": 80,
            "total": 100,
            "free": 20,
            "utilization": 0.80,
            "allocated_gb": 8.0,
            "total_gb": 10.0,
        }

        config = OOMConfig(warn_threshold=0.85)
        detector = OOMDetector(config)
        detector.cuda_available = True

        self.assertFalse(detector.should_warn())

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_should_reduce_true(self, mock_stats):
        """Test should_reduce returns True when above threshold."""
        mock_stats.return_value = {
            "allocated": 95,
            "total": 100,
            "free": 5,
            "utilization": 0.95,
            "allocated_gb": 9.5,
            "total_gb": 10.0,
        }

        config = OOMConfig(reduce_threshold=0.90)
        detector = OOMDetector(config)
        detector.cuda_available = True

        self.assertTrue(detector.should_reduce())

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_should_critical_true(self, mock_stats):
        """Test should_critical returns True when above threshold."""
        mock_stats.return_value = {
            "allocated": 97,
            "total": 100,
            "free": 3,
            "utilization": 0.97,
            "allocated_gb": 9.7,
            "total_gb": 10.0,
        }

        config = OOMConfig(critical_threshold=0.95)
        detector = OOMDetector(config)
        detector.cuda_available = True

        self.assertTrue(detector.should_critical())


class TestOOMDetectorReduceBatch(unittest.TestCase):
    """Test OOMDetector reduce_batch method."""

    def test_reduce_batch_by_half(self):
        """Test batch size is reduced by half."""
        config = OOMConfig(min_batch_size=1)
        detector = OOMDetector(config)

        result = detector.reduce_batch(16)
        self.assertEqual(result, 8)

        result = detector.reduce_batch(8)
        self.assertEqual(result, 4)

    def test_reduce_batch_respects_minimum(self):
        """Test batch size respects minimum."""
        config = OOMConfig(min_batch_size=4)
        detector = OOMDetector(config)

        result = detector.reduce_batch(8)
        self.assertEqual(result, 4)

        result = detector.reduce_batch(6)
        self.assertEqual(result, 4)

        result = detector.reduce_batch(4)
        self.assertEqual(result, 4)

    def test_reduce_batch_at_minimum(self):
        """Test batch size at minimum."""
        config = OOMConfig(min_batch_size=4)
        detector = OOMDetector(config)

        result = detector.reduce_batch(4)
        self.assertEqual(result, 4)


class TestOOMDetectorStats(unittest.TestCase):
    """Test OOMDetector statistics methods."""

    def test_get_stats(self):
        """Test get_stats returns correct values."""
        detector = OOMDetector()
        detector.cuda_available = False
        detector._check_count = 10
        detector._warning_count = 5
        detector._reduction_count = 2
        detector._last_utilization = 0.87

        stats = detector.get_stats()

        self.assertEqual(stats["check_count"], 10)
        self.assertEqual(stats["warning_count"], 5)
        self.assertEqual(stats["reduction_count"], 2)
        self.assertEqual(stats["last_utilization"], 0.87)
        self.assertEqual(stats["cuda_available"], False)

    def test_reset_stats(self):
        """Test reset_stats clears counters."""
        detector = OOMDetector()
        detector._check_count = 10
        detector._warning_count = 5
        detector._reduction_count = 2

        detector.reset_stats()

        self.assertEqual(detector._check_count, 0)
        self.assertEqual(detector._warning_count, 0)
        self.assertEqual(detector._reduction_count, 0)


class TestOOMDetectorClearCache(unittest.TestCase):
    """Test OOMDetector force_clear_cache method."""

    @patch("torch.cuda.empty_cache")
    def test_clear_cache_when_cuda_available(self, mock_clear):
        """Test cache clear when CUDA is available."""
        detector = OOMDetector()
        detector.cuda_available = True

        detector.force_clear_cache()
        mock_clear.assert_called_once()

    @patch("torch.cuda.empty_cache")
    def test_clear_cache_when_cuda_unavailable(self, mock_clear):
        """Test cache clear when CUDA is not available."""
        detector = OOMDetector()
        detector.cuda_available = False

        detector.force_clear_cache()
        mock_clear.assert_not_called()


class TestGetDetectorFromEnv(unittest.TestCase):
    """Test get_detector_from_env function."""

    def test_default_env_values(self):
        """Test detector with default environment values."""
        detector = get_detector_from_env()
        self.assertEqual(detector.config.warn_threshold, 0.85)
        self.assertEqual(detector.config.reduce_threshold, 0.90)
        self.assertEqual(detector.config.critical_threshold, 0.95)
        self.assertEqual(detector.config.min_batch_size, 1)
        self.assertTrue(detector.config.enabled)

    @patch.dict(os.environ, {"OOM_WARN_THRESHOLD": "0.75"})
    def test_custom_warn_threshold(self):
        """Test custom warning threshold from env."""
        detector = get_detector_from_env()
        self.assertEqual(detector.config.warn_threshold, 0.75)

    @patch.dict(os.environ, {"OOM_REDUCE_THRESHOLD": "0.80"})
    def test_custom_reduce_threshold(self):
        """Test custom reduce threshold from env."""
        detector = get_detector_from_env()
        self.assertEqual(detector.config.reduce_threshold, 0.80)

    @patch.dict(os.environ, {"OOM_ENABLED": "0"})
    def test_disabled_from_env(self):
        """Test disabled from environment."""
        detector = get_detector_from_env()
        self.assertFalse(detector.config.enabled)

    @patch.dict(os.environ, {"OOM_ENABLED": "false"})
    def test_disabled_from_env_false_string(self):
        """Test disabled from environment with 'false' string."""
        detector = get_detector_from_env()
        self.assertFalse(detector.config.enabled)

    @patch.dict(os.environ, {"OOM_MIN_BATCH": "8"})
    def test_custom_min_batch(self):
        """Test custom min batch from env."""
        detector = get_detector_from_env()
        self.assertEqual(detector.config.min_batch_size, 8)


class TestCheckMemoryBeforeRollout(unittest.TestCase):
    """Test check_memory_before_rollout function."""

    def test_no_action_needed(self):
        """Test when no action is needed."""
        detector = OOMDetector()
        detector.cuda_available = False

        should_reduce, batch_size = check_memory_before_rollout(detector)

        self.assertFalse(should_reduce)
        self.assertEqual(batch_size, 0)

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_critical_triggered(self, mock_stats):
        """Test critical level triggers reduction."""
        mock_stats.return_value = {
            "allocated": 97,
            "total": 100,
            "free": 3,
            "utilization": 0.97,
            "allocated_gb": 9.7,
            "total_gb": 10.0,
        }

        config = OOMConfig(critical_threshold=0.95, min_batch_size=4)
        detector = OOMDetector(config)
        detector.cuda_available = True

        should_reduce, batch_size = check_memory_before_rollout(detector)

        self.assertTrue(should_reduce)
        self.assertEqual(batch_size, 4)

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_reduce_triggered(self, mock_stats):
        """Test reduce level triggers reduction."""
        mock_stats.return_value = {
            "allocated": 92,
            "total": 100,
            "free": 8,
            "utilization": 0.92,
            "allocated_gb": 9.2,
            "total_gb": 10.0,
        }

        config = OOMConfig(reduce_threshold=0.90, critical_threshold=0.95)
        detector = OOMDetector(config)
        detector.cuda_available = True

        should_reduce, batch_size = check_memory_before_rollout(detector)

        self.assertTrue(should_reduce)
        self.assertEqual(batch_size, 0)

    @patch.object(OOMDetector, "_get_memory_stats")
    def test_warning_only(self, mock_stats):
        """Test warning level only logs, no reduction."""
        mock_stats.return_value = {
            "allocated": 87,
            "total": 100,
            "free": 13,
            "utilization": 0.87,
            "allocated_gb": 8.7,
            "total_gb": 10.0,
        }

        config = OOMConfig(warn_threshold=0.85, reduce_threshold=0.90)
        detector = OOMDetector(config)
        detector.cuda_available = True

        should_reduce, batch_size = check_memory_before_rollout(detector)

        self.assertFalse(should_reduce)
        self.assertEqual(batch_size, 0)

    def test_creates_detector_if_none(self):
        """Test function creates detector if None passed."""
        should_reduce, batch_size = check_memory_before_rollout(None)
        self.assertIsInstance(should_reduce, bool)
        self.assertIsInstance(batch_size, int)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
