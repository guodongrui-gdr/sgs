"""
Unit tests for TensorBoardLoggingCallback in train_self_play.py.

Tests verify:
- Callback reads metrics from model.logger.name_to_value
- metrics_logger.log_training_metrics() called with correct values
- Missing metrics (None values) are handled properly
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.train_self_play import TensorBoardLoggingCallback


class MockLogger:
    """Mock logger with name_to_value dict."""

    def __init__(self, values=None):
        self.name_to_value = values or {}


class TestTensorBoardLoggingCallback:
    """Test suite for TensorBoardLoggingCallback."""

    def setup_method(self):
        """Set up mocks for each test."""
        self.mock_metrics_logger = Mock()
        self.mock_metrics_logger.writer = Mock()
        self.mock_metrics_logger.step_count = 0

        self.callback = TensorBoardLoggingCallback(
            metrics_logger=self.mock_metrics_logger, log_freq=1000, verbose=0
        )

        # Mock the parent class attributes
        self.callback.n_calls = 1000  # Match log_freq to trigger logging
        self.callback.num_timesteps = 10000

    def test_callback_reads_metrics_from_model_logger(self):
        """Test that callback reads metrics from model.logger.name_to_value."""
        # Create mock model with logger containing training metrics
        mock_model = Mock()
        mock_model.logger = MockLogger(
            {
                "train/loss": 0.5,
                "train/entropy_loss": -1.0,
                "train/learning_rate": 3e-4,
                "train/value_loss": 0.2,
                "train/policy_gradient_loss": 0.1,
                "train/approx_kl": 0.01,
                "train/clip_fraction": 0.1,
                "train/explained_variance": 0.95,
            }
        )

        self.callback.model = mock_model

        # Execute
        self.callback._on_rollout_end()

        # Verify set_step was called
        self.mock_metrics_logger.set_step.assert_called_once_with(10000)

        # Verify log_training_metrics was called
        self.mock_metrics_logger.log_training_metrics.assert_called_once()

    def test_log_training_metrics_called_with_correct_values(self):
        """Test that log_training_metrics() is called with expected kwargs."""
        # Create mock model with all training metrics
        mock_model = Mock()
        mock_model.logger = MockLogger(
            {
                "train/loss": 0.5,
                "train/entropy_loss": -1.0,
                "train/learning_rate": 3e-4,
                "train/value_loss": 0.2,
                "train/policy_gradient_loss": 0.1,
                "train/approx_kl": 0.01,
                "train/clip_fraction": 0.1,
                "train/explained_variance": 0.95,
            }
        )

        self.callback.model = mock_model

        # Execute
        self.callback._on_rollout_end()

        # Verify log_training_metrics called with correct kwargs
        call_args = self.mock_metrics_logger.log_training_metrics.call_args

        assert call_args.kwargs["loss"] == 0.5
        assert call_args.kwargs["entropy"] == -1.0
        assert call_args.kwargs["learning_rate"] == 3e-4
        assert call_args.kwargs["value_loss"] == 0.2
        assert call_args.kwargs["policy_loss"] == 0.1
        assert call_args.kwargs["approx_kl"] == 0.01
        assert call_args.kwargs["clip_fraction"] == 0.1
        assert call_args.kwargs["explained_variance"] == 0.95

    def test_missing_metrics_returns_none(self):
        """Test that missing metrics (dict.get returns None) are handled properly."""
        # Create mock model with only some metrics
        mock_model = Mock()
        mock_model.logger = MockLogger(
            {
                "train/loss": 0.5,
                # Missing: entropy_loss, learning_rate, etc.
            }
        )

        self.callback.model = mock_model

        # Execute
        self.callback._on_rollout_end()

        # Verify log_training_metrics called with None for missing values
        call_args = self.mock_metrics_logger.log_training_metrics.call_args

        assert call_args.kwargs["loss"] == 0.5
        assert call_args.kwargs["entropy"] is None
        assert call_args.kwargs["learning_rate"] is None
        assert call_args.kwargs["value_loss"] is None
        assert call_args.kwargs["policy_loss"] is None
        assert call_args.kwargs["approx_kl"] is None
        assert call_args.kwargs["clip_fraction"] is None
        assert call_args.kwargs["explained_variance"] is None

    def test_on_step_returns_true(self):
        """Test that _on_step() returns True (stub implementation)."""
        # Execute
        result = self.callback._on_step()

        # Verify callback returns True to continue training
        assert result is True

    def test_no_logging_when_not_at_log_freq(self):
        """Test that logging only happens at log_freq intervals (deprecated - now logged on rollout end)."""
        # This test is now a placeholder since _on_rollout_end handles logging
        # Set n_calls to not match log_freq - but this doesn't matter anymore
        self.callback.n_calls = 500  # Not a multiple of 1000

        mock_model = Mock()
        mock_model.logger = MockLogger(
            {
                "train/loss": 0.5,
            }
        )

        self.callback.model = mock_model

        # Execute
        self.callback._on_rollout_end()

        # Verify set_step was still called (logging happens regardless of n_calls now)
        self.mock_metrics_logger.set_step.assert_called_once_with(10000)

        # Verify log_training_metrics was called
        self.mock_metrics_logger.log_training_metrics.assert_called_once()

    def test_episode_metrics_logged_when_available(self):
        """Test that episode metrics are logged when available in values."""
        mock_model = Mock()
        mock_model.logger = MockLogger(
            {
                "train/loss": 0.5,
                "rollout/ep_rew_mean": 100.0,
                "rollout/ep_len_mean": 50.0,
            }
        )

        self.callback.model = mock_model

        # Mock log_episode_metrics function
        with patch("train.train_self_play.log_episode_metrics") as mock_log_ep:
            self.callback._on_rollout_end()

            # Verify log_episode_metrics was called
            mock_log_ep.assert_called_once_with(
                self.mock_metrics_logger.writer, 10000, 100.0, 50
            )

    def test_episode_metrics_not_logged_when_none(self):
        """Test that episode metrics are NOT logged when values are None."""
        mock_model = Mock()
        mock_model.logger = MockLogger(
            {
                "train/loss": 0.5,
                "rollout/ep_rew_mean": None,
                "rollout/ep_len_mean": None,
            }
        )

        self.callback.model = mock_model

        # Mock log_episode_metrics function
        with patch("train.train_self_play.log_episode_metrics") as mock_log_ep:
            self.callback._on_rollout_end()

            # Verify log_episode_metrics was NOT called (values are None)
            mock_log_ep.assert_not_called()

    def test_model_without_logger_attribute(self):
        """Test callback handles model without logger attribute."""
        mock_model = Mock()
        # No logger attribute
        del mock_model.logger

        self.callback.model = mock_model

        # Execute - should not raise
        self.callback._on_rollout_end()

        # Verify set_step was still called
        self.mock_metrics_logger.set_step.assert_called_once_with(10000)

        # Verify log_training_metrics was NOT called (no logger data)
        self.mock_metrics_logger.log_training_metrics.assert_not_called()

    def test_model_with_none_logger(self):
        """Test callback handles model with None logger."""
        mock_model = Mock()
        mock_model.logger = None

        self.callback.model = mock_model

        # Execute - should not raise
        self.callback._on_rollout_end()

        # Verify set_step was still called
        self.mock_metrics_logger.set_step.assert_called_once_with(10000)

        # Verify log_training_metrics was NOT called (logger is None)
        self.mock_metrics_logger.log_training_metrics.assert_not_called()

    def test_model_logger_without_name_to_value(self):
        """Test callback handles logger without name_to_value attribute."""
        mock_model = Mock(spec=[])
        mock_model.logger = Mock(spec=[])  # No name_to_value attribute

        self.callback.model = mock_model

        # Execute - should not raise
        self.callback._on_rollout_end()

        # Verify set_step was still called
        self.mock_metrics_logger.set_step.assert_called_once_with(10000)

        # Verify log_training_metrics was NOT called
        self.mock_metrics_logger.log_training_metrics.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
