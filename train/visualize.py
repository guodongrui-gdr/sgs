"""
TensorBoard logging utilities for SGS RL training.

Provides functions for setting up TensorBoard logging and logging
various training metrics including loss, entropy, learning rate,
win rates, and ELO ratings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning(
        "tensorboard not available. Install with: pip install tensorboard. "
        "Logging will be disabled."
    )
    SummaryWriter = None


# Default log directory base
DEFAULT_LOG_DIR = Path("train/logs")


def setup_tensorboard(
    run_name: str,
    log_dir: Optional[Union[str, Path]] = None,
    purge_step: Optional[int] = None,
):
    """
    Set up TensorBoard SummaryWriter for training logging.

    Creates the log directory at train/logs/{run_name}/tensorboard and
    returns a SummaryWriter instance for logging metrics.

    Args:
        run_name: Name of the training run (used in directory naming)
        log_dir: Optional custom base log directory (default: train/logs/)
        purge_step: Optional step to purge all data before

    Returns:
        SummaryWriter instance configured for the run, or None if tensorboard unavailable

    Example:
        >>> writer = setup_tensorboard("my_experiment")
        >>> writer.add_scalar("train/loss", 0.5, step=0)
    """
    if not TENSORBOARD_AVAILABLE:
        logger.warning("TensorBoard not available, returning None")
        return None

    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    else:
        log_dir = Path(log_dir)

    # Create the specific run directory
    run_log_dir = log_dir / run_name / "tensorboard"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_log_dir), purge_step=purge_step)

    return writer


def log_training_metrics(
    writer,
    step: int,
    loss: Optional[float] = None,
    entropy: Optional[float] = None,
    learning_rate: Optional[float] = None,
    value_loss: Optional[float] = None,
    policy_loss: Optional[float] = None,
    approx_kl: Optional[float] = None,
    clip_fraction: Optional[float] = None,
    explained_variance: Optional[float] = None,
) -> None:
    """
    Log training metrics to TensorBoard.

    Logs various training metrics including loss, entropy, and learning rate
    at the specified training step. All metrics are optional - only provided
    values will be logged.

    Args:
        writer: SummaryWriter instance (can be None if tensorboard unavailable)
        step: Current training step (used as x-axis)
        loss: Total loss value
        entropy: Policy entropy value
        learning_rate: Current learning rate
        value_loss: Value function loss
        policy_loss: Policy gradient loss
        approx_kl: Approximate KL divergence
        clip_fraction: Fraction of clipped surrogate loss
        explained_variance: Value function explained variance

    Example:
        >>> writer = setup_tensorboard("run_1")
        >>> log_training_metrics(writer, step=1000, loss=0.5, entropy=0.2, lr=3e-4)
    """
    if writer is None:
        return

    if loss is not None:
        writer.add_scalar("train/loss", loss, step)

    if entropy is not None:
        writer.add_scalar("train/entropy", entropy, step)

    if learning_rate is not None:
        writer.add_scalar("train/learning_rate", learning_rate, step)

    if value_loss is not None:
        writer.add_scalar("train/value_loss", value_loss, step)

    if policy_loss is not None:
        writer.add_scalar("train/policy_loss", policy_loss, step)

    if approx_kl is not None:
        writer.add_scalar("train/approx_kl", approx_kl, step)

    if clip_fraction is not None:
        writer.add_scalar("train/clip_fraction", clip_fraction, step)

    if explained_variance is not None:
        writer.add_scalar("train/explained_variance", explained_variance, step)

    # Flush to ensure data is written
    writer.flush()


def log_win_rate(
    writer,
    step: int,
    win_rate: float,
    total_games: Optional[int] = None,
    identity: Optional[str] = None,
) -> None:
    """
    Log win rate metrics from evaluation results.

    Logs win rate (0.0 to 1.0) and optionally the total number of games
    and per-identity win rates.

    Args:
        writer: SummaryWriter instance (can be None if tensorboard unavailable)
        step: Current training step
        win_rate: Win rate as float (0.0 to 1.0)
        total_games: Optional total number of games played
        identity: Optional identity string (主公/忠臣/反贼/内奸) for per-identity tracking

    Example:
        >>> log_win_rate(writer, step=50000, win_rate=0.65, total_games=100)
        >>> log_win_rate(writer, step=50000, win_rate=0.70, identity="主公")
    """
    if writer is None:
        return

    if identity:
        writer.add_scalar(f"eval/win_rate_{identity}", win_rate, step)
    else:
        writer.add_scalar("eval/win_rate", win_rate, step)

    if total_games is not None:
        writer.add_scalar("eval/total_games", total_games, step)

    writer.flush()


def log_elo(
    writer,
    step: int,
    elo_rating: float,
    version: Optional[int] = None,
    elo_change: Optional[float] = None,
    opponent_elo: Optional[float] = None,
) -> None:
    """
    Log ELO rating changes for agent pool.

    Logs ELO rating for an agent and optionally the version number,
    ELO change from last update, and opponent ELO for comparison.

    Args:
        writer: SummaryWriter instance (can be None if tensorboard unavailable)
        step: Current training step
        elo_rating: Current ELO rating
        version: Optional agent version identifier
        elo_change: Optional ELO rating change
        opponent_elo: Optional opponent ELO rating

    Example:
        >>> log_elo(writer, step=100000, elo_rating=1250, version=5, elo_change=+25)
    """
    if writer is None:
        return

    writer.add_scalar("elo/rating", elo_rating, step)

    if version is not None:
        writer.add_scalar("elo/version", version, step)

    if elo_change is not None:
        writer.add_scalar("elo/change", elo_change, step)

    if opponent_elo is not None:
        writer.add_scalar("elo/opponent_rating", opponent_elo, step)
        # Calculate and log rating difference
        writer.add_scalar("elo/rating_diff", elo_rating - opponent_elo, step)

    writer.flush()


def log_episode_metrics(
    writer,
    step: int,
    episode_reward: float,
    episode_length: int,
    episode_num: Optional[int] = None,
) -> None:
    """
    Log per-episode metrics.

    Args:
        writer: SummaryWriter instance (can be None if tensorboard unavailable)
        step: Current training step
        episode_reward: Total reward for the episode
        episode_length: Number of steps in the episode
        episode_num: Optional episode number
    """
    if writer is None:
        return

    writer.add_scalar("rollout/ep_rew_mean", episode_reward, step)
    writer.add_scalar("rollout/ep_len_mean", episode_length, step)

    if episode_num is not None:
        writer.add_scalar("rollout/episodes", episode_num, step)

    writer.flush()


def log_identity_win_rates(
    writer,
    step: int,
    identity_win_rates: Dict[str, float],
) -> None:
    """
    Log win rates for all identities simultaneously.

    Args:
        writer: SummaryWriter instance (can be None if tensorboard unavailable)
        step: Current training step
        identity_win_rates: Dict mapping identity names to win rates
            e.g., {"主公": 0.65, "忠臣": 0.45, "反贼": 0.55, "内奸": 0.20}
    """
    if writer is None:
        return

    for identity, win_rate in identity_win_rates.items():
        writer.add_scalar(f"eval/win_rate_{identity}", win_rate, step)

    writer.flush()


def log_hyperparameters(
    writer,
    hyperparams: Dict[str, Union[int, float, str]],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Log hyperparameters and optional associated metrics.

    Args:
        writer: SummaryWriter instance (can be None if tensorboard unavailable)
        hyperparams: Dictionary of hyperparameter names and values
        metrics: Optional dictionary of metric names and values
    """
    if writer is None:
        return

    # Convert all values to strings for hparams
    hparams = {k: str(v) for k, v in hyperparams.items()}

    if metrics:
        writer.add_hparams(hparams, metrics)
    else:
        writer.add_hparams(hparams, {})

    writer.flush()


def close_tensorboard(writer) -> None:
    """
    Close TensorBoard writer and flush all pending data.

    Args:
        writer: SummaryWriter instance to close (can be None)
    """
    if writer is None:
        return

    writer.flush()
    writer.close()


# Convenience class for batch logging
class MetricsLogger:
    """
    Convenience wrapper for logging metrics during training.

    Provides a simpler interface for common logging patterns and
    handles step tracking automatically.

    Example:
        >>> logger = MetricsLogger("my_run")
        >>> logger.log_training_metrics(loss=0.5, entropy=0.2, lr=3e-4)
        >>> logger.log_win_rate(0.65, total_games=100)
        >>> logger.log_elo(1250, elo_change=+25)
        >>> logger.step()  # Increment step counter
        >>> logger.close()
    """

    def __init__(
        self,
        run_name: str,
        log_dir: Optional[Union[str, Path]] = None,
        start_step: int = 0,
    ):
        """
        Initialize MetricsLogger.

        Args:
            run_name: Name of the training run
            log_dir: Optional custom log directory
            start_step: Starting step number (default: 0)
        """
        self.writer = setup_tensorboard(run_name, log_dir)
        self.step_count = start_step

    def step(self, increment: int = 1) -> None:
        """Increment the step counter."""
        self.step_count += increment

    def set_step(self, step: int) -> None:
        """Set the step counter to a specific value."""
        self.step_count = step

    def log_training_metrics(
        self,
        loss: Optional[float] = None,
        entropy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Log training metrics at current step."""
        log_training_metrics(
            self.writer,
            self.step_count,
            loss=loss,
            entropy=entropy,
            learning_rate=learning_rate,
            **kwargs,
        )

    def log_win_rate(
        self,
        win_rate: float,
        total_games: Optional[int] = None,
        identity: Optional[str] = None,
    ) -> None:
        """Log win rate at current step."""
        log_win_rate(self.writer, self.step_count, win_rate, total_games, identity)

    def log_elo(
        self,
        elo_rating: float,
        version: Optional[int] = None,
        elo_change: Optional[float] = None,
        opponent_elo: Optional[float] = None,
    ) -> None:
        """Log ELO rating at current step."""
        log_elo(
            self.writer,
            self.step_count,
            elo_rating,
            version,
            elo_change,
            opponent_elo,
        )

    def log_identity_win_rates(self, identity_win_rates: Dict[str, float]) -> None:
        """Log all identity win rates at current step."""
        log_identity_win_rates(self.writer, self.step_count, identity_win_rates)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        close_tensorboard(self.writer)
