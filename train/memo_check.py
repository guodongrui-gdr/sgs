"""
OOM Detection Module for MAPPO Training

Monitors CUDA memory usage during training and provides warnings
when approaching out-of-memory conditions.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class OOMConfig:
    """Configuration for OOM detection thresholds."""

    warn_threshold: float = 0.85
    """Fraction of total memory at which to emit warning."""

    reduce_threshold: float = 0.90
    """Fraction of total memory at which to auto-reduce batch size."""

    critical_threshold: float = 0.95
    """Fraction of total memory considered critical."""

    min_batch_size: int = 1
    """Minimum batch size to auto-reduce to."""

    enabled: bool = True
    """Whether OOM detection is enabled."""


class OOMDetector:
    """
    Monitors CUDA memory usage and provides OOM detection/warning.

    Usage:
        detector = OOMDetector()
        if detector.should_warn():
            detector.log_warning()
        if detector.should_reduce():
            new_batch = detector.reduce_batch(current_batch)
    """

    def __init__(self, config: Optional[OOMConfig] = None):
        """
        Initialize OOM detector.

        Args:
            config: Configuration for thresholds. Uses defaults if None.
        """
        self.config = config or OOMConfig()
        self._check_count = 0
        self._warning_count = 0
        self._reduction_count = 0
        self._last_utilization = 0.0

        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA not available. OOMDetector will report 0 utilization.")

    def _get_memory_stats(self) -> dict:
        """
        Get current CUDA memory statistics.

        Returns:
            Dictionary with memory statistics in bytes and GB.
        """
        if not self.cuda_available:
            return {
                "allocated": 0,
                "total": 0,
                "reserved": 0,
                "free": 0,
                "utilization": 0.0,
                "allocated_gb": 0.0,
                "total_gb": 0.0,
            }

        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved()
        free = total - allocated

        return {
            "allocated": allocated,
            "total": total,
            "reserved": reserved,
            "free": free,
            "utilization": allocated / total if total > 0 else 0.0,
            "allocated_gb": allocated / (1024**3),
            "total_gb": total / (1024**3),
        }

    def check(self) -> dict:
        """
        Perform a memory check and return status.

        Returns:
            Dictionary with memory stats and status flags.
        """
        if not self.config.enabled:
            return {
                "enabled": False,
                "should_warn": False,
                "should_reduce": False,
                "should_critical": False,
                "utilization": 0.0,
            }

        self._check_count += 1
        stats = self._get_memory_stats()
        self._last_utilization = stats["utilization"]

        should_warn = stats["utilization"] >= self.config.warn_threshold
        should_reduce = stats["utilization"] >= self.config.reduce_threshold
        should_critical = stats["utilization"] >= self.config.critical_threshold

        if should_warn:
            self._warning_count += 1
        if should_reduce:
            self._reduction_count += 1

        return {
            "enabled": True,
            "should_warn": should_warn,
            "should_reduce": should_reduce,
            "should_critical": should_critical,
            "utilization": stats["utilization"],
            "allocated_gb": stats["allocated_gb"],
            "total_gb": stats["total_gb"],
            "free_gb": stats["free"] / (1024**3),
        }

    def should_warn(self) -> bool:
        """
        Check if memory usage exceeds warning threshold.

        Returns:
            True if allocated / total > warn_threshold.
        """
        result = self.check()
        return result.get("should_warn", False)

    def should_reduce(self) -> bool:
        """
        Check if memory usage exceeds auto-reduce threshold.

        Returns:
            True if allocated / total > reduce_threshold.
        """
        result = self.check()
        return result.get("should_reduce", False)

    def should_critical(self) -> bool:
        """
        Check if memory usage is at critical level.

        Returns:
            True if allocated / total > critical_threshold.
        """
        result = self.check()
        return result.get("should_critical", False)

    def log_warning(self, level: Optional[str] = None) -> None:
        """
        Log a memory warning with current usage.

        Args:
            level: Optional override for warning level ('warn', 'critical').
        """
        stats = self._get_memory_stats()
        util_pct = stats["utilization"] * 100
        allocated_gb = stats["allocated_gb"]
        total_gb = stats["total_gb"]
        free_gb = stats["free"] / (1024**3)

        if (
            level == "critical"
            or stats["utilization"] >= self.config.critical_threshold
        ):
            logger.warning(
                f"[OOM CRITICAL] Memory at {util_pct:.1f}% ({allocated_gb:.2f}GB / {total_gb:.2f}GB). "
                f"Free: {free_gb:.2f}GB. Immediate action recommended!"
            )
        elif level == "reduce" or stats["utilization"] >= self.config.reduce_threshold:
            logger.warning(
                f"[OOM WARNING] Memory at {util_pct:.1f}% ({allocated_gb:.2f}GB / {total_gb:.2f}GB). "
                f"Free: {free_gb:.2f}GB. Consider reducing batch size."
            )
        else:
            logger.warning(
                f"[OOM NOTICE] Memory at {util_pct:.1f}% ({allocated_gb:.2f}GB / {total_gb:.2f}GB). "
                f"Free: {free_gb:.2f}GB."
            )

    def reduce_batch(self, current_batch_size: int) -> int:
        """
        Auto-reduce batch size by 2x, respecting minimum.

        Args:
            current_batch_size: Current batch size.

        Returns:
            New batch size (at least min_batch_size).
        """
        new_size = max(current_batch_size // 2, self.config.min_batch_size)

        if new_size < current_batch_size:
            logger.warning(
                f"[OOM AUTO-REDUCE] Batch size: {current_batch_size} -> {new_size} "
                f"(memory utilization: {self._last_utilization * 100:.1f}%)"
            )
        elif new_size == current_batch_size == self.config.min_batch_size:
            logger.error(
                f"[OOM AUTO-REDUCE] Cannot reduce further! Already at minimum batch size {new_size}. "
                f"(memory utilization: {self._last_utilization * 100:.1f}%)"
            )

        return new_size

    def get_stats(self) -> dict:
        """
        Get accumulated statistics from this detector.

        Returns:
            Dictionary with check counts and utilization history.
        """
        return {
            "check_count": self._check_count,
            "warning_count": self._warning_count,
            "reduction_count": self._reduction_count,
            "last_utilization": self._last_utilization,
            "cuda_available": self.cuda_available,
        }

    def reset_stats(self) -> None:
        """Reset accumulated statistics."""
        self._check_count = 0
        self._warning_count = 0
        self._reduction_count = 0

    def force_clear_cache(self) -> None:
        """Force clear CUDA cache to free memory. Use sparingly."""
        if self.cuda_available:
            torch.cuda.empty_cache()
            logger.info("[OOM] CUDA cache cleared.")


def get_detector_from_env() -> OOMDetector:
    """
    Create OOMDetector from environment variables.

    Environment variables:
        OOM_WARN_THRESHOLD: Warning threshold (default 0.85)
        OOM_REDUCE_THRESHOLD: Auto-reduce threshold (default 0.90)
        OOM_CRITICAL_THRESHOLD: Critical threshold (default 0.95)
        OOM_MIN_BATCH: Minimum batch size (default 1)
        OOM_ENABLED: Enable/disable detection (default 1)

    Returns:
        Configured OOMDetector instance.
    """
    config = OOMConfig(
        warn_threshold=float(os.environ.get("OOM_WARN_THRESHOLD", "0.85")),
        reduce_threshold=float(os.environ.get("OOM_REDUCE_THRESHOLD", "0.90")),
        critical_threshold=float(os.environ.get("OOM_CRITICAL_THRESHOLD", "0.95")),
        min_batch_size=int(os.environ.get("OOM_MIN_BATCH", "1")),
        enabled=os.environ.get("OOM_ENABLED", "1").lower() in ("1", "true", "yes"),
    )
    return OOMDetector(config)


# Convenience function for integration in training loop
def check_memory_before_rollout(
    detector: Optional[OOMDetector] = None,
) -> tuple[bool, int]:
    """
    Check memory before rollout and return if action needed.

    Args:
        detector: Optional detector instance (creates new if None).

    Returns:
        Tuple of (should_reduce: bool, recommended_batch_size: int).
    """
    if detector is None:
        detector = get_detector_from_env()

    result = detector.check()

    if result.get("should_critical", False):
        detector.log_warning("critical")
        return True, detector.config.min_batch_size
    elif result.get("should_reduce", False):
        detector.log_warning("reduce")
        return True, 0  # Signal to reduce (actual size determined later)
    elif result.get("should_warn", False):
        detector.log_warning()

    return False, 0
