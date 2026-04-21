#!/usr/bin/env python3
"""
MAPPO Training Performance Benchmark Script

Captures baseline performance metrics for MAPPO training to enable
before/after comparison for optimization work.

Usage:
    .venv/bin/python tests/benchmark_training.py --mode mappo --steps 1000 --n-envs 1
    .venv/bin/python tests/benchmark_training.py --mode mappo --steps 100 --n-envs 1 --output results.json
"""

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from threading import Thread
import queue

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics"""

    # Timing
    total_time_sec: float = 0.0
    training_start_time: str = ""
    training_end_time: str = ""

    # Throughput
    total_steps: int = 0
    total_samples: int = 0
    steps_per_sec: float = 0.0
    samples_per_sec: float = 0.0

    # GPU metrics
    gpu_util_avg: float = 0.0
    gpu_util_max: float = 0.0
    gpu_util_min: float = 100.0
    gpu_util_samples: List[float] = field(default_factory=list)
    gpu_memory_avg_mb: float = 0.0

    # Training config
    n_envs: int = 0
    steps_per_rollout: int = 0
    batch_size: int = 0

    # Episode metrics
    episodes_completed: int = 0
    avg_episode_length: float = 0.0
    episode_lengths: List[int] = field(default_factory=list)

    # System info
    device: str = ""
    cuda_available: bool = False
    gpu_name: str = ""

    def __post_init__(self):
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to string if needed
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkMetrics":
        """Create from dictionary"""
        return cls(**data)


class GPUUtilizationMonitor:
    """Monitor GPU utilization in background thread"""

    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.samples: List[Dict[str, float]] = []
        self._stop_event = False
        self._thread: Optional[Thread] = None
        self._queue: queue.Queue = queue.Queue()

    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_event:
            try:
                # Query GPU utilization via nvidia-smi
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    line = result.stdout.strip()
                    if line:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                util = float(parts[0].strip())
                                mem_mb = float(parts[1].strip())
                                self.samples.append(
                                    {
                                        "utilization": util,
                                        "memory_mb": mem_mb,
                                        "timestamp": time.time(),
                                    }
                                )
                            except ValueError:
                                pass
                else:
                    # nvidia-smi not available, try torch
                    self._query_torch()

            except Exception as e:
                logger.debug(f"GPU query failed: {e}")

            time.sleep(self.sample_interval)

    def _query_torch(self):
        """Fallback to torch CUDA metrics"""
        try:
            import torch

            if torch.cuda.is_available():
                # Get memory stats
                mem_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                mem_reserved = torch.cuda.memory_reserved() / (1024**2)  # MB

                # Estimate utilization based on memory activity
                self.samples.append(
                    {
                        "utilization": -1,  # Unknown
                        "memory_mb": mem_allocated,
                        "timestamp": time.time(),
                    }
                )
        except ImportError:
            pass

    def start(self):
        """Start monitoring"""
        self._stop_event = False
        self._thread = Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("GPU monitoring started")

    def stop(self):
        """Stop monitoring"""
        self._stop_event = True
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info(f"GPU monitoring stopped, collected {len(self.samples)} samples")

    def get_average_utilization(self) -> float:
        """Get average GPU utilization"""
        if not self.samples:
            return 0.0
        utils = [s["utilization"] for s in self.samples if s["utilization"] >= 0]
        return sum(utils) / len(utils) if utils else 0.0

    def get_average_memory(self) -> float:
        """Get average GPU memory usage"""
        if not self.samples:
            return 0.0
        return sum(s["memory_mb"] for s in self.samples) / len(self.samples)

    def get_max_utilization(self) -> float:
        """Get max GPU utilization"""
        if not self.samples:
            return 0.0
        utils = [s["utilization"] for s in self.samples if s["utilization"] >= 0]
        return max(utils) if utils else 0.0

    def get_min_utilization(self) -> float:
        """Get min GPU utilization"""
        if not self.samples:
            return 100.0
        utils = [s["utilization"] for s in self.samples if s["utilization"] >= 0]
        return min(utils) if utils else 100.0


def get_gpu_memory_usage() -> float:
    """Get current GPU memory utilization percentage."""
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            if reserved > 0:
                return (allocated / reserved) * 100.0
            return 0.0
    except Exception:
        pass
    return 0.0


_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[INFO] Shutdown requested, finishing current iteration...")


def benchmark_mappo_training(
    steps: int = 1000,
    n_envs: int = 1,
    use_mappo: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark MAPPO training performance.

    Args:
        steps: Total training steps to run
        n_envs: Number of parallel environments
        use_mappo: Use MAPPO mode (True) or IPPO mode (False)

    Returns:
        Dictionary with benchmark metrics
    """
    global _shutdown_requested
    _shutdown_requested = False

    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    results = {
        "mode": "mappo" if use_mappo else "ippo",
        "steps_requested": steps,
        "n_envs": n_envs,
        "steps_completed": 0,
        "total_time": 0.0,
        "steps_per_second": 0.0,
        "time_per_rollout": 0.0,
        "time_per_update": 0.0,
        "gpu_memory_usage": 0.0,
        "rollout_times": [],
        "update_times": [],
        "episodes_completed": 0,
        "shutdown_early": False,
    }

    try:
        from train.train_mappo_world_model import (
            MAPPOWorldModelTrainer,
            MAPPOWorldModelConfig,
        )

        config = MAPPOWorldModelConfig(
            steps_total=steps,
            n_envs=n_envs,
            steps_per_rollout=min(256, steps),
            batch_size=64,
            use_mappo=use_mappo,
            use_world_model=False,
            use_team_rewards=False,
            checkpoint_interval=steps + 1,
            eval_interval=steps + 1,
        )

        print(f"\n=== Benchmark: MAPPO Training (mode={results['mode']}) ===")
        print(f"  Steps: {steps}")
        print(f"  Environments: {n_envs}")
        print(f"  Steps per rollout: {config.steps_per_rollout}")

        trainer = MAPPOWorldModelTrainer(config)

        start_time = time.time()
        rollout_times = []
        update_times = []

        if n_envs > 1:
            env = trainer._create_vec_env(n_envs, use_subprocess=False)
        else:
            env = trainer._create_env()
        steps_per_rollout = config.steps_per_rollout
        step_count = 0

        print("\n  Running benchmark...")

        while step_count < steps and not _shutdown_requested:
            rollout_start = time.time()
            rollout = trainer._collect_real_rollout(env, steps_per_rollout)
            rollout_time = time.time() - rollout_start
            rollout_times.append(rollout_time)

            update_start = time.time()
            metrics = trainer.agent.update(
                rollout,
                n_epochs=config.n_epochs,
                batch_size=config.batch_size,
            )
            update_time = time.time() - update_start
            update_times.append(update_time)

            step_count += steps_per_rollout

            gpu_mem = get_gpu_memory_usage()
            if gpu_mem > results["gpu_memory_usage"]:
                results["gpu_memory_usage"] = gpu_mem

            print(
                f"    Step {step_count}/{steps} - "
                f"rollout: {rollout_time:.3f}s, update: {update_time:.3f}s"
            )

        total_time = time.time() - start_time

        results["steps_completed"] = step_count
        results["total_time"] = total_time
        results["episodes_completed"] = trainer._episode_count
        results["rollout_times"] = rollout_times
        results["update_times"] = update_times
        results["shutdown_early"] = _shutdown_requested

        if step_count > 0:
            results["steps_per_second"] = step_count / total_time
            results["time_per_rollout"] = (
                sum(rollout_times) / len(rollout_times) if rollout_times else 0.0
            )
            results["time_per_update"] = (
                sum(update_times) / len(update_times) if update_times else 0.0
            )

        print(f"\n  === Results ===")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Steps completed: {step_count}")
        print(f"  Steps/second: {results['steps_per_second']:.2f}")
        print(f"  Avg time/rollout: {results['time_per_rollout']:.3f}s")
        print(f"  Avg time/update: {results['time_per_update']:.3f}s")
        print(f"  GPU memory usage: {results['gpu_memory_usage']:.1f}%")
        print(f"  Episodes completed: {results['episodes_completed']}")

        env.close()

    except ImportError as e:
        print(f"  ERROR: Failed to import training modules: {e}")
        results["error"] = str(e)
    except Exception as e:
        print(f"  ERROR: Benchmark failed: {e}")
        results["error"] = str(e)
    finally:
        signal.signal(signal.SIGINT, original_handler)

    return results


class TrainingBenchmark:
    """Benchmark runner for training scripts"""

    def __init__(
        self,
        steps: int = 1000,
        n_envs: int = 8,
        steps_per_rollout: int = 64,
        batch_size: int = 64,
    ):
        self.steps = steps
        self.n_envs = n_envs
        self.steps_per_rollout = steps_per_rollout
        self.batch_size = batch_size
        self.metrics = BenchmarkMetrics(
            n_envs=n_envs,
            steps_per_rollout=steps_per_rollout,
            batch_size=batch_size,
        )
        self.gpu_monitor = GPUUtilizationMonitor(sample_interval=1.0)

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        info = {
            "cuda_available": False,
            "gpu_name": "",
            "device": "cpu",
        }

        try:
            import torch

            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["device"] = f"cuda:{torch.cuda.current_device()}"
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        # Try nvidia-smi as fallback
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                info["gpu_name"] = result.stdout.strip().split("\n")[0]
        except Exception:
            pass

        return info

    def _extract_episode_metrics(self, log_file: Optional[Path]) -> Dict[str, Any]:
        """Extract episode metrics from training logs"""
        metrics = {
            "episodes_completed": 0,
            "episode_lengths": [],
        }

        # This would parse log files if available
        # For now, we rely on the training script output
        return metrics

    def run_training(self) -> BenchmarkMetrics:
        """Run training and collect metrics"""
        logger.info("=" * 60)
        logger.info("Starting Training Benchmark")
        logger.info("=" * 60)
        logger.info(f"Steps: {self.steps}")
        logger.info(f"Environments: {self.n_envs}")
        logger.info(f"Steps per rollout: {self.steps_per_rollout}")
        logger.info(f"Batch size: {self.batch_size}")

        # Get GPU info
        gpu_info = self._get_gpu_info()
        self.metrics.cuda_available = gpu_info["cuda_available"]
        self.metrics.gpu_name = gpu_info["gpu_name"]
        self.metrics.device = gpu_info["device"]

        logger.info(f"Device: {self.metrics.device}")
        if self.metrics.gpu_name:
            logger.info(f"GPU: {self.metrics.gpu_name}")

        # Start GPU monitoring
        if self.metrics.cuda_available:
            self.gpu_monitor.start()

        # Build training command
        train_script = (
            Path(__file__).parent.parent / "train" / "train_mappo_world_model.py"
        )

        cmd = [
            sys.executable,
            str(train_script),
            "--steps",
            str(self.steps),
            "--n-envs",
            str(self.n_envs),
            "--steps-per-rollout",
            str(self.steps_per_rollout),
            "--batch-size",
            str(self.batch_size),
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        # Run training with timing
        self.metrics.training_start_time = datetime.now().isoformat()
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            # Capture stdout/stderr for analysis
            stdout = result.stdout
            stderr = result.stderr

            # Parse output for metrics
            self._parse_training_output(stdout, stderr)

            if result.returncode != 0:
                logger.warning(f"Training exited with code {result.returncode}")
                logger.warning(f"stderr: {stderr[:500]}")

        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
        except Exception as e:
            logger.error(f"Training failed: {e}")

        end_time = time.time()
        self.metrics.training_end_time = datetime.now().isoformat()

        # Stop GPU monitoring
        if self.metrics.cuda_available:
            self.gpu_monitor.stop()

        # Calculate timing metrics
        self.metrics.total_time_sec = end_time - start_time
        self.metrics.total_steps = self.steps
        self.metrics.total_samples = self.steps * self.n_envs

        if self.metrics.total_time_sec > 0:
            self.metrics.steps_per_sec = (
                self.metrics.total_steps / self.metrics.total_time_sec
            )
            self.metrics.samples_per_sec = (
                self.metrics.total_samples / self.metrics.total_time_sec
            )

        # Calculate GPU metrics
        if self.metrics.cuda_available and self.gpu_monitor.samples:
            self.metrics.gpu_util_avg = self.gpu_monitor.get_average_utilization()
            self.metrics.gpu_util_max = self.gpu_monitor.get_max_utilization()
            self.metrics.gpu_util_min = self.gpu_monitor.get_min_utilization()
            self.metrics.gpu_memory_avg_mb = self.gpu_monitor.get_average_memory()
            self.metrics.gpu_util_samples = [
                s["utilization"] for s in self.gpu_monitor.samples
            ]

        return self.metrics

    def _parse_training_output(self, stdout: str, stderr: str):
        """Parse training output for metrics"""
        # Look for episode information in output
        # This is a simplified parser - can be extended based on actual output format

        lines = stdout.split("\n") + stderr.split("\n")

        for line in lines:
            # Look for episode completion markers
            if "episode" in line.lower() and "completed" in line.lower():
                self.metrics.episodes_completed += 1

            # Look for episode length info
            if "episode length" in line.lower():
                try:
                    # Try to extract number
                    import re

                    match = re.search(r"(\d+)", line)
                    if match:
                        self.metrics.episode_lengths.append(int(match.group(1)))
                except Exception:
                    pass

        # Calculate average episode length
        if self.metrics.episode_lengths:
            self.metrics.avg_episode_length = sum(self.metrics.episode_lengths) / len(
                self.metrics.episode_lengths
            )

    def save_baseline(self, output_path: Path):
        """Save baseline metrics to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "benchmark_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "metrics": self.metrics.to_dict(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Baseline metrics saved to {output_path}")

    def print_report(self):
        """Print benchmark report"""
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Time: {self.metrics.total_time_sec:.2f} seconds")
        logger.info(f"Total Steps: {self.metrics.total_steps}")
        logger.info(f"Total Samples: {self.metrics.total_samples}")
        logger.info(f"Steps/sec: {self.metrics.steps_per_sec:.2f}")
        logger.info(f"Samples/sec: {self.metrics.samples_per_sec:.2f}")
        logger.info(f"\nGPU Utilization:")
        logger.info(f"  Average: {self.metrics.gpu_util_avg:.1f}%")
        logger.info(f"  Min: {self.metrics.gpu_util_min:.1f}%")
        logger.info(f"  Max: {self.metrics.gpu_util_max:.1f}%")
        logger.info(f"\nEpisode Metrics:")
        logger.info(f"  Completed: {self.metrics.episodes_completed}")
        logger.info(f"  Avg Length: {self.metrics.avg_episode_length:.1f}")
        logger.info("=" * 60)


@dataclass
class BenchmarkComparison:
    """Container for benchmark comparison results"""

    # Speedup ratios (after / before)
    samples_speedup_ratio: float = 0.0
    steps_speedup_ratio: float = 0.0

    # Absolute improvements
    time_improvement_sec: float = 0.0
    time_improvement_pct: float = 0.0
    samples_improvement_abs: float = 0.0
    samples_improvement_pct: float = 0.0
    steps_improvement_abs: float = 0.0
    steps_improvement_pct: float = 0.0
    gpu_util_improvement_abs: float = 0.0
    gpu_util_improvement_pct: float = 0.0

    # Raw values for reference
    before_samples_per_sec: float = 0.0
    after_samples_per_sec: float = 0.0
    before_steps_per_sec: float = 0.0
    after_steps_per_sec: float = 0.0
    before_time_sec: float = 0.0
    after_time_sec: float = 0.0
    before_gpu_util: float = 0.0
    after_gpu_util: float = 0.0

    # Metadata
    before_path: str = ""
    after_path: str = ""
    comparison_timestamp: str = ""
    benchmark_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkComparison":
        """Create from dictionary"""
        return cls(**data)


def compare_baselines(current: BenchmarkMetrics, baseline_path: Path):
    """Compare current metrics with saved baseline"""
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE COMPARISON")
    logger.info("=" * 60)

    if not baseline_path.exists():
        logger.error(f"Baseline file not found: {baseline_path}")
        return

    with open(baseline_path, "r") as f:
        data = json.load(f)

    baseline = BenchmarkMetrics.from_dict(data["metrics"])

    # Calculate improvements
    time_diff = current.total_time_sec - baseline.total_time_sec
    time_pct = (
        (time_diff / baseline.total_time_sec * 100)
        if baseline.total_time_sec > 0
        else 0
    )

    steps_diff = current.steps_per_sec - baseline.steps_per_sec
    steps_pct = (
        (steps_diff / baseline.steps_per_sec * 100) if baseline.steps_per_sec > 0 else 0
    )

    samples_diff = current.samples_per_sec - baseline.samples_per_sec
    samples_pct = (
        (samples_diff / baseline.samples_per_sec * 100)
        if baseline.samples_per_sec > 0
        else 0
    )

    logger.info(
        f"Time: {current.total_time_sec:.2f}s vs {baseline.total_time_sec:.2f}s "
        f"({time_diff:+.2f}s, {time_pct:+.1f}%)"
    )
    logger.info(
        f"Steps/sec: {current.steps_per_sec:.2f} vs {baseline.steps_per_sec:.2f} "
        f"({steps_diff:+.2f}, {steps_pct:+.1f}%)"
    )
    logger.info(
        f"Samples/sec: {current.samples_per_sec:.2f} vs {baseline.samples_per_sec:.2f} "
        f"({samples_diff:+.2f}, {samples_pct:+.1f}%)"
    )

    if current.gpu_util_avg > 0 and baseline.gpu_util_avg > 0:
        util_diff = current.gpu_util_avg - baseline.gpu_util_avg
        logger.info(
            f"GPU Util: {current.gpu_util_avg:.1f}% vs {baseline.gpu_util_avg:.1f}% "
            f"({util_diff:+.1f}%)"
        )

    logger.info("=" * 60)


def compare_two_benchmarks(
    before_path: Path, after_path: Path, output_path: Path
) -> BenchmarkComparison:
    """
    Compare two benchmark JSON files and generate comparison report.

    Calculates speedup ratios and improvements between before/after metrics.

    Args:
        before_path: Path to baseline metrics JSON (before optimization)
        after_path: Path to optimized metrics JSON (after optimization)
        output_path: Path to save comparison report

    Returns:
        BenchmarkComparison object with all comparison metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPARISON REPORT")
    logger.info("=" * 60)

    # Validate inputs
    if not before_path.exists():
        logger.error(f"Before baseline file not found: {before_path}")
        raise FileNotFoundError(f"Before file does not exist: {before_path}")

    if not after_path.exists():
        logger.error(f"After baseline file not found: {after_path}")
        raise FileNotFoundError(f"After file does not exist: {after_path}")

    # Load metrics
    with open(before_path, "r") as f:
        before_data = json.load(f)
    with open(after_path, "r") as f:
        after_data = json.load(f)

    before = BenchmarkMetrics.from_dict(before_data["metrics"])
    after = BenchmarkMetrics.from_dict(after_data["metrics"])

    # Calculate comparison metrics
    comparison = BenchmarkComparison()
    comparison.comparison_timestamp = datetime.now().isoformat()
    comparison.before_path = str(before_path)
    comparison.after_path = str(after_path)

    # Store raw values
    comparison.before_samples_per_sec = before.samples_per_sec
    comparison.after_samples_per_sec = after.samples_per_sec
    comparison.before_steps_per_sec = before.steps_per_sec
    comparison.after_steps_per_sec = after.steps_per_sec
    comparison.before_time_sec = before.total_time_sec
    comparison.after_time_sec = after.total_time_sec
    comparison.before_gpu_util = before.gpu_util_avg
    comparison.after_gpu_util = after.gpu_util_avg

    # Calculate speedup ratios (after / before)
    comparison.samples_speedup_ratio = (
        after.samples_per_sec / before.samples_per_sec
        if before.samples_per_sec > 0
        else 0.0
    )
    comparison.steps_speedup_ratio = (
        after.steps_per_sec / before.steps_per_sec if before.steps_per_sec > 0 else 0.0
    )

    # Calculate time improvement (lower is better)
    comparison.time_improvement_sec = before.total_time_sec - after.total_time_sec
    comparison.time_improvement_pct = (
        comparison.time_improvement_sec / before.total_time_sec * 100
        if before.total_time_sec > 0
        else 0.0
    )

    # Calculate throughput improvements
    comparison.samples_improvement_abs = after.samples_per_sec - before.samples_per_sec
    comparison.samples_improvement_pct = (
        comparison.samples_improvement_abs / before.samples_per_sec * 100
        if before.samples_per_sec > 0
        else 0.0
    )

    comparison.steps_improvement_abs = after.steps_per_sec - before.steps_per_sec
    comparison.steps_improvement_pct = (
        comparison.steps_improvement_abs / before.steps_per_sec * 100
        if before.steps_per_sec > 0
        else 0.0
    )

    # Calculate GPU utilization improvement
    comparison.gpu_util_improvement_abs = after.gpu_util_avg - before.gpu_util_avg
    comparison.gpu_util_improvement_pct = (
        comparison.gpu_util_improvement_abs / before.gpu_util_avg * 100
        if before.gpu_util_avg > 0
        else 0.0
    )

    # Print comparison report
    logger.info(f"\nBefore: {before_path}")
    logger.info(f"After:  {after_path}")
    logger.info("-" * 60)

    logger.info("\nSPEEDUP RATIOS:")
    logger.info(
        f"  Samples/sec: {comparison.samples_speedup_ratio:.2f}x "
        f"({before.samples_per_sec:.2f} → {after.samples_per_sec:.2f})"
    )
    logger.info(
        f"  Steps/sec:   {comparison.steps_speedup_ratio:.2f}x "
        f"({before.steps_per_sec:.2f} → {after.steps_per_sec:.2f})"
    )

    logger.info("\nTIME IMPROVEMENT:")
    logger.info(
        f"  Total: {comparison.time_improvement_sec:+.2f}s "
        f"({comparison.time_improvement_pct:+.1f}%)"
    )
    logger.info(
        f"  Before: {before.total_time_sec:.2f}s, After: {after.total_time_sec:.2f}s"
    )

    logger.info("\nTHROUGHPUT IMPROVEMENT:")
    logger.info(
        f"  Samples/sec: {comparison.samples_improvement_abs:+.2f} "
        f"({comparison.samples_improvement_pct:+.1f}%)"
    )
    logger.info(
        f"  Steps/sec:   {comparison.steps_improvement_abs:+.2f} "
        f"({comparison.steps_improvement_pct:+.1f}%)"
    )

    logger.info("\nGPU UTILIZATION IMPROVEMENT:")
    logger.info(
        f"  Absolute: {comparison.gpu_util_improvement_abs:+.1f}% "
        f"({before.gpu_util_avg:.1f}% → {after.gpu_util_avg:.1f}%)"
    )
    logger.info(f"  Relative: {comparison.gpu_util_improvement_pct:+.1f}%")

    # Performance verdict
    logger.info("\n" + "=" * 60)
    if comparison.samples_speedup_ratio >= 2.0:
        logger.info("✓ EXCELLENT: Achieved 2x+ speedup target")
    elif comparison.samples_speedup_ratio >= 1.5:
        logger.info("✓ GOOD: Achieved 1.5x+ speedup")
    elif comparison.samples_speedup_ratio > 1.0:
        logger.info("→ MODERATE: Some improvement achieved")
    elif comparison.samples_speedup_ratio == 1.0:
        logger.info("= NEUTRAL: No change in performance")
    else:
        logger.info("✗ REGRESSION: Performance decreased")
    logger.info("=" * 60)

    # Save comparison report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_version": comparison.benchmark_version,
        "comparison_timestamp": comparison.comparison_timestamp,
        "before_file": str(before_path),
        "after_file": str(after_path),
        "comparison": comparison.to_dict(),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nComparison report saved to: {output_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="MAPPO Training Performance Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mappo", "ippo"],
        default="mappo",
        help="Training mode: mappo (centralized) or ippo (decentralized)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Total training steps to benchmark (default: 1000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: stdout only)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--before",
        type=str,
        metavar="PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--after",
        type=str,
        metavar="PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--steps-per-rollout",
        type=int,
        default=64,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default="train/logs/benchmark_comparison.json",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MAPPO Training Performance Benchmark")
    print("=" * 60)

    results = benchmark_mappo_training(
        steps=args.steps,
        n_envs=args.n_envs,
        use_mappo=(args.mode == "mappo"),
    )

    json_output = json.dumps(results, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("JSON Output:")
    print("=" * 60)
    print(json_output)

    if "error" in results:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    main()
