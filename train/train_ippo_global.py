"""
IPPO Global State Training Script - Phase 0 Validation

Implements IPPO (Independent PPO) with global state baseline for SGS.

Features:
- MaskablePPO from sb3_contrib with action masking
- Global state observation (concatenated agent states)
- Independent policy learning per agent
- Baseline metrics logging: win_rate, sample_efficiency, coordination_score
- TensorBoard logging
- Checkpoint saving

Usage:
    # Standard training (10K steps)
    python train/train_ippo_global.py --steps 10000 --n-envs 4

    # Quick test
    python train/train_ippo_global.py --steps 100 --n-envs 1

    # Full training
    python train/train_ippo_global.py --steps 100000 --eval-interval 10000
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable_baselines3 not available")

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    logger.warning("sb3_contrib not available")

from ai.multi_agent_env_global import GlobalStateWrapper, GlobalStateConfig
from ai.gym_wrapper import SGSConfig


class MetricsCallback(BaseCallback):
    """
    Custom callback for logging IPPO baseline metrics.

    Logs:
    - win_rate: Game win percentage
    - sample_efficiency: Steps per episode
    - coordination_score: Identity-based win alignment
    - episode_length: Average episode steps
    - episode_reward: Average episode reward
    """

    def __init__(
        self,
        log_dir: Path,
        eval_interval: int = 1000,
        num_eval_episodes: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

        self._episode_rewards = []
        self._episode_lengths = []
        self._win_count = 0
        self._total_episodes = 0
        self._identity_wins: Dict[str, int] = {}
        self._last_eval_step = 0

        self.metrics_file = log_dir / "metrics.csv"
        self._init_metrics_file()

    def _init_metrics_file(self):
        header = (
            "step,win_rate,avg_reward,avg_length,sample_efficiency,coordination_score\n"
        )
        with open(self.metrics_file, "w") as f:
            f.write(header)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if info is None:
                continue
            if isinstance(info, dict):
                wrapped_info = info.get("infos", [info])
            else:
                wrapped_info = [info]

            for i in wrapped_info:
                if isinstance(i, dict) and "winner" in i:
                    winner = i.get("winner", "")
                    if winner:
                        self._win_count += 1
                        self._identity_wins[winner] = (
                            self._identity_wins.get(winner, 0) + 1
                        )
                        self._total_episodes += 1

        rewards = self.locals.get("rewards", [])
        for reward in rewards:
            if reward is not None and isinstance(reward, (int, float)):
                self._episode_rewards.append(reward)

        if (
            self.n_calls % self.eval_interval == 0
            and self.n_calls > self._last_eval_step
        ):
            self._last_eval_step = self.n_calls
            self._log_metrics()

        return True

    def _on_rollout_end(self) -> None:
        ep_info = self.model.ep_info_buffer
        if ep_info:
            for ep in ep_info:
                self._episode_lengths.append(ep.get("l", 0))
                self._episode_rewards.append(ep.get("r", 0))
                self._total_episodes += 1

    def _log_metrics(self):
        avg_reward = np.mean(self._episode_rewards) if self._episode_rewards else 0.0
        avg_length = np.mean(self._episode_lengths) if self._episode_lengths else 0.0

        win_rate = self._win_count / max(self._total_episodes, 1)

        sample_efficiency = avg_length if avg_length > 0 else 0.0

        coordination_score = self._compute_coordination_score()

        line = f"{self.num_timesteps},{win_rate:.4f},{avg_reward:.4f},{avg_length:.1f},{sample_efficiency:.1f},{coordination_score:.4f}\n"
        with open(self.metrics_file, "a") as f:
            f.write(line)

        logger.info(
            f"Step {self.num_timesteps}: win_rate={win_rate:.2%}, "
            f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}, "
            f"coordination={coordination_score:.2f}"
        )

    def _compute_coordination_score(self) -> float:
        if not self._identity_wins:
            return 0.5

        expected_wins = {"主公": 0.25, "忠臣": 0.25, "反贼": 0.40, "内奸": 0.10}
        total_wins = sum(self._identity_wins.values())
        if total_wins == 0:
            return 0.5

        score = 0.0
        for identity, expected in expected_wins.items():
            actual = self._identity_wins.get(identity, 0) / total_wins
            score += min(actual, expected) / expected if expected > 0 else 0

        return score / len(expected_wins)


def mask_fn(env) -> np.ndarray:
    """Extract action mask for MaskablePPO"""
    if hasattr(env, "action_masks"):
        return env.action_masks()
    return np.ones(20, dtype=np.float32)


def make_env(player_num: int, rank: int = 0):
    """Create environment factory for VecEnv"""

    def _init():
        config = GlobalStateConfig(player_num=player_num)
        env = GlobalStateWrapper(config)
        return env

    return _init


def create_vec_env(n_envs: int, player_num: int, use_subprocess: bool = True):
    """Create vectorized environment for training"""
    env_fns = [make_env(player_num, rank=i) for i in range(n_envs)]

    if use_subprocess and n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    return env


class ActionMaskerWrapper:
    """Wrapper to add action masking for MaskablePPO"""

    def __init__(self, env):
        self.env = env

    def action_masks(self):
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()
        return np.ones(20, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def train_ippo_global(
    steps: int = 10000,
    n_envs: int = 4,
    eval_interval: int = 1000,
    player_num: int = 5,
    log_dir: Optional[Path] = None,
    test_mode: bool = False,
) -> Optional[any]:
    """
    Main training function for IPPO global baseline.

    Args:
        steps: Total training steps
        n_envs: Number of parallel environments
        eval_interval: Evaluation/logging interval
        player_num: Number of players
        log_dir: Custom log directory
        test_mode: Quick test mode

    Returns:
        Trained MaskablePPO model
    """
    if not SB3_AVAILABLE or not MASKABLE_PPO_AVAILABLE:
        raise ImportError("Required: pip install stable-baselines3 sb3-contrib")

    if test_mode:
        steps = min(steps, 100)
        n_envs = 1
        eval_interval = 10
        logger.info("Test mode: reduced steps and single env")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = Path("train/logs") / f"ippo_global_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Log directory: {log_dir}")

    logger.info(f"Creating {n_envs} parallel environments...")
    env = create_vec_env(n_envs, player_num, use_subprocess=(n_envs > 1))

    single_state_dim = 3066
    global_state_dim = single_state_dim * player_num
    action_dim = 20

    logger.info(
        f"State dimensions: single={single_state_dim}, global={global_state_dim}"
    )

    from gymnasium import spaces

    observation_space = spaces.Dict(
        {
            "global_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(global_state_dim,), dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, shape=(action_dim,), dtype=np.float32
            ),
        }
    )
    action_space = spaces.Discrete(action_dim)

    model = MaskablePPO(
        "MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
    )

    metrics_callback = MetricsCallback(
        log_dir=log_dir,
        eval_interval=eval_interval,
        verbose=1,
    )

    callbacks = CallbackList([metrics_callback])

    logger.info(f"Starting IPPO global training for {steps} steps...")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=steps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        emergency_path = log_dir / "interrupted_model.zip"
        model.save(str(emergency_path))
        logger.info(f"Saved emergency model to {emergency_path}")

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    final_path = log_dir / "ippo_global_model.zip"
    model.save(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    env.close()

    evidence_path = log_dir / "training_evidence.log"
    with open(evidence_path, "w") as f:
        f.write(f"IPPO Global Training Evidence\n")
        f.write(f"============================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total steps: {steps}\n")
        f.write(f"Environments: {n_envs}\n")
        f.write(f"Players: {player_num}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Global state dim: {global_state_dim}\n")
        f.write(f"Model saved: {final_path}\n")
        f.write(f"\nMetrics logged to: {metrics_callback.metrics_file}\n")

    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="IPPO Global State Training for SGS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Total training steps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Evaluation/logging interval",
    )
    parser.add_argument(
        "--player-num",
        type=int,
        default=5,
        help="Number of players",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom log directory",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Quick test mode (100 steps)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point"""
    args = parse_args()

    try:
        model = train_ippo_global(
            steps=args.steps,
            n_envs=args.n_envs,
            eval_interval=args.eval_interval,
            player_num=args.player_num,
            log_dir=Path(args.log_dir) if args.log_dir else None,
            test_mode=args.test_mode,
        )
        logger.info("Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
