"""
自博弈训练器 - 使用自博弈进行强化学习训练

功能:
- 自博弈训练循环
- 策略池集成
- 周期性评估
- 训练监控
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    BaseCallback = object

try:
    from sb3_contrib import MaskablePPO

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    MaskablePPO = None

from ai.gym_wrapper import SGSEnv, SGSConfig
from ai.multi_agent_env import SelfPlayEnv, make_multi_agent_env
from ai.policy_pool import PolicyPool, MatchHistory


@dataclass
class SelfPlayConfig:
    total_timesteps: int = 10_000_000
    save_freq: int = 100_000
    eval_freq: int = 50_000
    update_opponent_freq: int = 25_000
    n_eval_games: int = 100

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    player_num: int = 5
    max_rounds: int = 100

    pool_size: int = 10
    sample_latest_prob: float = 0.3
    sample_best_prob: float = 0.3

    log_dir: str = ""
    seed: int = 42

    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = str(
                Path("logs") / f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )


class SelfPlayTrainer:
    """
    自博弈训练器

    训练智能体通过与自己或其他历史版本对战来提升
    """

    def __init__(self, config: SelfPlayConfig = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for self-play training")

        self.config = config or SelfPlayConfig()

        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.policy_pool = PolicyPool(
            pool_dir=str(self.log_dir / "policy_pool"),
            max_size=self.config.pool_size,
            sample_latest_prob=self.config.sample_latest_prob,
            sample_best_prob=self.config.sample_best_prob,
        )

        self.match_history = MatchHistory()

        self.env: Optional[SelfPlayEnv] = None
        self.model: Optional[Any] = None
        self.current_policy_version: int = 0

        self._step_count: int = 0
        self._episode_count: int = 0
        self._win_count: int = 0
        self._total_reward: float = 0.0

        self._training_metrics: List[Dict] = []

    def setup(self):
        sgs_config = SGSConfig(
            player_num=self.config.player_num,
            max_rounds=self.config.max_rounds,
        )

        self.env = make_multi_agent_env(
            player_num=self.config.player_num,
            training_agent_idx=0,
        )

        if MASKABLE_PPO_AVAILABLE:
            self.model = MaskablePPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                seed=self.config.seed,
                tensorboard_log=str(self.log_dir / "tensorboard"),
                verbose=1,
            )
        else:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                seed=self.config.seed,
                tensorboard_log=str(self.log_dir / "tensorboard"),
                verbose=1,
            )

    def train(self):
        if self.model is None:
            self.setup()

        print(
            f"Starting self-play training for {self.config.total_timesteps} timesteps"
        )
        print(f"Log directory: {self.log_dir}")

        start_time = time.time()

        while self._step_count < self.config.total_timesteps:
            obs, _ = self.env.reset(seed=self.config.seed + self._episode_count)

            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action = self._select_action(obs)
                next_obs, reward, terminated, truncated, info = (
                    self.env.step_with_policy(self.model)
                )

                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                self._step_count += 1

                if self._step_count % self.config.save_freq == 0:
                    self._save_checkpoint()

                if self._step_count % self.config.eval_freq == 0:
                    self._evaluate()

                if self._step_count % self.config.update_opponent_freq == 0:
                    self._update_opponents()

                obs = next_obs

            self._episode_count += 1
            self._total_reward += episode_reward

            winner = info.get("winner")
            training_identity = self.env.players[0].identity if self.env.players else ""

            if winner and self._check_win(training_identity, winner):
                self._win_count += 1

            self._log_episode(episode_reward, episode_length, info)

        total_time = time.time() - start_time

        print(f"\nTraining completed!")
        print(f"Total timesteps: {self._step_count}")
        print(f"Total episodes: {self._episode_count}")
        print(f"Win rate: {self._win_count / self._episode_count:.2%}")
        print(f"Average reward: {self._total_reward / self._episode_count:.2f}")
        print(f"Total time: {total_time / 3600:.2f} hours")

        self._save_final_model()

    def _select_action(self, obs: Dict) -> int:
        legal_actions = self.env.get_legal_actions(0)

        if not legal_actions:
            return 0

        action_mask = self.env.get_action_mask(0)

        if MASKABLE_PPO_AVAILABLE:
            action, _ = self.model.predict(
                obs, action_masks=action_mask, deterministic=False
            )
        else:
            action, _ = self.model.predict(obs, deterministic=False)

        if action not in legal_actions:
            action = np.random.choice(legal_actions)

        return action

    def _check_win(self, training_identity: str, winner: str) -> bool:
        if training_identity in ["主公", "忠臣"] and winner == "主公":
            return True
        if training_identity == "反贼" and winner == "反贼":
            return True
        if training_identity == "内奸" and winner == "内奸":
            return True
        return False

    def _save_checkpoint(self):
        checkpoint_path = (
            self.log_dir / "checkpoints" / f"model_step_{self._step_count}"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(checkpoint_path))

        self.current_policy_version += 1
        self.policy_pool.add_policy(
            str(checkpoint_path),
            parent_version=self.current_policy_version - 1,
        )

        print(f"Saved checkpoint at step {self._step_count}")

        self._save_metrics()

    def _evaluate(self):
        print(f"\nEvaluating at step {self._step_count}...")

        wins = 0
        total_reward = 0.0

        for _ in range(self.config.n_eval_games):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action_mask = self.env.get_action_mask(0)
                if MASKABLE_PPO_AVAILABLE:
                    action, _ = self.model.predict(
                        obs, action_masks=action_mask, deterministic=True
                    )
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = self.env.step_with_policy(
                    self.model
                )
                done = terminated or truncated
                episode_reward += reward

            total_reward += episode_reward

            winner = info.get("winner")
            training_identity = self.env.players[0].identity if self.env.players else ""
            if winner and self._check_win(training_identity, winner):
                wins += 1

        win_rate = wins / self.config.n_eval_games
        avg_reward = total_reward / self.config.n_eval_games

        print(f"Evaluation results ({self.config.n_eval_games} games):")
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Average reward: {avg_reward:.2f}")

        self._training_metrics.append(
            {
                "step": self._step_count,
                "episode": self._episode_count,
                "win_rate": win_rate,
                "avg_reward": avg_reward,
            }
        )

    def _update_opponents(self):
        print(f"Updating opponents at step {self._step_count}")

        if len(self.policy_pool) > 0:
            sampled_policy = self.policy_pool.sample_policy()
            if sampled_policy:
                print(
                    f"  Sampled policy v{sampled_policy.version} (ELO: {sampled_policy.elo_rating:.0f})"
                )

    def _log_episode(self, episode_reward: float, episode_length: int, info: Dict):
        if self._episode_count % 100 == 0:
            win_rate = self._win_count / max(self._episode_count, 1)
            avg_reward = self._total_reward / max(self._episode_count, 1)
            print(
                f"Episode {self._episode_count} | "
                f"Step {self._step_count} | "
                f"Win rate: {win_rate:.2%} | "
                f"Avg reward: {avg_reward:.2f}"
            )

    def _save_metrics(self):
        metrics_path = self.log_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._training_metrics, f, indent=2)

    def _save_final_model(self):
        final_path = self.log_dir / "final_model"
        self.model.save(str(final_path))

        self.policy_pool.add_policy(str(final_path))

        print(f"Final model saved to {final_path}")

    def load_model(self, path: str):
        if MASKABLE_PPO_AVAILABLE:
            try:
                self.model = MaskablePPO.load(path, env=self.env)
                print(f"Loaded MaskablePPO model from {path}")
                return
            except Exception:
                pass

        self.model = PPO.load(path, env=self.env)
        print(f"Loaded PPO model from {path}")


def run_self_play(
    total_timesteps: int = 10_000_000,
    player_num: int = 5,
    log_dir: str = None,
    **kwargs,
):
    config = SelfPlayConfig(
        total_timesteps=total_timesteps,
        player_num=player_num,
        log_dir=log_dir,
        **kwargs,
    )

    trainer = SelfPlayTrainer(config)
    trainer.train()

    return trainer.model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-play training for SGS")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--player-num", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_self_play(
        total_timesteps=args.timesteps,
        player_num=args.player_num,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        seed=args.seed,
    )
