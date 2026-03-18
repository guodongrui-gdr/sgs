"""
Stable-Baselines3 训练脚本

支持:
- PPO/MaskablePPO 训练
- 自博弈训练
- 模型保存/加载
- TensorBoard 日志
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger("engine.game_engine").setLevel(logging.INFO)
logging.getLogger("ai.gym_wrapper").setLevel(logging.INFO)

import numpy as np

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    DQN = None
    A2C = None

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from sb3_contrib.common.maskable.evaluation import evaluate_policy

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    MaskablePPO = None

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.gym_wrapper import SGSEnv, SGSConfig, GYM_AVAILABLE
from ai.state_encoder import EncodingConfig
from ai.action_encoder import ActionConfig
from ai.reward import RewardConfig


class TrainingConfig:
    def __init__(
        self,
        total_timesteps: int = 1_000_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.05,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 42,
        n_envs: int = 8,
        algorithm: str = "ppo",
        use_masking: bool = True,
        use_transformer: bool = False,
        checkpoint_freq: int = 10000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        log_dir: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        if n_envs > 1 and n_steps == 2048:
            self.n_steps = max(256, 2048 // n_envs)
        else:
            self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.n_envs = n_envs
        self.algorithm = algorithm
        self.use_masking = use_masking
        self.use_transformer = use_transformer
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = log_dir or self._default_log_dir()
        self.model_path = model_path

    def _default_log_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path(__file__).parent / "logs" / f"sgs_{timestamp}")


class SelfPlayCallback(BaseCallback):
    def __init__(
        self,
        update_freq: int = 10000,
        save_freq: int = 50000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_freq = save_freq
        self.policy_pool: List[str] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self._update_opponents()

        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()

        return True

    def _update_opponents(self):
        if self.verbose > 0:
            print(f"Updating opponents at step {self.n_calls}")

    def _save_checkpoint(self):
        if self.verbose > 0:
            print(f"Saving checkpoint at step {self.n_calls}")


def make_sgs_env(config: SGSConfig, seed: int = 0) -> Callable:
    def _init():
        env = SGSEnv(config)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def create_env(
    config: SGSConfig, n_envs: int = 1, seed: int = 0, use_subprocess: bool = False
) -> Any:
    if n_envs == 1 or not use_subprocess:
        return DummyVecEnv([make_sgs_env(config, seed + i) for i in range(n_envs)])
    else:
        return SubprocVecEnv([make_sgs_env(config, seed + i) for i in range(n_envs)])


def create_model(
    env: Any,
    config: TrainingConfig,
    use_masking: bool = True,
):
    policy = "MultiInputPolicy"
    policy_kwargs = None

    if config.use_transformer:
        try:
            from ai.models.transformer_policy import (
                TransformerFeaturesExtractor,
                TransformerConfig,
            )

            transformer_config = TransformerConfig()
            policy_kwargs = dict(
                features_extractor_class=TransformerFeaturesExtractor,
                features_extractor_kwargs={
                    "config": transformer_config,
                    "state_dim": env.observation_space["state"].shape[0]
                    if hasattr(env.observation_space, "spaces")
                    else 3000,
                },
            )
            print("Using Transformer policy network")
        except ImportError as e:
            print(f"Warning: Transformer not available, falling back to MLP: {e}")

    if use_masking and MASKABLE_PPO_AVAILABLE:
        model = MaskablePPO(
            policy,
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            tensorboard_log=config.log_dir,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    elif config.algorithm == "ppo" and SB3_AVAILABLE:
        model = PPO(
            policy,
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            tensorboard_log=config.log_dir,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    elif config.algorithm == "dqn" and SB3_AVAILABLE:
        model = DQN(
            policy,
            env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            gamma=config.gamma,
            seed=config.seed,
            tensorboard_log=config.log_dir,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    elif config.algorithm == "a2c" and SB3_AVAILABLE:
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            gamma=config.gamma,
            seed=config.seed,
            tensorboard_log=config.log_dir,
            verbose=1,
        )
    else:
        raise ImportError(
            "No RL algorithm available. Install stable-baselines3 or sb3-contrib."
        )

    return model


def train(config: TrainingConfig, sgs_config: SGSConfig = None):
    if not GYM_AVAILABLE:
        raise ImportError(
            "gymnasium or gym is required. Install with: pip install gymnasium"
        )

    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        )

    sgs_config = sgs_config or SGSConfig()

    os.makedirs(config.log_dir, exist_ok=True)

    env = create_env(sgs_config, config.n_envs, config.seed)

    use_masking = config.use_masking and MASKABLE_PPO_AVAILABLE
    model = create_model(env, config, use_masking)

    if config.model_path and os.path.exists(config.model_path):
        print(f"Loading model from {config.model_path}")
        if use_masking and MASKABLE_PPO_AVAILABLE:
            model = MaskablePPO.load(config.model_path, env=env)
        elif config.algorithm == "ppo":
            model = PPO.load(config.model_path, env=env)
        elif config.algorithm == "dqn":
            model = DQN.load(config.model_path, env=env)
        elif config.algorithm == "a2c":
            model = A2C.load(config.model_path, env=env)

    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=os.path.join(config.log_dir, "checkpoints"),
        name_prefix="sgs_model",
    )
    callbacks.append(checkpoint_callback)

    eval_env = create_env(sgs_config, 1, config.seed + 1000, use_subprocess=False)
    if use_masking and MASKABLE_PPO_AVAILABLE:
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(config.log_dir, "best_model"),
            log_path=os.path.join(config.log_dir, "eval"),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
        )
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(config.log_dir, "best_model"),
            log_path=os.path.join(config.log_dir, "eval"),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
        )
    callbacks.append(eval_callback)

    self_play_callback = SelfPlayCallback()
    callbacks.append(self_play_callback)

    print(f"Starting training for {config.total_timesteps} timesteps...")
    print(f"Log directory: {config.log_dir}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Use masking: {use_masking}")

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_model_path = os.path.join(config.log_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    env.close()
    eval_env.close()

    return model


def evaluate(model_path: str, n_episodes: int = 100, config: SGSConfig = None):
    if not GYM_AVAILABLE:
        raise ImportError("gymnasium or gym is required")

    config = config or SGSConfig()
    env = SGSEnv(config)

    if MASKABLE_PPO_AVAILABLE:
        model = MaskablePPO.load(model_path, env=env)
        use_masking = True
    elif SB3_AVAILABLE:
        model = PPO.load(model_path, env=env)
        use_masking = False
    else:
        raise ImportError("No RL algorithm available")

    wins = 0
    total_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if use_masking:
                action_masks = env.action_masks()
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)
        if "winner" in info:
            player_identity = info.get("player_identity", "")
            winner = info.get("winner", "")
            if player_identity in ["主公", "忠臣"] and winner == "主公":
                wins += 1
            elif player_identity == "反贼" and winner == "反贼":
                wins += 1
            elif player_identity == "内奸" and winner == "内奸":
                wins += 1

    win_rate = wins / n_episodes
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"Evaluation results over {n_episodes} episodes:")
    print(f"  Win rate: {win_rate:.2%}")
    print(f"  Average reward: {avg_reward:.2f} +/- {std_reward:.2f}")

    env.close()

    return win_rate, avg_reward, std_reward


def main():
    parser = argparse.ArgumentParser(description="Train SGS RL agent")
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "evaluate"]
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--algorithm", type=str, default="ppo", choices=["ppo", "dqn", "a2c"]
    )
    parser.add_argument("--use-masking", action="store_true", default=True)
    parser.add_argument(
        "--use-transformer",
        action="store_true",
        default=False,
        help="Use Transformer policy network instead of MLP",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--player-num", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    sgs_config = SGSConfig(
        player_num=args.player_num,
        max_rounds=args.max_rounds,
    )

    if args.mode == "train":
        training_config = TrainingConfig(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            algorithm=args.algorithm,
            use_masking=args.use_masking,
            use_transformer=args.use_transformer,
            seed=args.seed,
            log_dir=args.log_dir,
            model_path=args.model_path,
            learning_rate=args.lr,
        )

        train(training_config, sgs_config)

    elif args.mode == "evaluate":
        if not args.model_path:
            print("Error: --model-path is required for evaluation mode")
            return

        evaluate(args.model_path, args.n_eval_episodes, sgs_config)


if __name__ == "__main__":
    main()
