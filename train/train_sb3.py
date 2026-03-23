"""
Stable-Baselines3 训练脚本

支持:
- PPO/MaskablePPO 训练
- 自博弈训练
- 模型保存/加载
- TensorBoard 日志
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Callable

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger("engine.game_engine").setLevel(logging.WARNING)
logging.getLogger("ai.gym_wrapper").setLevel(logging.WARNING)

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
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )
    from stable_baselines3.common.utils import get_linear_fn, constant_fn

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    DQN = None
    A2C = None
    get_linear_fn = None
    constant_fn = None

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


class TrainingConfig:
    def __init__(
        self,
        total_timesteps: int = 1_000_000,
        learning_rate: float = 5e-4,
        lr_schedule_type: str = "cosine",
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.98,
        clip_range: float = 0.2,
        ent_coef: float = 0.05,
        vf_coef: float = 0.25,
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
        norm_reward: bool = False,
        clip_reward: float = 100.0,
        normalize_advantage: bool = True,
    ):
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.lr_schedule_type = lr_schedule_type
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
        self.norm_reward = norm_reward
        self.clip_reward = clip_reward
        self.normalize_advantage = normalize_advantage

    def _default_log_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path(__file__).parent / "logs" / f"sgs_{timestamp}")

    def get_lr_schedule(self):
        if self.lr_schedule_type == "linear_decay":
            return get_linear_fn(self.learning_rate, self.learning_rate * 0.1, 1.0)
        elif self.lr_schedule_type == "linear_warmup":
            return get_linear_fn(self.learning_rate * 0.1, self.learning_rate, 1.0)
        elif self.lr_schedule_type == "constant":
            return constant_fn(self.learning_rate)
        elif self.lr_schedule_type == "cosine":
            return self._cosine_schedule
        elif self.lr_schedule_type == "exponential":
            return self._exponential_schedule
        elif self.lr_schedule_type == "step":
            return self._step_schedule
        else:
            return self.learning_rate

    def _cosine_schedule(self, progress_remaining: float) -> float:
        import math

        progress = 1.0 - progress_remaining
        return self.learning_rate * 0.1 + (
            self.learning_rate - self.learning_rate * 0.1
        ) * 0.5 * (1 + math.cos(math.pi * progress))

    def _exponential_schedule(self, progress_remaining: float) -> float:
        import math

        progress = 1.0 - progress_remaining
        gamma = 0.95
        return self.learning_rate * (gamma ** (progress * 10))

    def _step_schedule(self, progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        if progress < 0.33:
            return self.learning_rate
        elif progress < 0.66:
            return self.learning_rate * 0.5
        else:
            return self.learning_rate * 0.1


class SelfPlayCallback(BaseCallback):
    def __init__(
        self,
        update_freq: int = 25000,
        save_freq: int = 100000,
        pool_size: int = 10,
        log_dir: str = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_freq = save_freq
        self.pool_size = pool_size
        self.log_dir = log_dir or "./policy_pool"
        self.policy_pool = None
        self._policy_version = 0

    def _on_training_start(self) -> None:
        from ai.policy_pool import PolicyPool

        self.policy_pool = PolicyPool(
            pool_dir=self.log_dir,
            max_size=self.pool_size,
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0 and self.n_calls > 0:
            self._update_opponents()

        if self.n_calls % self.save_freq == 0 and self.n_calls > 0:
            self._save_checkpoint()

        return True

    def _update_opponents(self):
        if self.policy_pool is None:
            return

        if len(self.policy_pool) > 0:
            sampled = self.policy_pool.sample_policy()
            if sampled and self.verbose > 0:
                print(
                    f"[SelfPlay] Step {self.n_calls}: sampled policy v{sampled.version} (ELO: {sampled.elo_rating:.0f})"
                )

    def _save_checkpoint(self):
        if self.model is None:
            return

        checkpoint_path = os.path.join(
            self.log_dir, f"policy_v{self._policy_version}.zip"
        )
        self.model.save(checkpoint_path)

        if self.policy_pool is not None:
            self.policy_pool.add_policy(
                checkpoint_path, parent_version=self._policy_version - 1
            )

        self._policy_version += 1

        if self.verbose > 0:
            print(
                f"[SelfPlay] Step {self.n_calls}: saved policy v{self._policy_version}"
            )


class AsyncEvalCallback(BaseCallback):
    """异步评估回调 - 在后台进程中运行评估，不阻塞训练"""

    def __init__(
        self,
        eval_env_factory: Callable,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        best_model_save_path: str = None,
        log_path: str = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env_factory = eval_env_factory
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self._eval_process = None
        self._last_eval_step = 0
        self._best_mean_reward = -np.inf
        self.eval_results = []
        self._temp_model_path = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            if self._eval_process is None or not self._eval_process.is_alive():
                self._start_async_eval()
        return True

    def _start_async_eval(self):
        import multiprocessing as mp
        import tempfile

        # 保存当前模型到临时文件
        if self._temp_model_path is None:
            self._temp_model_path = tempfile.mktemp(suffix=".zip")
        self.model.save(self._temp_model_path)

        def run_eval(model_path, env_factory, n_episodes, result_queue):
            try:
                from sb3_contrib import MaskablePPO
                from stable_baselines3.common.vec_env import DummyVecEnv

                env = env_factory()
                model = MaskablePPO.load(model_path, env=env)

                episode_rewards = []
                for _ in range(n_episodes):
                    obs = env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        # 获取底层环境的 action_masks
                        e = env.envs[0]
                        while hasattr(e, "env"):
                            e = e.env
                        action_masks = e.action_masks()

                        action, _ = model.predict(
                            obs, deterministic=True, action_masks=action_masks
                        )
                        obs, reward, done, info = env.step(action)
                        total_reward += reward[0]
                    episode_rewards.append(total_reward)

                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                result_queue.put((mean_reward, std_reward))
            except Exception as e:
                result_queue.put((None, str(e)))

        self._result_queue = mp.Queue()
        self._eval_process = mp.Process(
            target=run_eval,
            args=(
                self._temp_model_path,
                self.eval_env_factory,
                self.n_eval_episodes,
                self._result_queue,
            ),
            daemon=True,
        )
        self._eval_process.start()

    def _on_rollout_end(self) -> None:
        # 检查之前的评估是否完成
        if self._eval_process is not None and not self._eval_process.is_alive():
            try:
                result = self._result_queue.get_nowait()
                if result[0] is not None:
                    mean_reward, std_reward = result
                    self.eval_results.append(
                        (self.num_timesteps, mean_reward, std_reward)
                    )

                    if self.verbose > 0:
                        print(
                            f"Eval at step {self.num_timesteps}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                        )

                    if mean_reward > self._best_mean_reward:
                        self._best_mean_reward = mean_reward
                        if self.best_model_save_path and self.model:
                            self.model.save(
                                os.path.join(self.best_model_save_path, "best_model")
                            )

                    if self.log_path:
                        with open(
                            os.path.join(self.log_path, "eval_results.txt"), "a"
                        ) as f:
                            f.write(
                                f"{self.num_timesteps},{mean_reward},{std_reward}\n"
                            )
            except:
                pass
            self._eval_process = None

    def _on_training_end(self) -> None:
        if self._eval_process and self._eval_process.is_alive():
            self._eval_process.join(timeout=30)
        # 清理临时文件
        if self._temp_model_path and os.path.exists(self._temp_model_path):
            os.remove(self._temp_model_path)


def make_sgs_env(config: SGSConfig, seed: int = 0) -> Callable:
    def _init():
        env = SGSEnv(config)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def create_env(
    config: SGSConfig,
    n_envs: int = 1,
    seed: int = 0,
    use_subprocess: bool = False,
    norm_reward: bool = False,
    clip_reward: float = 100.0,
    gamma: float = 0.99,
) -> VecNormalize:
    if n_envs == 1 or not use_subprocess:
        env = DummyVecEnv([make_sgs_env(config, seed + i) for i in range(n_envs)])
    else:
        env = SubprocVecEnv([make_sgs_env(config, seed + i) for i in range(n_envs)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=clip_reward,
        gamma=gamma,
        norm_obs_keys=["state"],
    )
    return env


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
        lr_schedule = config.get_lr_schedule()
        model = MaskablePPO(
            policy,
            env,
            learning_rate=lr_schedule,
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
        lr_schedule = config.get_lr_schedule()
        model = PPO(
            policy,
            env,
            learning_rate=lr_schedule,
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
            learning_rate=lr_schedule,
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


def train(
    config: TrainingConfig,
    sgs_config: SGSConfig = None,
    reset_num_timesteps: bool = True,
):
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

    vec_normalize_path = os.path.join(config.log_dir, "vec_normalize.pkl")

    env = create_env(
        sgs_config,
        config.n_envs,
        config.seed,
        norm_reward=config.norm_reward,
        clip_reward=config.clip_reward,
        gamma=config.gamma,
    )

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

        if not reset_num_timesteps:
            import re

            match = re.search(r"(\d+)_steps", config.model_path)
            if match:
                loaded_steps = int(match.group(1))
                model.num_timesteps = loaded_steps
                print(f"Restored num_timesteps to {loaded_steps}")

            if os.path.exists(vec_normalize_path):
                env = VecNormalize.load(vec_normalize_path, env)
                print(f"Loaded VecNormalize stats from {vec_normalize_path}")

    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=os.path.join(config.log_dir, "checkpoints"),
        name_prefix="sgs_model",
    )
    callbacks.append(checkpoint_callback)

    # 创建评估环境工厂函数
    def eval_env_factory():
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from ai.gym_wrapper import SGSEnv

        def make_env():
            env = SGSEnv(sgs_config)
            env.reset(seed=config.seed + 1000)
            return Monitor(env)

        return DummyVecEnv([make_env])

    async_eval_callback = AsyncEvalCallback(
        eval_env_factory=eval_env_factory,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        best_model_save_path=os.path.join(config.log_dir, "best_model"),
        log_path=os.path.join(config.log_dir, "eval"),
        verbose=1,
    )
    callbacks.append(async_eval_callback)

    self_play_callback = SelfPlayCallback(
        update_freq=25000,
        save_freq=100000,
        pool_size=10,
        log_dir=os.path.join(config.log_dir, "policy_pool"),
    )
    callbacks.append(self_play_callback)

    print(f"Starting training for {config.total_timesteps} timesteps...")
    print(f"Log directory: {config.log_dir}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Use masking: {use_masking}")
    if not reset_num_timesteps:
        print(f"Resuming training (keeping timesteps counter)")

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    final_model_path = os.path.join(config.log_dir, "final_model")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    env.save(vec_normalize_path)
    print(f"VecNormalize stats saved to {vec_normalize_path}")

    env.close()

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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from checkpoint, inheriting training state (timesteps, learning rate schedule)",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--player-num", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=[
            "linear_decay",
            "linear_warmup",
            "constant",
            "cosine",
            "exponential",
            "step",
        ],
        help="Learning rate schedule: cosine (default), linear_decay, linear_warmup, constant, exponential, step",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.98, help="GAE lambda")
    parser.add_argument(
        "--ent-coef", type=float, default=0.05, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.25, help="Value function coefficient"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048, help="Number of steps per rollout"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="Number of epochs per update"
    )
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize rewards"
    )
    parser.add_argument(
        "--clip-reward", type=float, default=100.0, help="Reward clipping value"
    )

    args = parser.parse_args()

    sgs_config = SGSConfig(
        player_num=args.player_num,
        max_rounds=args.max_rounds,
    )

    if args.mode == "train":
        resume_from_timesteps = 0
        if args.model_path and os.path.exists(args.model_path):
            import re

            match = re.search(r"(\d+)_steps", args.model_path)
            if match:
                resume_from_timesteps = int(match.group(1))
                print(f"Resuming from {resume_from_timesteps} timesteps")

        training_config = TrainingConfig(
            total_timesteps=args.timesteps + resume_from_timesteps,
            n_envs=args.n_envs,
            algorithm=args.algorithm,
            use_masking=args.use_masking,
            use_transformer=args.use_transformer,
            seed=args.seed,
            log_dir=args.log_dir,
            model_path=args.model_path,
            learning_rate=args.lr,
            lr_schedule_type=args.lr_schedule,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            norm_reward=args.norm_reward,
            clip_reward=args.clip_reward,
        )

        train(training_config, sgs_config, reset_num_timesteps=not args.resume)

    elif args.mode == "evaluate":
        if not args.model_path:
            print("Error: --model-path is required for evaluation mode")
            return

        evaluate(args.model_path, args.n_eval_episodes, sgs_config)


if __name__ == "__main__":
    main()
