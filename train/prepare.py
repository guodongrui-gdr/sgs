"""
prepare.py - 固定配置和评估函数（不可编辑）

这个文件包含：
- 固定常量（时间预算、评估配置）
- 环境创建函数
- 评估函数
- 运行时工具

agent 不能修改此文件，只能修改 train.py。
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Callable

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger("engine.game_engine").setLevel(logging.WARNING)
logging.getLogger("ai.gym_wrapper").setLevel(logging.WARNING)

import numpy as np

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    from sb3_contrib import MaskablePPO

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.gym_wrapper import SGSEnv, SGSConfig

# ============ 固定常量（agent 不可修改） ============

FIXED_TIME_BUDGET = 600  # 10分钟（秒）
EVAL_EPISODES = 50  # 评估回合数
PLAYER_NUM = 5  # 固定玩家数
MAX_ROUNDS = 15  # 固定回合数（快速实验）

MAX_VRAM_MB = 8192  # VRAM 上限（MB）
MIN_EVAL_EPISODES = 10  # 最小评估回合数

# 评估阈值（用于判断改进）
MIN_WIN_RATE_IMPROVEMENT = 0.01  # 胜率改进阈值
MIN_REWARD_IMPROVEMENT = 1.0  # 奖励改进阈值

# ============ 环境创建函数（agent 不可修改） ============


def make_env(seed: int = 42) -> SGSEnv:
    """
    创建环境 - agent 不能修改环境配置

    Args:
        seed: 随机种子

    Returns:
        SGSEnv 环境实例
    """
    config = SGSConfig(
        player_num=PLAYER_NUM,
        max_rounds=MAX_ROUNDS,
    )
    env = SGSEnv(config)
    env.reset(seed=seed)
    return Monitor(env)


def make_vec_env(n_envs: int = 1, seed: int = 42) -> DummyVecEnv:
    """
    创建向量化环境

    Args:
        n_envs: 环境数量
        seed: 随机种子

    Returns:
        DummyVecEnv 实例
    """

    def make_env_fn(rank: int):
        def _init():
            env = make_env(seed + rank)
            return env

        return _init

    return DummyVecEnv([make_env_fn(i) for i in range(n_envs)])


# ============ 评估函数（agent 不可修改） ============


def evaluate_model(
    model,
    n_episodes: int = EVAL_EPISODES,
    deterministic: bool = True,
    verbose: bool = True,
) -> Tuple[float, float, float, dict]:
    """
    评估模型 - agent 不能修改评估逻辑

    Args:
        model: RL 模型
        n_episodes: 评估回合数
        deterministic: 是否使用确定性策略
        verbose: 是否打印详细信息

    Returns:
        (win_rate, mean_reward, std_reward, info_dict)
    """
    env = make_env(seed=1000)

    wins = 0
    episode_rewards = []
    episode_lengths = []

    use_masking = MASKABLE_PPO_AVAILABLE and hasattr(model, "action_masks")

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            if use_masking:
                action_masks = env.action_masks()
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

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
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    if verbose:
        print(f"Evaluation over {n_episodes} episodes:")
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Mean episode length: {mean_length:.1f}")

    env.close()

    info_dict = {
        "win_rate": win_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "n_episodes": n_episodes,
    }

    return win_rate, mean_reward, std_reward, info_dict


# ============ 内存监控工具 ============


def get_memory_usage_mb() -> float:
    """
    获取当前进程内存使用（MB）

    使用标准库方法，不依赖 psutil
    """
    try:
        import resource

        # 获取 RSS（ Resident Set Size）
        mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux 上单位是 KB，转换为 MB
        if mem_bytes > 0:
            return mem_bytes / 1024.0
        return 0.0
    except:
        # 如果无法获取，返回 0
        return 0.0


def get_peak_memory_mb() -> float:
    """
    获取峰值内存使用（MB）

    使用标准库方法，不依赖 psutil
    """
    return get_memory_usage_mb()


# ============ 结果输出函数（agent 不可修改） ============


def print_results(
    win_rate: float,
    mean_reward: float,
    std_reward: float,
    training_seconds: float,
    total_steps: int,
    peak_memory_mb: float,
    fps: float,
    model_params: Optional[dict] = None,
):
    """
    输出标准格式结果 - agent 不能修改输出格式

    这个格式必须保持一致，以便 grep 提取结果
    """
    print("---")
    print(f"win_rate:         {win_rate:.6f}")
    print(f"mean_reward:      {mean_reward:.6f}")
    print(f"std_reward:       {std_reward:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_steps:      {total_steps}")
    print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
    print(f"fps:              {fps:.1f}")

    if model_params:
        for key, value in model_params.items():
            print(f"{key}: {value}")


# ============ 时间控制工具 ============


class TimeBudgetTracker:
    """
    时间预算跟踪器 - 确保训练在固定时间内停止
    """

    def __init__(self, budget_seconds: float = FIXED_TIME_BUDGET):
        self.budget_seconds = budget_seconds
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def remaining(self) -> float:
        return self.budget_seconds - self.elapsed()

    def should_stop(self) -> bool:
        return self.elapsed() >= self.budget_seconds


# ============ 验证函数 ============


def verify_environment():
    """
    验证环境是否正常工作
    """
    try:
        env = make_env(seed=42)
        obs, _ = env.reset()

        # 获取底层环境（Monitor 包装器）
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        if MASKABLE_PPO_AVAILABLE:
            action_masks = base_env.action_masks()
            valid_actions = np.where(action_masks > 0)[0]
            if len(valid_actions) > 0:
                action = valid_actions[0]
                obs, reward, terminated, truncated, info = env.step(action)

        env.close()
        return True
    except Exception as e:
        print(f"Environment verification failed: {e}")
        return False


def verify_dependencies():
    """
    验证依赖是否安装
    """
    if not SB3_AVAILABLE:
        print("Error: stable-baselines3 not installed")
        return False

    if not MASKABLE_PPO_AVAILABLE:
        print("Warning: sb3-contrib not installed, masking disabled")

    return True


# ============ 主入口（用于验证） ============

if __name__ == "__main__":
    print("=== Environment Verification ===")

    if verify_dependencies():
        print("✓ Dependencies OK")

    if verify_environment():
        print("✓ Environment OK")

    print("\nFixed constants:")
    print(f"  Time budget: {FIXED_TIME_BUDGET}s ({FIXED_TIME_BUDGET / 60:.1f} min)")
    print(f"  Eval episodes: {EVAL_EPISODES}")
    print(f"  Player num: {PLAYER_NUM}")
    print(f"  Max rounds: {MAX_ROUNDS}")
    print(f"  Max VRAM: {MAX_VRAM_MB}MB")
