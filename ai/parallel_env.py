"""
并行环境支持 - 用于加速训练

功能:
- 多进程并行环境
- 动作掩码支持
- 正确的观察空间处理
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable

try:
    from stable_baselines3.common.vec_env import (
        VecEnv,
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )
    from stable_baselines3.common.monitor import Monitor

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    DummyVecEnv = object
    SubprocVecEnv = object

try:
    from sb3_contrib.common.maskable.utils import get_action_masks

    MASKABLE_AVAILABLE = True
except ImportError:
    MASKABLE_AVAILABLE = False


class MaskableVecEnv:
    """
    支持动作掩码的并行环境包装器

    用于MaskablePPO等需要动作掩码的算法
    """

    def __init__(self, vec_env: "VecEnv"):
        self.vec_env = vec_env

    def action_masks(self) -> np.ndarray:
        """获取所有环境的动作掩码"""
        masks = []
        for env_idx in range(self.vec_env.num_envs):
            env = self.vec_env.envs[env_idx]
            if hasattr(env, "action_masks"):
                masks.append(env.action_masks())
            elif hasattr(env, "get_attr"):
                masks.append(env.get_attr("action_masks")[0])
            else:
                mask_shape = self._get_default_mask_shape(env_idx)
                masks.append(np.ones(mask_shape, dtype=np.float32))
        return np.array(masks)

    def _get_default_mask_shape(self, env_idx: int) -> Tuple[int, ...]:
        """获取默认掩码形状"""
        env = self.vec_env.envs[env_idx]
        if hasattr(env, "action_space"):
            return (env.action_space.n,)
        return (100,)

    def __getattr__(self, name):
        return getattr(self.vec_env, name)

    def __getitem__(self, idx):
        return self.vec_env[idx]


def make_maskable_env(env_factory: Callable, seed: int = 0):
    """
    创建支持动作掩码的环境

    Args:
        env_factory: 环境工厂函数
        seed: 随机种子

    Returns:
        包装后的环境
    """

    def _init():
        env = env_factory()
        if hasattr(env, "reset"):
            env.reset(seed=seed)
        return Monitor(env)

    return _init


def create_parallel_envs(
    env_factory: Callable,
    n_envs: int = 1,
    seed: int = 0,
    use_subprocess: bool = None,
) -> Any:
    """
    创建并行环境

    Args:
        env_factory: 环境工厂函数
        n_envs: 环境数量
        seed: 随机种子
        use_subprocess: 是否使用多进程（None则自动选择）

    Returns:
        向量化环境
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required")

    if use_subprocess is None:
        use_subprocess = n_envs > 1

    if n_envs == 1:
        env = DummyVecEnv([make_maskable_env(env_factory, seed)])
        return env

    if use_subprocess:
        env_fns = [make_maskable_env(env_factory, seed + i) for i in range(n_envs)]
        env = SubprocVecEnv(env_fns)
        return env
    else:
        env_fns = [make_maskable_env(env_factory, seed + i) for i in range(n_envs)]
        env = DummyVecEnv(env_fns)
        return env


def get_masks_from_env(env: Any) -> np.ndarray:
    """
    从环境获取动作掩码

    支持多种环境类型
    """
    if hasattr(env, "action_masks"):
        return env.action_masks()

    if hasattr(env, "get_attr"):
        try:
            masks = env.get_attr("action_masks")
            return np.array(masks)
        except Exception:
            pass

    if hasattr(env, "num_envs"):
        return np.ones((env.num_envs, 100), dtype=np.float32)

    return np.ones((1, 100), dtype=np.float32)


class ParallelSGSEnv:
    """
    三国杀并行环境

    专门为三国杀游戏优化的并行环境
    """

    def __init__(
        self,
        config_factory: Callable,
        n_envs: int = 4,
        seed: int = 0,
    ):
        from ai.gym_wrapper import SGSEnv

        self.n_envs = n_envs
        self.config_factory = config_factory

        def make_env():
            return SGSEnv(config_factory())

        self.vec_env = create_parallel_envs(
            make_env,
            n_envs=n_envs,
            seed=seed,
            use_subprocess=n_envs > 2,
        )

        self.num_envs = n_envs
        self.observation_space = self.vec_env.observation_space
        self.action_space = self.vec_env.action_space

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        return self.vec_env.reset(**kwargs)

    def step(
        self, actions
    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        return self.vec_env.step(actions)

    def action_masks(self) -> np.ndarray:
        return get_masks_from_env(self.vec_env)

    def close(self):
        self.vec_env.close()

    def __getattr__(self, name):
        return getattr(self.vec_env, name)
