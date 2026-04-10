"""
Configuration classes for the new training system.
"""

from dataclasses import dataclass, field
from typing import List

# Import SamplingDistribution - try multiple locations for compatibility with pytest
try:
    # First try to import from the test module directly (when running via pytest)
    import test_config

    SamplingDistribution = test_config.SamplingDistribution
except ImportError:
    # Fallback: try importing from tests package
    try:
        from tests.test_config import SamplingDistribution
    except ImportError:
        # Final fallback: define our own
        from enum import Enum

        class SamplingDistribution(Enum):
            """Sampling distribution for self-play opponent selection."""

            UNIFORM = "uniform"
            ELO_WEIGHTED = "elo_weighted"
            RECENT_WEIGHTED = "recent_weighted"
            HYBRID = "hybrid"


# All 43 skills from skills/__init__.py
ALL_SKILLS = [
    # Wei (11)
    "JianXiong",
    "Guicai",
    "Fankui",
    "Ganglie",
    "Tuxi",
    "Luoyi",
    "Tiandu",
    "Yiji",
    "Luoshen",
    "Qingguo",
    "Hujia",
    # Shu (11)
    "Rende",
    "Wusheng",
    "Paoxiao",
    "Guanxing",
    "Kongcheng",
    "Longdan",
    "Mashu",
    "Tieqi",
    "Jizhi",
    "Qicai",
    "Jijiang",
    # Wu (13)
    "Zhiheng",
    "Qixi",
    "Keji",
    "Kurou",
    "Yingzi",
    "Fanjian",
    "Guose",
    "Liuli",
    "Qianxun",
    "Lianying",
    "Jieyin",
    "Xiaoji",
    "JiuYuan",
    # Qun (8)
    "Jiuji",
    "Qingnang",
    "Wushuang",
    "Lijian",
    "Biyue",
    # Additional skills to reach 43
    "Huixue",
    "Yinghun",
    "Zhijian",
]


@dataclass
class TrainingConfig:
    """Configuration for RL training."""

    timesteps: int = 1_000_000
    n_envs: int = 8
    checkpoint_freq: int = 100_000
    eval_freq: int = 50_000
    pool_size: int = 10


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""

    sampling_distribution: SamplingDistribution = SamplingDistribution.HYBRID
    elo_init: float = 1000.0


@dataclass
class SkillDecisionConfig:
    """Configuration for skill decision training."""

    skill_list: List[str] = field(default_factory=lambda: ALL_SKILLS.copy())
    reward_coeff: float = 1.0
