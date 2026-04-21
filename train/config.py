"""
Unified Configuration for SGS RL Training

All training configurations in one place.
Supports multiple training modes: Independent, IPPO, World Model, Self-Play.

Note: MAPPO is removed because SGS (SanGuoSha) is turn-based,
      making simultaneous multi-agent steps inappropriate.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TrainingMode(Enum):
    """Training mode selection.

    Note: MAPPO is excluded because SanGuoSha is turn-based.
    """

    INDEPENDENT = "independent"  # Independent multi-agent training (default)
    IPPO = "ippo"  # Independent PPO (fully decentralized)
    WORLD_MODEL_TRAIN = "world_model_train"  # Train World Model (Phase 1)
    IPPO_WORLD_MODEL = "ippo_world_model"  # IPPO with World Model imagination (Phase 2)
    SELF_PLAY = "self_play"  # Self-play with agent pool


@dataclass
class TrainingConfig:
    """Unified configuration for all training modes."""

    # === Basic Training Parameters ===
    mode: TrainingMode = TrainingMode.INDEPENDENT
    steps_total: int = 100000
    n_envs: int = 8
    steps_per_rollout: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # === Environment ===
    num_agents: int = 5
    local_state_dim: int = 2670
    global_state_dim: int = 2670 * 5
    action_dim: int = 20
    max_rounds: int = 15
    use_action_mask: bool = True
    use_shaping: bool = True
    reveal_all_identities: bool = False

    # === Checkpointing & Logging ===
    checkpoint_interval: int = 10000
    eval_interval: int = 5000
    log_interval: int = 1000
    num_eval_episodes: int = 10
    output_dir: str = "train/logs"
    evidence_dir: str = ".sisyphus/evidence"

    # === Team Rewards ===
    use_team_rewards: bool = False
    lord_loyalist_coordination_bonus: float = 2.0
    rebel_focus_fire_bonus: float = 1.5
    protect_lord_bonus: float = 3.0

    # === Health & Safety ===
    entropy_threshold: float = 0.01
    value_std_threshold: float = 0.001
    max_safe_envs: int = 32

    # === World Model ===
    use_world_model: bool = False
    imagination_ratio: float = 0.3
    imagination_horizon: int = 15
    imagination_batch_size: int = 32
    dynamics_checkpoint_path: str = "train/logs/dynamics_training/dynamics_final.pt"

    # === World Model Training (world_model_train mode only) ===
    state_dim: int = 2670
    latent_dim: int = 128
    hidden_dim: int = 256
    stochastic_dim: int = 64
    action_embed_dim: int = 64
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    dynamics_weight: float = 1.0
    reward_weight: float = 0.5
    state_error_threshold: float = 0.1
    reward_error_threshold: float = 5.0
    # Checkpoint to collect rollouts from for world model training
    policy_checkpoint_path: str = "train/logs/independent/independent_step_100000.pt"

    # === Self-Play (self_play mode only) ===
    pool_size: int = 10
    sampling_distribution: str = "hybrid"
    elo_init: float = 1000.0

    # === Device ===
    device: Optional[str] = None

    # === Reproducibility ===
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.steps_total < 0:
            raise ValueError("steps_total must be non-negative")
        if self.n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least 2")


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training (legacy, for compatibility)."""

    sampling_distribution: str = "hybrid"
    elo_init: float = 1000.0


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
class SkillDecisionConfig:
    """Configuration for skill decision training."""

    skill_list: List[str] = field(default_factory=lambda: ALL_SKILLS.copy())
    reward_coeff: float = 1.0
