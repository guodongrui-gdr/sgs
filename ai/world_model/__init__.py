"""
World Model Module - RSSM-based imagination training

Architecture:
- RSSEncoder: State compression (3066-dim → 128-dim latent)
- DynamicsModel: (z_t, a_t) → z_{t+1} prediction
- RewardModel: (z_t, a_t) → r_t prediction
- ImaginedEnvironment: Gym-like wrapper for imagination rollouts

Usage:
    from ai.world_model import RSSEncoder, DynamicsModel, RewardModel
    from ai.world_model import RSSMConfig, DynamicsConfig, RewardModelConfig
    from ai.world_model import ImaginedEnvironment, ImaginationConfig
"""

from ai.world_model.rssm import (
    RSSEncoder,
    RSSEncoderConfig,
    RSSMDecoder,
    RSSMVAE,
    RSSMEncoderOutput,
    compute_kl_loss,
    compute_reconstruction_loss,
    create_rssm_encoder,
    create_rssm_decoder,
    create_rssm_vae,
)
from ai.world_model.dynamics import (
    DynamicsModel,
    DynamicsConfig,
    ActionEmbedding,
)
from ai.world_model.reward import (
    RewardModel,
    RewardModelConfig,
)
from ai.world_model.imagination_env import (
    ImaginedEnvironment,
    ImaginationConfig,
    ImaginationState,
    create_imagination_env,
)

__all__ = [
    "RSSEncoder",
    "RSSEncoderConfig",
    "RSSMDecoder",
    "RSSMVAE",
    "RSSMEncoderOutput",
    "compute_kl_loss",
    "compute_reconstruction_loss",
    "create_rssm_encoder",
    "create_rssm_decoder",
    "create_rssm_vae",
    "DynamicsModel",
    "DynamicsConfig",
    "ActionEmbedding",
    "RewardModel",
    "RewardModelConfig",
    "ImaginedEnvironment",
    "ImaginationConfig",
    "ImaginationState",
    "create_imagination_env",
]
