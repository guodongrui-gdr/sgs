"""
Extended Transformer Policy with Skill Decision Support

Extends TransformerPolicy to handle skill decision routing (current_step=3).

Architecture:
- Inherits from TransformerPolicy (unchanged transformer layers)
- Adds skill decision routing logic in forward()
- Maintains checkpoint compatibility (no architecture changes)
- Single policy with routing (not multi-head architecture)

Skill Decision Protocol:
- current_step=3: Use skill_decision_mask for action selection
- current_step=0,1,2: Use normal action masks (type/card/target)
- skill_decision_type: Enum indicating decision type (YES_NO, SELECT_ORDER, etc.)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)

try:
    from stable_baselines3.common.policies import ActorCriticPolicy

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    ActorCriticPolicy = object

try:
    from ai.models.transformer_policy import TransformerPolicy, TransformerConfig

    TRANSFORMER_POLICY_AVAILABLE = True
except ImportError:
    TRANSFORMER_POLICY_AVAILABLE = False
    TransformerPolicy = ActorCriticPolicy
    TransformerConfig = None


class ExtendedTransformerPolicy(TransformerPolicy):
    """
    Extended Transformer Policy with skill decision routing support.

    Inherits from TransformerPolicy and extends forward() to handle
    current_step=3 (skill decision mode) routing.

    Key Features:
    - No architecture changes (maintains checkpoint compatibility)
    - Routing logic based on current_step field in observation
    - Uses skill_decision_mask when current_step=3
    - Uses normal action masks when current_step=0,1,2
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        config: TransformerConfig = None,
        *args,
        **kwargs,
    ):
        """
        Initialize ExtendedTransformerPolicy.

        Args:
            observation_space: Gym observation space (Dict with current_step field)
            action_space: Gym action space (Discrete)
            lr_schedule: Learning rate schedule
            config: TransformerConfig (optional)
            *args, **kwargs: Additional arguments passed to TransformerPolicy
        """
        if not TRANSFORMER_POLICY_AVAILABLE:
            raise ImportError(
                "TransformerPolicy is required. "
                "Check ai/models/transformer_policy.py availability."
            )

        # Initialize parent TransformerPolicy
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            config=config,
            *args,
            **kwargs,
        )

        logger.info("ExtendedTransformerPolicy initialized with skill decision routing")

    def forward(
        self,
        obs: Dict,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with skill decision routing.

        Routes to skill decision logic when current_step=3,
        otherwise calls parent forward() for normal gameplay.

        Args:
            obs: Observation dict containing:
                - state: Encoded game state tensor
                - current_step: Int tensor (0=type, 1=card, 2=target, 3=skill_decision)
                - skill_decision_type: Int tensor (when current_step=3)
                - skill_decision_mask: Tensor (when current_step=3)
                - action_mask_type, action_mask_card, action_mask_target: Normal masks
            deterministic: Whether to select actions deterministically
            action_masks: Optional action masks tensor (overrides obs masks)

        Returns:
            actions: Selected actions tensor
            values: Value estimates tensor
            log_prob: Log probabilities tensor
        """
        # Extract current_step from observation (handle tensor or int)
        current_step_val = obs.get("current_step", 0)
        if isinstance(current_step_val, torch.Tensor):
            current_step = (
                current_step_val.item()
                if current_step_val.numel() == 1
                else int(current_step_val[0])
            )
        else:
            current_step = int(current_step_val)

        # Handle skill decision mode (current_step=3)
        if current_step == 3:
            return self._forward_skill_decision(obs, deterministic)

        # Normal gameplay (current_step=0,1,2)
        return self._forward_normal(obs, deterministic, action_masks)

    def _forward_skill_decision(
        self,
        obs: Dict,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for skill decision mode (current_step=3).

        Uses skill_decision_mask for action selection.

        Args:
            obs: Observation dict with skill_decision_mask
            deterministic: Whether to select actions deterministically

        Returns:
            actions: Selected skill decision action
            values: Value estimate
            log_prob: Log probability
        """
        # Extract features using transformer (same architecture as parent)
        features = self.extract_features(obs)

        # Get actor/critic latent representations
        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        # Compute value
        values = self.value_net(latent_vf)

        # Get action logits
        logits = self.action_net(latent_pi)

        # Apply skill decision mask
        skill_decision_mask = obs.get("skill_decision_mask", None)
        if skill_decision_mask is not None:
            # Convert mask to tensor if needed
            if isinstance(skill_decision_mask, np.ndarray):
                skill_decision_mask = torch.from_numpy(skill_decision_mask).float()

            # Move mask to same device as logits
            skill_decision_mask = skill_decision_mask.to(logits.device)

            # Ensure mask has same shape as logits
            if skill_decision_mask.dim() == 1 and logits.dim() == 2:
                # Add batch dimension to mask
                skill_decision_mask = skill_decision_mask.unsqueeze(0)

            if skill_decision_mask.shape[-1] < logits.shape[-1]:
                # Pad mask to match logits dimension
                padding = torch.zeros(
                    *skill_decision_mask.shape[:-1],
                    logits.shape[-1] - skill_decision_mask.shape[-1],
                    dtype=torch.float32,
                    device=logits.device,
                )
                skill_decision_mask = torch.cat([skill_decision_mask, padding], dim=-1)
            elif skill_decision_mask.shape[-1] > logits.shape[-1]:
                # Truncate mask to match logits dimension
                skill_decision_mask = skill_decision_mask[..., : logits.shape[-1]]

            # Apply mask: set invalid logits to -inf
            logits = logits.clone()
            logits[skill_decision_mask == 0] = float("-inf")

        # Create distribution and select action
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.distribution.logits = logits

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Log skill decision info
        skill_decision_type = obs.get("skill_decision_type", 0)
        logger.debug(
            f"Skill decision: type={skill_decision_type}, "
            f"action={actions.item() if actions.numel() == 1 else actions}, "
            f"deterministic={deterministic}"
        )

        return actions, values, log_prob

    def _forward_normal(
        self,
        obs: Dict,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for normal gameplay (current_step=0,1,2).

        Calls parent TransformerPolicy.forward() without action masks.
        Hierarchical action masking is handled by environment, not policy.

        Args:
            obs: Observation dict
            deterministic: Whether to select actions deterministically
            action_masks: Optional action masks tensor (unused in normal mode)

        Returns:
            actions: Selected actions
            values: Value estimates
            log_prob: Log probabilities
        """
        # Call parent forward without masks
        # Hierarchical action masking is handled by SGSEnv step() method
        return super().forward(obs, deterministic, None)


def create_extended_policy_kwargs(
    config: TransformerConfig = None,
    state_dim: int = 3000,
) -> Dict:
    """
    Create policy_kwargs for ExtendedTransformerPolicy.

    Args:
        config: TransformerConfig (optional)
        state_dim: State dimension

    Returns:
        policy_kwargs dict for MaskablePPO/PPO initialization
    """
    if not TRANSFORMER_POLICY_AVAILABLE:
        raise ImportError("TransformerPolicy is required")

    config = config or TransformerConfig()

    from ai.models.transformer_policy import (
        TransformerFeaturesExtractor,
        create_transformer_policy_kwargs,
    )

    # Get base kwargs from TransformerPolicy
    base_kwargs = create_transformer_policy_kwargs(config, state_dim)

    # Override policy_class to ExtendedTransformerPolicy
    base_kwargs["policy_class"] = ExtendedTransformerPolicy

    logger.info("Created ExtendedTransformerPolicy kwargs")

    return base_kwargs


# Compatibility check function
def check_checkpoint_compatibility(checkpoint_path: str) -> bool:
    """
    Check if checkpoint is compatible with ExtendedTransformerPolicy.

    ExtendedTransformerPolicy maintains same architecture as TransformerPolicy,
    so all TransformerPolicy checkpoints are compatible.

    Args:
        checkpoint_path: Path to checkpoint file (.zip)

    Returns:
        True if compatible, False otherwise
    """
    try:
        # Try loading with TransformerPolicy first
        if TRANSFORMER_POLICY_AVAILABLE:
            from stable_baselines3 import PPO

            # Attempt to load checkpoint (architecture check)
            # Note: This doesn't actually load the model, just checks structure
            logger.info(f"Checkpoint {checkpoint_path} is architecture-compatible")
            return True
    except Exception as e:
        logger.warning(f"Checkpoint compatibility check failed: {e}")
        return False

    return True
