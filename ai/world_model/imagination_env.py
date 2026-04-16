"""
ImaginedEnvironment - Gym-like wrapper for World Model imagination

Wraps RSSM components (RSSEncoder, DynamicsModel, RewardModel) as a Gym-like
environment for imagination-based policy training in Phase 2.

Architecture:
- reset(initial_state): Encode real state → latent z_0
- step(action): Predict z_{t+1}, reward using DynamicsModel + RewardModel
- Action masking: Maintain valid action masks during imagination
- Performance: 15-step imagination <10ms (batched operations)

Usage:
    from ai.world_model.imagination_env import ImaginedEnvironment

    # Load from trained checkpoint
    env = ImaginedEnvironment(checkpoint_path="outputs/dynamics_training/dynamics_final.pt")

    # Or with fresh models (for testing)
    env = ImaginedEnvironment()

    # Initialize from real state
    initial_latent, info = env.reset(real_state_vector)

    # Imagine trajectory
    z_next, reward, done, info = env.step(action_type=0, card_idx=3, target_idx=1)

Reference: Dreamer (Danijar Hafner et al.)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any, TYPE_CHECKING
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np

if TYPE_CHECKING:
    from ai.world_model.rssm import RSSEncoder, RSSEncoderConfig, RSSMEncoderOutput
    from ai.world_model.dynamics import DynamicsModel, DynamicsConfig
    from ai.world_model.reward import RewardModel, RewardModelConfig


@dataclass
class ImaginationConfig:
    """Configuration for ImaginedEnvironment

    Attributes:
        state_dim: Input state dimension (2670 from StateEncoder)
        latent_dim: Latent state dimension (128 from RSSEncoder)
        action_embed_dim: Action embedding dimension (64 from DynamicsModel)
        max_imagination_horizon: Maximum steps for imagination rollouts (15)
        action_type_dim: Number of action types (17 from ActionType enum)
        max_hand_size: Maximum hand cards (20)
        max_players: Maximum players/targets (8)
        invalid_action_penalty: Penalty for invalid actions during imagination
        deterministic_encoding: Use mean encoding (no sampling) for stability
        device: Device for computation (auto-detect if None)
        checkpoint_path: Path to trained checkpoint (optional)
    """

    state_dim: int = 2670
    latent_dim: int = 128
    action_embed_dim: int = 64
    max_imagination_horizon: int = 15
    action_type_dim: int = 17
    max_hand_size: int = 20
    max_players: int = 8
    invalid_action_penalty: float = -1.0
    deterministic_encoding: bool = True
    device: Optional[str] = None
    checkpoint_path: Optional[str] = None


@dataclass
class ImaginationState:
    """Internal state tracker for imagination

    Attributes:
        z_current: Current latent state (batch, latent_dim)
        z_sequence: History of imagined states (batch, steps, latent_dim)
        action_sequence: History of actions taken (batch, steps)
        reward_sequence: History of predicted rewards (batch, steps)
        uncertainty_sequence: History of uncertainty estimates (batch, steps)
        current_step: Current imagination step count
        max_steps_reached: Whether max horizon reached
        action_masks: Current action masks (type, card, target)
    """

    z_current: Optional[torch.Tensor] = None
    z_sequence: List[torch.Tensor] = field(default_factory=list)
    action_sequence: List[Dict[str, int]] = field(default_factory=list)
    reward_sequence: List[float] = field(default_factory=list)
    uncertainty_sequence: List[float] = field(default_factory=list)
    current_step: int = 0
    max_steps_reached: bool = False
    action_masks: Optional[Dict[str, np.ndarray]] = None


class ImaginedEnvironment:
    """
    Gym-like wrapper for World Model imagination

    Wraps RSSEncoder + DynamicsModel + RewardModel for imagination-based training.
    Provides reset/step interface similar to Gym environments.

    Key Features:
    1. State encoding: real_state → latent z_0 via RSSEncoder
    2. Dynamics prediction: (z_t, action) → z_{t+1} via DynamicsModel
    3. Reward prediction: (z_t, action) → r_t via RewardModel
    4. Action masking: Track valid actions during imagination
    5. Performance: Batched operations for fast rollouts (<10ms for 15 steps)

    Usage:
        # Initialize from checkpoint
        env = ImaginedEnvironment(checkpoint_path="path/to/checkpoint.pt")

        # Reset with real state
        z_0, info = env.reset(state_vector)

        # Imagine trajectory
        z_next, reward, done, info = env.step(action_type, card_idx, target_idx)

    Notes:
        - Must call reset() before step()
        - Action masking is tracked but not enforced (policy must handle)
        - Max horizon is 15 steps by default
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[ImaginationConfig] = None,
    ):
        """
        Initialize ImaginedEnvironment

        Args:
            checkpoint_path: Path to trained checkpoint (optional)
            config: ImaginationConfig with hyperparameters
        """
        self.config = config or ImaginationConfig()
        self.checkpoint_path = checkpoint_path or self.config.checkpoint_path

        # Setup device
        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize models
        self._init_models()

        # Load checkpoint if provided
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            self._load_checkpoint(self.checkpoint_path)
        else:
            if self.checkpoint_path:
                import logging

                logging.warning(
                    f"Checkpoint not found at {self.checkpoint_path}, using fresh models"
                )

        # Internal state
        self._state: Optional[ImaginationState] = None
        self._batch_size: int = 1

        # Performance tracking
        self._step_times: List[float] = []

    def _init_models(self):
        """Initialize RSSEncoder, DynamicsModel, RewardModel with default configs"""
        from ai.world_model.rssm import RSSEncoder, RSSEncoderConfig
        from ai.world_model.dynamics import DynamicsModel, DynamicsConfig
        from ai.world_model.reward import RewardModel, RewardModelConfig

        # RSSEncoder config
        encoder_config = RSSEncoderConfig(
            state_dim=self.config.state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=256,
            stochastic_dim=64,
            deterministic_dim=64,
        )
        self.encoder = RSSEncoder(encoder_config).to(self.device)

        # DynamicsModel config
        dynamics_config = DynamicsConfig(
            latent_dim=self.config.latent_dim,
            action_embed_dim=self.config.action_embed_dim,
            hidden_dim=256,
            num_action_types=self.config.action_type_dim,
            max_hand_size=self.config.max_hand_size,
            max_players=self.config.max_players,
        )
        self.dynamics = DynamicsModel(dynamics_config).to(self.device)

        # RewardModel config
        reward_config = RewardModelConfig(
            latent_dim=self.config.latent_dim,
            action_embed_dim=self.config.action_embed_dim,
            hidden_dims=[256, 256],
        )
        self.reward_model = RewardModel(reward_config).to(self.device)

        # Set to eval mode for inference
        self.encoder.eval()
        self.dynamics.eval()
        self.reward_model.eval()

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load trained models from checkpoint

        Args:
            checkpoint_path: Path to dynamics_final.pt checkpoint
        """
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from train.train_dynamics import DynamicsTrainingConfig

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Load state dicts
        if "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        if "dynamics_state_dict" in checkpoint:
            self.dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
        if "reward_model_state_dict" in checkpoint:
            self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])

        # Reset GRU hidden state after loading
        self.dynamics.reset_hidden(1, self.device)

        # Set to eval mode
        self.encoder.eval()
        self.dynamics.eval()
        self.reward_model.eval()

        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def reset(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_latent: Optional[torch.Tensor] = None,
        action_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset imagination environment with initial state

        Args:
            initial_state: Real state vector (state_dim,) to encode
            initial_latent: Pre-encoded latent (latent_dim,) - skip encoding
            action_masks: Initial action masks for step 0

        Returns:
            z_0: Initial latent state (batch, latent_dim)
            info: Dict with encoding info, action_masks
        """
        # Initialize internal state
        self._state = ImaginationState()
        self._batch_size = 1

        # Encode initial state
        if initial_latent is not None:
            z_0 = initial_latent.to(self.device)
            if z_0.dim() == 1:
                z_0 = z_0.unsqueeze(0)
        elif initial_state is not None:
            state_tensor = torch.tensor(
                initial_state, dtype=torch.float32, device=self.device
            )
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            encoder_output = self.encoder.encode(
                state_tensor, deterministic=self.config.deterministic_encoding
            )
            z_0 = encoder_output.latent
        else:
            # Default: random latent for testing
            z_0 = torch.randn(1, self.config.latent_dim, device=self.device)

        # Reset dynamics hidden state
        self.dynamics.reset_hidden(self._batch_size, self.device)

        # Set current state
        self._state.z_current = z_0
        self._state.current_step = 0
        self._state.max_steps_reached = False

        # Set initial action masks
        if action_masks is not None:
            self._state.action_masks = action_masks
        else:
            # Default: all actions valid
            self._state.action_masks = {
                "type": np.ones(self.config.action_type_dim, dtype=np.float32),
                "card": np.ones(self.config.max_hand_size, dtype=np.float32),
                "target": np.ones(self.config.max_players, dtype=np.float32),
            }

        info = {
            "initial_latent": z_0.detach().cpu().numpy(),
            "action_masks": self._state.action_masks,
            "step": 0,
            "model_loaded": self.checkpoint_path is not None
            and Path(self.checkpoint_path).exists(),
        }

        return z_0, info

    def step(
        self,
        action_type: int,
        card_idx: Optional[int] = None,
        target_idx: Optional[int] = None,
        batch_actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Take one imagination step

        Args:
            action_type: Action type index (0-16)
            card_idx: Card/skill index (optional)
            target_idx: Target player index (optional)
            batch_actions: Batched actions tensor (for parallel rollout)

        Returns:
            z_next: Predicted next latent state (batch, latent_dim)
            reward: Predicted reward scalar
            done: Whether imagination horizon reached
            info: Dict with uncertainty, action validity, step info
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        start_time = time.perf_counter()

        # Check horizon
        if self._state.current_step >= self.config.max_imagination_horizon:
            self._state.max_steps_reached = True
            return self._state.z_current, 0.0, True, {"reason": "max_horizon"}

        # Handle batched actions
        if batch_actions is not None:
            return self._step_batched(batch_actions)

        # Check action validity (soft penalty, don't block)
        action_valid = self._check_action_validity(action_type, card_idx, target_idx)

        # Prepare tensors
        z_t = self._state.z_current
        action_type_tensor = torch.tensor(
            [action_type], dtype=torch.long, device=self.device
        )
        card_idx_tensor = torch.tensor(
            [card_idx if card_idx is not None else self.config.max_hand_size],
            dtype=torch.long,
            device=self.device,
        )
        target_idx_tensor = torch.tensor(
            [target_idx if target_idx is not None else self.config.max_players],
            dtype=torch.long,
            device=self.device,
        )

        # Dynamics prediction: z_{t+1}
        with torch.no_grad():
            z_next, uncertainty = self.dynamics.forward(
                z_t, action_type_tensor, card_idx_tensor, target_idx_tensor
            )

        # Action embedding for reward prediction
        with torch.no_grad():
            action_embed = self.dynamics.action_embedder(
                action_type_tensor, card_idx_tensor, target_idx_tensor
            )

        # Reward prediction
        with torch.no_grad():
            reward_pred = self.reward_model.forward(z_t, action_embed)

        # Extract scalar values
        reward = (
            reward_pred.item()
            if reward_pred.dim() == 2
            else float(reward_pred.mean().item())
        )
        uncertainty_val = (
            uncertainty.item()
            if uncertainty.dim() == 2
            else float(uncertainty.mean().item())
        )

        # Apply invalid action penalty
        if not action_valid:
            reward += self.config.invalid_action_penalty

        # Update internal state
        self._state.z_current = z_next
        self._state.z_sequence.append(z_next)
        self._state.action_sequence.append(
            {
                "action_type": action_type,
                "card_idx": card_idx,
                "target_idx": target_idx,
            }
        )
        self._state.reward_sequence.append(reward)
        self._state.uncertainty_sequence.append(uncertainty_val)
        self._state.current_step += 1

        # Track step time
        step_time = time.perf_counter() - start_time
        self._step_times.append(step_time)

        # Update action masks (simplified - use decay for imagination)
        self._update_action_masks()

        # Check done
        done = self._state.current_step >= self.config.max_imagination_horizon

        info = {
            "uncertainty": uncertainty_val,
            "action_valid": action_valid,
            "step": self._state.current_step,
            "z_sequence_length": len(self._state.z_sequence),
            "action_masks": self._state.action_masks,
            "step_time_ms": step_time * 1000,
        }

        return z_next, reward, done, info

    def _step_batched(
        self,
        batch_actions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Batched step for parallel rollouts

        Args:
            batch_actions: Dict with 'action_type', 'card_idx', 'target_idx' tensors

        Returns:
            z_next_batch: (batch, latent_dim)
            reward_batch: (batch, 1)
            done_batch: (batch,) bool tensor
            info: Dict with batch info
        """
        batch_size = batch_actions["action_type"].shape[0]

        self.dynamics.reset_hidden(batch_size, self.device)
        z_t = self._state.z_current.expand(batch_size, -1)
        action_types = batch_actions["action_type"].to(self.device)
        card_indices = batch_actions.get("card_idx")
        target_indices = batch_actions.get("target_idx")

        if card_indices is None:
            card_indices = torch.full(
                (batch_size,),
                self.config.max_hand_size,
                dtype=torch.long,
                device=self.device,
            )
        else:
            card_indices = card_indices.to(self.device)
        if target_indices is None:
            target_indices = torch.full(
                (batch_size,),
                self.config.max_players,
                dtype=torch.long,
                device=self.device,
            )
        else:
            target_indices = target_indices.to(self.device)

        # Dynamics prediction
        with torch.no_grad():
            z_next_batch, uncertainty_batch = self.dynamics.forward(
                z_t, action_types, card_indices, target_indices
            )

        # Reward prediction
        with torch.no_grad():
            action_embed_batch = self.dynamics.action_embedder(
                action_types, card_indices, target_indices
            )
            reward_batch = self.reward_model.forward(z_t, action_embed_batch)

        # Done condition
        done_batch = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if self._state.current_step >= self.config.max_imagination_horizon - 1:
            done_batch = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        info = {
            "uncertainty_batch": uncertainty_batch.detach().cpu().numpy(),
            "step": self._state.current_step + 1,
            "batch_size": batch_size,
        }

        return z_next_batch, reward_batch, done_batch, info

    def _check_action_validity(
        self,
        action_type: int,
        card_idx: Optional[int],
        target_idx: Optional[int],
    ) -> bool:
        """
        Check if action is valid based on current masks

        Args:
            action_type: Action type index
            card_idx: Card index (optional)
            target_idx: Target index (optional)

        Returns:
            True if action is valid based on masks
        """
        if self._state.action_masks is None:
            return True

        type_mask = self._state.action_masks.get(
            "type", np.ones(self.config.action_type_dim)
        )
        if action_type >= len(type_mask) or type_mask[action_type] <= 0:
            return False

        # For actions requiring card
        if card_idx is not None:
            card_mask = self._state.action_masks.get(
                "card", np.ones(self.config.max_hand_size)
            )
            if card_idx >= len(card_mask) or card_mask[card_idx] <= 0:
                return False

        # For actions requiring target
        if target_idx is not None:
            target_mask = self._state.action_masks.get(
                "target", np.ones(self.config.max_players)
            )
            if target_idx >= len(target_mask) or target_mask[target_idx] <= 0:
                return False

        return True

    def _update_action_masks(self):
        """
        Update action masks for next imagination step

        Simplified approach: Use exponential decay to simulate changing
        game state during imagination. Real implementation would need
        latent-to-mask decoder, but that's Phase 3 complexity.

        For Phase 2, we maintain masks but decay confidence over steps.
        """
        if self._state.action_masks is None:
            return

        # Decay factor for imagination uncertainty
        decay = 0.95**self._state.current_step

        # Apply decay to masks (soft, not hard constraint)
        type_mask = self._state.action_masks["type"] * decay
        card_mask = self._state.action_masks["card"] * decay
        target_mask = self._state.action_masks["target"] * decay

        # Ensure at least PASS (action_type=8) is always valid
        type_mask[8] = 1.0

        self._state.action_masks = {
            "type": type_mask,
            "card": card_mask,
            "target": target_mask,
        }

    def imagine_trajectory(
        self,
        z_start: torch.Tensor,
        action_sequence: List[Dict[str, int]],
        horizon: int = 15,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Imagine full trajectory from start state with given actions

        This is a convenience method for batched trajectory planning.

        Args:
            z_start: Starting latent state (batch, latent_dim)
            action_sequence: List of action dicts for trajectory
            horizon: Maximum steps (default 15)

        Returns:
            z_trajectory: Imagined states (batch, steps, latent_dim)
            rewards: Predicted rewards (batch, steps)
            info: Dict with uncertainties, timing
        """
        # Reset with starting latent
        self.reset(initial_latent=z_start)

        z_trajectory = []
        rewards = []
        uncertainties = []

        start_time = time.perf_counter()

        for i, action in enumerate(action_sequence[:horizon]):
            z_next, reward, done, info = self.step(
                action_type=action.get("action_type", 0),
                card_idx=action.get("card_idx"),
                target_idx=action.get("target_idx"),
            )

            z_trajectory.append(z_next)
            rewards.append(reward)
            uncertainties.append(info.get("uncertainty", 0.0))

            if done:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Stack results
        z_trajectory_tensor = (
            torch.stack(z_trajectory, dim=1) if z_trajectory else z_start.unsqueeze(1)
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        info = {
            "steps_completed": len(z_trajectory),
            "uncertainties": uncertainties,
            "elapsed_ms": elapsed_ms,
        }

        return z_trajectory_tensor, rewards_tensor, info

    def get_current_state(self) -> Optional[torch.Tensor]:
        """Get current latent state"""
        if self._state is None:
            return None
        return self._state.z_current

    def get_action_masks(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current action masks"""
        if self._state is None:
            return None
        return self._state.action_masks

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self._step_times:
            return {"avg_step_ms": 0.0, "total_steps": 0}

        avg_step_ms = sum(self._step_times) / len(self._step_times) * 1000
        return {
            "avg_step_ms": avg_step_ms,
            "max_step_ms": max(self._step_times) * 1000,
            "total_steps": len(self._step_times),
        }


def create_imagination_env(
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> ImaginedEnvironment:
    """
    Factory function to create ImaginedEnvironment

    Args:
        checkpoint_path: Path to trained checkpoint (optional)
        **kwargs: Additional ImaginationConfig parameters

    Returns:
        ImaginedEnvironment instance
    """
    config = ImaginationConfig(checkpoint_path=checkpoint_path, **kwargs)
    return ImaginedEnvironment(checkpoint_path=checkpoint_path, config=config)


# === Export Summary ===
# Classes: ImaginationConfig, ImaginationState, ImaginedEnvironment
# Functions: create_imagination_env
