"""
Reward Model - Predicts rewards from latent state + action embeddings

Architecture:
- Input: (latent_state, action_embedding, event_type)
- MLP: input=192 (128+64) → hidden=[256,256] → output=1
- Identity-aware: latent encodes identity relationships

Usage:
    from ai.world_model.reward import RewardModel, RewardModelConfig

    config = RewardModelConfig()
    model = RewardModel(config)

    # Predict reward from latent + action
    reward = model(z_t, action_embed, event_type="damage")

Reward Event Types:
- damage: damage_dealt (+3), damage_taken (-1.5)
- kill: kill_enemy (+15), kill_ally (-20)
- game_over: victory (+100), defeat (-100)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# REWARD EVENT TYPES
# ============================================================================


class RewardEventType:
    """Reward event type constants"""

    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_TAKEN = "damage_taken"
    KILL_ENEMY = "kill_enemy"
    KILL_ALLY = "kill_ally"
    GAME_OVER_VICTORY = "game_over_victory"
    GAME_OVER_DEFEAT = "game_over_defeat"
    SURVIVE = "survive"
    SKILL_ACTIVATION = "skill_activation"

    # All event types
    ALL_TYPES = [
        DAMAGE_DEALT,
        DAMAGE_TAKEN,
        KILL_ENEMY,
        KILL_ALLY,
        GAME_OVER_VICTORY,
        GAME_OVER_DEFEAT,
        SURVIVE,
        SKILL_ACTIVATION,
    ]


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class RewardModelConfig:
    """
    Configuration for Reward Model

    Architecture:
    - Input: latent_dim + action_embed_dim = 192
    - Hidden: [256, 256] with LayerNorm
    - Output: 1 (scalar reward)

    Reward clipping: [-50, +50] to match RewardConfig.clip_reward
    """

    # Input dimensions
    latent_dim: int = 128  # From RSSEncoder
    action_embed_dim: int = 64  # From ActionEmbedding
    event_embed_dim: int = 16  # Event type embedding

    # MLP architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # Normalization
    use_layer_norm: bool = True
    dropout: float = 0.1

    # Activation
    activation: str = "elu"  # "elu", "relu", "gelu"

    # Output
    output_dim: int = 1

    # Reward clipping (matches ai/reward.py RewardConfig.clip_reward)
    reward_clip_min: float = -50.0
    reward_clip_max: float = 50.0

    # Event conditioning
    use_event_conditioning: bool = True
    num_event_types: int = 8  # Number of distinct event types

    # Identity-aware reward patterns (from ai/reward.py RewardConfig)
    # These are reference values, actual predictions come from network
    reference_rewards: Dict[str, float] = field(
        default_factory=lambda: {
            "victory": 100.0,
            "defeat": -100.0,
            "damage_dealt": 3.0,
            "damage_taken": -1.5,
            "kill_enemy": 15.0,
            "kill_ally": -20.0,
            "lord_kill_loyalist": -30.0,
            "survive_per_turn": 0.5,
            "skill_activation": 0.2,
        }
    )


# ============================================================================
# REWARD MODEL
# ============================================================================


class RewardModel(nn.Module):
    """
    Reward Model - Predicts scalar reward from latent state + action

    Architecture:
    1. Concatenate latent + action embedding (+ optional event type)
    2. MLP with LayerNorm: 192 → [256, 256] → 1
    3. Clip output to [-50, +50]

    Identity-Aware Design:
    - Latent state encodes identity information from game state
    - Network learns identity-aware reward patterns through training
    - Event type conditioning allows specialization

    Usage:
        model = RewardModel(config)

        # Basic usage: latent + action
        reward = model(z_t, action_embed)

        # With event type conditioning
        reward = model(z_t, action_embed, event_type="damage_dealt")

        # Batch prediction
        rewards = model(z_batch, action_batch, event_types)
    """

    def __init__(self, config: Optional[RewardModelConfig] = None):
        super().__init__()

        self.config = config or RewardModelConfig()
        c = self.config

        # Calculate input dimension
        input_dim = c.latent_dim + c.action_embed_dim
        if c.use_event_conditioning:
            input_dim += c.event_embed_dim

        # Activation function
        self.activation = self._get_activation(c.activation)

        # Event type embedding (optional)
        if c.use_event_conditioning:
            self.event_embedding = nn.Embedding(c.num_event_types, c.event_embed_dim)
            self.event_to_idx = self._create_event_mapping()
        else:
            self.event_embedding = None
            self.event_to_idx = None

        # === MLP Network ===
        layers = []
        in_dim = input_dim

        for i, hidden_dim in enumerate(c.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if c.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if c.dropout > 0:
                layers.append(nn.Dropout(c.dropout))
            in_dim = hidden_dim

        # Output layer (scalar reward)
        layers.append(nn.Linear(in_dim, c.output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        return activations.get(name, nn.ELU())

    def _create_event_mapping(self) -> Dict[str, int]:
        """Map event type strings to indices"""
        mapping = {}
        for i, event_type in enumerate(RewardEventType.ALL_TYPES):
            mapping[event_type] = i
        # Add aliases
        mapping["damage"] = mapping[RewardEventType.DAMAGE_DEALT]
        mapping["kill"] = mapping[RewardEventType.KILL_ENEMY]
        mapping["victory"] = mapping[RewardEventType.GAME_OVER_VICTORY]
        mapping["defeat"] = mapping[RewardEventType.GAME_OVER_DEFEAT]
        mapping["game_over"] = mapping[RewardEventType.GAME_OVER_VICTORY]
        return mapping

    def _init_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_event_embedding(
        self,
        event_type: Optional[str] = None,
        event_idx: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get event type embedding

        Args:
            event_type: String event type (e.g., "damage_dealt")
            event_idx: Pre-computed event index tensor

        Returns:
            Event embedding tensor or None
        """
        if not self.config.use_event_conditioning or self.event_embedding is None:
            return None

        if event_idx is not None:
            return self.event_embedding(event_idx)

        if event_type is not None and self.event_to_idx is not None:
            idx = self.event_to_idx.get(event_type, 0)
            # Get batch size from device (create scalar tensor)
            return self.event_embedding.weight[idx].unsqueeze(0)

        return None

    def forward(
        self,
        z_t: torch.Tensor,
        action_embed: torch.Tensor,
        event_type: Optional[str] = None,
        event_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict reward from latent state + action embedding

        Args:
            z_t: Latent state tensor (batch, latent_dim) or (batch, seq_len, latent_dim)
            action_embed: Action embedding tensor (batch, action_embed_dim) or (batch, seq_len, action_embed_dim)
            event_type: Optional event type string for conditioning
            event_idx: Optional pre-computed event indices (batch,) or (batch, seq_len)

        Returns:
            Predicted reward scalar (batch, 1) or (batch, seq_len, 1)

        Note:
            - Identity relationships are encoded in latent state z_t
            - Event conditioning allows specialization for different reward types
            - Output is clipped to [-50, +50] range
        """
        # Handle sequence input
        if z_t.dim() == 3:
            batch_size, seq_len, latent_dim = z_t.shape

            # Process each timestep
            outputs = []
            for t in range(seq_len):
                reward = self.forward(
                    z_t[:, t, :],
                    action_embed[:, t, :] if action_embed.dim() == 3 else action_embed,
                    event_type,
                    event_idx[:, t]
                    if event_idx is not None and event_idx.dim() == 2
                    else event_idx,
                )
                outputs.append(reward)

            return torch.stack(outputs, dim=1)

        # Standard batch processing
        batch_size = z_t.shape[0]
        c = self.config

        # Concatenate inputs
        inputs = [z_t, action_embed]

        # Add event embedding
        if c.use_event_conditioning and self.event_embedding is not None:
            event_embed = self._get_event_embedding(event_type, event_idx)
            if event_embed is None:
                # Use zero embedding when no event type specified
                event_embed = torch.zeros(
                    batch_size, c.event_embed_dim, device=z_t.device
                )
            elif event_embed.dim() == 2 and event_embed.shape[0] == 1:
                # Expand to batch size
                event_embed = event_embed.expand(batch_size, -1)
            inputs.append(event_embed)

        # Concatenate all inputs
        x = torch.cat(inputs, dim=-1)

        # MLP forward pass
        reward = self.mlp(x)

        # Clip to valid range
        reward = torch.clamp(
            reward,
            min=self.config.reward_clip_min,
            max=self.config.reward_clip_max,
        )

        return reward

    def predict_batch(
        self,
        z_batch: torch.Tensor,
        action_batch: torch.Tensor,
        event_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Batch prediction with multiple event types

        Args:
            z_batch: Batch of latent states (batch, latent_dim)
            action_batch: Batch of action embeddings (batch, action_embed_dim)
            event_types: List of event type strings (length batch)

        Returns:
            Batch of predicted rewards (batch, 1)
        """
        if event_types is None:
            return self.forward(z_batch, action_batch)

        # Convert event types to indices
        if self.event_to_idx is not None:
            event_indices = torch.tensor(
                [self.event_to_idx.get(et, 0) for et in event_types],
                dtype=torch.long,
                device=z_batch.device,
            )
            return self.forward(z_batch, action_batch, event_idx=event_indices)

        return self.forward(z_batch, action_batch)

    def predict_sequence(
        self,
        z_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        event_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict rewards for a sequence of states/actions

        Args:
            z_sequence: Sequence of latent states (batch, seq_len, latent_dim)
            action_sequence: Sequence of action embeddings (batch, seq_len, action_embed_dim)
            event_sequence: Sequence of event indices (batch, seq_len)

        Returns:
            Sequence of predicted rewards (batch, seq_len, 1)
        """
        return self.forward(z_sequence, action_sequence, event_idx=event_sequence)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compute_reward_loss(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute reward prediction loss (MSE)

    Args:
        predicted: Predicted rewards (batch, 1)
        actual: Actual rewards (batch, 1)
        reduction: "mean", "sum", or "none"

    Returns:
        Reward prediction loss
    """
    loss = F.mse_loss(predicted, actual, reduction="none")

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def create_reward_model(
    latent_dim: int = 128,
    action_embed_dim: int = 64,
    hidden_dims: Optional[List[int]] = None,
    **kwargs,
) -> RewardModel:
    """
    Factory function to create Reward Model

    Args:
        latent_dim: Latent state dimension
        action_embed_dim: Action embedding dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional config parameters

    Returns:
        RewardModel instance
    """
    hidden_dims = hidden_dims or [256, 256]

    config = RewardModelConfig(
        latent_dim=latent_dim,
        action_embed_dim=action_embed_dim,
        hidden_dims=hidden_dims,
        **kwargs,
    )

    return RewardModel(config)


# ============================================================================
# TESTING UTILITIES
# ============================================================================


def test_reward_model():
    """
    Test RewardModel with synthetic data

    Verifies:
    - Output shape is correct
    - Clipping works
    - Event conditioning works
    - Prediction error is reasonable
    """
    print("Testing RewardModel...")

    config = RewardModelConfig()
    model = RewardModel(config)

    # Test basic prediction
    batch_size = 32
    z_t = torch.randn(batch_size, config.latent_dim)
    action_embed = torch.randn(batch_size, config.action_embed_dim)

    reward = model(z_t, action_embed)
    assert reward.shape == (batch_size, 1), (
        f"Expected shape (32, 1), got {reward.shape}"
    )
    assert reward.min() >= config.reward_clip_min, (
        f"Reward below min clip: {reward.min()}"
    )
    assert reward.max() <= config.reward_clip_max, (
        f"Reward above max clip: {reward.max()}"
    )
    print(
        f"✓ Basic prediction: shape={reward.shape}, range=[{reward.min():.2f}, {reward.max():.2f}]"
    )

    # Test event conditioning
    reward_damage = model(z_t, action_embed, event_type="damage_dealt")
    reward_kill = model(z_t, action_embed, event_type="kill_enemy")
    reward_victory = model(z_t, action_embed, event_type="victory")

    print(
        f"✓ Event conditioning: damage={reward_damage.mean():.2f}, kill={reward_kill.mean():.2f}, victory={reward_victory.mean():.2f}"
    )

    # Test sequence prediction
    seq_len = 10
    z_seq = torch.randn(batch_size, seq_len, config.latent_dim)
    action_seq = torch.randn(batch_size, seq_len, config.action_embed_dim)

    reward_seq = model(z_seq, action_seq)
    assert reward_seq.shape == (batch_size, seq_len, 1), (
        f"Expected shape (32, 10, 1), got {reward_seq.shape}"
    )
    print(f"✓ Sequence prediction: shape={reward_seq.shape}")

    # Test batch with event types
    event_types = ["damage_dealt"] * 16 + ["kill_enemy"] * 16
    reward_batch = model.predict_batch(z_t, action_embed, event_types)
    assert reward_batch.shape == (batch_size, 1), (
        f"Expected shape (32, 1), got {reward_batch.shape}"
    )
    print(f"✓ Batch prediction with events: shape={reward_batch.shape}")

    print("✓ All tests passed!")
    return model


# === Export Summary ===
# Classes: RewardModelConfig, RewardModel
# Functions: compute_reward_loss, create_reward_model, test_reward_model
# Constants: RewardEventType
