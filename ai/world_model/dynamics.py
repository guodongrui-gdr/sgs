"""
Dynamics Model - GRU-based latent state prediction for World Model

Predicts next latent state given current state and action:
    (z_t, action) -> z_{t+1} + uncertainty

Architecture:
- Hierarchical action embedding: type(16) + card(32) + target(16) = 64 dims
- GRU sequence model: 2 layers, hidden_dim=256
- State predictor: hidden -> z_next (128 dims)
- Uncertainty head: hidden -> sigma (bounded [0, 1])

Usage:
    from ai.world_model.dynamics import DynamicsModel, DynamicsConfig

    config = DynamicsConfig()
    model = DynamicsModel(config)
    z_next, uncertainty = model.forward(z_t, action_type, card_idx, target_idx)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DynamicsConfig:
    """Configuration for DynamicsModel

    Attributes:
        latent_dim: Dimension of latent state (from RSSEncoder output)
        action_embed_dim: Total action embedding dimension
        type_embed_dim: Embedding dimension for action type component
        card_embed_dim: Embedding dimension for card component
        target_embed_dim: Embedding dimension for target component
        hidden_dim: Hidden dimension for GRU and prediction heads
        gru_layers: Number of GRU layers
        num_action_types: Number of possible action types (ActionType enum)
        max_hand_size: Maximum number of hand cards
        max_players: Maximum number of players (targets)
        dropout: Dropout rate for regularization
    """

    latent_dim: int = 128
    action_embed_dim: int = 64  # type(16) + card(32) + target(16)
    type_embed_dim: int = 16
    card_embed_dim: int = 32
    target_embed_dim: int = 16
    hidden_dim: int = 256
    gru_layers: int = 2
    num_action_types: int = 17  # From ActionType enum (Phase 1 expanded)
    max_hand_size: int = 20
    max_players: int = 8
    dropout: float = 0.1

    def __post_init__(self):
        """Validate configuration consistency"""
        # Verify action embedding dimensions sum correctly
        total_embed = self.type_embed_dim + self.card_embed_dim + self.target_embed_dim
        if total_embed != self.action_embed_dim:
            raise ValueError(
                f"Action embedding dimensions must sum to action_embed_dim: "
                f"{self.type_embed_dim} + {self.card_embed_dim} + {self.target_embed_dim} "
                f"= {total_embed} != {self.action_embed_dim}"
            )


class ActionEmbedding(nn.Module):
    """Hierarchical action embedding module

    Encodes three-component action (type, card, target) into unified embedding.
    Uses separate nn.Embedding layers for each component, then concatenates.

    Args:
        config: DynamicsConfig with embedding dimensions and vocab sizes

    Input:
        action_type: int tensor of action type indices [batch_size]
        card_idx: int tensor of card indices [batch_size] (optional)
        target_idx: int tensor of target indices [batch_size] (optional)

    Output:
        action_embed: float tensor [batch_size, action_embed_dim]
    """

    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config

        # Action type embedding: 17 types -> 16 dims
        self.type_embedding = nn.Embedding(
            num_embeddings=config.num_action_types,
            embedding_dim=config.type_embed_dim,
            padding_idx=0,  # ActionType.PASS/use as padding
        )

        # Card embedding: 20 cards -> 32 dims
        # Use padding_idx for when card is not needed (e.g., END_TURN)
        self.card_embedding = nn.Embedding(
            num_embeddings=config.max_hand_size + 1,  # +1 for padding
            embedding_dim=config.card_embed_dim,
            padding_idx=config.max_hand_size,  # Last index is padding
        )

        # Target embedding: 8 players -> 16 dims
        # Use padding_idx for when target is not needed
        self.target_embedding = nn.Embedding(
            num_embeddings=config.max_players + 1,  # +1 for padding
            embedding_dim=config.target_embed_dim,
            padding_idx=config.max_players,  # Last index is padding
        )

        # Optional: Learnable "no selection" embedding
        # This provides meaningful representation for actions without card/target
        self.no_card_embed = nn.Parameter(torch.zeros(1, config.card_embed_dim))
        self.no_target_embed = nn.Parameter(torch.zeros(1, config.target_embed_dim))

    def forward(
        self,
        action_type: torch.Tensor,
        card_idx: Optional[torch.Tensor] = None,
        target_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode hierarchical action into unified embedding

        Args:
            action_type: [batch_size] tensor of action type indices
            card_idx: [batch_size] tensor of card indices (optional)
            target_idx: [batch_size] tensor of target indices (optional)

        Returns:
            [batch_size, action_embed_dim] unified action embedding
        """
        batch_size = action_type.shape[0]
        device = action_type.device

        # Embed action type (always present)
        type_embed = self.type_embedding(action_type)  # [batch, type_embed_dim]

        # Embed card index (use padding/no_card for missing)
        if card_idx is not None:
            card_embed = self.card_embedding(card_idx)  # [batch, card_embed_dim]
        else:
            card_embed = self.no_card_embed.expand(batch_size, -1).to(device)

        # Embed target index (use padding/no_target for missing)
        if target_idx is not None:
            target_embed = self.target_embedding(
                target_idx
            )  # [batch, target_embed_dim]
        else:
            target_embed = self.no_target_embed.expand(batch_size, -1).to(device)

        # Concatenate all embeddings
        action_embed = torch.cat([type_embed, card_embed, target_embed], dim=-1)

        return action_embed  # [batch, action_embed_dim]


class DynamicsModel(nn.Module):
    """GRU-based dynamics model for latent state prediction

    Predicts next latent state z_{t+1} given current state z_t and action a_t.
    Also outputs uncertainty estimate for exploration guidance.

    Architecture:
    1. Action embedding: hierarchical encoding (type + card + target)
    2. State-action fusion: concatenate z_t and action_embed
    3. GRU: 2-layer sequence processing
    4. State predictor: hidden -> z_next (MLP)
    5. Uncertainty head: hidden -> sigma (sigmoid bounded)

    Args:
        config: DynamicsConfig with all hyperparameters

    Methods:
        forward(z_t, action_type, card_idx, target_idx) -> (z_next, uncertainty)
        forward_sequence(z_sequence, action_sequence) -> (z_next_sequence, uncertainties)
        reset_hidden() -> reset GRU hidden state for new episode
    """

    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config

        # Action embedding module
        self.action_embedder = ActionEmbedding(config)

        # Input fusion: latent + action -> GRU input
        gru_input_dim = config.latent_dim + config.action_embed_dim
        self.input_proj = nn.Linear(gru_input_dim, config.hidden_dim)

        # GRU backbone: 2 layers
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )

        # State predictor head: hidden -> z_next
        self.state_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        # Uncertainty head: hidden -> sigma (bounded [0, 1])
        # Higher uncertainty -> more exploration needed
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),  # Bound output to [0, 1]
        )

        # Hidden state for sequence processing
        self._hidden_state: Optional[torch.Tensor] = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param, gain=1.0)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        z_t: torch.Tensor,
        action_type: torch.Tensor,
        card_idx: Optional[torch.Tensor] = None,
        target_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent state and uncertainty

        Args:
            z_t: [batch_size, latent_dim] current latent state
            action_type: [batch_size] action type index
            card_idx: [batch_size] card index (optional)
            target_idx: [batch_size] target index (optional)

        Returns:
            z_next: [batch_size, latent_dim] predicted next latent state
            uncertainty: [batch_size, 1] uncertainty estimate (bounded [0, 1])
        """
        batch_size = z_t.shape[0]

        # Embed action components
        action_embed = self.action_embedder(action_type, card_idx, target_idx)

        # Fuse state and action
        # [batch, latent_dim + action_embed_dim]
        fused_input = torch.cat([z_t, action_embed], dim=-1)

        # Project to GRU input dimension
        gru_input = self.input_proj(fused_input)  # [batch, hidden_dim]

        # GRU processing
        # Add time dimension for single-step: [batch, 1, hidden_dim]
        gru_input_seq = gru_input.unsqueeze(1)

        # Process through GRU
        gru_output, new_hidden = self.gru(gru_input_seq, self._hidden_state)

        # Update hidden state for next call
        self._hidden_state = new_hidden

        # Extract output: [batch, hidden_dim]
        hidden = gru_output.squeeze(1)

        # Predict next latent state
        z_next = self.state_predictor(hidden)  # [batch, latent_dim]

        # Predict uncertainty
        uncertainty = self.uncertainty_head(hidden)  # [batch, 1]

        return z_next, uncertainty

    def forward_sequence(
        self,
        z_sequence: torch.Tensor,
        action_types: torch.Tensor,
        card_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        reset_hidden: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence of state-action pairs

        Args:
            z_sequence: [batch, seq_len, latent_dim] sequence of latent states
            action_types: [batch, seq_len] sequence of action types
            card_indices: [batch, seq_len] sequence of card indices (optional)
            target_indices: [batch, seq_len] sequence of target indices (optional)
            reset_hidden: Whether to reset hidden state at start

        Returns:
            z_next_sequence: [batch, seq_len, latent_dim] predicted next states
            uncertainties: [batch, seq_len, 1] uncertainty estimates
        """
        batch_size, seq_len = z_sequence.shape[:2]

        # Reset hidden state if requested
        if reset_hidden:
            self.reset_hidden(batch_size, z_sequence.device)

        # Embed all actions at once for efficiency
        # Process step-by-step through GRU for sequence modeling
        z_next_list = []
        uncertainty_list = []

        for t in range(seq_len):
            # Get t-th timestep inputs
            z_t = z_sequence[:, t, :]  # [batch, latent_dim]
            action_type_t = action_types[:, t]  # [batch]

            card_idx_t = None
            if card_indices is not None:
                card_idx_t = card_indices[:, t]

            target_idx_t = None
            if target_indices is not None:
                target_idx_t = target_indices[:, t]

            # Predict next state
            z_next_t, uncertainty_t = self.forward(
                z_t, action_type_t, card_idx_t, target_idx_t
            )

            z_next_list.append(z_next_t)
            uncertainty_list.append(uncertainty_t)

        # Stack outputs
        z_next_sequence = torch.stack(
            z_next_list, dim=1
        )  # [batch, seq_len, latent_dim]
        uncertainties = torch.stack(uncertainty_list, dim=1)  # [batch, seq_len, 1]

        return z_next_sequence, uncertainties

    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset GRU hidden state for new episode/sequence

        Args:
            batch_size: Number of parallel sequences
            device: Device to create hidden state on
        """
        if device is None:
            device = next(self.parameters()).device

        self._hidden_state = torch.zeros(
            self.config.gru_layers, batch_size, self.config.hidden_dim, device=device
        )

    def get_hidden(self) -> Optional[torch.Tensor]:
        """Get current hidden state"""
        return self._hidden_state

    def set_hidden(self, hidden: torch.Tensor):
        """Set hidden state (for continuing sequences)"""
        self._hidden_state = hidden

    def imagine_trajectory(
        self,
        z_start: torch.Tensor,
        action_sequence: torch.Tensor,
        card_sequence: Optional[torch.Tensor] = None,
        target_sequence: Optional[torch.Tensor] = None,
        horizon: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Imagine future trajectory from starting state

        Used for planning/exploration in Phase 2 imagination-based training.

        Args:
            z_start: [batch, latent_dim] starting latent state
            action_sequence: [batch, horizon] planned actions
            card_sequence: [batch, horizon] planned card selections
            target_sequence: [batch, horizon] planned targets
            horizon: Number of steps to imagine

        Returns:
            z_trajectory: [batch, horizon, latent_dim] imagined states
            uncertainties: [batch, horizon, 1] uncertainty estimates
        """
        batch_size = z_start.shape[0]
        device = z_start.device

        # Reset for new trajectory
        self.reset_hidden(batch_size, device)

        z_trajectory = []
        uncertainties = []
        z_current = z_start

        for t in range(horizon):
            # Get action at step t
            action_type_t = action_sequence[:, t]

            card_idx_t = None
            if card_sequence is not None:
                card_idx_t = card_sequence[:, t]

            target_idx_t = None
            if target_sequence is not None:
                target_idx_t = target_sequence[:, t]

            # Predict next state
            z_next, uncertainty = self.forward(
                z_current, action_type_t, card_idx_t, target_idx_t
            )

            z_trajectory.append(z_next)
            uncertainties.append(uncertainty)

            # Update current state for next step
            z_current = z_next

        # Stack outputs
        z_trajectory = torch.stack(z_trajectory, dim=1)  # [batch, horizon, latent_dim]
        uncertainties = torch.stack(uncertainties, dim=1)  # [batch, horizon, 1]

        return z_trajectory, uncertainties


class DynamicsLoss(nn.Module):
    """Loss function for training DynamicsModel

    Combines:
    - State prediction loss: MSE between predicted and actual z_next
    - Uncertainty regularization: penalize overconfident predictions

    Args:
        state_loss_weight: Weight for state prediction MSE
        uncertainty_penalty: Weight for uncertainty regularization
        target_uncertainty: Target uncertainty level (encourage exploration)
    """

    def __init__(
        self,
        state_loss_weight: float = 1.0,
        uncertainty_penalty: float = 0.1,
        target_uncertainty: float = 0.3,
    ):
        super().__init__()
        self.state_loss_weight = state_loss_weight
        self.uncertainty_penalty = uncertainty_penalty
        self.target_uncertainty = target_uncertainty

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        uncertainty: torch.Tensor,
        accuracy: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute dynamics model loss

        Args:
            z_pred: [batch, latent_dim] predicted next state
            z_target: [batch, latent_dim] actual next state
            uncertainty: [batch, 1] predicted uncertainty
            accuracy: [batch] actual prediction accuracy (optional)

        Returns:
            loss: Scalar loss value
            info: Dict with loss components for logging
        """
        # State prediction MSE loss
        state_loss = F.mse_loss(z_pred, z_target)

        # Uncertainty regularization
        # Penalize predictions that are too confident (low uncertainty)
        # when they might be inaccurate
        uncertainty_loss = F.mse_loss(
            uncertainty, torch.full_like(uncertainty, self.target_uncertainty)
        )

        # Optional: Accuracy-weighted uncertainty loss
        # If accuracy is provided, encourage higher uncertainty for inaccurate predictions
        if accuracy is not None:
            # accuracy: 0 for wrong predictions, 1 for correct
            # Encourage: high uncertainty when accuracy is low
            target_uncertainty_dynamic = 1.0 - accuracy.unsqueeze(-1)
            uncertainty_loss_dynamic = F.mse_loss(
                uncertainty, target_uncertainty_dynamic * self.target_uncertainty
            )
            uncertainty_loss = uncertainty_loss + uncertainty_loss_dynamic

        # Total loss
        total_loss = (
            self.state_loss_weight * state_loss
            + self.uncertainty_penalty * uncertainty_loss
        )

        # Logging info
        info = {
            "state_loss": state_loss.item(),
            "uncertainty_loss": uncertainty_loss.item(),
            "total_loss": total_loss.item(),
            "mean_uncertainty": uncertainty.mean().item(),
            "z_pred_norm": z_pred.norm(dim=-1).mean().item(),
            "z_target_norm": z_target.norm(dim=-1).mean().item(),
        }

        return total_loss, info


def create_dynamics_model(config: Optional[DynamicsConfig] = None) -> DynamicsModel:
    """Factory function to create DynamicsModel

    Args:
        config: Optional DynamicsConfig (uses default if None)

    Returns:
        DynamicsModel instance
    """
    if config is None:
        config = DynamicsConfig()

    return DynamicsModel(config)
