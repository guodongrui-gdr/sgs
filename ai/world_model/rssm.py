"""
RSSM Encoder - Recurrent State Space Model for World Model

Architecture:
- RSSEncoder: Compresses 2670-dim state → 128-dim latent (stochastic + deterministic)
- Uses VAE-style encoding with mean + log_std for stochastic component
- RSSMDecoder: Reconstructs state from latent for training verification

Reference: Dreamer (Danijar Hafner et al.)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSEncoderConfig:
    """Configuration for RSSM Encoder"""

    state_dim: int = 2670  # Input state dimension (actual from StateEncoder)
    latent_dim: int = 128  # Total latent dimension (stochastic + deterministic)
    hidden_dim: int = 256  # Hidden layer dimension

    # Stochastic component dimensions
    stochastic_dim: int = 64  # Stochastic latent dimension
    deterministic_dim: int = 64  # Deterministic latent dimension

    # VAE parameters
    kl_weight: float = 1.0  # KL divergence weight
    free_bits: float = 1.0  # Free bits for KL regularization

    # Network architecture
    num_encoder_layers: int = 3  # Number of encoder MLP layers
    num_decoder_layers: int = 3  # Number of decoder MLP layers

    # Normalization
    use_layer_norm: bool = True
    dropout: float = 0.1

    # Activation
    activation: str = "elu"  # "elu", "relu", "gelu"


class RSSMEncoderOutput:
    """Container for encoder output"""

    def __init__(
        self,
        stochastic: torch.Tensor,
        deterministic: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        latent: torch.Tensor,
    ):
        self.stochastic = stochastic  # (batch, stochastic_dim)
        self.deterministic = deterministic  # (batch, deterministic_dim)
        self.mean = mean  # (batch, stochastic_dim)
        self.log_std = log_std  # (batch, stochastic_dim)
        self.latent = latent  # (batch, latent_dim) = concat(stochastic, deterministic)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "stochastic": self.stochastic,
            "deterministic": self.deterministic,
            "mean": self.mean,
            "log_std": self.log_std,
            "latent": self.latent,
        }


class RSSEncoder(nn.Module):
    """
    RSSM Encoder - VAE-style state compression

    Compresses high-dimensional game state (3066-dim) to compact latent (128-dim).

    Architecture:
    1. Encoder MLP: state → hidden → (mean, log_std) + deterministic
    2. Sample stochastic component: z_stoch = mean + std * noise
    3. Concatenate: z = [z_stoch, z_det]

    The stochastic component allows the world model to handle uncertainty
    in game dynamics (e.g., opponent behavior, card draws).
    """

    def __init__(self, config: Optional[RSSEncoderConfig] = None):
        super().__init__()

        self.config = config or RSSEncoderConfig()
        c = self.config

        # Validate dimensions
        assert c.stochastic_dim + c.deterministic_dim == c.latent_dim, (
            f"stochastic_dim + deterministic_dim must equal latent_dim: "
            f"{c.stochastic_dim} + {c.deterministic_dim} != {c.latent_dim}"
        )

        # Activation function
        self.activation = self._get_activation(c.activation)

        # === Encoder Network ===
        # Shared encoder layers
        encoder_layers = []
        in_dim = c.state_dim

        for i in range(c.num_encoder_layers - 1):
            out_dim = c.hidden_dim if i < c.num_encoder_layers - 2 else c.hidden_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if c.use_layer_norm:
                encoder_layers.append(nn.LayerNorm(out_dim))
            encoder_layers.append(self.activation)
            if c.dropout > 0:
                encoder_layers.append(nn.Dropout(c.dropout))
            in_dim = out_dim

        self.shared_encoder = nn.Sequential(*encoder_layers)

        # Stochastic component head (VAE)
        self.stochastic_mean_head = nn.Linear(c.hidden_dim, c.stochastic_dim)
        self.stochastic_log_std_head = nn.Linear(c.hidden_dim, c.stochastic_dim)

        # Deterministic component head
        self.deterministic_head = nn.Linear(c.hidden_dim, c.deterministic_dim)

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

    def _init_weights(self):
        """Initialize network weights with orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> RSSMEncoderOutput:
        """
        Encode state to latent representation

        Args:
            state: Game state tensor (batch, state_dim) or (batch, seq_len, state_dim)
            deterministic: If True, use mean directly (no sampling)

        Returns:
            RSSMEncoderOutput containing:
            - stochastic: sampled stochastic component
            - deterministic: deterministic component
            - mean: mean of stochastic distribution
            - log_std: log standard deviation
            - latent: concatenated latent (batch, latent_dim)
        """
        # Handle sequence input (for temporal processing)
        if state.dim() == 3:
            batch_size, seq_len, state_dim = state.shape
            # Encode each timestep
            outputs = []
            for t in range(seq_len):
                out = self.encode(state[:, t, :], deterministic)
                outputs.append(out)
            # Stack outputs - return first element's attributes stacked
            # This is a simplified version; full implementation would handle sequences properly
            stochastic = torch.stack([o.stochastic for o in outputs], dim=1)
            det = torch.stack([o.deterministic for o in outputs], dim=1)
            mean = torch.stack([o.mean for o in outputs], dim=1)
            log_std = torch.stack([o.log_std for o in outputs], dim=1)
            latent = torch.stack([o.latent for o in outputs], dim=1)
            return RSSMEncoderOutput(stochastic, det, mean, log_std, latent)

        # Standard batch encoding
        batch_size = state.shape[0]

        # Shared encoding
        hidden = self.shared_encoder(state)

        # Stochastic component (VAE)
        mean = self.stochastic_mean_head(hidden)
        log_std = self.stochastic_log_std_head(hidden)

        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)

        # Sample stochastic component
        if deterministic:
            # Use mean directly for deterministic inference
            stochastic = mean
        else:
            # Sample: z = mean + std * noise
            noise = torch.randn_like(mean)
            stochastic = mean + std * noise

        # Deterministic component
        deterministic_out = self.deterministic_head(hidden)

        # Concatenate to form full latent
        latent = torch.cat([stochastic, deterministic_out], dim=-1)

        return RSSMEncoderOutput(
            stochastic=stochastic,
            deterministic=deterministic_out,
            mean=mean,
            log_std=log_std,
            latent=latent,
        )

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> RSSMEncoderOutput:
        """Forward pass - alias for encode()"""
        return self.encode(state, deterministic)

    def get_stochastic_distribution(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and std of stochastic distribution without sampling"""
        hidden = self.shared_encoder(state)
        mean = self.stochastic_mean_head(hidden)
        log_std = self.stochastic_log_std_head(hidden)
        log_std = torch.clamp(log_std, min=-10, max=2)
        return mean, torch.exp(log_std)


class RSSMDecoder(nn.Module):
    """
    RSSM Decoder - Reconstructs state from latent

    Used for training verification: encode → decode should approximate original state.
    Reconstruction loss ensures latent captures essential state information.
    """

    def __init__(self, config: Optional[RSSEncoderConfig] = None):
        super().__init__()

        self.config = config or RSSEncoderConfig()
        c = self.config

        # Activation
        self.activation = self._get_activation(c.activation)

        # === Decoder Network ===
        decoder_layers = []
        in_dim = c.latent_dim

        for i in range(c.num_decoder_layers - 1):
            out_dim = c.hidden_dim if i < c.num_decoder_layers - 2 else c.hidden_dim
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if c.use_layer_norm:
                decoder_layers.append(nn.LayerNorm(out_dim))
            decoder_layers.append(self.activation)
            if c.dropout > 0:
                decoder_layers.append(nn.Dropout(c.dropout))
            in_dim = out_dim

        # Final reconstruction layer
        decoder_layers.append(nn.Linear(in_dim, c.state_dim))

        self.decoder = nn.Sequential(*decoder_layers)

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

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstructed state

        Args:
            latent: Latent representation (batch, latent_dim)

        Returns:
            Reconstructed state (batch, state_dim)
        """
        return self.decoder(latent)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass - alias for decode()"""
        return self.decode(latent)

    def reconstruct(
        self, stochastic: torch.Tensor, deterministic: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct state from stochastic and deterministic components

        Args:
            stochastic: Stochastic component (batch, stochastic_dim)
            deterministic: Deterministic component (batch, deterministic_dim)

        Returns:
            Reconstructed state (batch, state_dim)
        """
        latent = torch.cat([stochastic, deterministic], dim=-1)
        return self.decode(latent)


def compute_kl_loss(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    free_bits: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence loss for VAE regularization

    KL divergence between learned distribution N(mean, std^2) and
    prior N(0, 1).

    Formula: KL = 0.5 * (mean^2 + std^2 - 1 - log(std^2))

    Args:
        mean: Mean of stochastic distribution (batch, stochastic_dim)
        log_std: Log standard deviation (batch, stochastic_dim)
        free_bits: Minimum KL per dimension (prevents posterior collapse)
        reduction: "mean", "sum", or "none"

    Returns:
        KL divergence loss scalar or tensor
    """
    # Clamp log_std to prevent numerical issues
    log_std = torch.clamp(log_std, min=-10, max=2)
    std = torch.exp(log_std)

    # KL divergence: KL(N(mean, std^2) || N(0, 1))
    # = 0.5 * (mean^2 + std^2 - log(std^2) - 1)
    kl_per_dim = 0.5 * (mean.pow(2) + std.pow(2) - 2 * log_std - 1)

    # Free bits: ensure KL is at least free_bits per dimension
    # This prevents posterior collapse (latent becoming deterministic)
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Reduction
    if reduction == "mean":
        return kl_per_dim.mean()
    elif reduction == "sum":
        return kl_per_dim.sum()
    else:
        return kl_per_dim


def compute_reconstruction_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute reconstruction loss (MSE)

    Args:
        reconstructed: Reconstructed state (batch, state_dim)
        original: Original state (batch, state_dim)
        reduction: "mean", "sum", or "none"

    Returns:
        Reconstruction loss
    """
    loss = F.mse_loss(reconstructed, original, reduction="none")

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class RSSMVAE(nn.Module):
    """
    Complete RSSM VAE module combining encoder and decoder

    Useful for training as a single module.
    """

    def __init__(self, config: Optional[RSSEncoderConfig] = None):
        super().__init__()

        self.config = config or RSSEncoderConfig()

        self.encoder = RSSEncoder(self.config)
        self.decoder = RSSMDecoder(self.config)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, RSSMEncoderOutput]:
        """
        Encode and decode state

        Args:
            state: Input state (batch, state_dim)
            deterministic: If True, use mean directly

        Returns:
            reconstructed: Reconstructed state (batch, state_dim)
            encoder_output: Encoder output with latent components
        """
        encoder_output = self.encoder.encode(state, deterministic)
        reconstructed = self.decoder.decode(encoder_output.latent)
        return reconstructed, encoder_output

    def compute_losses(
        self,
        state: torch.Tensor,
        encoder_output: Optional[RSSMEncoderOutput] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for training

        Args:
            state: Input state (batch, state_dim)
            encoder_output: Pre-computed encoder output (optional)

        Returns:
            Dictionary of losses:
            - reconstruction_loss: MSE between original and reconstructed
            - kl_loss: KL divergence for regularization
            - total_loss: Weighted sum of losses
        """
        if encoder_output is None:
            encoder_output = self.encoder.encode(state)

        reconstructed = self.decoder.decode(encoder_output.latent)

        # Reconstruction loss
        recon_loss = compute_reconstruction_loss(reconstructed, state)

        # KL loss
        kl_loss = compute_kl_loss(
            encoder_output.mean,
            encoder_output.log_std,
            free_bits=self.config.free_bits,
        )

        # Total loss
        total_loss = recon_loss + self.config.kl_weight * kl_loss

        return {
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

    def encode(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> RSSMEncoderOutput:
        """Encode state to latent"""
        return self.encoder.encode(state, deterministic)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to state"""
        return self.decoder.decode(latent)


# === Utility Functions ===


def create_rssm_encoder(
    state_dim: int = 3066,
    latent_dim: int = 128,
    hidden_dim: int = 256,
    stochastic_dim: int = 64,
    **kwargs,
) -> RSSEncoder:
    """
    Factory function to create RSSM encoder with custom dimensions

    Args:
        state_dim: Input state dimension
        latent_dim: Total latent dimension
        hidden_dim: Hidden layer dimension
        stochastic_dim: Stochastic component dimension
        **kwargs: Additional config parameters

    Returns:
        RSSEncoder instance
    """
    deterministic_dim = latent_dim - stochastic_dim

    config = RSSEncoderConfig(
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        stochastic_dim=stochastic_dim,
        deterministic_dim=deterministic_dim,
        **kwargs,
    )

    return RSSEncoder(config)


def create_rssm_decoder(
    state_dim: int = 3066,
    latent_dim: int = 128,
    hidden_dim: int = 256,
    **kwargs,
) -> RSSMDecoder:
    """
    Factory function to create RSSM decoder

    Args:
        state_dim: Output state dimension
        latent_dim: Input latent dimension
        hidden_dim: Hidden layer dimension
        **kwargs: Additional config parameters

    Returns:
        RSSMDecoder instance
    """
    config = RSSEncoderConfig(
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        **kwargs,
    )

    return RSSMDecoder(config)


def create_rssm_vae(
    state_dim: int = 3066,
    latent_dim: int = 128,
    hidden_dim: int = 256,
    stochastic_dim: int = 64,
    **kwargs,
) -> RSSMVAE:
    """
    Factory function to create complete RSSM VAE

    Args:
        state_dim: State dimension
        latent_dim: Total latent dimension
        hidden_dim: Hidden layer dimension
        stochastic_dim: Stochastic component dimension
        **kwargs: Additional config parameters

    Returns:
        RSSMVAE instance
    """
    deterministic_dim = latent_dim - stochastic_dim

    config = RSSEncoderConfig(
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        stochastic_dim=stochastic_dim,
        deterministic_dim=deterministic_dim,
        **kwargs,
    )

    return RSSMVAE(config)


# === Export Summary ===
# Classes: RSSEncoderConfig, RSSEncoder, RSSMDecoder, RSSMVAE, RSSMEncoderOutput
# Functions: compute_kl_loss, compute_reconstruction_loss, create_*
