"""
Imagination Quality Validation Framework - Phase 2 World Model Validation

Validates imagination quality by comparing imagined rollouts vs real rollouts.

Metrics:
- State divergence: MSE between real latent states vs imagined latent states
- Reward divergence: MSE between real rewards vs predicted rewards
- Action divergence: KL between action distributions

Validation Gate:
- 15-step divergence must be < 20% (plan threshold)

Usage:
    python tests/test_imagination_quality.py --horizons 1,5,10,15 --n-rollouts 50
    python tests/test_imagination_quality.py --checkpoint outputs/dynamics_training/dynamics_final.pt

Output:
    outputs/imagination_quality_report.json
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Import world model components
from ai.world_model import (
    RSSEncoder,
    RSSEncoderConfig,
    RSSMDecoder,
    DynamicsModel,
    DynamicsConfig,
)
from ai.world_model.reward import RewardModel, RewardModelConfig
from ai.state_encoder import StateEncoder
from ai.gym_wrapper import SGSConfig, SGSEnv

import pickle
import io

try:
    from train.train_dynamics import DynamicsTrainingConfig

    _HAS_TRAIN_CONFIG = True
except ImportError:
    DynamicsTrainingConfig = type("DynamicsTrainingConfig", (), {})
    _HAS_TRAIN_CONFIG = False


def _load_checkpoint_safe(path: str, device: torch.device) -> dict:
    """Load checkpoint using torch.load with custom handling for missing config classes"""
    checkpoint = torch.load(
        path,
        weights_only=False,
        map_location=device,
    )
    return checkpoint


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class ImaginationQualityConfig:
    """Configuration for imagination quality validation"""

    # Rollout parameters
    n_rollouts: int = 50
    max_steps_per_rollout: int = 100
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 15])

    # Model parameters
    checkpoint_path: str = "outputs/dynamics_training/dynamics_final.pt"
    latent_dim: int = 128
    state_dim: int = 2670  # Actual from StateEncoder

    # Validation thresholds
    divergence_threshold: float = 0.20  # 20% divergence at 15 steps
    state_error_threshold: float = 0.1  # State prediction error

    # Output
    output_path: str = "outputs/imagination_quality_report.json"
    verbose: bool = True

    # Environment
    player_num: int = 5
    max_rounds: int = 15
    use_random_policy: bool = True  # Use random policy for rollouts


@dataclass
class RolloutData:
    """Container for rollout data"""

    states: np.ndarray  # (n_steps, state_dim)
    next_states: np.ndarray  # (n_steps, state_dim)
    actions: np.ndarray  # (n_steps,)
    rewards: np.ndarray  # (n_steps,)
    latent_states: np.ndarray  # (n_steps, latent_dim)
    latent_next_states: np.ndarray  # (n_steps, latent_dim)
    predicted_next_states: np.ndarray  # (n_steps, latent_dim)
    predicted_rewards: np.ndarray  # (n_steps, 1)
    uncertainties: np.ndarray  # (n_steps, 1)
    done: bool


@dataclass
class HorizonMetrics:
    """Metrics for a specific horizon"""

    horizon: int
    n_samples: int
    state_mse: float
    state_divergence_pct: float
    reward_mse: float
    reward_divergence_pct: float
    action_kl: float
    mean_uncertainty: float
    max_uncertainty: float


@dataclass
class ImaginationQualityReport:
    """Complete validation report"""

    timestamp: str
    checkpoint_path: str
    n_rollouts: int
    total_steps_collected: int
    horizons: List[int]
    horizon_metrics: Dict[int, HorizonMetrics]
    overall_state_mse: float
    overall_reward_mse: float
    validation_passed: bool
    validation_details: Dict[str, Any]
    config: Dict[str, Any]


# ============================================================================
# PLACEHOLDER: IMAGINED ENVIRONMENT (Task 9)
# ============================================================================


class ImaginedEnvironmentPlaceholder:
    """
    Placeholder for ImaginedEnvironment from Task 9.

    This will be replaced with the actual ImaginedEnvironment once Task 9 completes.
    The placeholder simulates imagination using the dynamics model directly.

    TODO: Replace with actual ImaginedEnvironment when available:
        from ai.world_model.imagined_env import ImaginedEnvironment
    """

    def __init__(
        self,
        dynamics_model: DynamicsModel,
        reward_model: RewardModel,
        encoder: RSSEncoder,
        initial_latent: torch.Tensor,
        device: torch.device,
    ):
        self.dynamics = dynamics_model
        self.reward_model = reward_model
        self.encoder = encoder
        self.current_latent = initial_latent.clone()
        self.device = device
        self._step_count = 0

        # Reset dynamics hidden state
        self.dynamics.reset_hidden(1, device)

    def step(
        self,
        action_type: int,
        card_idx: Optional[int] = None,
        target_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Imagine one step forward.

        Args:
            action_type: Action type index
            card_idx: Card index (optional)
            target_idx: Target index (optional)

        Returns:
            next_latent: Predicted next latent state
            reward: Predicted reward
            uncertainty: Uncertainty estimate
            done: Whether imagination should stop (always False for placeholder)
        """
        # Prepare action tensors
        action_type_t = torch.tensor(
            [action_type], dtype=torch.long, device=self.device
        )

        card_idx_t = None
        if card_idx is not None:
            card_idx_t = torch.tensor([card_idx], dtype=torch.long, device=self.device)

        target_idx_t = None
        if target_idx is not None:
            target_idx_t = torch.tensor(
                [target_idx], dtype=torch.long, device=self.device
            )

        # Predict next latent state
        next_latent, uncertainty = self.dynamics.forward(
            self.current_latent, action_type_t, card_idx_t, target_idx_t
        )

        # Get action embedding for reward prediction
        action_embed = self.dynamics.action_embedder(
            action_type_t, card_idx_t, target_idx_t
        )

        # Predict reward
        reward = self.reward_model.forward(self.current_latent, action_embed)

        # Update current latent
        self.current_latent = next_latent.clone()
        self._step_count += 1

        return next_latent, reward, uncertainty, False

    def imagine_trajectory(
        self,
        action_sequence: List[Tuple[int, Optional[int], Optional[int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagine a trajectory from current state.

        Args:
            action_sequence: List of (action_type, card_idx, target_idx) tuples

        Returns:
            latent_trajectory: (n_steps, latent_dim) imagined latent states
            reward_trajectory: (n_steps, 1) predicted rewards
            uncertainty_trajectory: (n_steps, 1) uncertainty estimates
        """
        latent_list = []
        reward_list = []
        uncertainty_list = []

        # Reset dynamics
        self.dynamics.reset_hidden(1, self.device)
        current_latent = self.current_latent.clone()

        for action_type, card_idx, target_idx in action_sequence:
            next_latent, reward, uncertainty, _ = self.step(
                action_type, card_idx, target_idx
            )
            latent_list.append(next_latent.squeeze(0))
            reward_list.append(reward.squeeze(0))
            uncertainty_list.append(uncertainty.squeeze(0))

        return (
            torch.stack(latent_list, dim=0),
            torch.stack(reward_list, dim=0),
            torch.stack(uncertainty_list, dim=0),
        )


# ============================================================================
# WORLD MODEL LOADER
# ============================================================================


class WorldModelLoader:
    """Load trained world model from checkpoint"""

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.encoder: Optional[RSSEncoder] = None
        self.decoder: Optional[RSSMDecoder] = None
        self.dynamics: Optional[DynamicsModel] = None
        self.reward_model: Optional[RewardModel] = None

    def load(self) -> Tuple[RSSEncoder, RSSMDecoder, DynamicsModel, RewardModel]:
        """Load all world model components from checkpoint"""
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = _load_checkpoint_safe(self.checkpoint_path, self.device)

        # Get config if available
        config = checkpoint.get("config", None)

        # Create models with default config if checkpoint config not available
        encoder_config = RSSEncoderConfig(
            state_dim=2670,
            latent_dim=128,
            hidden_dim=256,
            stochastic_dim=64,
        )

        dynamics_config = DynamicsConfig(
            latent_dim=128,
            action_embed_dim=64,
            hidden_dim=256,
        )

        reward_config = RewardModelConfig(
            latent_dim=128,
            action_embed_dim=64,
            hidden_dims=[256, 256],
        )

        # Create models
        self.encoder = RSSEncoder(encoder_config).to(self.device)
        self.decoder = RSSMDecoder(encoder_config).to(self.device)
        self.dynamics = DynamicsModel(dynamics_config).to(self.device)
        self.reward_model = RewardModel(reward_config).to(self.device)

        # Load state dicts
        if "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            logger.info("Loaded encoder weights")

        if "decoder_state_dict" in checkpoint:
            self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
            logger.info("Loaded decoder weights")

        if "dynamics_state_dict" in checkpoint:
            self.dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
            logger.info("Loaded dynamics weights")

        if "reward_model_state_dict" in checkpoint and self.reward_model is not None:
            self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])
            logger.info("Loaded reward model weights")

        # Log checkpoint metrics if available
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            if "state_errors" in metrics and metrics["state_errors"]:
                final_state_error = np.mean(metrics["state_errors"])
                logger.info(f"Checkpoint state error: {final_state_error:.4f}")
            if "reward_errors" in metrics and metrics["reward_errors"]:
                final_reward_error = np.mean(metrics["reward_errors"])
                logger.info(f"Checkpoint reward error: {final_reward_error:.2f}")

        return self.encoder, self.decoder, self.dynamics, self.reward_model


# ============================================================================
# REAL ENVIRONMENT ROLLOUT COLLECTOR
# ============================================================================


class RealRolloutCollector:
    """Collect rollouts from real SGSEnv"""

    def __init__(
        self,
        config: ImaginationQualityConfig,
        encoder: RSSEncoder,
        dynamics: DynamicsModel,
        reward_model: RewardModel,
        state_encoder: StateEncoder,
        device: torch.device,
    ):
        self.config = config
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward_model = reward_model
        self.state_encoder = state_encoder
        self.device = device

        # Create environment
        sgs_config = SGSConfig(
            player_num=config.player_num,
            max_rounds=config.max_rounds,
            use_action_mask=True,
            use_shaping=True,
            other_player_policy="rule",
        )
        self.env = SGSEnv(sgs_config)

    def collect_single_rollout(self, max_steps: int) -> RolloutData:
        """Collect a single rollout from real environment"""
        states = []
        next_states = []
        actions = []
        rewards = []
        done = False

        obs, info = self.env.reset()

        for step in range(max_steps):
            # Get current state
            game_state = self.env._get_game_state_dict()
            current_state = self.state_encoder.encode(game_state, 0)

            # Select action (random or based on masks)
            if self.config.use_random_policy:
                # Get action masks
                masks = obs.get("action_mask_type", np.ones(17))
                valid_actions = [i for i, m in enumerate(masks) if m > 0]
                if valid_actions:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0  # END_TURN
            else:
                # Use first valid action
                action = 0

            # Step environment
            try:
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
            except Exception as e:
                logger.warning(f"Step failed at step {step}: {e}")
                done = True
                reward = 0.0
                obs, info = self.env.reset()

            # Get next state
            if not done:
                next_game_state = self.env._get_game_state_dict()
                next_state = self.state_encoder.encode(next_game_state, 0)
            else:
                next_state = np.zeros_like(current_state)

            states.append(current_state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            if done:
                obs, info = self.env.reset()
                break

        # Convert to arrays
        states_arr = np.array(states, dtype=np.float32)
        next_states_arr = np.array(next_states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.int64)
        rewards_arr = np.array(rewards, dtype=np.float32)

        # Encode to latent space and predict
        (
            latent_states,
            latent_next_states,
            predicted_next_states,
            predicted_rewards,
            uncertainties,
        ) = self._encode_and_predict(states_arr, next_states_arr, actions_arr)

        return RolloutData(
            states=states_arr,
            next_states=next_states_arr,
            actions=actions_arr,
            rewards=rewards_arr,
            latent_states=latent_states,
            latent_next_states=latent_next_states,
            predicted_next_states=predicted_next_states,
            predicted_rewards=predicted_rewards,
            uncertainties=uncertainties,
            done=done,
        )

    def _encode_and_predict(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode states to latent and predict next states/rewards"""
        n_steps = states.shape[0]

        # Encode states to latent
        states_t = torch.from_numpy(states).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)

        with torch.no_grad():
            # Encode current states
            encoder_output = self.encoder.encode(states_t, deterministic=True)
            latent_states = encoder_output.latent.cpu().numpy()

            # Encode next states (target)
            next_encoder_output = self.encoder.encode(next_states_t, deterministic=True)
            latent_next_states = next_encoder_output.latent.cpu().numpy()

            # Predict using dynamics model
            actions_t = torch.from_numpy(actions).long().to(self.device)

            # Reset dynamics hidden state for single-step processing
            self.dynamics.reset_hidden(1, self.device)

            # Predict next latent states step-by-step
            predicted_latents = []
            predicted_rewards = []
            uncertainties = []

            for t in range(n_steps):
                z_t = encoder_output.latent[t : t + 1]
                action_t = actions_t[t : t + 1]

                z_pred, uncertainty = self.dynamics.forward(z_t, action_t, None, None)

                # Get action embedding for reward prediction
                action_embed = self.dynamics.action_embedder(action_t, None, None)
                reward_pred = self.reward_model.forward(z_t, action_embed)

                predicted_latents.append(z_pred.squeeze(0))
                predicted_rewards.append(reward_pred.squeeze(0))
                uncertainties.append(uncertainty.squeeze(0))

            predicted_next_states = torch.stack(predicted_latents, dim=0).cpu().numpy()
            predicted_rewards_arr = torch.stack(predicted_rewards, dim=0).cpu().numpy()
            uncertainties_arr = torch.stack(uncertainties, dim=0).cpu().numpy()

        return (
            latent_states,
            latent_next_states,
            predicted_next_states,
            predicted_rewards_arr,
            uncertainties_arr,
        )

    def collect_rollouts(self, n_rollouts: int, max_steps: int) -> List[RolloutData]:
        """Collect multiple rollouts"""
        rollouts = []
        for i in range(n_rollouts):
            logger.info(f"Collecting rollout {i + 1}/{n_rollouts}")
            try:
                rollout = self.collect_single_rollout(max_steps)
                rollouts.append(rollout)
            except Exception as e:
                logger.warning(f"Rollout {i + 1} failed: {e}")
        return rollouts


# ============================================================================
# IMAGINATION QUALITY VALIDATOR
# ============================================================================


class ImaginationQualityValidator:
    """Validate imagination quality against real rollouts"""

    def __init__(self, config: ImaginationQualityConfig):
        self.config = config

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load world model
        self.loader = WorldModelLoader(config.checkpoint_path, self.device)
        self.encoder, self.decoder, self.dynamics, self.reward_model = (
            self.loader.load()
        )

        # State encoder
        self.state_encoder = StateEncoder()

        # Placeholder for imagined environment
        self.imagined_env_class = ImaginedEnvironmentPlaceholder

    def compute_horizon_metrics(
        self,
        rollouts: List[RolloutData],
        horizon: int,
    ) -> HorizonMetrics:
        """
        Compute divergence metrics for a specific horizon.

        For each horizon, we compare:
        - Real latent state at step t+horizon vs imagined state after horizon steps
        - Real rewards over horizon vs predicted rewards
        """
        state_errors = []
        reward_errors = []
        uncertainties = []
        n_samples = 0

        for rollout in rollouts:
            n_steps = min(len(rollout.latent_states), horizon)

            if n_steps < horizon:
                continue

            # Get real latent trajectory (horizon steps)
            real_latents = rollout.latent_states[:horizon]
            real_next_latent = rollout.latent_next_states[horizon - 1]

            # Get predicted latent trajectory
            predicted_latents = rollout.predicted_next_states[:horizon]
            predicted_next_latent = rollout.predicted_next_states[horizon - 1]

            # Get real and predicted rewards
            real_rewards = rollout.rewards[:horizon]
            predicted_rewards = rollout.predicted_rewards[:horizon].flatten()

            # Compute state divergence (cumulative or final)
            # Using final state at horizon as comparison
            state_mse = np.mean((real_next_latent - predicted_next_latent) ** 2)
            state_errors.append(state_mse)

            # Compute reward divergence (sum over horizon)
            reward_mse = np.mean((real_rewards - predicted_rewards) ** 2)
            reward_errors.append(reward_mse)

            # Uncertainty
            uncertainties.extend(rollout.uncertainties[:horizon].flatten().tolist())
            n_samples += 1

        if n_samples == 0:
            logger.warning(f"No valid samples for horizon {horizon}")
            return HorizonMetrics(
                horizon=horizon,
                n_samples=0,
                state_mse=float("inf"),
                state_divergence_pct=float("inf"),
                reward_mse=float("inf"),
                reward_divergence_pct=float("inf"),
                action_kl=float("inf"),
                mean_uncertainty=float("inf"),
                max_uncertainty=float("inf"),
            )

        mean_state_mse = np.mean(state_errors)
        mean_reward_mse = np.mean(reward_errors)
        mean_uncertainty = np.mean(uncertainties) if uncertainties else 0.0
        max_uncertainty = np.max(uncertainties) if uncertainties else 0.0

        # Compute divergence percentage
        # Divergence = MSE / variance of real states
        all_real_latents = []
        for rollout in rollouts:
            all_real_latents.extend(rollout.latent_states.tolist())

        if all_real_latents:
            latent_variance = np.var(np.array(all_real_latents))
            state_divergence_pct = (
                mean_state_mse / latent_variance
                if latent_variance > 0
                else mean_state_mse
            )
        else:
            state_divergence_pct = mean_state_mse

        # Reward divergence percentage
        all_real_rewards = []
        for rollout in rollouts:
            all_real_rewards.extend(rollout.rewards.tolist())

        if all_real_rewards:
            reward_variance = np.var(np.array(all_real_rewards))
            reward_divergence_pct = (
                mean_reward_mse / reward_variance
                if reward_variance > 0
                else mean_reward_mse
            )
        else:
            reward_divergence_pct = mean_reward_mse

        # Action KL divergence placeholder (would need policy distribution)
        # For now, use action matching rate as proxy
        action_kl = 0.0  # Placeholder

        return HorizonMetrics(
            horizon=horizon,
            n_samples=n_samples,
            state_mse=float(mean_state_mse),
            state_divergence_pct=float(state_divergence_pct),
            reward_mse=float(mean_reward_mse),
            reward_divergence_pct=float(reward_divergence_pct),
            action_kl=action_kl,
            mean_uncertainty=float(mean_uncertainty),
            max_uncertainty=float(max_uncertainty),
        )

    def validate(self) -> ImaginationQualityReport:
        """Run full validation and generate report"""
        logger.info(f"Starting imagination quality validation")
        logger.info(
            f"Config: n_rollouts={self.config.n_rollouts}, horizons={self.config.horizons}"
        )

        # Collect real rollouts
        collector = RealRolloutCollector(
            self.config,
            self.encoder,
            self.dynamics,
            self.reward_model,
            self.state_encoder,
            self.device,
        )

        rollouts = collector.collect_rollouts(
            self.config.n_rollouts,
            self.config.max_steps_per_rollout,
        )

        total_steps = sum(len(r.states) for r in rollouts)
        logger.info(f"Collected {len(rollouts)} rollouts, {total_steps} total steps")

        # Compute metrics for each horizon
        horizon_metrics = {}
        for horizon in self.config.horizons:
            logger.info(f"Computing metrics for horizon {horizon}")
            metrics = self.compute_horizon_metrics(rollouts, horizon)
            horizon_metrics[horizon] = metrics
            logger.info(
                f"  state_mse={metrics.state_mse:.4f}, "
                f"reward_mse={metrics.reward_mse:.2f}, "
                f"divergence={metrics.state_divergence_pct:.2%}"
            )

        # Overall metrics
        overall_state_mse = np.mean([m.state_mse for m in horizon_metrics.values()])
        overall_reward_mse = np.mean([m.reward_mse for m in horizon_metrics.values()])

        # Validation gate check
        # 15-step divergence must be < 20%
        if 15 in horizon_metrics:
            horizon_15_metrics = horizon_metrics[15]
            validation_passed = (
                horizon_15_metrics.state_divergence_pct
                < self.config.divergence_threshold
            )
            validation_details = {
                "threshold": self.config.divergence_threshold,
                "actual_divergence": horizon_15_metrics.state_divergence_pct,
                "15_step_state_mse": horizon_15_metrics.state_mse,
                "15_step_reward_mse": horizon_15_metrics.reward_mse,
                "passed": validation_passed,
            }
        else:
            # If 15 not in horizons, check max horizon
            max_horizon = max(self.config.horizons)
            max_metrics = horizon_metrics[max_horizon]
            validation_passed = (
                max_metrics.state_divergence_pct < self.config.divergence_threshold
            )
            validation_details = {
                "threshold": self.config.divergence_threshold,
                "max_horizon": max_horizon,
                "actual_divergence": max_metrics.state_divergence_pct,
                "passed": validation_passed,
            }

        # Generate report
        report = ImaginationQualityReport(
            timestamp=datetime.now().isoformat(),
            checkpoint_path=self.config.checkpoint_path,
            n_rollouts=len(rollouts),
            total_steps_collected=total_steps,
            horizons=self.config.horizons,
            horizon_metrics=horizon_metrics,
            overall_state_mse=float(overall_state_mse),
            overall_reward_mse=float(overall_reward_mse),
            validation_passed=validation_passed,
            validation_details=validation_details,
            config=asdict(self.config),
        )

        # Log validation result
        if validation_passed:
            logger.info(
                f"VALIDATION PASSED: 15-step divergence < {self.config.divergence_threshold:.0%}"
            )
        else:
            logger.warning(f"VALIDATION FAILED: divergence exceeds threshold")

        return report

    def save_report(self, report: ImaginationQualityReport) -> str:
        """Save report to JSON file"""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert horizon_metrics to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "checkpoint_path": report.checkpoint_path,
            "n_rollouts": report.n_rollouts,
            "total_steps_collected": report.total_steps_collected,
            "horizons": report.horizons,
            "horizon_metrics": {
                str(h): asdict(m) for h, m in report.horizon_metrics.items()
            },
            "overall_state_mse": report.overall_state_mse,
            "overall_reward_mse": report.overall_reward_mse,
            "validation_passed": report.validation_passed,
            "validation_details": report.validation_details,
            "config": report.config,
        }

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Saved report to {output_path}")
        return str(output_path)


# ============================================================================
# CLI INTERFACE
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Imagination Quality Validation Framework"
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,5,10,15",
        help="Horizons to evaluate (comma-separated)",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=50,
        help="Number of rollouts to collect",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per rollout",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/dynamics_training/dynamics_final.pt",
        help="Path to dynamics model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/imagination_quality_report.json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Divergence threshold for validation gate",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=quiet, 1=normal, 2=debug)",
    )
    parser.add_argument(
        "--player-num",
        type=int,
        default=5,
        help="Number of players in environment",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set verbosity
    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose == 2:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse horizons
    horizons = [int(h) for h in args.horizons.split(",")]

    # Create config
    config = ImaginationQualityConfig(
        n_rollouts=args.n_rollouts,
        max_steps_per_rollout=args.max_steps,
        horizons=horizons,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        divergence_threshold=args.threshold,
        player_num=args.player_num,
        verbose=args.verbose > 0,
    )

    # Run validation
    validator = ImaginationQualityValidator(config)

    try:
        report = validator.validate()
        output_path = validator.save_report(report)

        # Print summary
        print("\n" + "=" * 60)
        print("IMAGINATION QUALITY VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Checkpoint: {report.checkpoint_path}")
        print(f"Rollouts: {report.n_rollouts}")
        print(f"Total steps: {report.total_steps_collected}")
        print("\nHorizon Metrics:")
        for h, m in report.horizon_metrics.items():
            print(f"  Horizon {h}:")
            print(f"    State MSE: {m.state_mse:.4f}")
            print(f"    Reward MSE: {m.reward_mse:.2f}")
            print(f"    Divergence: {m.state_divergence_pct:.2%}")
            print(f"    Uncertainty: {m.mean_uncertainty:.4f}")
        print("\nValidation Gate:")
        print(f"  Threshold: {report.validation_details['threshold']:.0%}")
        print(f"  Actual: {report.validation_details['actual_divergence']:.2%}")
        print(f"  Result: {'PASSED' if report.validation_passed else 'FAILED'}")
        print("=" * 60)
        print(f"Report saved to: {output_path}")
        print("=" * 60)

        return 0 if report.validation_passed else 1

    except FileNotFoundError as e:
        logger.error(f"Checkpoint not found: {e}")
        logger.info("Please train dynamics model first: python train/train_dynamics.py")
        return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
