"""
Dynamics Model Training Script - World Model Phase 1

Trains RSSEncoder + DynamicsModel + RewardModel for state/reward prediction.
Collects rollouts from trained MAPPO policy for training data.

Architecture:
- RSSEncoder: 3066-dim state → 128-dim latent (VAE-style)
- DynamicsModel: (z_t, action) → z_{t+1} + uncertainty
- RewardModel: (z_t, action) → reward prediction

Validation Gate:
- State prediction MSE < 0.1 (10% error threshold)
- Reward prediction MSE < 5.0

Usage:
    # Full training (50K steps)
    python train/train_dynamics.py --steps 50000

    # Quick test (1000 steps)
    python train/train_dynamics.py --steps 1000 --n-envs 1
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from ai.world_model.rssm import (
    RSSEncoder,
    RSSEncoderConfig,
    RSSMDecoder,
    RSSMVAE,
    compute_kl_loss,
    compute_reconstruction_loss,
)
from ai.world_model.dynamics import DynamicsModel, DynamicsConfig, DynamicsLoss
from ai.world_model.reward import RewardModel, RewardModelConfig, compute_reward_loss
from ai.mappo import MAPPOAgent, MAPPOAgentConfig
from ai.mappo.centralized_critic import CentralizedCriticConfig
from ai.mappo.mappo_policy import MAPPOActorConfig
from ai.state_encoder import StateEncoder
from ai.gym_wrapper import SGSConfig, SGSEnv


@dataclass
class DynamicsTrainingConfig:
    """Configuration for Dynamics Model training"""

    total_steps: int = 50000
    n_envs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-4
    log_interval: int = 1000
    checkpoint_interval: int = 10000

    # Model dimensions (must match world model architecture)
    state_dim: int = 2670  # Actual state dimension from StateEncoder
    latent_dim: int = 128
    hidden_dim: int = 256
    stochastic_dim: int = 64
    action_embed_dim: int = 64

    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    dynamics_weight: float = 1.0
    reward_weight: float = 0.5

    # Validation thresholds
    state_error_threshold: float = 0.1  # 10% error
    reward_error_threshold: float = 5.0

    # MAPPO rollout collection
    num_agents: int = 5
    local_state_dim: int = 2670
    global_state_dim: int = 2670 * 5
    action_dim: int = 20
    rollout_steps: int = 128

    # Paths
    mappo_checkpoint_path: str = "train/logs/mappo_50k_fixed_v2/mappo_checkpoint.pt"
    output_dir: str = "train/logs/dynamics_training"
    evidence_dir: str = ".sisyphus/evidence"

    device: Optional[str] = None


class RolloutCollector:
    """Collects rollouts from trained MAPPO policy"""

    def __init__(
        self,
        mappo_checkpoint_path: str,
        num_agents: int,
        local_state_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.device = device
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.state_encoder = StateEncoder()

        # Load MAPPO agent
        self.mappo_agent = self._load_mappo_agent(mappo_checkpoint_path)
        self.envs: List[SGSEnv] = []

    def _load_mappo_agent(self, checkpoint_path: str) -> MAPPOAgent:
        """Load trained MAPPO agent from checkpoint"""
        if not Path(checkpoint_path).exists():
            logger.warning(f"MAPPO checkpoint not found at {checkpoint_path}")
            logger.info("Creating new MAPPO agent (will use random actions)")
            config = MAPPOAgentConfig(
                num_agents=self.num_agents,
                actor_config=MAPPOActorConfig(
                    local_state_dim=self.local_state_dim,
                    action_dim=self.action_dim,
                ),
                critic_config=CentralizedCriticConfig(
                    local_state_dim=self.local_state_dim,
                    num_agents=self.num_agents,
                    global_state_dim=self.local_state_dim * self.num_agents,
                    action_dim=self.action_dim,
                ),
            )
            return MAPPOAgent(config).to(self.device)

        logger.info(f"Loading MAPPO checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location=self.device
        )

        config = checkpoint.get(
            "config",
            MAPPOAgentConfig(
                num_agents=self.num_agents,
                actor_config=MAPPOActorConfig(
                    local_state_dim=self.local_state_dim,
                    action_dim=self.action_dim,
                ),
                critic_config=CentralizedCriticConfig(
                    local_state_dim=self.local_state_dim,
                    num_agents=self.num_agents,
                    global_state_dim=self.local_state_dim * self.num_agents,
                    action_dim=self.action_dim,
                ),
            ),
        )

        agent = MAPPOAgent(config).to(self.device)

        if "actors_state_dict" in checkpoint:
            for i, actor in enumerate(agent.actors):
                actor.load_state_dict(checkpoint["actors_state_dict"][i])
        if "critic_state_dict" in checkpoint:
            agent.shared_critic.load_state_dict(checkpoint["critic_state_dict"])

        logger.info("MAPPO agent loaded successfully")
        return agent

    def create_envs(self, n_envs: int) -> List[SGSEnv]:
        """Create game environments"""
        envs = []
        for i in range(n_envs):
            sgs_config = SGSConfig(
                player_num=self.num_agents,
                max_rounds=15,
                use_action_mask=True,
                use_shaping=True,
                other_player_policy="rule",
            )
            env = SGSEnv(sgs_config)
            envs.append(env)
        self.envs = envs
        return envs

    def _get_local_obs(self, env: SGSEnv) -> np.ndarray:
        """Get local observations for all agents"""
        game_state = env._get_game_state_dict()
        local_obs = np.zeros((self.num_agents, self.local_state_dim), dtype=np.float32)
        for i in range(self.num_agents):
            local_obs[i] = self.state_encoder.encode(game_state, i)
        return local_obs

    def _get_action_masks(self, env: SGSEnv) -> np.ndarray:
        """Get action masks for all agents"""
        masks = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        game_state = env._get_game_state_dict()
        for i in range(self.num_agents):
            if i < len(env.players) and env.players[i].is_alive:
                type_mask, card_mask, target_mask = (
                    env.action_mask_generator.generate_masks(
                        game_state, env.players[i], env.engine, 0, None
                    )
                )
                combined_mask = np.concatenate([type_mask, card_mask, target_mask])
                if len(combined_mask) < self.action_dim:
                    combined_mask = np.pad(
                        combined_mask, (0, self.action_dim - len(combined_mask))
                    )
                masks[i] = combined_mask[: self.action_dim]
        return masks

    def collect_rollouts(self, env: SGSEnv, n_steps: int) -> Dict[str, np.ndarray]:
        """Collect rollout data from MAPPO policy"""
        obs, info = env.reset()

        states = []  # Raw game states (3066-dim)
        next_states = []
        actions = []  # Action indices
        action_types = []  # Action type indices
        rewards = []  # Observed rewards
        action_type = 0  # Default action type

        for step in range(n_steps):
            # Get current state for all agents
            game_state = env._get_game_state_dict()
            current_state = self.state_encoder.encode(
                game_state, 0
            )  # Agent 0 perspective

            # Get action from MAPPO
            local_obs = self._get_local_obs(env)
            masks = self._get_action_masks(env)

            local_obs_t = (
                torch.from_numpy(local_obs).unsqueeze(0).float().to(self.device)
            )
            masks_t = torch.from_numpy(masks).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                joint_action_t, _, _, _ = self.mappo_agent.get_actions(
                    local_obs_t, masks_t, deterministic=False
                )

            actions_np = joint_action_t.squeeze(0).cpu().numpy()

            # Execute action for current player
            current_player_idx = env.current_player_idx
            if (
                current_player_idx < len(env.players)
                and env.players[current_player_idx].is_alive
            ):
                action_type = int(actions_np[current_player_idx])
                try:
                    step_result = env.step(action_type)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                except Exception as e:
                    logger.warning(f"Action failed: {e}")
                    obs, info = env.reset()
                    done = True
                    reward = 0.0
            else:
                obs, info = env.reset()
                done = True
                reward = 0.0

            # Get next state
            if not done:
                next_game_state = env._get_game_state_dict()
                next_state = self.state_encoder.encode(next_game_state, 0)
            else:
                next_state = np.zeros_like(current_state)

            # Record data
            states.append(current_state)
            next_states.append(next_state)
            actions.append(action_type)
            action_types.append(action_type)
            rewards.append(reward)

            if done:
                obs, info = env.reset()

        return {
            "states": np.array(states, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "action_types": np.array(action_types, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
        }


class DynamicsTrainer:
    """Trains RSSEncoder + DynamicsModel + RewardModel"""

    def __init__(self, config: DynamicsTrainingConfig):
        self.config = config

        # Device setup
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")

        # Create world model components
        self._setup_models()

        # Create rollout collector
        self.rollout_collector = RolloutCollector(
            config.mappo_checkpoint_path,
            config.num_agents,
            config.local_state_dim,
            config.action_dim,
            self.device,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_model.parameters()),
            lr=config.learning_rate,
        )

        # Metrics tracking
        self._step_count = 0
        self._state_errors: List[float] = []
        self._reward_errors: List[float] = []
        self._kl_losses: List[float] = []
        self._recon_losses: List[float] = []

        self._writer: Optional[SummaryWriter] = None

        # Rollout buffer
        self._rollout_buffer: List[Dict[str, np.ndarray]] = []

    def _setup_models(self):
        """Setup all world model components"""
        c = self.config

        # RSSM Encoder (VAE-style)
        encoder_config = RSSEncoderConfig(
            state_dim=c.state_dim,
            latent_dim=c.latent_dim,
            hidden_dim=c.hidden_dim,
            stochastic_dim=c.stochastic_dim,
            deterministic_dim=c.latent_dim - c.stochastic_dim,
            kl_weight=c.kl_weight,
        )
        self.encoder = RSSEncoder(encoder_config).to(self.device)

        # RSSM Decoder
        self.decoder = RSSMDecoder(encoder_config).to(self.device)

        # Dynamics Model
        dynamics_config = DynamicsConfig(
            latent_dim=c.latent_dim,
            action_embed_dim=c.action_embed_dim,
            hidden_dim=c.hidden_dim,
        )
        self.dynamics = DynamicsModel(dynamics_config).to(self.device)

        # Reward Model
        reward_config = RewardModelConfig(
            latent_dim=c.latent_dim,
            action_embed_dim=c.action_embed_dim,
            hidden_dims=[c.hidden_dim, c.hidden_dim],
        )
        self.reward_model = RewardModel(reward_config).to(self.device)

        logger.info(
            f"Encoder params: {sum(p.numel() for p in self.encoder.parameters())}"
        )
        logger.info(
            f"Decoder params: {sum(p.numel() for p in self.decoder.parameters())}"
        )
        logger.info(
            f"Dynamics params: {sum(p.numel() for p in self.dynamics.parameters())}"
        )
        logger.info(
            f"Reward params: {sum(p.numel() for p in self.reward_model.parameters())}"
        )

    def collect_rollouts(
        self, n_envs: int, steps_per_env: int
    ) -> Dict[str, torch.Tensor]:
        """Collect training data from MAPPO policy"""
        envs = self.rollout_collector.create_envs(n_envs)

        all_states = []
        all_next_states = []
        all_actions = []
        all_rewards = []

        for env in envs:
            rollout = self.rollout_collector.collect_rollouts(env, steps_per_env)
            all_states.append(rollout["states"])
            all_next_states.append(rollout["next_states"])
            all_actions.append(rollout["action_types"])
            all_rewards.append(rollout["rewards"])

        for env in envs:
            env.close()

        # Concatenate and convert to tensors
        states = np.concatenate(all_states, axis=0)
        next_states = np.concatenate(all_next_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)

        return {
            "states": torch.from_numpy(states).float().to(self.device),
            "next_states": torch.from_numpy(next_states).float().to(self.device),
            "actions": torch.from_numpy(actions).long().to(self.device),
            "rewards": torch.from_numpy(rewards).float().to(self.device),
        }

    def compute_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all losses for world model training"""
        c = self.config

        batch_size = states.shape[0]
        self.dynamics.reset_hidden(batch_size, states.device)

        encoder_output = self.encoder.encode(states, deterministic=False)

        # Reconstruction loss (encoder-decoder)
        reconstructed = self.decoder.decode(encoder_output.latent)
        recon_loss = compute_reconstruction_loss(reconstructed, states)

        # KL loss (VAE regularization)
        kl_loss = compute_kl_loss(
            encoder_output.mean,
            encoder_output.log_std,
            free_bits=self.encoder.config.free_bits,
        )

        # Encode next state (target)
        next_encoder_output = self.encoder.encode(next_states, deterministic=True)
        z_next_target = next_encoder_output.latent

        # Dynamics prediction
        z_t = encoder_output.latent
        action_type = actions  # Simplified: use action type directly
        z_pred, uncertainty = self.dynamics.forward(z_t, action_type, None, None)

        # Dynamics loss (state prediction MSE)
        dynamics_loss = F.mse_loss(z_pred, z_next_target)

        # Reward prediction
        # Build action embedding manually (simplified)
        action_embed = self.dynamics.action_embedder(actions, None, None)
        reward_pred = self.reward_model.forward(z_t, action_embed)
        reward_target = rewards.unsqueeze(-1)

        # Reward loss
        reward_loss = compute_reward_loss(reward_pred, reward_target)

        # Total loss
        total_loss = (
            c.reconstruction_weight * recon_loss
            + c.kl_weight * kl_loss
            + c.dynamics_weight * dynamics_loss
            + c.reward_weight * reward_loss
        )

        # Metrics
        state_error = dynamics_loss.item()
        reward_error = reward_loss.item()

        metrics = {
            "reconstruction_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "state_prediction_error": state_error,
            "reward_prediction_error": reward_error,
            "total_loss": total_loss.item(),
            "mean_uncertainty": uncertainty.mean().item(),
        }

        return total_loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()

        loss, metrics = self.compute_loss(
            batch["states"],
            batch["next_states"],
            batch["actions"],
            batch["rewards"],
        )

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_model.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return metrics

    def validate(self) -> Dict[str, float]:
        """Run validation to check prediction errors"""
        # Collect fresh rollouts for validation
        val_data = self.collect_rollouts(1, 64)

        with torch.no_grad():
            _, metrics = self.compute_loss(
                val_data["states"],
                val_data["next_states"],
                val_data["actions"],
                val_data["rewards"],
            )

        return metrics

    def train(self, log_dir: Optional[str] = None) -> Dict[str, float]:
        """Main training loop"""
        c = self.config

        # Setup logging
        tb_dir = (
            Path(log_dir)
            if log_dir
            else Path(c.output_dir)
            / f"dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(tb_dir))

        logger.info(
            f"Starting Dynamics training for {c.total_steps} steps on {self.device}"
        )
        logger.info(f"Config: batch_size={c.batch_size}, lr={c.learning_rate}")
        logger.info(f"TensorBoard logs: {tb_dir}")

        start_time = time.time()
        pbar = tqdm(total=c.total_steps, desc="Training", unit="steps")

        # Initial rollout collection
        logger.info("Collecting initial rollouts...")
        rollout_data = self.collect_rollouts(c.n_envs, c.rollout_steps)
        n_samples = rollout_data["states"].shape[0]
        logger.info(f"Collected {n_samples} samples")

        while self._step_count < c.total_steps:
            # Sample batch from rollout data
            indices = np.random.choice(
                n_samples, min(c.batch_size, n_samples), replace=False
            )
            batch = {
                "states": rollout_data["states"][indices],
                "next_states": rollout_data["next_states"][indices],
                "actions": rollout_data["actions"][indices],
                "rewards": rollout_data["rewards"][indices],
            }

            # Train step
            metrics = self.train_step(batch)

            # Track metrics
            self._state_errors.append(metrics["state_prediction_error"])
            self._reward_errors.append(metrics["reward_prediction_error"])
            self._kl_losses.append(metrics["kl_loss"])
            self._recon_losses.append(metrics["reconstruction_loss"])

            self._step_count += 1

            # Logging
            pbar.update(1)
            pbar.set_postfix(
                state_err=f"{metrics['state_prediction_error']:.4f}",
                reward_err=f"{metrics['reward_prediction_error']:.2f}",
                kl=f"{metrics['kl_loss']:.4f}",
            )

            if self._writer:
                self._writer.add_scalar(
                    "train/state_prediction_error",
                    metrics["state_prediction_error"],
                    self._step_count,
                )
                self._writer.add_scalar(
                    "train/reward_prediction_error",
                    metrics["reward_prediction_error"],
                    self._step_count,
                )
                self._writer.add_scalar(
                    "train/kl_loss", metrics["kl_loss"], self._step_count
                )
                self._writer.add_scalar(
                    "train/reconstruction_loss",
                    metrics["reconstruction_loss"],
                    self._step_count,
                )
                self._writer.add_scalar(
                    "train/total_loss", metrics["total_loss"], self._step_count
                )
                self._writer.add_scalar(
                    "train/mean_uncertainty",
                    metrics["mean_uncertainty"],
                    self._step_count,
                )

            # Log interval
            if self._step_count % c.log_interval == 0:
                logger.info(
                    f"[Step {self._step_count}] "
                    f"state_err={metrics['state_prediction_error']:.4f}, "
                    f"reward_err={metrics['reward_prediction_error']:.2f}, "
                    f"kl={metrics['kl_loss']:.4f}, "
                    f"recon={metrics['reconstruction_loss']:.4f}"
                )

            # Refresh rollouts periodically
            if (
                self._step_count % (c.rollout_steps * 2) == 0
                and self._step_count < c.total_steps - c.rollout_steps
            ):
                logger.info("Refreshing rollouts...")
                rollout_data = self.collect_rollouts(c.n_envs, c.rollout_steps)
                n_samples = rollout_data["states"].shape[0]

            # Checkpoint
            if self._step_count % c.checkpoint_interval == 0:
                self.save_checkpoint(
                    str(tb_dir / f"dynamics_checkpoint_{self._step_count}.pt")
                )

        pbar.close()
        training_time = time.time() - start_time

        # Final validation
        logger.info("Running final validation...")
        val_metrics = self.validate()

        # Compute final averages
        final_state_error = np.mean(self._state_errors[-100:])
        final_reward_error = np.mean(self._reward_errors[-100:])
        final_kl = np.mean(self._kl_losses[-100:])
        final_recon = np.mean(self._recon_losses[-100:])

        # Check acceptance gate
        state_ok = final_state_error < c.state_error_threshold
        reward_ok = final_reward_error < c.reward_error_threshold
        passed = state_ok and reward_ok

        final_metrics = {
            "total_steps": self._step_count,
            "training_time": training_time,
            "final_state_error": final_state_error,
            "final_reward_error": final_reward_error,
            "final_kl_loss": final_kl,
            "final_recon_loss": final_recon,
            "state_error_threshold": c.state_error_threshold,
            "reward_error_threshold": c.reward_error_threshold,
            "state_ok": state_ok,
            "reward_ok": reward_ok,
            "passed": passed,
        }

        logger.info(f"Training completed in {training_time:.1f}s")
        logger.info(
            f"Final state error: {final_state_error:.4f} (threshold: {c.state_error_threshold})"
        )
        logger.info(
            f"Final reward error: {final_reward_error:.2f} (threshold: {c.reward_error_threshold})"
        )

        if passed:
            logger.info("VALIDATION PASSED: World model ready for Phase 2")
        else:
            logger.warning("VALIDATION FAILED: Prediction errors exceed thresholds")

        # Save final checkpoint
        self.save_checkpoint(str(tb_dir / "dynamics_final.pt"))

        # Save evidence
        self.save_evidence(tb_dir)

        if self._writer:
            self._writer.close()

        return final_metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "step": self._step_count,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "dynamics_state_dict": self.dynamics.state_dict(),
                "reward_model_state_dict": self.reward_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "metrics": {
                    "state_errors": self._state_errors[-100:],
                    "reward_errors": self._reward_errors[-100:],
                    "kl_losses": self._kl_losses[-100:],
                    "recon_losses": self._recon_losses[-100:],
                },
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def save_evidence(self, output_dir: Path):
        """Save training evidence"""
        evidence_path = Path(self.config.evidence_dir) / "dynamics_training.log"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)

        evidence = []
        evidence.append("Dynamics Model Training Evidence Log")
        evidence.append(f"Generated: {datetime.now().isoformat()}")
        evidence.append("=" * 60)
        evidence.append("Configuration:")
        evidence.append(f"  - Total steps: {self.config.total_steps}")
        evidence.append(f"  - Batch size: {self.config.batch_size}")
        evidence.append(f"  - Learning rate: {self.config.learning_rate}")
        evidence.append(f"  - State dim: {self.config.state_dim}")
        evidence.append(f"  - Latent dim: {self.config.latent_dim}")
        evidence.append(f"  - Hidden dim: {self.config.hidden_dim}")
        evidence.append("=" * 60)
        evidence.append("Training Metrics (last 100 steps):")
        evidence.append(
            f"  - State prediction error: {np.mean(self._state_errors[-100:]) if self._state_errors else 'N/A':.4f}"
        )
        evidence.append(
            f"  - Reward prediction error: {np.mean(self._reward_errors[-100:]) if self._reward_errors else 'N/A':.2f}"
        )
        evidence.append(
            f"  - KL loss: {np.mean(self._kl_losses[-100:]) if self._kl_losses else 'N/A':.4f}"
        )
        evidence.append(
            f"  - Reconstruction loss: {np.mean(self._recon_losses[-100:]) if self._recon_losses else 'N/A':.4f}"
        )
        evidence.append("=" * 60)
        evidence.append("Validation Gate:")
        evidence.append(
            f"  - State error threshold: {self.config.state_error_threshold}"
        )
        evidence.append(
            f"  - Reward error threshold: {self.config.reward_error_threshold}"
        )

        final_state_error = (
            np.mean(self._state_errors[-100:]) if self._state_errors else float("inf")
        )
        final_reward_error = (
            np.mean(self._reward_errors[-100:]) if self._reward_errors else float("inf")
        )

        state_ok = final_state_error < self.config.state_error_threshold
        reward_ok = final_reward_error < self.config.reward_error_threshold

        evidence.append(f"  - State OK: {state_ok} (error={final_state_error:.4f})")
        evidence.append(f"  - Reward OK: {reward_ok} (error={final_reward_error:.2f})")
        evidence.append("=" * 60)

        if state_ok and reward_ok:
            evidence.append("ASSERTION PASSED: World model validation gate passed")
            evidence.append("Phase 2 (imagination-based training) can proceed")
        else:
            evidence.append("ASSERTION FAILED: World model validation gate failed")
            if not state_ok:
                evidence.append(
                    f"  - State error {final_state_error:.4f} >= threshold {self.config.state_error_threshold}"
                )
            if not reward_ok:
                evidence.append(
                    f"  - Reward error {final_reward_error:.2f} >= threshold {self.config.reward_error_threshold}"
                )

        evidence.append("=" * 60)
        evidence.append(f"Output directory: {output_dir}")

        with open(evidence_path, "w") as f:
            f.write("\n".join(evidence))

        logger.info(f"Saved evidence to {evidence_path}")

        # Also save to output_dir
        with open(output_dir / "dynamics_training.log", "w") as f:
            f.write("\n".join(evidence))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamics Model Training for World Model"
    )
    parser.add_argument("--steps", type=int, default=50000, help="Total training steps")
    parser.add_argument(
        "--n-envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=1000, help="Log interval")
    parser.add_argument(
        "--mappo-checkpoint",
        type=str,
        default="train/logs/mappo_50k_fixed_v2/mappo_checkpoint.pt",
        help="MAPPO checkpoint path for rollout collection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train/logs/dynamics_training",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda, cpu, or None for auto)"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)

    config = DynamicsTrainingConfig(
        total_steps=args.steps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_interval=args.log_interval,
        mappo_checkpoint_path=args.mappo_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = DynamicsTrainer(config)

    try:
        metrics = trainer.train(log_dir=str(output_dir))

        if metrics["passed"]:
            logger.info("Training completed successfully - validation gate passed")
            return 0
        else:
            logger.warning("Training completed but validation gate failed")
            return 1

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()

        # Save partial evidence
        evidence_path = Path(config.evidence_dir) / "dynamics_training_failed.log"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(evidence_path, "w") as f:
            f.write(f"Dynamics training failed: {e}\n")
            f.write(traceback.format_exc())

        return 1


if __name__ == "__main__":
    sys.exit(main())
