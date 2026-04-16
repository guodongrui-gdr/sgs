"""
MAPPO + World Model Integration Training Script

Full integration of World Model imagination with MAPPO multi-agent training.
Supports configurable flags for independent feature control.

Features:
- MAPPO mode: Centralized critic with decentralized actors
- IPPO mode: Independent PPO (decentralized training)
- World Model: Imagination-based training using dynamics model
- Team Rewards: Team-based reward allocation with coordination bonuses

Configuration Options:
- use_world_model: Enable/disable imagination-based training
- use_mappo: MAPPO (centralized) vs IPPO (decentralized) mode
- use_team_rewards: Enable/disable team reward allocation

Usage:
    # Full integration (all features)
    python train/train_mappo_world_model.py --steps 100000 --use-world-model true --use-mappo true --use-team-rewards true

    # MAPPO only (no imagination, no team rewards)
    python train/train_mappo_world_model.py --steps 100000 --use-mappo true

    # World Model only (IPPO + imagination)
    python train/train_mappo_world_model.py --steps 100000 --use-world-model true

    # Quick test
    python train/train_mappo_world_model.py --steps 1000 --n-envs 1

Output:
    outputs/mappo_world_model/{timestamp}/
    - Checkpoints (mappo_world_model_checkpoint_{step}.pt)
    - TensorBoard logs
    - Coordination metrics
    - Training evidence log
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ai.gym_wrapper import SGSEnv

# Import for unpickling dynamics checkpoints
from train.train_dynamics import DynamicsTrainingConfig


class TrainingMode(Enum):
    """Training mode selection"""

    MAPPO = "mappo"  # Centralized critic + decentralized actors
    IPPO = "ippo"  # Independent PPO (fully decentralized)


@dataclass
class MAPPOWorldModelConfig:
    """Configuration for MAPPO + World Model integration training"""

    # Training parameters
    steps_total: int = 100000
    n_envs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Feature flags
    use_world_model: bool = False  # Enable imagination-based training
    use_mappo: bool = True  # MAPPO vs IPPO mode
    use_team_rewards: bool = False  # Enable team reward allocation

    # Checkpoint and logging
    checkpoint_interval: int = 10000
    eval_interval: int = 5000
    log_interval: int = 1000
    num_eval_episodes: int = 10

    # Model dimensions (from existing architecture)
    num_agents: int = 5
    local_state_dim: int = 2670
    global_state_dim: int = 2670 * 5
    action_dim: int = 20
    latent_dim: int = 128

    # World Model parameters
    imagination_ratio: float = 0.3  # Ratio of imagination vs real steps
    imagination_horizon: int = 15  # Max steps for imagination rollouts
    imagination_batch_size: int = 32  # Batch size for imagination

    # Team reward parameters
    lord_loyalist_coordination_bonus: float = 2.0
    rebel_focus_fire_bonus: float = 1.5
    protect_lord_bonus: float = 3.0

    # Health thresholds
    entropy_threshold: float = 0.01
    value_std_threshold: float = 0.001

    # Paths
    mappo_checkpoint_path: str = "train/logs/mappo_50k_fixed_v2/mappo_checkpoint.pt"
    dynamics_checkpoint_path: str = "train/logs/dynamics_training/dynamics_final.pt"
    output_dir: str = "train/logs/mappo_world_model"
    evidence_dir: str = ".sisyphus/evidence"

    device: Optional[str] = None  # "cuda", "cuda:0", "cpu", or None for auto


# === Component Loaders ===


class MAPPOCheckpointLoader:
    """Loads MAPPO agent from checkpoint"""

    def __init__(
        self, checkpoint_path: str, config: MAPPOWorldModelConfig, device: torch.device
    ):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint and return agent state"""
        if not Path(self.checkpoint_path).exists():
            logger.warning(f"MAPPO checkpoint not found at {self.checkpoint_path}")
            logger.info("Creating new MAPPO agent (will train from scratch)")
            return None

        logger.info(f"Loading MAPPO checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(
            self.checkpoint_path, weights_only=False, map_location=self.device
        )

        logger.info("MAPPO checkpoint loaded successfully")
        return checkpoint


class DynamicsCheckpointLoader:
    """Loads World Model dynamics from checkpoint"""

    def __init__(
        self, checkpoint_path: str, config: MAPPOWorldModelConfig, device: torch.device
    ):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint and return dynamics state"""
        if not Path(self.checkpoint_path).exists():
            logger.warning(f"Dynamics checkpoint not found at {self.checkpoint_path}")
            logger.info("World Model imagination disabled (no dynamics model)")
            return None

        logger.info(f"Loading Dynamics checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(
            self.checkpoint_path, weights_only=False, map_location=self.device
        )

        # Check validation status
        metrics = checkpoint.get("metrics", {})
        state_error = metrics.get("state_errors", [])
        if state_error:
            final_error = (
                np.mean(state_error[-10:])
                if len(state_error) >= 10
                else np.mean(state_error)
            )
            logger.info(f"Dynamics model state error: {final_error:.4f}")

        logger.info("Dynamics checkpoint loaded successfully")
        return checkpoint


# === Coordination Metrics Tracker ===


@dataclass
class CoordinationMetrics:
    """Coordination metrics for team-based gameplay"""

    # Lord team coordination
    lord_loyalist_coordination_count: int = 0
    lord_team_protect_events: int = 0
    lord_team_shared_target_count: int = 0

    # Rebel team coordination
    rebel_focus_fire_count: int = 0
    rebel_shared_target_events: int = 0
    rebel_damage_to_lord_total: float = 0.0

    # Win rates by team
    lord_team_wins: int = 0
    rebel_team_wins: int = 0
    spy_wins: int = 0
    total_episodes: int = 0

    # Harm penalties
    harm_teammate_events: int = 0

    # Performance
    avg_episode_length: float = 0.0
    avg_lord_team_reward: float = 0.0
    avg_rebel_team_reward: float = 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get metrics summary for logging"""
        total = self.total_episodes if self.total_episodes > 0 else 1

        return {
            "coordination/lord_loyalist_freq": self.lord_loyalist_coordination_count
            / total,
            "coordination/lord_protect_freq": self.lord_team_protect_events / total,
            "coordination/rebel_focus_fire_freq": self.rebel_focus_fire_count / total,
            "coordination/rebel_shared_target_freq": self.rebel_shared_target_events
            / total,
            "coordination/harm_teammate_freq": self.harm_teammate_events / total,
            "win_rate/lord_team": self.lord_team_wins / total,
            "win_rate/rebel_team": self.rebel_team_wins / total,
            "win_rate/spy": self.spy_wins / total,
            "performance/avg_episode_length": self.avg_episode_length,
            "reward/avg_lord_team": self.avg_lord_team_reward,
            "reward/avg_rebel_team": self.avg_rebel_team_reward,
        }

    def reset_episode(self):
        """Reset episode-level counters"""
        self.lord_loyalist_coordination_count = 0
        self.lord_team_protect_events = 0
        self.lord_team_shared_target_count = 0
        self.rebel_focus_fire_count = 0
        self.rebel_shared_target_events = 0
        self.rebel_damage_to_lord_total = 0.0
        self.harm_teammate_events = 0


class CoordinationTracker:
    """Tracks coordination events during training/evaluation"""

    def __init__(self, config: MAPPOWorldModelConfig):
        self.config = config
        self.metrics = CoordinationMetrics()
        self._episode_rewards: Dict[str, List[float]] = {
            "lord_team": [],
            "rebel_team": [],
        }
        self._episode_lengths: List[int] = []

    def record_coordination_event(
        self,
        event_type: str,
        team: str,
        agent_idx: int,
        target_idx: Optional[int] = None,
        damage_value: float = 0.0,
    ):
        """Record a coordination event"""
        if event_type == "lord_loyalist_coordination":
            self.metrics.lord_loyalist_coordination_count += 1
        elif event_type == "lord_protect":
            self.metrics.lord_team_protect_events += 1
        elif event_type == "shared_target":
            if team == "lord":
                self.metrics.lord_team_shared_target_count += 1
            elif team == "rebel":
                self.metrics.rebel_shared_target_events += 1
        elif event_type == "focus_fire":
            self.metrics.rebel_focus_fire_count += 1
            if target_idx == 0:  # Targeting lord
                self.metrics.rebel_damage_to_lord_total += damage_value
        elif event_type == "harm_teammate":
            self.metrics.harm_teammate_events += 1

    def record_episode_result(
        self,
        winner: Optional[str],
        episode_length: int,
        team_rewards: Dict[str, float],
    ):
        """Record episode result"""
        self.metrics.total_episodes += 1
        self._episode_lengths.append(episode_length)

        if winner == "lord" or winner == "lord_team":
            self.metrics.lord_team_wins += 1
        elif winner == "rebel" or winner == "rebel_team":
            self.metrics.rebel_team_wins += 1
        elif winner == "spy" or winner == "内奸":
            self.metrics.spy_wins += 1

        # Track team rewards
        if "lord_team" in team_rewards:
            self._episode_rewards["lord_team"].append(team_rewards["lord_team"])
        if "rebel_team" in team_rewards:
            self._episode_rewards["rebel_team"].append(team_rewards["rebel_team"])

        # Update averages
        self.metrics.avg_episode_length = float(
            np.mean(self._episode_lengths) if self._episode_lengths else 0.0
        )
        self.metrics.avg_lord_team_reward = float(
            np.mean(self._episode_rewards["lord_team"])
            if self._episode_rewards["lord_team"]
            else 0.0
        )
        self.metrics.avg_rebel_team_reward = float(
            np.mean(self._episode_rewards["rebel_team"])
            if self._episode_rewards["rebel_team"]
            else 0.0
        )

    def get_metrics(self) -> CoordinationMetrics:
        """Get current metrics"""
        return self.metrics

    def reset(self):
        """Reset all metrics"""
        self.metrics = CoordinationMetrics()
        self._episode_rewards = {"lord_team": [], "rebel_team": []}
        self._episode_lengths = []


# === Integration Trainer ===


class MAPPOWorldModelTrainer:
    """
    Integrated trainer for MAPPO + World Model

    Handles:
    1. MAPPO/IPPO training with centralized/decentralized critics
    2. World Model imagination-based training
    3. Team reward allocation with coordination bonuses
    4. Coordination metrics tracking and logging
    """

    def __init__(self, config: MAPPOWorldModelConfig):
        self.config = config

        # Device setup
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Configuration:")
        logger.info(
            f"  - Training mode: {TrainingMode.MAPPO if config.use_mappo else TrainingMode.IPPO}"
        )
        logger.info(f"  - World Model: {config.use_world_model}")
        logger.info(f"  - Team Rewards: {config.use_team_rewards}")
        logger.info(f"  - Total steps: {config.steps_total}")

        # Initialize components
        self._init_agent()
        self._init_world_model()
        self._init_team_rewards()
        self._init_coordination_tracker()

        # Training state
        self._step_count = 0
        self._episode_count = 0
        self._entropy_history: List[float] = []
        self._value_std_history: List[float] = []
        self._loss_history: List[float] = []

        # TensorBoard writer
        self._writer: Optional[SummaryWriter] = None

    def _init_agent(self):
        """Initialize MAPPO/IPPO agent"""
        from ai.mappo import MAPPOAgent, MAPPOAgentConfig
        from ai.mappo.centralized_critic import CentralizedCriticConfig
        from ai.mappo.mappo_policy import MAPPOActorConfig
        from ai.state_encoder import StateEncoder

        # Load checkpoint if available
        loader = MAPPOCheckpointLoader(
            self.config.mappo_checkpoint_path, self.config, self.device
        )
        checkpoint = loader.load()

        if checkpoint and "agent_config" in checkpoint:
            agent_config = checkpoint["agent_config"]
        else:
            agent_config = MAPPOAgentConfig(
                num_agents=self.config.num_agents,
                actor_config=MAPPOActorConfig(
                    local_state_dim=self.config.local_state_dim,
                    action_dim=self.config.action_dim,
                ),
                critic_config=CentralizedCriticConfig(
                    local_state_dim=self.config.local_state_dim,
                    num_agents=self.config.num_agents,
                    global_state_dim=self.config.global_state_dim,
                    action_dim=self.config.action_dim,
                ),
                learning_rate=self.config.learning_rate,
                ppo_gamma=self.config.gamma,
                ppo_gae_lambda=self.config.gae_lambda,
                entropy_threshold=self.config.entropy_threshold,
                value_std_threshold=self.config.value_std_threshold,
            )

        self.agent = MAPPOAgent(agent_config).to(self.device)
        self.state_encoder = StateEncoder()

        # Load weights if checkpoint available
        if checkpoint:
            agent_state = checkpoint.get("agent_state", checkpoint)
            if "actors_state_dict" in agent_state:
                for i, actor in enumerate(self.agent.actors):
                    if i < len(agent_state["actors_state_dict"]):
                        actor.load_state_dict(agent_state["actors_state_dict"][i])
            if "critic_state_dict" in agent_state:
                self.agent.shared_critic.load_state_dict(
                    agent_state["critic_state_dict"]
                )
            logger.info("Loaded MAPPO weights from checkpoint")

        self.agent_config = agent_config

    def _init_world_model(self):
        """Initialize World Model components"""
        if not self.config.use_world_model:
            self.imagination_env = None
            self.encoder = None
            self.dynamics = None
            self.reward_model = None
            logger.info("World Model disabled - training without imagination")
            return

        try:
            from ai.world_model.imagination_env import (
                ImaginedEnvironment,
                ImaginationConfig,
            )

            # Load dynamics checkpoint
            loader = DynamicsCheckpointLoader(
                self.config.dynamics_checkpoint_path, self.config, self.device
            )
            checkpoint = loader.load()

            if checkpoint:
                # Create imagination environment with checkpoint
                self.imagination_env = ImaginedEnvironment(
                    checkpoint_path=self.config.dynamics_checkpoint_path,
                    config=ImaginationConfig(
                        state_dim=self.config.local_state_dim,
                        latent_dim=self.config.latent_dim,
                        max_imagination_horizon=self.config.imagination_horizon,
                        device=str(self.device),
                    ),
                )
                logger.info(f"ImaginedEnvironment initialized with checkpoint")
            else:
                # Placeholder: Create without checkpoint (will not work properly)
                logger.warning(
                    "Creating placeholder ImaginedEnvironment without checkpoint"
                )
                self.imagination_env = ImaginedEnvironment(
                    config=ImaginationConfig(
                        state_dim=self.config.local_state_dim,
                        latent_dim=self.config.latent_dim,
                        max_imagination_horizon=self.config.imagination_horizon,
                        device=str(self.device),
                    )
                )

            # Extract models for direct access (optional)
            self.encoder = self.imagination_env.encoder
            self.dynamics = self.imagination_env.dynamics
            self.reward_model = self.imagination_env.reward_model

        except ImportError as e:
            logger.warning(f"World Model components not available: {e}")
            logger.info("Falling back to training without imagination")
            self.imagination_env = None
            self.encoder = None
            self.dynamics = None
            self.reward_model = None

    def _init_team_rewards(self):
        """Initialize team reward allocator"""
        if not self.config.use_team_rewards:
            self.team_reward_allocator = None
            logger.info("Team Rewards disabled - using individual rewards")
            return

        try:
            from ai.mappo.team_rewards import TeamRewardAllocator, TeamRewardConfig

            reward_config = TeamRewardConfig(
                lord_loyalist_coordination_bonus=self.config.lord_loyalist_coordination_bonus,
                rebel_focus_fire_bonus=self.config.rebel_focus_fire_bonus,
                protect_lord_bonus=self.config.protect_lord_bonus,
            )

            self.team_reward_allocator = TeamRewardAllocator(
                config=reward_config,
                num_agents=self.config.num_agents,
            )
            logger.info("TeamRewardAllocator initialized")

        except ImportError as e:
            logger.warning(f"Team Reward components not available: {e}")
            logger.info("Falling back to individual rewards")
            self.team_reward_allocator = None

    def _init_coordination_tracker(self):
        """Initialize coordination metrics tracker"""
        self.coordination_tracker = CoordinationTracker(self.config)

    def _create_env(self) -> "SGSEnv":
        """Create a single SGS environment"""
        from ai.gym_wrapper import SGSConfig, SGSEnv

        sgs_config = SGSConfig(
            player_num=self.config.num_agents,
            max_rounds=15,
            use_action_mask=True,
            use_shaping=True,
            other_player_policy="rule",
        )
        return SGSEnv(sgs_config)

    def _get_global_state(self, env: "SGSEnv") -> np.ndarray:
        """Get global state for all agents"""
        game_state = env._get_game_state_dict()

        local_states = []
        for i in range(self.config.num_agents):
            local_state = self.state_encoder.encode(game_state, i)
            local_states.append(local_state)

        return np.concatenate(local_states)

    def _get_action_masks(self, env: "SGSEnv") -> np.ndarray:
        """Get action masks for all agents"""
        masks = np.zeros(
            (self.config.num_agents, self.config.action_dim), dtype=np.float32
        )

        game_state = env._get_game_state_dict()

        for i in range(self.config.num_agents):
            if i < len(env.players) and env.players[i].is_alive:
                type_mask, card_mask, target_mask = (
                    env.action_mask_generator.generate_masks(
                        game_state, env.players[i], env.engine, 0, None
                    )
                )
                combined_mask = np.concatenate([type_mask, card_mask, target_mask])
                if len(combined_mask) < self.config.action_dim:
                    combined_mask = np.pad(
                        combined_mask, (0, self.config.action_dim - len(combined_mask))
                    )
                masks[i] = combined_mask[: self.config.action_dim]

        return masks

    def _collect_real_rollout(
        self, env: "SGSEnv", n_steps: int
    ) -> Dict[str, torch.Tensor]:
        """Collect rollout from real environment"""
        obs, info = env.reset()

        # Initialize team reward allocator if enabled
        if self.team_reward_allocator and hasattr(env, "players"):
            self.team_reward_allocator.initialize_teams(env.players)

        global_states = []
        local_observations = []
        joint_actions = []
        old_log_probs = []
        rewards = []
        dones = []
        action_masks = []
        values = []

        team_rewards_episode = {"lord_team": 0.0, "rebel_team": 0.0}

        for step in range(n_steps):
            global_state = self._get_global_state(env)
            local_obs = np.zeros(
                (self.config.num_agents, self.config.local_state_dim), dtype=np.float32
            )

            game_state = env._get_game_state_dict()
            for i in range(self.config.num_agents):
                local_obs[i] = self.state_encoder.encode(game_state, i)

            masks = self._get_action_masks(env)

            global_state_t = (
                torch.from_numpy(global_state).unsqueeze(0).float().to(self.device)
            )
            local_obs_t = (
                torch.from_numpy(local_obs).unsqueeze(0).float().to(self.device)
            )
            masks_t = torch.from_numpy(masks).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                joint_action_t, log_probs_t, entropies_t, mean_entropy = (
                    self.agent.get_actions(local_obs_t, masks_t, deterministic=False)
                )

                value_t = self.agent.get_centralized_value(
                    global_state_t, joint_action_t
                )

            actions_np = joint_action_t.squeeze(0).cpu().numpy()

            current_player_idx = env.current_player_idx
            episode_reward = 0.0

            if (
                current_player_idx < len(env.players)
                and env.players[current_player_idx].is_alive
            ):
                action_type = int(actions_np[current_player_idx])
                try:
                    step_result = env.step(action_type)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                    episode_reward = reward

                    # Track coordination if team rewards enabled
                    if self.team_reward_allocator and self.config.use_team_rewards:
                        self._track_coordination(
                            env, current_player_idx, action_type, info
                        )
                        team_rewards_episode["lord_team"] += info.get(
                            "lord_team_reward", 0.0
                        )
                        team_rewards_episode["rebel_team"] += info.get(
                            "rebel_team_reward", 0.0
                        )

                except Exception as e:
                    logger.warning(f"Action failed: {e}")
                    obs, info = env.reset()
                    done = True
                    episode_reward = 0.0
            else:
                obs, info = env.reset()
                done = True
                episode_reward = 0.0

            global_states.append(global_state)
            local_observations.append(local_obs)
            joint_actions.append(actions_np)
            old_log_probs.append(log_probs_t.squeeze(0).cpu().numpy())
            rewards.append(episode_reward)
            dones.append(done)
            action_masks.append(masks)
            values.append(value_t.item())

            self._entropy_history.append(mean_entropy.item())

            if done:
                # Record episode result
                winner = info.get("winner") if isinstance(info, dict) else None
                self.coordination_tracker.record_episode_result(
                    winner=winner,
                    episode_length=step + 1,
                    team_rewards=team_rewards_episode,
                )
                team_rewards_episode = {"lord_team": 0.0, "rebel_team": 0.0}

                obs, info = env.reset()
                self._episode_count += 1

                # Reinitialize team reward allocator
                if self.team_reward_allocator and hasattr(env, "players"):
                    self.team_reward_allocator.initialize_teams(env.players)

        # Convert to tensors
        global_states_t = (
            torch.from_numpy(np.array(global_states)).float().to(self.device)
        )
        local_observations_t = (
            torch.from_numpy(np.array(local_observations)).float().to(self.device)
        )
        joint_actions_t = (
            torch.from_numpy(np.array(joint_actions)).long().to(self.device)
        )
        old_log_probs_t = (
            torch.from_numpy(np.array(old_log_probs)).float().to(self.device)
        )
        rewards_t = torch.from_numpy(np.array(rewards)).float().to(self.device)
        dones_t = torch.from_numpy(np.array(dones)).float().to(self.device)
        action_masks_t = (
            torch.from_numpy(np.array(action_masks)).float().to(self.device)
        )
        values_t = torch.from_numpy(np.array(values)).float().to(self.device)

        # Compute advantages
        next_global_state = self._get_global_state(env)
        next_global_state_t = (
            torch.from_numpy(next_global_state).unsqueeze(0).float().to(self.device)
        )

        with torch.no_grad():
            dummy_actions = torch.zeros(
                1, self.config.num_agents, dtype=torch.long, device=self.device
            )
            next_value = self.agent.get_centralized_value(
                next_global_state_t, dummy_actions
            ).item()

        next_values = np.zeros(len(rewards))
        for i in range(len(rewards)):
            if dones[i]:
                next_values[i] = 0
            elif i == len(rewards) - 1:
                next_values[i] = next_value
            else:
                next_values[i] = values[i + 1]

        next_values_t = torch.from_numpy(next_values).float().to(self.device)

        advantages, returns = self.agent.compute_gae(
            values_t, rewards_t, dones_t, next_values_t
        )

        return {
            "global_states": global_states_t,
            "local_observations": local_observations_t,
            "joint_actions": joint_actions_t,
            "old_log_probs": old_log_probs_t,
            "old_values": values_t,
            "advantages": advantages,
            "returns": returns,
            "action_masks": action_masks_t,
        }

    def _collect_imagination_rollout(
        self,
        real_state: np.ndarray,
        n_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Collect rollout from imagination environment"""
        if self.imagination_env is None:
            return {}

        # Reset imagination from real state
        z_0, info = self.imagination_env.reset(initial_state=real_state)

        z_sequence = []
        rewards_sequence = []
        uncertainties_sequence = []

        for step in range(n_steps):
            # Get action from agent (using latent as pseudo-observation)
            # Note: This is simplified - real implementation needs latent-to-obs mapping
            local_obs_t = (
                z_0.unsqueeze(0)
                .expand(1, self.config.num_agents, -1)
                .float()
                .to(self.device)
            )

            # Generate action masks from imagination state
            masks = info.get("action_masks", {})
            if masks:
                type_mask = masks.get("type", np.ones(self.config.action_dim))
                masks_t = (
                    torch.from_numpy(type_mask)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                )
                masks_t = masks_t.expand(1, self.config.num_agents, -1)
            else:
                masks_t = (
                    torch.ones(1, self.config.num_agents, self.config.action_dim)
                    .float()
                    .to(self.device)
                )

            with torch.no_grad():
                joint_action_t, _, _, _ = self.agent.get_actions(
                    local_obs_t, masks_t, deterministic=False
                )

            actions_np = joint_action_t.squeeze(0).cpu().numpy()

            # Take imagination step (use first agent's action)
            action_type = int(actions_np[0])
            z_next, reward, done, step_info = self.imagination_env.step(
                action_type=action_type
            )

            z_sequence.append(z_next)
            rewards_sequence.append(reward)
            uncertainties_sequence.append(step_info.get("uncertainty", 0.0))

            if done:
                break

            z_0 = z_next
            info = step_info

        if not z_sequence:
            return {}

        # Convert to tensors
        z_sequence_t = torch.stack(z_sequence, dim=0).squeeze(1).to(self.device)
        rewards_t = torch.tensor(rewards_sequence, dtype=torch.float32).to(self.device)
        uncertainties_t = torch.tensor(uncertainties_sequence, dtype=torch.float32).to(
            self.device
        )

        return {
            "z_sequence": z_sequence_t,
            "imagined_rewards": rewards_t,
            "uncertainties": uncertainties_t,
        }

    def _track_coordination(
        self,
        env: "SGSEnv",
        agent_idx: int,
        action_type: int,
        info: Dict,
    ):
        """Track coordination events for metrics"""
        if not self.team_reward_allocator:
            return

        from ai.mappo.team_rewards import ActionContext, Team

        if agent_idx >= len(env.players):
            return
        player = env.players[agent_idx]
        identity = player.identity if hasattr(player, "identity") else "unknown"
        team = self.team_reward_allocator.get_team(agent_idx)

        action_type_str = self._get_action_type_str(action_type)
        target_idx = info.get("target_idx")
        target_identity = None
        if target_idx is not None and target_idx < len(env.players):
            target_identity = env.players[target_idx].identity

        action_context = ActionContext(
            agent_idx=agent_idx,
            agent_identity=identity,
            action_type=action_type_str,
            target_idx=target_idx,
            target_identity=target_identity,
            damage_value=info.get("damage", 0.0),
            card_type=info.get("card_type"),
        )

        self.team_reward_allocator.record_action(action_context)
        team_bonus = self.team_reward_allocator.compute_team_bonus(action_context)

        stats = self.team_reward_allocator.get_coordination_stats()

        if action_type_str == "attack" and target_idx is not None:
            target_team = self.team_reward_allocator.get_team(target_idx)

            if team == Team.LORD and target_team in [Team.REBEL, Team.SPY]:
                self.coordination_tracker.record_coordination_event(
                    "lord_loyalist_coordination", "lord", agent_idx
                )

            if team == Team.REBEL and target_team == Team.LORD:
                self.coordination_tracker.record_coordination_event(
                    "focus_fire", "rebel", agent_idx, target_idx=target_idx
                )

        info["team_bonus"] = team_bonus

    def _get_action_type_str(self, action_type: int) -> str:
        """Convert action type enum to string"""
        from ai.action_encoder import ActionType

        action_map = {
            ActionType.USE_CARD: "attack",
            ActionType.END_TURN: "pass",
            ActionType.USE_SKILL: "skill",
            ActionType.RESPOND_SHAN: "defend",
            ActionType.RESPOND_SHA: "attack",
            ActionType.RESPOND_TAO: "heal",
            ActionType.RESPOND_WUXIE: "skill",
            ActionType.DISCARD: "discard",
            ActionType.JUDGE_MODIFY: "skill",
            ActionType.PASS: "pass",
        }
        return action_map.get(action_type, "unknown")

    def _run_evaluation_episode(self, ep_idx: int) -> Tuple[float, int, Optional[str]]:
        """Run single evaluation episode"""
        eval_env = self._create_env()

        obs, info = eval_env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        # Initialize team reward allocator
        if self.team_reward_allocator:
            self.team_reward_allocator.initialize_teams(eval_env.players)

        while not done and episode_steps < 500:
            game_state = eval_env._get_game_state_dict()

            local_obs = np.zeros(
                (self.config.num_agents, self.config.local_state_dim),
                dtype=np.float32,
            )
            masks = np.zeros(
                (self.config.num_agents, self.config.action_dim), dtype=np.float32
            )

            for i in range(self.config.num_agents):
                if i < len(eval_env.players) and eval_env.players[i].is_alive:
                    local_obs[i] = self.state_encoder.encode(game_state, i)
                    type_mask, card_mask, target_mask = (
                        eval_env.action_mask_generator.generate_masks(
                            game_state,
                            eval_env.players[i],
                            eval_env.engine,
                            0,
                            None,
                        )
                    )
                    combined = np.concatenate([type_mask, card_mask, target_mask])
                    if len(combined) < self.config.action_dim:
                        combined = np.pad(
                            combined, (0, self.config.action_dim - len(combined))
                        )
                    masks[i] = combined[: self.config.action_dim]

            local_obs_t = (
                torch.from_numpy(local_obs).unsqueeze(0).float().to(self.device)
            )
            masks_t = torch.from_numpy(masks).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                joint_action_t, _, _, _ = self.agent.get_actions(
                    local_obs_t, masks_t, deterministic=True
                )

            actions_np = joint_action_t.squeeze(0).cpu().numpy()

            current_player_idx = eval_env.current_player_idx
            if (
                current_player_idx < len(eval_env.players)
                and eval_env.players[current_player_idx].is_alive
            ):
                action = int(actions_np[current_player_idx])
                try:
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                except Exception:
                    obs, info = eval_env.reset()
                    done = True
                    episode_reward = 0.0

            episode_steps += 1

        eval_env.close()

        winner = info.get("winner") if isinstance(info, dict) else None
        return episode_reward, episode_steps, winner

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation episodes"""
        results = []

        with ThreadPoolExecutor(
            max_workers=min(self.config.num_eval_episodes, 8)
        ) as executor:
            futures = [
                executor.submit(self._run_evaluation_episode, ep)
                for ep in range(self.config.num_eval_episodes)
            ]

            for future in as_completed(futures):
                try:
                    reward, steps, winner = future.result()
                    results.append((reward, steps, winner))
                except Exception as e:
                    logger.warning(f"Eval episode failed: {e}")
                    results.append((0.0, 0, None))

        total_reward = sum(r[0] for r in results)
        episode_lengths = [r[1] for r in results]

        # Count wins by team
        lord_wins = 0
        rebel_wins = 0
        spy_wins = 0

        for r in results:
            winner = r[2]
            if winner in ["lord", "lord_team", "主公"]:
                lord_wins += 1
            elif winner in ["rebel", "rebel_team", "反贼"]:
                rebel_wins += 1
            elif winner in ["spy", "内奸"]:
                spy_wins += 1

        return {
            "win_rate": (lord_wins + rebel_wins + spy_wins) / len(results)
            if results
            else 0.0,
            "lord_win_rate": lord_wins / len(results) if results else 0.0,
            "rebel_win_rate": rebel_wins / len(results) if results else 0.0,
            "spy_win_rate": spy_wins / len(results) if results else 0.0,
            "avg_reward": total_reward / len(results) if results else 0.0,
            "avg_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        }

    def train(self, log_dir: Optional[str] = None) -> Dict[str, float]:
        """Main training loop"""
        env = self._create_env()

        # Suppress verbose engine logs
        logging.getLogger("engine").setLevel(logging.ERROR)
        logging.getLogger("ai.gym_wrapper").setLevel(logging.ERROR)

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "mappo" if self.config.use_mappo else "ippo"
        wm_str = "_wm" if self.config.use_world_model else ""
        tr_str = "_team" if self.config.use_team_rewards else ""

        output_dir = (
            Path(log_dir)
            if log_dir
            else Path(self.config.output_dir)
            / f"{mode_str}{wm_str}{tr_str}_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(output_dir))

        logger.info(f"Starting training for {self.config.steps_total} steps...")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"TensorBoard logs: {output_dir}")

        # Training parameters
        steps_per_rollout = 64
        start_time = time.time()
        pbar = tqdm(total=self.config.steps_total, desc="Training", unit="steps")

        while self._step_count < self.config.steps_total:
            # Real environment rollout
            real_rollout = self._collect_real_rollout(env, steps_per_rollout)

            # Imagination rollout if enabled
            imagination_rollout = {}
            if self.config.use_world_model and self.imagination_env:
                # Get current state for imagination
                game_state = env._get_game_state_dict()
                current_state = self.state_encoder.encode(game_state, 0)

                imagination_rollout = self._collect_imagination_rollout(
                    current_state,
                    int(steps_per_rollout * self.config.imagination_ratio),
                )

            # Combine rollouts for update
            combined_rollout = real_rollout

            # Add imagination rewards if available (simplified integration)
            if imagination_rollout and "imagined_rewards" in imagination_rollout:
                # Placeholder: Blend imagined rewards with real rewards
                # Full integration would involve latent-to-state mapping
                imagined_rewards = imagination_rollout["imagined_rewards"]
                logger.debug(
                    f"Imagination rewards: {imagined_rewards.mean().item():.2f}"
                )

            # Update agent
            metrics = self.agent.update(
                combined_rollout,
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
            )

            self._loss_history.append(metrics["policy_loss"])
            self._step_count += steps_per_rollout

            # Health check
            health = self.agent.check_training_health()

            # Progress update
            pbar.update(steps_per_rollout)
            pbar.set_postfix(
                loss=f"{metrics['policy_loss']:.4f}",
                entropy=f"{health.get('recent_entropy', 0):.4f}",
                eps=self._episode_count,
            )

            # TensorBoard logging
            if self._writer:
                self._writer.add_scalar(
                    "train/policy_loss", metrics["policy_loss"], self._step_count
                )
                self._writer.add_scalar(
                    "train/value_loss", metrics.get("value_loss", 0), self._step_count
                )
                self._writer.add_scalar(
                    "train/entropy", health.get("recent_entropy", 0), self._step_count
                )
                self._writer.add_scalar(
                    "train/value_std",
                    health.get("recent_value_std", 0),
                    self._step_count,
                )
                self._writer.add_scalar(
                    "train/episodes", self._episode_count, self._step_count
                )

                # Coordination metrics
                coordination_metrics = (
                    self.coordination_tracker.get_metrics().get_summary()
                )
                for key, value in coordination_metrics.items():
                    self._writer.add_scalar(key, value, self._step_count)

                # World model metrics if enabled
                if self.config.use_world_model and imagination_rollout:
                    if "uncertainties" in imagination_rollout:
                        avg_uncertainty = (
                            imagination_rollout["uncertainties"].mean().item()
                        )
                        self._writer.add_scalar(
                            "world_model/avg_uncertainty",
                            avg_uncertainty,
                            self._step_count,
                        )

            # Evaluation
            if self._step_count % self.config.eval_interval < steps_per_rollout:
                pbar.write(f"[Step {self._step_count}] Running evaluation...")
                eval_metrics = self._evaluate()
                pbar.write(
                    f"[Eval] win_rate={eval_metrics['win_rate']:.2%} "
                    f"lord={eval_metrics['lord_win_rate']:.2%} "
                    f"rebel={eval_metrics['rebel_win_rate']:.2%} "
                    f"avg_reward={eval_metrics['avg_reward']:.2f}"
                )

                if self._writer:
                    self._writer.add_scalar(
                        "eval/win_rate", eval_metrics["win_rate"], self._step_count
                    )
                    self._writer.add_scalar(
                        "eval/lord_win_rate",
                        eval_metrics["lord_win_rate"],
                        self._step_count,
                    )
                    self._writer.add_scalar(
                        "eval/rebel_win_rate",
                        eval_metrics["rebel_win_rate"],
                        self._step_count,
                    )
                    self._writer.add_scalar(
                        "eval/spy_win_rate",
                        eval_metrics["spy_win_rate"],
                        self._step_count,
                    )
                    self._writer.add_scalar(
                        "eval/avg_reward", eval_metrics["avg_reward"], self._step_count
                    )
                    self._writer.add_scalar(
                        "eval/avg_length", eval_metrics["avg_length"], self._step_count
                    )

            # Health warnings
            if not health["entropy_ok"]:
                pbar.write(f"[WARN] Entropy low: {health['recent_entropy']:.4f}")
            if not health["value_std_ok"]:
                pbar.write(f"[WARN] Value_std low: {health['recent_value_std']:.4f}")

            # Checkpoint
            if self._step_count % self.config.checkpoint_interval < steps_per_rollout:
                checkpoint_path = (
                    output_dir / f"mappo_world_model_checkpoint_{self._step_count}.pt"
                )
                self.save_checkpoint(str(checkpoint_path))

        pbar.close()
        training_time = time.time() - start_time

        # Final health check
        final_health = self.agent.check_training_health()

        # Final evaluation
        logger.info("Running final evaluation...")
        final_eval = self._evaluate()

        final_metrics = {
            "total_steps": self._step_count,
            "total_episodes": self._episode_count,
            "training_time": training_time,
            "final_entropy": final_health.get("recent_entropy", 0.0),
            "final_value_std": final_health.get("recent_value_std", 0.0),
            "entropy_ok": final_health["entropy_ok"],
            "value_std_ok": final_health["value_std_ok"],
            "overall_ok": final_health["overall_ok"],
            "final_win_rate": final_eval["win_rate"],
            "lord_win_rate": final_eval["lord_win_rate"],
            "rebel_win_rate": final_eval["rebel_win_rate"],
            "spy_win_rate": final_eval["spy_win_rate"],
        }

        logger.info(f"Training completed in {training_time:.1f}s!")
        logger.info(f"Final win rate: {final_eval['win_rate']:.2%}")
        logger.info(f"Lord win rate: {final_eval['lord_win_rate']:.2%}")
        logger.info(f"Rebel win rate: {final_eval['rebel_win_rate']:.2%}")

        # Save final checkpoint
        final_checkpoint_path = output_dir / "mappo_world_model_final.pt"
        self.save_checkpoint(str(final_checkpoint_path))

        # Save evidence
        self.save_evidence(output_dir, final_metrics)

        env.close()
        if self._writer:
            self._writer.close()

        return final_metrics

    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            "step": self._step_count,
            "episode": self._episode_count,
            "config": self.config,
            "agent_state": {
                "actors_state_dict": [
                    actor.state_dict() for actor in self.agent.actors
                ],
                "critic_state_dict": self.agent.shared_critic.state_dict(),
            },
            "agent_config": self.agent_config,
            "training_metrics": {
                "entropy_history": self._entropy_history[-100:],
                "value_std_history": self._value_std_history[-100:],
                "loss_history": self._loss_history[-100:],
            },
            "coordination_metrics": self.coordination_tracker.get_metrics().get_summary(),
            "feature_flags": {
                "use_world_model": self.config.use_world_model,
                "use_mappo": self.config.use_mappo,
                "use_team_rewards": self.config.use_team_rewards,
            },
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def save_evidence(self, output_dir: Path, final_metrics: Dict[str, float]):
        """Save training evidence log"""
        evidence_path = (
            Path(self.config.evidence_dir) / "mappo_world_model_training.log"
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)

        evidence = []
        evidence.append("MAPPO + World Model Integration Training Evidence Log")
        evidence.append(f"Generated: {datetime.now().isoformat()}")
        evidence.append("=" * 60)
        evidence.append("Configuration:")
        evidence.append(
            f"  - Training mode: {TrainingMode.MAPPO if self.config.use_mappo else TrainingMode.IPPO}"
        )
        evidence.append(f"  - World Model enabled: {self.config.use_world_model}")
        evidence.append(f"  - Team Rewards enabled: {self.config.use_team_rewards}")
        evidence.append(f"  - Total steps: {self.config.steps_total}")
        evidence.append(f"  - Batch size: {self.config.batch_size}")
        evidence.append(f"  - Learning rate: {self.config.learning_rate}")
        evidence.append("=" * 60)
        evidence.append("Training Metrics:")
        evidence.append(f"  - Total episodes: {self._episode_count}")
        evidence.append(
            f"  - Training time: {final_metrics.get('training_time', 0):.1f}s"
        )
        evidence.append(
            f"  - Final entropy: {final_metrics.get('final_entropy', 0):.4f}"
        )
        evidence.append(
            f"  - Final value_std: {final_metrics.get('final_value_std', 0):.4f}"
        )
        evidence.append("=" * 60)
        evidence.append("Evaluation Results:")
        evidence.append(
            f"  - Final win rate: {final_metrics.get('final_win_rate', 0):.2%}"
        )
        evidence.append(
            f"  - Lord win rate: {final_metrics.get('lord_win_rate', 0):.2%}"
        )
        evidence.append(
            f"  - Rebel win rate: {final_metrics.get('rebel_win_rate', 0):.2%}"
        )
        evidence.append(f"  - Spy win rate: {final_metrics.get('spy_win_rate', 0):.2%}")
        evidence.append("=" * 60)
        evidence.append("Coordination Metrics:")
        coord_metrics = self.coordination_tracker.get_metrics().get_summary()
        for key, value in coord_metrics.items():
            evidence.append(f"  - {key}: {value:.4f}")
        evidence.append("=" * 60)
        evidence.append("Health Check:")
        evidence.append(f"  - Entropy OK: {final_metrics.get('entropy_ok', False)}")
        evidence.append(f"  - Value_std OK: {final_metrics.get('value_std_ok', False)}")
        evidence.append(f"  - Overall OK: {final_metrics.get('overall_ok', False)}")
        evidence.append("=" * 60)
        evidence.append(f"Output directory: {output_dir}")

        with open(evidence_path, "w") as f:
            f.write("\n".join(evidence))

        # Also save to output_dir
        with open(output_dir / "training_evidence.log", "w") as f:
            f.write("\n".join(evidence))

        logger.info(f"Saved evidence to {evidence_path}")


# === CLI Interface ===


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MAPPO + World Model Integration Training"
    )

    # Training parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total training steps (default: 100000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)"
    )

    # Feature flags
    parser.add_argument(
        "--use-world-model",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable World Model imagination training (default: false)",
    )
    parser.add_argument(
        "--use-mappo",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use MAPPO (centralized) vs IPPO (decentralized) (default: true)",
    )
    parser.add_argument(
        "--use-team-rewards",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable team reward allocation (default: false)",
    )

    # Checkpoint paths
    parser.add_argument(
        "--mappo-checkpoint",
        type=str,
        default="train/logs/mappo_50k_fixed_v2/mappo_checkpoint.pt",
        help="Path to MAPPO checkpoint",
    )
    parser.add_argument(
        "--dynamics-checkpoint",
        type=str,
        default="train/logs/dynamics_training/dynamics_final.pt",
        help="Path to Dynamics model checkpoint",
    )

    # Logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train/logs/mappo_world_model",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Checkpoint save interval (default: 10000)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Evaluation interval (default: 5000)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda, cpu, or None for auto)"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level (0=warning, 1=info)"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Set verbosity
    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)

    # Parse feature flags
    use_world_model = args.use_world_model.lower() == "true"
    use_mappo = args.use_mappo.lower() == "true"
    use_team_rewards = args.use_team_rewards.lower() == "true"

    # Create configuration
    config = MAPPOWorldModelConfig(
        steps_total=args.steps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_world_model=use_world_model,
        use_mappo=use_mappo,
        use_team_rewards=use_team_rewards,
        mappo_checkpoint_path=args.mappo_checkpoint,
        dynamics_checkpoint_path=args.dynamics_checkpoint,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        device=args.device,
    )

    # Create trainer
    trainer = MAPPOWorldModelTrainer(config)

    try:
        metrics = trainer.train()

        # Report results
        if metrics["overall_ok"]:
            logger.info("Training completed successfully - health checks passed")
        else:
            logger.warning("Training completed but health checks failed")

        logger.info(f"Final win rate: {metrics['final_win_rate']:.2%}")
        logger.info(f"Lord win rate: {metrics['lord_win_rate']:.2%}")
        logger.info(f"Rebel win rate: {metrics['rebel_win_rate']:.2%}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()

        # Save failure evidence
        evidence_path = Path(config.evidence_dir) / "mappo_world_model_failed.log"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(evidence_path, "w") as f:
            f.write(f"Training failed: {e}\n")
            f.write(traceback.format_exc())

        return 1


if __name__ == "__main__":
    sys.exit(main())
