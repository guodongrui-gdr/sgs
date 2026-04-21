"""
Unified SGS RL Training Script

Supports multiple training modes:
- Independent: Multi-agent training with independent learning (default)
- IPPO: Independent PPO (decentralized)
- World Model Train: Train World Model dynamics/reward models (Phase 1)
- IPPO + World Model: IPPO with imagination-based training (Phase 2)
- Self-Play: Self-play training with agent pool

Usage:
    # Independent training (default)
    python train/train.py --mode independent --steps 100000

    # IPPO training
    python train/train.py --mode ippo --steps 100000 --n-envs 8

    # World Model training (Phase 1)
    python train/train.py --mode world_model_train --steps 50000

    # Quick test
    python train/train.py --mode independent --steps 1000 --n-envs 1
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose environment and engine logs during training
logging.getLogger("ai.gym_wrapper").setLevel(logging.ERROR)
logging.getLogger("engine.game_engine").setLevel(logging.ERROR)

from ai.gym_wrapper import SGSConfig, SGSEnv
from ai.state_encoder import StateEncoder
from ai.independent_trainer import IndependentAgentTrainer, IndependentAgentConfig
from train.config import TrainingConfig, TrainingMode


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

    def __init__(self, config: TrainingConfig):
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


class Trainer:
    """Main trainer for independent multi-agent learning."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Device setup
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        logger.info(f"Using device: {self.device}")

        # Create environment
        env_config = SGSConfig(
            player_num=config.num_agents,
            max_rounds=config.max_rounds,
            use_action_mask=config.use_action_mask,
            use_shaping=config.use_shaping,
        )
        self.env = SGSEnv(env_config)

        # State encoder
        self.state_encoder = StateEncoder(env_config.state_config)

        # Agent trainer
        agent_config = IndependentAgentConfig(
            num_agents=config.num_agents,
            local_state_dim=config.local_state_dim,
            action_dim=config.action_dim,
            global_state_dim=config.global_state_dim,
            device=self.device,
        )
        self.trainer = IndependentAgentTrainer(agent_config).to(self.device)

        self._episode_count = 0
        self._step_count = 0

        log_dir = Path(config.output_dir) / "independent"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Team reward allocator (optional)
        self.team_reward_allocator = None
        if config.use_team_rewards:
            from ai.mappo.team_rewards import TeamRewardAllocator, TeamRewardConfig

            reward_config = TeamRewardConfig(
                lord_loyalist_coordination_bonus=config.lord_loyalist_coordination_bonus,
                rebel_focus_fire_bonus=config.rebel_focus_fire_bonus,
                protect_lord_bonus=config.protect_lord_bonus,
            )
            self.team_reward_allocator = TeamRewardAllocator(
                config=reward_config, num_agents=config.num_agents
            )
            logger.info("TeamRewardAllocator initialized")

        # Coordination tracking
        self.coordination_tracker = CoordinationTracker(config)

    def _get_global_state(self, env: SGSEnv) -> np.ndarray:
        """Get global state by encoding all agents."""
        game_state = env._get_game_state_dict()
        local_obs = np.zeros((5, 2670), dtype=np.float32)
        for i in range(5):
            local_obs[i] = self.state_encoder.encode(game_state, i)
        return local_obs.flatten()

    def _get_action_mask(self, env: SGSEnv, player_idx: int) -> np.ndarray:
        """Get action mask for a specific player."""
        game_state = env._get_game_state_dict()
        player = env.players[player_idx]
        type_mask, card_mask, target_mask = env.action_mask_generator.generate_masks(
            game_state, player, env.engine, 0, None
        )
        combined = np.concatenate([type_mask, card_mask, target_mask])
        if len(combined) < 45:
            combined = np.pad(combined, (0, 45 - len(combined)))
        return combined[:45]

    def _get_action_type_str(self, action_type: int) -> str:
        """Convert action type enum to string."""
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

    def _track_coordination(
        self,
        env: SGSEnv,
        agent_idx: int,
        action_type: int,
        info: Dict,
    ):
        """Track coordination events for metrics."""
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

    def collect_rollout(self, n_steps: int) -> int:
        """
        Collect rollout data for n_steps.
        Only stores data for the current active agent each step.

        Returns:
            Number of steps actually collected
        """
        obs, info = self.env.reset()
        if self.team_reward_allocator:
            self.team_reward_allocator.initialize_teams(self.env.players)
        steps_collected = 0

        for step in range(n_steps):
            current_player_idx = self.env.current_player_idx
            players = self.env.players

            if (
                current_player_idx < len(players)
                and players[current_player_idx].is_alive
            ):
                # Get global state
                global_state = self._get_global_state(self.env)

                # Get local observation for current player
                game_state = self.env._get_game_state_dict()
                local_obs = self.state_encoder.encode(game_state, current_player_idx)

                # Get action mask
                mask = self._get_action_mask(self.env, current_player_idx)

                # Get action and log_prob
                action, log_prob = self.trainer.get_action(
                    current_player_idx, local_obs, mask, deterministic=False
                )

                # Get value
                value = self.trainer.get_value(global_state)

                # Step environment
                try:
                    obs, reward, done, truncated, info = self.env.step(action)
                    step_done = done or truncated
                except Exception as e:
                    logger.warning(f"Action failed: {e}")
                    obs, info = self.env.reset()
                    step_done = True
                    reward = 0.0

                # Store transition
                self.trainer.store_transition(
                    agent_idx=current_player_idx,
                    global_state=global_state,
                    local_obs=local_obs,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    done=step_done,
                    mask=mask,
                    value=value,
                )

                steps_collected += 1

                if self.team_reward_allocator and self.config.use_team_rewards:
                    self._track_coordination(self.env, current_player_idx, action, info)

                if step_done:
                    team_rewards = {"lord_team": 0.0, "rebel_team": 0.0}
                    if self.team_reward_allocator and self.config.use_team_rewards:
                        team_rewards["lord_team"] = info.get("lord_team_reward", 0.0)
                        team_rewards["rebel_team"] = info.get("rebel_team_reward", 0.0)
                    winner = info.get("winner") if isinstance(info, dict) else None
                    self.coordination_tracker.record_episode_result(
                        winner, steps_collected, team_rewards
                    )
                    self.coordination_tracker.metrics.reset_episode()
                    self._episode_count += 1
                    obs, info = self.env.reset()
                    if self.team_reward_allocator:
                        self.team_reward_allocator.initialize_teams(self.env.players)
            else:
                # Dead agent, skip
                try:
                    obs, reward, done, truncated, info = self.env.step(0)
                    if done or truncated:
                        obs, info = self.env.reset()
                except:
                    obs, info = self.env.reset()

        return steps_collected

    def _run_evaluation_episode(self, ep_idx: int) -> tuple:
        """Run single evaluation episode (deterministic)."""
        eval_env = SGSEnv(SGSConfig())

        obs, info = eval_env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        if self.team_reward_allocator:
            self.team_reward_allocator.initialize_teams(eval_env.players)

        while not done and episode_steps < 5000:
            current_player_idx = eval_env.current_player_idx
            players = eval_env.players

            if (
                current_player_idx < len(players)
                and players[current_player_idx].is_alive
            ):
                game_state = eval_env._get_game_state_dict()
                local_obs = self.state_encoder.encode(game_state, current_player_idx)
                mask = self._get_action_mask(eval_env, current_player_idx)

                action, _ = self.trainer.get_action(
                    current_player_idx, local_obs, mask, deterministic=True
                )

                try:
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated

                    if self.team_reward_allocator and self.config.use_team_rewards:
                        self._track_coordination(
                            eval_env, current_player_idx, action, info
                        )
                except Exception:
                    obs, info = eval_env.reset()
                    done = True
                    episode_reward = 0.0

            episode_steps += 1

        eval_env.close()

        winner = info.get("winner") if isinstance(info, dict) else None
        return episode_reward, episode_steps, winner

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation episodes with ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def train(self) -> Dict[str, float]:
        """Main training loop."""
        logger.info("Starting independent multi-agent training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total steps: {self.config.steps_total}")
        logger.info(f"Steps per rollout: {self.config.steps_per_rollout}")

        start_time = time.time()
        total_steps = 0

        pbar = tqdm(total=self.config.steps_total, desc="Training", unit="steps")

        while total_steps < self.config.steps_total:
            rollout_start = time.time()
            steps = self.collect_rollout(self.config.steps_per_rollout)
            rollout_time = time.time() - rollout_start
            total_steps += steps

            update_start = time.time()
            metrics = self.trainer.update_all_agents(
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
            )
            update_time = time.time() - update_start

            steps_per_s = steps / rollout_time if rollout_time > 0 else 0

            pbar.update(steps)
            pbar.set_postfix(eps=self._episode_count, steps_s=f"{steps_per_s:.1f}")

            if metrics:
                policy_losses = []
                value_losses = []
                entropies = []
                for key, value in metrics.items():
                    if "policy_loss" in key:
                        policy_losses.append(value)
                    elif "value_loss" in key:
                        value_losses.append(value)
                    elif "entropy" in key:
                        entropies.append(value)

                if policy_losses:
                    self.writer.add_scalar(
                        "train/policy_loss", np.mean(policy_losses), total_steps
                    )
                if value_losses:
                    self.writer.add_scalar(
                        "train/value_loss", np.mean(value_losses), total_steps
                    )
                if entropies:
                    self.writer.add_scalar(
                        "train/entropy", np.mean(entropies), total_steps
                    )

                self.writer.add_scalar(
                    "training/rollout_time_s", rollout_time, total_steps
                )
                self.writer.add_scalar(
                    "training/update_time_s", update_time, total_steps
                )
                self.writer.add_scalar(
                    "training/steps_per_second", steps_per_s, total_steps
                )

            coordination_metrics = self.coordination_tracker.get_metrics().get_summary()
            for key, value in coordination_metrics.items():
                self.writer.add_scalar(key, value, total_steps)

            if total_steps % self.config.eval_interval < self.config.steps_per_rollout:
                pbar.write(f"[Step {total_steps}] Running evaluation...")
                eval_metrics = self._evaluate()
                pbar.write(
                    f"[Eval] win_rate={eval_metrics['win_rate']:.2%} "
                    f"lord={eval_metrics['lord_win_rate']:.2%} "
                    f"rebel={eval_metrics['rebel_win_rate']:.2%} "
                    f"avg_reward={eval_metrics['avg_reward']:.2f}"
                )
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f"eval/{key}", value, total_steps)

            if (
                total_steps % self.config.checkpoint_interval
                < self.config.steps_per_rollout
            ):
                checkpoint_path = f"checkpoints/independent_step_{total_steps}.pt"
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                self.trainer.save(checkpoint_path)
                pbar.write(f"Checkpoint saved: {checkpoint_path}")

        pbar.close()

        total_time = time.time() - start_time
        logger.info(f"Training complete: {total_steps} steps in {total_time:.1f}s")
        logger.info(f"Average: {total_steps / total_time:.1f} steps/s")

        self.writer.close()

        return {"total_steps": total_steps, "total_time": total_time}


def main():
    """Main entry point for unified SGS RL training.

    Supports multiple training modes:
    - independent: Independent multi-agent training (default, recommended for beginners)
    - ippo: Independent PPO with decentralized training
    - world_model_train: Train World Model dynamics/reward models (Phase 1)
    - ippo_world_model: IPPO with World Model imagination (Phase 2, needs Phase 1)
    - self_play: Self-play training with agent pool

    Note: MAPPO is excluded because SanGuoSha is turn-based,
          making simultaneous multi-agent steps inappropriate.
    """
    parser = argparse.ArgumentParser(description="SGS RL Unified Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="independent",
        choices=[
            "independent",
            "ippo",
            "world_model_train",
            "ippo_world_model",
            "self_play",
        ],
        help="Training mode (default: independent)",
    )
    parser.add_argument(
        "--steps", type=int, default=100000, help="Total training steps"
    )
    parser.add_argument("--steps-per-rollout", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu, auto if None)"
    )
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument(
        "--use-team-rewards", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument("--lord-loyalist-coordination-bonus", type=float, default=2.0)
    parser.add_argument("--rebel-focus-fire-bonus", type=float, default=1.5)
    parser.add_argument("--protect-lord-bonus", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # World Model flags
    parser.add_argument(
        "--use-world-model", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument("--dynamics-checkpoint", type=str, default=None)
    parser.add_argument("--policy-checkpoint", type=str, default=None)

    # Self-play flags
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Self-play total timesteps (overrides --steps)",
    )

    args = parser.parse_args()

    # Parse flags
    use_team_rewards = args.use_team_rewards.lower() == "true"
    use_world_model = args.use_world_model.lower() == "true"

    # Map mode string to enum
    mode_map = {
        "independent": TrainingMode.INDEPENDENT,
        "ippo": TrainingMode.IPPO,
        "world_model_train": TrainingMode.WORLD_MODEL_TRAIN,
        "ippo_world_model": TrainingMode.IPPO_WORLD_MODEL,
        "self_play": TrainingMode.SELF_PLAY,
    }
    mode = mode_map.get(args.mode, TrainingMode.INDEPENDENT)

    config = TrainingConfig(
        mode=mode,
        steps_total=args.steps,
        steps_per_rollout=args.steps_per_rollout,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_epochs=args.n_epochs,
        device=args.device,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        num_eval_episodes=args.num_eval_episodes,
        use_team_rewards=use_team_rewards,
        lord_loyalist_coordination_bonus=args.lord_loyalist_coordination_bonus,
        rebel_focus_fire_bonus=args.rebel_focus_fire_bonus,
        protect_lord_bonus=args.protect_lord_bonus,
        use_world_model=use_world_model,
        seed=args.seed,
    )

    if args.dynamics_checkpoint:
        config.dynamics_checkpoint_path = args.dynamics_checkpoint
    if args.policy_checkpoint:
        config.policy_checkpoint_path = args.policy_checkpoint
    if args.pool_size:
        config.pool_size = args.pool_size

    logger.info("=" * 60)
    logger.info(f"SGS RL Training - Mode: {mode.value.upper()}")
    logger.info("=" * 60)
    logger.info(
        f"Steps: {config.steps_total}, Envs: {config.n_envs}, LR: {config.learning_rate}"
    )
    logger.info(f"Device: {config.device or 'auto'}")
    if use_world_model:
        logger.info(f"World Model enabled, dynamics: {config.dynamics_checkpoint_path}")
    if use_team_rewards:
        logger.info("Team Rewards enabled")
    logger.info("=" * 60)

    # Route to appropriate trainer
    if mode == TrainingMode.INDEPENDENT:
        trainer = Trainer(config)
        trainer.train()
    elif mode == TrainingMode.IPPO:
        logger.info("IPPO mode: Using decentralized training")
        from train.train_mappo_world_model import (
            MAPPOWorldModelTrainer,
            MAPPOWorldModelConfig,
        )

        ippo_config = MAPPOWorldModelConfig(
            steps_total=config.steps_total,
            n_envs=config.n_envs,
            use_mappo=False,  # Force IPPO (decentralized)
            use_world_model=False,
            use_team_rewards=config.use_team_rewards,
            device=config.device,
        )
        trainer = MAPPOWorldModelTrainer(ippo_config)
        trainer.train()
    elif mode == TrainingMode.WORLD_MODEL_TRAIN:
        logger.info("World Model Training (Phase 1)")
        from train.train_dynamics import DynamicsTrainer, DynamicsTrainingConfig

        wm_config = DynamicsTrainingConfig(
            total_steps=config.steps_total,
            n_envs=config.n_envs,
            device=config.device,
        )
        trainer = DynamicsTrainer(wm_config)
        trainer.train()
    elif mode == TrainingMode.IPPO_WORLD_MODEL:
        logger.info("IPPO + World Model (Phase 2)")
        from train.train_mappo_world_model import (
            MAPPOWorldModelTrainer,
            MAPPOWorldModelConfig,
        )

        ippo_wm_config = MAPPOWorldModelConfig(
            steps_total=config.steps_total,
            n_envs=config.n_envs,
            use_mappo=False,  # Force IPPO (decentralized)
            use_world_model=True,
            dynamics_checkpoint_path=config.dynamics_checkpoint_path,
            use_team_rewards=config.use_team_rewards,
            device=config.device,
        )
        trainer = MAPPOWorldModelTrainer(ippo_wm_config)
        trainer.train()
    elif mode == TrainingMode.SELF_PLAY:
        logger.info("Self-Play mode")
        timesteps = args.timesteps or config.steps_total
        from train.train_self_play import train

        train(timesteps=timesteps, n_envs=config.n_envs)
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    logger.info(f"Training completed: {mode.value}")


if __name__ == "__main__":
    main()
