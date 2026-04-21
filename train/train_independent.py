"""
Independent Multi-Agent Training Script

Trains each agent independently with shared global value function.
Each step only collects data for the current active agent.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
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

# Suppress verbose environment and engine logs during training
logging.getLogger("ai.gym_wrapper").setLevel(logging.ERROR)
logging.getLogger("engine.game_engine").setLevel(logging.ERROR)

from ai.gym_wrapper import SGSConfig, SGSEnv
from ai.state_encoder import StateEncoder
from ai.independent_trainer import IndependentAgentTrainer, IndependentAgentConfig


class IndependentTrainingConfig:
    """Training configuration."""

    def __init__(
        self,
        steps_total: int = 100000,
        steps_per_rollout: int = 256,
        n_envs: int = 1,
        batch_size: int = 64,
        n_epochs: int = 10,
        eval_interval: int = 10000,
        checkpoint_interval: int = 10000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.steps_total = steps_total
        self.steps_per_rollout = steps_per_rollout
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.device = device


class IndependentTrainer:
    """Main trainer for independent multi-agent learning."""

    def __init__(self, config: Optional[IndependentTrainingConfig] = None):
        self.config = config or IndependentTrainingConfig()
        self.device = self.config.device

        # Create environment
        env_config = SGSConfig()
        self.env = SGSEnv(env_config)

        # State encoder
        self.state_encoder = StateEncoder(env_config.state_config)

        # Agent trainer
        agent_config = IndependentAgentConfig(
            num_agents=5,
            local_state_dim=2670,
            action_dim=45,
            global_state_dim=2670 * 5,
            device=self.device,
        )
        self.trainer = IndependentAgentTrainer(agent_config).to(self.device)

        self._episode_count = 0
        self._step_count = 0

        log_dir = Path("logs/independent_agent")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

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

    def collect_rollout(self, n_steps: int) -> int:
        """
        Collect rollout data for n_steps.
        Only stores data for the current active agent each step.

        Returns:
            Number of steps actually collected
        """
        obs, info = self.env.reset()
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

                if step_done:
                    self._episode_count += 1
                    obs, info = self.env.reset()
            else:
                # Dead agent, skip
                try:
                    obs, reward, done, truncated, info = self.env.step(0)
                    if done or truncated:
                        obs, info = self.env.reset()
                except:
                    obs, info = self.env.reset()

        return steps_collected

    def train(self) -> Dict[str, float]:
        """Main training loop."""
        logger.info("Starting independent multi-agent training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total steps: {self.config.steps_total}")
        logger.info(f"Steps per rollout: {self.config.steps_per_rollout}")

        start_time = time.time()
        total_steps = 0

        while total_steps < self.config.steps_total:
            # Collect rollout
            rollout_start = time.time()
            steps = self.collect_rollout(self.config.steps_per_rollout)
            rollout_time = time.time() - rollout_start
            total_steps += steps

            # Update all agents
            update_start = time.time()
            metrics = self.trainer.update_all_agents(
                n_epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
            )
            update_time = time.time() - update_start

            # Log
            if metrics:
                steps_per_s = steps / rollout_time if rollout_time > 0 else 0
                logger.info(
                    f"Step {total_steps}/{self.config.steps_total} | "
                    f"Rollout: {rollout_time:.2f}s | "
                    f"Update: {update_time:.2f}s | "
                    f"Steps/s: {steps_per_s:.1f}"
                )

                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, total_steps)

                self.writer.add_scalar(
                    "training/rollout_time_s", rollout_time, total_steps
                )
                self.writer.add_scalar(
                    "training/update_time_s", update_time, total_steps
                )
                self.writer.add_scalar(
                    "training/steps_per_second", steps_per_s, total_steps
                )

            # Checkpoint
            if (
                total_steps % self.config.checkpoint_interval
                < self.config.steps_per_rollout
            ):
                checkpoint_path = f"checkpoints/independent_agent_step_{total_steps}.pt"
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                self.trainer.save(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        total_time = time.time() - start_time
        logger.info(f"Training complete: {total_steps} steps in {total_time:.1f}s")
        logger.info(f"Average: {total_steps / total_time:.1f} steps/s")

        self.writer.close()

        return {"total_steps": total_steps, "total_time": total_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--steps-per-rollout", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    config = IndependentTrainingConfig(
        steps_total=args.steps,
        steps_per_rollout=args.steps_per_rollout,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device=args.device,
    )

    trainer = IndependentTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
