"""
Baseline Measurement Script for SGS RL Training

Runs training with default parameters and measures win rates for all 3 factions
(Lord, Rebel, Spy) with configurable statistical reliability.

Usage:
    python train/optimization/baseline.py --steps 100000 --eval-episodes 100
    python train/optimization/baseline.py --steps 50000 --eval-episodes 50 --output logs/my_baseline.json
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("ai.gym_wrapper").setLevel(logging.ERROR)
logging.getLogger("engine.game_engine").setLevel(logging.ERROR)

from ai.gym_wrapper import SGSConfig, SGSEnv
from ai.state_encoder import StateEncoder
from ai.independent_trainer import IndependentAgentTrainer, IndependentAgentConfig
from train.config import TrainingConfig


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if args.eval_episodes <= 0:
        raise ValueError(f"--eval-episodes must be positive, got {args.eval_episodes}")


def run_evaluation_episode(
    trainer: IndependentAgentTrainer,
    state_encoder: StateEncoder,
    ep_idx: int,
) -> Tuple[float, int, str]:
    """Run a single evaluation episode deterministically."""
    eval_env = SGSEnv(SGSConfig())
    obs, info = eval_env.reset()
    episode_reward = 0.0
    episode_steps = 0
    done = False

    while not done and episode_steps < 5000:
        current_player_idx = eval_env.current_player_idx
        players = eval_env.players

        if current_player_idx < len(players) and players[current_player_idx].is_alive:
            game_state = eval_env._get_game_state_dict()
            local_obs = state_encoder.encode(game_state, current_player_idx)

            mask = (
                eval_env.action_masks() if hasattr(eval_env, "action_masks") else None
            )
            if mask is not None and isinstance(mask, tuple):
                mask = mask[0]

            action, _ = trainer.get_action(
                current_player_idx, local_obs, mask, deterministic=True
            )

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


def compute_baseline_metrics(
    trainer: IndependentAgentTrainer,
    state_encoder: StateEncoder,
    num_episodes: int,
) -> Dict[str, float]:
    """Run evaluation episodes and compute faction win rates."""
    results: List[Tuple[float, int, str]] = []

    logger.info(f"Running {num_episodes} evaluation episodes...")

    with ThreadPoolExecutor(max_workers=min(num_episodes, 8)) as executor:
        futures = [
            executor.submit(run_evaluation_episode, trainer, state_encoder, ep)
            for ep in range(num_episodes)
        ]

        for future in as_completed(futures):
            try:
                reward, steps, winner = future.result()
                results.append((reward, steps, winner))
            except Exception as e:
                logger.warning(f"Evaluation episode failed: {e}")
                results.append((0.0, 0, None))

    lord_wins = 0
    rebel_wins = 0
    spy_wins = 0

    for reward, steps, winner in results:
        if winner in ["lord", "lord_team", "主公"]:
            lord_wins += 1
        elif winner in ["rebel", "rebel_team", "反贼"]:
            rebel_wins += 1
        elif winner in ["spy", "内奸"]:
            spy_wins += 1

    total_reward = sum(r[0] for r in results)
    episode_lengths = [r[1] for r in results]

    n = len(results) if results else 1

    return {
        "lord_win_rate": lord_wins / n,
        "rebel_win_rate": rebel_wins / n,
        "spy_win_rate": spy_wins / n,
        "avg_reward": total_reward / n,
        "avg_episode_length": float(np.mean(episode_lengths))
        if episode_lengths
        else 0.0,
        "total_episodes": n,
        "lord_wins": lord_wins,
        "rebel_wins": rebel_wins,
        "spy_wins": spy_wins,
    }


def main():
    """Main entry point for baseline measurement."""
    parser = argparse.ArgumentParser(
        description="Run baseline training and measure faction win rates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=100000, help="Total training steps"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/baseline_win_rates.json",
        help="Output JSON file",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        sys.exit(1)

    if args.seed is not None:
        np.random.seed(args.seed)
        import torch

        torch.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("SGS RL Baseline Measurement")
    logger.info("=" * 60)
    logger.info(f"Training steps: {args.steps}")
    logger.info(f"Evaluation episodes: {args.eval_episodes}")
    logger.info(f"Output file: {args.output}")
    logger.info("=" * 60)

    config = TrainingConfig(
        steps_total=args.steps,
        num_eval_episodes=args.eval_episodes,
        device=args.device,
        seed=args.seed,
    )

    from train.train import Trainer

    logger.info("Starting training with default parameters...")
    trainer = Trainer(config)
    train_result = trainer.train()

    logger.info(f"Training complete: {train_result.get('total_steps', 0)} steps")

    metrics = compute_baseline_metrics(
        trainer=trainer.trainer,
        state_encoder=trainer.state_encoder,
        num_episodes=args.eval_episodes,
    )

    metrics["training_steps"] = train_result.get("total_steps", 0)
    metrics["training_time_s"] = train_result.get("total_time", 0.0)
    metrics["steps_per_second"] = train_result.get("total_steps", 0) / train_result.get(
        "total_time", 1.0
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("Baseline Results:")
    logger.info(f"  Lord win rate:   {metrics['lord_win_rate']:.2%}")
    logger.info(f"  Rebel win rate:  {metrics['rebel_win_rate']:.2%}")
    logger.info(f"  Spy win rate:    {metrics['spy_win_rate']:.2%}")
    logger.info(f"  Avg reward:      {metrics['avg_reward']:.4f}")
    logger.info(f"  Avg episode len: {metrics['avg_episode_length']:.1f}")
    logger.info(f"  Total episodes:  {metrics['total_episodes']}")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
