"""
Self-Play Training Script - Main training loop for SGS RL

Features:
- MaskablePPO with ExtendedTransformerPolicy
- SelfPlayWrapper environment with agent pool sampling
- Checkpoint saving at regular intervals
- Periodic evaluation against agent pool
- Training recovery from checkpoints
- TensorBoard logging
- Command-line configuration

Usage:
    # Standard training
    python train/train_self_play.py --timesteps 2000000 --n-envs 8

    # Quick test mode
    python train/train_self_play.py --test-mode

    # Resume from checkpoint
    python train/train_self_play.py --resume checkpoints/step_500000.zip
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

# Add project root to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Conditional imports for SB3
try:
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
        BaseCallback,
    )

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning(
        "stable_baselines3 not available. Install with: pip install stable-baselines3"
    )

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from sb3_contrib.common.wrappers import ActionMasker

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    logger.warning("sb3_contrib not available. Install with: pip install sb3-contrib")

# Project imports
from ai.self_play_wrapper import SelfPlayWrapper
from ai.gym_wrapper import SGSConfig
from train.config import TrainingConfig
from train.agent_pool_manager import AgentPoolManager
from train.policy_extension import (
    ExtendedTransformerPolicy,
    create_extended_policy_kwargs,
)
from train.evaluate import evaluate_model, EvaluationConfig
from train.recovery import (
    save_training_state,
    load_training_state,
    TrainingState,
    list_checkpoints,
)
from train.visualize import (
    setup_tensorboard,
    MetricsLogger,
    log_win_rate,
    log_elo,
    log_hyperparameters,
)


# ============================================================================
# Custom Callbacks
# ============================================================================


class SelfPlayCheckpointCallback(BaseCallback):
    """
    Custom checkpoint callback for self-play training.

    Saves checkpoints at regular intervals and adds trained agent to pool.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        agent_pool: AgentPoolManager,
        name_prefix: str = "step",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.agent_pool = agent_pool
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        # Check if we should save
        if self.n_calls % self.save_freq == 0:
            # Create checkpoint path
            checkpoint_path = (
                self.save_path / f"{self.name_prefix}_{self.num_timesteps}"
            )

            # Save model
            try:
                self.model.save(str(checkpoint_path))
                if self.verbose > 0:
                    logger.info(
                        f"Saved checkpoint at step {self.num_timesteps} to {checkpoint_path}"
                    )

                # Add to agent pool
                version = self.agent_pool.add_agent(
                    path=str(checkpoint_path) + ".zip",
                    elo_rating=1000.0
                    + self.num_timesteps / 10000,  # Simple ELO estimation
                    parent_version=self.agent_pool.get_latest_agent().version
                    if self.agent_pool.get_latest_agent()
                    else -1,
                    metadata={
                        "step": self.num_timesteps,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                if self.verbose > 0:
                    logger.info(f"Added agent version {version} to pool")

            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

        return True


class EvaluationCallback(BaseCallback):
    """
    Custom evaluation callback for self-play training.

    Runs periodic evaluation against agent pool and logs to TensorBoard.
    """

    def __init__(
        self,
        eval_freq: int,
        agent_pool: AgentPoolManager,
        metrics_logger: MetricsLogger,
        num_eval_games: int = 50,
        player_num: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.agent_pool = agent_pool
        self.metrics_logger = metrics_logger
        self.num_eval_games = num_eval_games
        self.player_num = player_num
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        # Check if we should evaluate
        if self.n_calls % self.eval_freq == 0 and self.n_calls > self._last_eval_step:
            self._last_eval_step = self.n_calls

            try:
                # Get best agent for comparison
                best_agent = self.agent_pool.get_best_agent()
                latest_agent = self.agent_pool.get_latest_agent()

                # Log pool statistics
                pool_size = len(self.agent_pool)
                if self.metrics_logger.writer is not None:
                    self.metrics_logger.writer.add_scalar(
                        "pool/size", pool_size, self.num_timesteps
                    )

                if best_agent:
                    self.metrics_logger.log_elo(
                        elo_rating=best_agent.elo_rating,
                        version=best_agent.version,
                    )

                if self.verbose > 0:
                    logger.info(
                        f"Evaluation at step {self.num_timesteps}: "
                        f"pool_size={pool_size}, best_elo={best_agent.elo_rating if best_agent else 'N/A'}"
                    )

            except Exception as e:
                logger.error(f"Evaluation callback error: {e}")

        return True


class TensorBoardLoggingCallback(BaseCallback):
    """
    Custom callback for logging training metrics to TensorBoard.

    Logs loss, entropy, learning rate, and other training statistics.
    """

    def __init__(
        self,
        metrics_logger: MetricsLogger,
        log_freq: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.metrics_logger = metrics_logger
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Log metrics at specified frequency
        if self.n_calls % self.log_freq == 0:
            # Update step in logger
            self.metrics_logger.set_step(self.num_timesteps)

            # Get training info from model
            if hasattr(self.model, "_stats_window_size"):
                # Access recent statistics if available
                pass

        return True


# ============================================================================
# Environment Creation Functions
# ============================================================================


def make_env(config: SGSConfig, rank: int = 0) -> callable:
    """
    Create environment factory for VecEnv.

    Args:
        config: SGS environment configuration
        rank: Process rank for multiprocessing

    Returns:
        Function that creates SelfPlayWrapper environment
    """

    def _init():
        env = SelfPlayWrapper(config)
        return env

    return _init


def create_vec_env(n_envs: int, config: SGSConfig, use_subprocess: bool = True) -> Any:
    """
    Create vectorized environment for training.

    Args:
        n_envs: Number of parallel environments
        config: SGS environment configuration
        use_subprocess: Use SubprocVecEnv (True) or DummyVecEnv (False)

    Returns:
        VecEnv instance
    """
    env_fns = [make_env(config, rank=i) for i in range(n_envs)]

    if use_subprocess and n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Optionally normalize observations and rewards
    # For Dict observation spaces, specify keys to normalize
    # SelfPlayWrapper has Dict obs with 'state' (Box) and other fields
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        norm_obs_keys=["state"],  # Only normalize state vector, not masks
    )

    return env


def mask_fn(env: Any) -> np.ndarray:
    """
    Extract action masks from environment for MaskablePPO.

    Args:
        env: Environment instance

    Returns:
        Combined action mask array
    """
    # SelfPlayWrapper inherits from SGSEnv which provides action masks
    if hasattr(env, "action_masks"):
        return env.action_masks()

    # Fallback: construct mask from observation
    obs = env.get_obs()
    if isinstance(obs, dict):
        type_mask = obs.get("action_mask_type", np.ones(10))
        card_mask = obs.get("action_mask_card", np.ones(20))
        target_mask = obs.get("action_mask_target", np.ones(5))
        # Combine masks - simplest approach
        return np.concatenate([type_mask, card_mask, target_mask])

    # Default: all actions valid
    return np.ones(100, dtype=np.float32)


# ============================================================================
# Training Functions
# ============================================================================


def create_model(
    env: Any,
    config: TrainingConfig,
    policy_kwargs: Optional[Dict] = None,
) -> Any:
    """
    Create MaskablePPO model with ExtendedTransformerPolicy.

    Args:
        env: Training environment
        config: Training configuration
        policy_kwargs: Optional custom policy kwargs

    Returns:
        MaskablePPO model instance
    """
    if not MASKABLE_PPO_AVAILABLE:
        raise ImportError("MaskablePPO required. Install sb3-contrib.")

    # Create policy kwargs
    if policy_kwargs is None:
        try:
            policy_kwargs = create_extended_policy_kwargs(state_dim=3000)
            # Remove keys that are invalid for MaskablePPO's policy
            # policy_class is not a valid policy_kwarg (used separately)
            policy_kwargs.pop("policy_class", None)
            # state_dim is already in features_extractor_kwargs
            policy_kwargs.pop("state_dim", None)
        except Exception as e:
            logger.warning(f"Failed to create ExtendedTransformerPolicy kwargs: {e}")
            logger.info("Using default policy instead")
            policy_kwargs = None

    # Create model - use MaskableActorCriticPolicy if available
    try:
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

        USE_MASKABLE_POLICY = True
    except ImportError:
        USE_MASKABLE_POLICY = False
        logger.warning(
            "MaskableActorCriticPolicy not available, falling back to MlpPolicy"
        )

    # Create model
    # Note: ExtendedTransformerPolicy inherits from ActorCriticPolicy (SB3)
    # MaskablePPO requires MaskableActorCriticPolicy (sb3_contrib)
    # SelfPlayWrapper has Dict observation space - use MultiInputPolicy
    if policy_kwargs is not None:
        model = MaskablePPO(
            "MultiInputPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=None,  # We use custom MetricsLogger
        )
        logger.info("Using MultiInputPolicy with custom features extractor")
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=None,
        )
        logger.info("Using default MultiInputPolicy")

    return model


def initialize_agent_pool(pool_size: int, log_dir: Path) -> AgentPoolManager:
    """
    Initialize agent pool with baseline policies.

    Args:
        pool_size: Maximum pool size
        log_dir: Directory for storing pool data

    Returns:
        Initialized AgentPoolManager
    """
    pool = AgentPoolManager(pool_size=pool_size)

    # Add baseline agents (RuleAI / Random)
    # For initial pool, we use placeholder paths that indicate RuleAI
    for i in range(min(3, pool_size)):  # Start with 3 baseline agents
        version = pool.add_agent(
            path="",  # Empty path -> will use RuleAI in SelfPlayWrapper
            elo_rating=1000.0 - i * 10,  # Slightly different ratings
            parent_version=-1,
            metadata={"type": "rule_baseline", "index": i},
        )
        logger.info(f"Added baseline agent version {version} to pool")

    return pool


def setup_callbacks(
    config: TrainingConfig,
    agent_pool: AgentPoolManager,
    metrics_logger: MetricsLogger,
    log_dir: Path,
) -> CallbackList:
    """
    Create callback list for training.

    Args:
        config: Training configuration
        agent_pool: Agent pool manager
        metrics_logger: Metrics logger instance
        log_dir: Log directory

    Returns:
        CallbackList with all callbacks
    """
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Self-play checkpoint callback
    checkpoint_callback = SelfPlayCheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(checkpoint_dir),
        agent_pool=agent_pool,
        name_prefix="step",
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvaluationCallback(
        eval_freq=config.eval_freq,
        agent_pool=agent_pool,
        metrics_logger=metrics_logger,
        num_eval_games=50,
        player_num=5,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # TensorBoard logging callback
    tb_callback = TensorBoardLoggingCallback(
        metrics_logger=metrics_logger,
        log_freq=1000,
        verbose=0,
    )
    callbacks.append(tb_callback)

    return CallbackList(callbacks)


def train(
    config: TrainingConfig,
    test_mode: bool = False,
    resume_path: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> Any:
    """
    Main training function.

    Args:
        config: Training configuration
        test_mode: Quick test mode (reduced timesteps)
        resume_path: Path to checkpoint for resuming
        log_dir: Custom log directory

    Returns:
        Trained model
    """
    if not SB3_AVAILABLE or not MASKABLE_PPO_AVAILABLE:
        raise ImportError(
            "Required dependencies not available. "
            "Install: pip install stable-baselines3 sb3-contrib"
        )

    # Override config for test mode
    if test_mode:
        config.timesteps = 1000
        config.n_envs = 1
        config.checkpoint_freq = 500
        config.eval_freq = 250
        logger.info("Test mode enabled: reduced timesteps and single env")

    # Setup log directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("train/logs") / f"self_play_{timestamp}"

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Log directory: {log_dir}")

    # Create environment config
    sgs_config = SGSConfig(
        player_num=5,
        max_rounds=15,
        use_action_mask=True,
        use_shaping=True,
        other_player_policy="rule",
    )

    # Create vectorized environment
    logger.info(f"Creating {config.n_envs} parallel environments...")
    env = create_vec_env(config.n_envs, sgs_config, use_subprocess=(config.n_envs > 1))

    # Initialize agent pool
    logger.info(f"Initializing agent pool (size={config.pool_size})...")
    agent_pool = initialize_agent_pool(config.pool_size, log_dir)

    # Setup TensorBoard logging
    run_name = log_dir.name
    metrics_logger = MetricsLogger(run_name, log_dir=log_dir)

    # Log hyperparameters
    hyperparams = {
        "timesteps": config.timesteps,
        "n_envs": config.n_envs,
        "checkpoint_freq": config.checkpoint_freq,
        "eval_freq": config.eval_freq,
        "pool_size": config.pool_size,
        "learning_rate": 3e-4,
        "gamma": 0.99,
    }
    # Use log_hyperparameters which handles None writer
    log_hyperparameters(metrics_logger.writer, hyperparams)

    # Create or load model
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from checkpoint: {resume_path}")
        try:
            state = load_training_state(resume_path)
            model = state.model
            # Restore training state
            starting_step = state.step_count
            remaining_steps = config.timesteps - starting_step
            logger.info(
                f"Resuming at step {starting_step}, remaining {remaining_steps} steps"
            )

            # Set environment
            model.set_env(env)

            # Update metrics logger step
            metrics_logger.set_step(starting_step)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh training instead")
            model = create_model(env, config)
    else:
        logger.info("Creating new model...")
        model = create_model(env, config)

    # Setup callbacks
    callbacks = setup_callbacks(config, agent_pool, metrics_logger, log_dir)

    # Start training
    logger.info(f"Starting training for {config.timesteps} timesteps...")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=config.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save emergency checkpoint
        emergency_path = log_dir / "checkpoints" / "interrupted.zip"
        model.save(str(emergency_path))
        logger.info(f"Saved emergency checkpoint to {emergency_path}")

    # Training complete
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time / 3600:.2f} hours")

    # Save final model
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    # Save VecNormalize stats
    vecnormalize_path = log_dir / "vecnormalize.pkl"
    env.save(str(vecnormalize_path))
    logger.info(f"Saved VecNormalize stats to {vecnormalize_path}")

    # Final evaluation
    logger.info("Running final evaluation...")
    try:
        result = evaluate_model(
            model_path=str(final_path),
            num_games=100,
            player_num=5,
            opponent_pool=list(agent_pool),
            vec_normalize_path=str(vecnormalize_path),
        )
        logger.info(f"Final win rate: {result.win_rate:.2%}")

        # Log final metrics
        metrics_logger.set_step(config.timesteps)
        metrics_logger.log_win_rate(result.win_rate, total_games=100)
        metrics_logger.log_identity_win_rates(result.identity_win_rates)

        # Save evaluation report
        result.save(str(log_dir / "evaluation_report.json"))

    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")

    # Close logging
    metrics_logger.close()

    # Cleanup
    env.close()

    return model


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Self-play training for SGS RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=10,
        help="Agent pool size",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="Checkpoint frequency (steps)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluation frequency (steps)",
    )

    # Mode flags
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Quick test mode (1000 steps, single env)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom log directory",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=info, 2=debug)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set verbosity
    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose == 2:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create config
    config = TrainingConfig(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        pool_size=args.pool_size,
    )

    # Determine log directory
    log_dir = Path(args.log_dir) if args.log_dir else None

    # Run training
    try:
        model = train(
            config=config,
            test_mode=args.test_mode,
            resume_path=args.resume,
            log_dir=log_dir,
        )
        logger.info("Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
