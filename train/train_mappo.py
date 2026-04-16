"""
MAPPO Training Script

Vanilla MAPPO (Multi-Agent PPO) training for Phase 0 validation.
Centralized critic with decentralized actors.

Usage:
    # Quick test (100 steps)
    python train/train_mappo.py --steps 100 --n-envs 1

    # Full training (10K steps)
    python train/train_mappo.py --steps 10000 --n-envs 4
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

import torch

try:
    import gymnasium as gym

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from ai.mappo import MAPPOAgent, MAPPOAgentConfig
from ai.mappo.centralized_critic import CentralizedCriticConfig
from ai.mappo.mappo_policy import MAPPOActorConfig
from ai.state_encoder import StateEncoder
from ai.gym_wrapper import SGSConfig, SGSEnv
from ai.action_encoder import ActionType, HierarchicalAction


class HierarchicalActionCollector:
    """Collect training data during hierarchical action execution.

    For each step of the hierarchical action (type -> card/skill -> target),
    collect the action, log_prob, and reward for PPO update.
    """

    def __init__(self, agent, state_encoder, action_encoder, mask_generator, device):
        self.agent = agent
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.mask_generator = mask_generator
        self.device = device

    def _get_agent_action_with_log_prob(
        self, game_state, player_idx, mask, deterministic=False
    ):
        local_obs = self.state_encoder.encode(game_state, player_idx)
        local_obs_t = (
            torch.from_numpy(local_obs)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        mask_t = (
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(self.device)
        )

        with torch.no_grad():
            joint_action_t, log_probs_t, _, _ = self.agent.get_actions(
                local_obs_t, mask_t, deterministic=deterministic
            )

        action = int(joint_action_t.squeeze(0).squeeze(0).cpu().numpy())
        log_prob = float(log_probs_t.squeeze(0).squeeze(0).cpu().numpy())

        return action, log_prob, local_obs, mask

    def execute_and_collect(self, env, game_state, player_idx, deterministic=False):
        """Execute hierarchical action and collect training data.

        Returns:
            list of dicts with keys: action, log_prob, local_obs, mask, reward
        """
        collected_data = []
        player = env.players[player_idx]
        total_reward = 0.0

        type_mask, _, _ = self.mask_generator.generate_masks(
            game_state, player, env.engine, 0, None
        )
        # Pad type_mask to action_dim (20) to match agent's expected input
        type_mask_padded = np.zeros(20, dtype=np.float32)
        type_mask_padded[: len(type_mask)] = type_mask

        action_type, log_prob, local_obs, mask = self._get_agent_action_with_log_prob(
            game_state, player_idx, type_mask_padded, deterministic
        )

        obs, reward, terminated, truncated, info = env.step(action_type)
        total_reward += reward
        done = terminated or truncated

        collected_data.append(
            {
                "action": action_type,
                "log_prob": log_prob,
                "local_obs": local_obs,
                "mask": mask,
                "reward": reward,
            }
        )

        if done or not self.action_encoder.needs_card(action_type):
            return collected_data, total_reward, done, info

        if env.current_step != 1 or env.pending_action is None:
            return collected_data, total_reward, done, info

        pending_action = env.pending_action
        card_mask, _, _ = self.mask_generator.generate_masks(
            game_state, player, env.engine, 1, pending_action
        )
        card_mask_padded = np.zeros(20, dtype=np.float32)
        card_mask_padded[: len(card_mask)] = card_mask

        card_idx, log_prob, local_obs, mask = self._get_agent_action_with_log_prob(
            game_state, player_idx, card_mask_padded, deterministic
        )

        obs, reward, terminated, truncated, info = env.step(card_idx)
        total_reward += reward
        done = terminated or truncated

        collected_data.append(
            {
                "action": card_idx,
                "log_prob": log_prob,
                "local_obs": local_obs,
                "mask": mask,
                "reward": reward,
            }
        )

        if done:
            return collected_data, total_reward, done, info

        card_or_skill = None
        if pending_action.action_type == ActionType.USE_CARD:
            if card_idx < len(player.hand_cards):
                card_or_skill = player.hand_cards[card_idx]
        elif pending_action.action_type == ActionType.USE_SKILL:
            if card_idx < len(player.skills):
                card_or_skill = player.skills[card_idx]

        if not self.action_encoder.needs_target(action_type, card_or_skill):
            return collected_data, total_reward, done, info

        if env.current_step != 2 or env.pending_action is None:
            return collected_data, total_reward, done, info

        pending_action = env.pending_action
        _, _, target_mask = self.mask_generator.generate_masks(
            game_state, player, env.engine, 2, pending_action
        )
        target_mask_padded = np.zeros(20, dtype=np.float32)
        target_mask_padded[: len(target_mask)] = target_mask

        target_idx, log_prob, local_obs, mask = self._get_agent_action_with_log_prob(
            game_state, player_idx, target_mask_padded, deterministic
        )

        obs, reward, terminated, truncated, info = env.step(target_idx)
        total_reward += reward
        done = terminated or truncated

        collected_data.append(
            {
                "action": target_idx,
                "log_prob": log_prob,
                "local_obs": local_obs,
                "mask": mask,
                "reward": reward,
            }
        )

        return collected_data, total_reward, done, info


@dataclass
class MAPPOTrainingConfig:
    total_steps: int = 10000
    n_envs: int = 1
    eval_interval: int = 1000
    num_eval_episodes: int = 10
    checkpoint_interval: int = 5000

    num_agents: int = 5
    local_state_dim: int = 2670
    global_state_dim: int = 2670 * 5
    action_dim: int = 20

    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    entropy_threshold: float = 0.01
    value_std_threshold: float = 0.001

    log_dir: Optional[str] = None
    device: Optional[str] = None  # "cuda", "cuda:0", "cpu", or None for auto


class MAPPOTrainer:
    def __init__(self, config: MAPPOTrainingConfig):
        self.config = config

        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")

        self.agent_config = MAPPOAgentConfig(
            num_agents=config.num_agents,
            actor_config=MAPPOActorConfig(
                local_state_dim=config.local_state_dim,
                action_dim=config.action_dim,
            ),
            critic_config=CentralizedCriticConfig(
                local_state_dim=config.local_state_dim,
                num_agents=config.num_agents,
                global_state_dim=config.global_state_dim,
                action_dim=config.action_dim,
            ),
            learning_rate=config.learning_rate,
            ppo_gamma=config.gamma,
            ppo_gae_lambda=config.gae_lambda,
            entropy_threshold=config.entropy_threshold,
            value_std_threshold=config.value_std_threshold,
        )

        self.agent = MAPPOAgent(self.agent_config).to(self.device)
        self.state_encoder = StateEncoder()

        self.envs: List[SGSEnv] = []

        self._step_count = 0
        self._episode_count = 0

        self._entropy_history: List[float] = []
        self._value_std_history: List[float] = []
        self._loss_history: List[float] = []

        self._writer: Optional[SummaryWriter] = None

    def _create_envs(self, n_envs: int) -> List[SGSEnv]:
        envs = []
        for i in range(n_envs):
            sgs_config = SGSConfig(
                player_num=self.config.num_agents,
                max_rounds=15,
                use_action_mask=True,
                use_shaping=True,
                other_player_policy="rule",
            )
            env = SGSEnv(sgs_config)
            envs.append(env)
        return envs

    def _get_global_state(self, env: SGSEnv) -> np.ndarray:
        game_state = env._get_game_state_dict()

        local_states = []
        for i in range(self.config.num_agents):
            local_state = self.state_encoder.encode(game_state, i)
            local_states.append(local_state)

        global_state = np.concatenate(local_states)
        return global_state

    def _get_action_masks(self, env: SGSEnv) -> np.ndarray:
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

    def _empty_rollout(self) -> Dict[str, torch.Tensor]:
        return {
            "global_states": torch.zeros(
                1, self.config.global_state_dim, device=self.device
            ),
            "local_observations": torch.zeros(
                1,
                self.config.num_agents,
                self.config.local_state_dim,
                device=self.device,
            ),
            "joint_actions": torch.zeros(
                1, self.config.num_agents, dtype=torch.long, device=self.device
            ),
            "old_log_probs": torch.zeros(1, self.config.num_agents, device=self.device),
            "old_values": torch.zeros(1, device=self.device),
            "advantages": torch.zeros(1, device=self.device),
            "returns": torch.zeros(1, device=self.device),
            "action_masks": torch.zeros(
                1, self.config.num_agents, self.config.action_dim, device=self.device
            ),
        }

    def _collect_rollout(self, env: SGSEnv, n_steps: int) -> Dict[str, torch.Tensor]:
        obs, info = env.reset()

        global_states = []
        local_observations = []
        joint_actions = []
        old_log_probs = []
        rewards = []
        dones = []
        action_masks = []
        values = []

        collector = HierarchicalActionCollector(
            self.agent,
            self.state_encoder,
            env.action_encoder,
            env.action_mask_generator,
            self.device,
        )

        for step in range(n_steps):
            global_state = self._get_global_state(env)
            game_state = env._get_game_state_dict()

            current_player_idx = env.current_player_idx
            if (
                current_player_idx < len(env.players)
                and env.players[current_player_idx].is_alive
            ):
                try:
                    collected_data, episode_reward, done, info = (
                        collector.execute_and_collect(
                            env, game_state, current_player_idx, deterministic=False
                        )
                    )

                    for data in collected_data:
                        global_states.append(global_state)
                        local_observations.append(data["local_obs"])
                        joint_actions.append([data["action"]] * self.config.num_agents)
                        old_log_probs.append(
                            [data["log_prob"]] * self.config.num_agents
                        )
                        rewards.append(data["reward"])
                        dones.append(done)
                        action_masks.append(data["mask"])

                        global_state_t = (
                            torch.from_numpy(global_state)
                            .unsqueeze(0)
                            .float()
                            .to(self.device)
                        )
                        joint_action_t = torch.tensor(
                            [[data["action"]] * self.config.num_agents],
                            dtype=torch.long,
                            device=self.device,
                        )
                        with torch.no_grad():
                            value_t = self.agent.get_centralized_value(
                                global_state_t, joint_action_t
                            )
                        values.append(value_t.item())

                    if len(collected_data) > 0:
                        entropy_estimate = -np.mean(
                            [d["log_prob"] for d in collected_data]
                        )
                        self._entropy_history.append(abs(entropy_estimate))

                except Exception as e:
                    logger.warning(f"Hierarchical action failed: {e}")
                    obs, info = env.reset()
                    done = True
                    episode_reward = 0.0
            else:
                obs, info = env.reset()
                done = True
                episode_reward = 0.0

            if done:
                obs, info = env.reset()
                self._episode_count += 1

        if len(global_states) == 0:
            return self._empty_rollout()

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

        # Expand single-agent data to multi-agent format for MAPPO
        # local_observations: [T, obs_dim] -> [T, num_agents, obs_dim]
        local_obs_expanded = local_observations_t.unsqueeze(1).expand(
            -1, self.config.num_agents, -1
        )
        # action_masks: [T, action_dim] -> [T, num_agents, action_dim]
        action_masks_expanded = action_masks_t.unsqueeze(1).expand(
            -1, self.config.num_agents, -1
        )

        return {
            "global_states": global_states_t,
            "local_observations": local_obs_expanded,
            "joint_actions": joint_actions_t,
            "old_log_probs": old_log_probs_t,
            "old_values": values_t,
            "advantages": advantages,
            "returns": returns,
            "action_masks": action_masks_expanded,
        }

    def _run_single_eval_episode(self, ep_idx: int) -> Tuple[float, int, bool]:
        eval_env = SGSEnv(
            SGSConfig(
                player_num=self.config.num_agents,
                max_rounds=15,
                use_action_mask=True,
                use_shaping=False,
                other_player_policy="rule",
            )
        )

        collector = HierarchicalActionCollector(
            self.agent,
            self.state_encoder,
            eval_env.action_encoder,
            eval_env.action_mask_generator,
            self.device,
        )

        obs, info = eval_env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done and episode_steps < 500:
            game_state = eval_env._get_game_state_dict()
            current_player_idx = eval_env.current_player_idx

            if (
                current_player_idx < len(eval_env.players)
                and eval_env.players[current_player_idx].is_alive
            ):
                try:
                    _, action_reward, action_done, info = collector.execute_and_collect(
                        eval_env, game_state, current_player_idx, deterministic=True
                    )
                    episode_reward += action_reward
                    done = action_done
                except Exception:
                    obs, info = eval_env.reset()
                    done = True
                    episode_reward = 0.0

            episode_steps += 1

        eval_env.close()

        winner = info.get("winner") if isinstance(info, dict) else None
        won = winner is not None

        return episode_reward, episode_steps, won

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation episodes in parallel."""
        results = []

        with ThreadPoolExecutor(
            max_workers=min(self.config.num_eval_episodes, 8)
        ) as executor:
            futures = [
                executor.submit(self._run_single_eval_episode, ep)
                for ep in range(self.config.num_eval_episodes)
            ]

            for future in as_completed(futures):
                try:
                    reward, steps, won = future.result()
                    results.append((reward, steps, won))
                except Exception as e:
                    logger.warning(f"Eval episode failed: {e}")
                    results.append((0.0, 0, False))

        total_reward = sum(r[0] for r in results)
        episode_lengths = [r[1] for r in results]
        wins = sum(1 for r in results if r[2])

        return {
            "win_rate": wins / len(results) if results else 0.0,
            "avg_reward": total_reward / len(results) if results else 0.0,
            "avg_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        }

    def train(self, log_dir: Optional[str] = None) -> Dict[str, float]:
        self.envs = self._create_envs(self.config.n_envs)

        total_steps = self.config.total_steps
        steps_per_rollout = 64

        logging.getLogger("engine").setLevel(logging.ERROR)
        logging.getLogger("engine.game_engine").setLevel(logging.ERROR)
        logging.getLogger("ai").setLevel(logging.ERROR)
        logging.getLogger("ai.gym_wrapper").setLevel(logging.ERROR)

        tb_dir = (
            Path(log_dir)
            if log_dir
            else Path("train/logs")
            / f"mappo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(tb_dir))

        logger.info(
            f"Starting MAPPO training for {total_steps} steps on {self.device}..."
        )
        logger.info(
            f"Config: n_envs={self.config.n_envs}, entropy_threshold={self.config.entropy_threshold}"
        )
        logger.info(f"TensorBoard logs: {tb_dir}")

        start_time = time.time()
        pbar = tqdm(total=total_steps, desc="Training", unit="steps")

        while self._step_count < total_steps:
            for env_idx, env in enumerate(self.envs):
                rollout = self._collect_rollout(env, steps_per_rollout)

                metrics = self.agent.update(
                    rollout,
                    n_epochs=self.config.n_epochs,
                    batch_size=self.config.batch_size,
                )

                self._loss_history.append(metrics["policy_loss"])

                self._step_count += steps_per_rollout

                health = self.agent.check_training_health()

                pbar.update(steps_per_rollout)
                pbar.set_postfix(
                    loss=f"{metrics['policy_loss']:.4f}",
                    entropy=f"{health.get('recent_entropy', 0):.4f}",
                    v_std=f"{health.get('recent_value_std', 0):.4f}",
                    eps=self._episode_count,
                )

                if self._writer:
                    self._writer.add_scalar(
                        "train/policy_loss", metrics["policy_loss"], self._step_count
                    )
                    self._writer.add_scalar(
                        "train/value_loss",
                        metrics.get("value_loss", 0),
                        self._step_count,
                    )
                    self._writer.add_scalar(
                        "train/entropy",
                        health.get("recent_entropy", 0),
                        self._step_count,
                    )
                    self._writer.add_scalar(
                        "train/value_std",
                        health.get("recent_value_std", 0),
                        self._step_count,
                    )
                    self._writer.add_scalar(
                        "train/episodes", self._episode_count, self._step_count
                    )

                if self._step_count % self.config.eval_interval < steps_per_rollout:
                    pbar.write(f"[Step {self._step_count}] Running evaluation...")
                    eval_metrics = self._evaluate()
                    pbar.write(
                        f"[Eval] win_rate={eval_metrics['win_rate']:.2%} "
                        f"avg_reward={eval_metrics['avg_reward']:.2f} "
                        f"avg_len={eval_metrics['avg_length']:.1f}"
                    )
                    if self._writer:
                        self._writer.add_scalar(
                            "eval/win_rate", eval_metrics["win_rate"], self._step_count
                        )
                        self._writer.add_scalar(
                            "eval/avg_reward",
                            eval_metrics["avg_reward"],
                            self._step_count,
                        )
                        self._writer.add_scalar(
                            "eval/avg_length",
                            eval_metrics["avg_length"],
                            self._step_count,
                        )

                if not health["entropy_ok"]:
                    pbar.write(
                        f"[WARN] Entropy low: {health['recent_entropy']:.4f} < {self.config.entropy_threshold}"
                    )
                if not health["value_std_ok"]:
                    pbar.write(
                        f"[WARN] Value_std low: {health['recent_value_std']:.4f} < {self.config.value_std_threshold}"
                    )

        pbar.close()
        training_time = time.time() - start_time

        final_health = self.agent.check_training_health()

        final_metrics = {
            "total_steps": self._step_count,
            "total_episodes": self._episode_count,
            "training_time": training_time,
            "final_entropy": final_health.get("recent_entropy", 0.0),
            "final_value_std": final_health.get("recent_value_std", 0.0),
            "entropy_ok": final_health["entropy_ok"],
            "value_std_ok": final_health["value_std_ok"],
            "overall_ok": final_health["overall_ok"],
        }

        logger.info(f"Training completed in {training_time:.1f}s!")
        logger.info(
            f"Final entropy: {final_metrics['final_entropy']:.4f} (threshold: {self.config.entropy_threshold})"
        )
        logger.info(
            f"Final value_std: {final_metrics['final_value_std']:.4f} (threshold: {self.config.value_std_threshold})"
        )

        if not final_health["overall_ok"]:
            logger.error("Training FAILED: entropy or value_std collapsed!")
        else:
            logger.info("Training HEALTHY: entropy and value_std within thresholds")

        for env in self.envs:
            env.close()

        if self._writer:
            self._writer.close()

        return final_metrics

    def save_checkpoint(self, path: str):
        self.agent.save(path)
        logger.info(f"Saved checkpoint to {path}")

    def save_evidence(self, path: str):
        evidence = []
        evidence.append(f"MAPPO Training Evidence Log")
        evidence.append(f"Generated: {datetime.now().isoformat()}")
        evidence.append(f"=" * 60)
        evidence.append(f"Configuration:")
        evidence.append(f"  - Total steps: {self.config.total_steps}")
        evidence.append(f"  - N_envs: {self.config.n_envs}")
        evidence.append(f"  - Num_agents: {self.config.num_agents}")
        evidence.append(f"  - Local_state_dim: {self.config.local_state_dim}")
        evidence.append(f"  - Global_state_dim: {self.config.global_state_dim}")
        evidence.append(f"  - Action_dim: {self.config.action_dim}")
        evidence.append(f"  - Entropy_threshold: {self.config.entropy_threshold}")
        evidence.append(f"  - Value_std_threshold: {self.config.value_std_threshold}")
        evidence.append(f"=" * 60)
        evidence.append(f"Training Metrics:")
        evidence.append(f"  - Total episodes: {self._episode_count}")
        evidence.append(f"  - Entropy history (last 10): {self._entropy_history[-10:]}")
        evidence.append(
            f"  - Value_std history (last 10): {self._value_std_history[-10:]}"
        )
        evidence.append(f"  - Loss history (last 10): {self._loss_history[-10:]}")
        evidence.append(f"=" * 60)
        final_health = self.agent.check_training_health()
        evidence.append(f"Final Health Check:")
        evidence.append(f"  - Entropy OK: {final_health['entropy_ok']}")
        evidence.append(f"  - Value_std OK: {final_health['value_std_ok']}")
        evidence.append(f"  - Overall OK: {final_health['overall_ok']}")
        evidence.append(
            f"  - Recent entropy: {final_health.get('recent_entropy', 'N/A')}"
        )
        evidence.append(
            f"  - Recent value_std: {final_health.get('recent_value_std', 'N/A')}"
        )
        evidence.append(f"=" * 60)

        if final_health["entropy_ok"]:
            evidence.append(
                f"ASSERTION PASSED: entropy > {self.config.entropy_threshold}"
            )
        else:
            evidence.append(
                f"ASSERTION FAILED: entropy <= {self.config.entropy_threshold}"
            )

        if final_health["value_std_ok"]:
            evidence.append(
                f"ASSERTION PASSED: value_std > {self.config.value_std_threshold}"
            )
        else:
            evidence.append(
                f"ASSERTION FAILED: value_std <= {self.config.value_std_threshold}"
            )

        with open(path, "w") as f:
            f.write("\n".join(evidence))

        logger.info(f"Saved evidence to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAPPO Training for SGS")
    parser.add_argument("--steps", type=int, default=100, help="Total training steps")
    parser.add_argument(
        "--eval-interval", type=int, default=10000, help="Evaluation interval"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--n-envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cuda:0, cpu, or None for auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)

    config = MAPPOTrainingConfig(
        total_steps=args.steps,
        n_envs=args.n_envs,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        device=args.device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else Path("train/logs") / f"mappo_{timestamp}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer = MAPPOTrainer(config)

    try:
        metrics = trainer.train(log_dir=str(log_dir))

        checkpoint_path = log_dir / "mappo_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        evidence_path = log_dir / "mappo_training.log"
        trainer.save_evidence(str(evidence_path))

        if not metrics["entropy_ok"]:
            logger.warning(
                f"Entropy low: {metrics['final_entropy']} <= {config.entropy_threshold}"
            )

        if not metrics["value_std_ok"]:
            logger.warning(
                f"Value_std low: {metrics['final_value_std']} <= {config.value_std_threshold}"
            )

        logger.info("Training completed with checkpoint saved.")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()

        evidence_path = log_dir / "task-2-mappo-training.log"
        trainer.save_evidence(str(evidence_path))

        return 1


if __name__ == "__main__":
    sys.exit(main())
