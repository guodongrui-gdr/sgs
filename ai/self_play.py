"""
自博弈训练器 - 使用自博弈进行强化学习训练

功能:
- 自博弈训练循环
- 策略池集成
- 周期性评估
- 训练监控
- 自博弈与规则AI混合训练
- 对手采样策略配置
- ELO评分跟踪
"""

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    BaseCallback = object

try:
    from sb3_contrib import MaskablePPO

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    MaskablePPO = None

from ai.gym_wrapper import SGSConfig
from ai.multi_agent_env import SelfPlayEnv, make_multi_agent_env
from ai.policy_pool import PolicyPool, MatchHistory, PoolConfig, SamplingStrategy
from ai.rule_ai import RuleAI, RuleAIConfig


class TrainingMode(Enum):
    """训练模式"""

    SELF_PLAY_ONLY = "self_play_only"  # 仅自博弈
    RULE_BASED_ONLY = "rule_based_only"  # 仅规则AI
    MIXED = "mixed"  # 混合训练
    ALTERNATING = "alternating"  # 交替训练


@dataclass
class SelfPlayConfig:
    """自博弈训练配置"""

    total_timesteps: int = 10_000_000
    save_freq: int = 100_000
    eval_freq: int = 50_000
    update_opponent_freq: int = 25_000
    n_eval_games: int = 100

    learning_rate: float = 5e-4
    lr_schedule: str = "cosine"
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.98
    ent_coef: float = 0.05
    vf_coef: float = 0.25
    clip_range: float = 0.2

    player_num: int = 5
    max_rounds: int = 100

    pool_size: int = 10

    # 对手采样配置
    sample_latest_ratio: float = 0.3  # 30%采样最新策略
    sample_best_ratio: float = 0.3  # 30%采样最高ELO策略
    sample_random_ratio: float = 0.4  # 40%随机采样(多样性)

    # 自博弈与规则AI混合配置
    training_mode: TrainingMode = TrainingMode.MIXED
    self_play_ratio: float = 0.5  # 自博弈比例(50%)
    rule_based_ratio: float = 0.5  # 规则AI比例(50%)
    alternating_epochs: int = 5  # 交替训练时的epoch周期

    # ELO评分配置
    elo_k_factor: float = 32.0
    elo_decay_rate: float = 0.01  # 每小时衰减1%
    elo_max_decay: float = 0.3  # 最大衰减30%
    initial_elo: float = 1000.0

    # 规则AI配置
    rule_ai_aggressiveness: float = 0.7
    rule_ai_defensiveness: float = 0.5

    log_dir: str = ""
    seed: int = 42

    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = str(
                Path("logs") / f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # 验证采样比例总和
        ratio_sum = (
            self.sample_latest_ratio + self.sample_best_ratio + self.sample_random_ratio
        )
        if abs(ratio_sum - 1.0) > 0.01:
            # 自动调整到100%
            self.sample_random_ratio = (
                1.0 - self.sample_latest_ratio - self.sample_best_ratio
            )

        # 验证训练比例
        mode_sum = self.self_play_ratio + self.rule_based_ratio
        if abs(mode_sum - 1.0) > 0.01:
            self.rule_based_ratio = 1.0 - self.self_play_ratio

    def get_pool_config(self) -> PoolConfig:
        """获取策略池配置"""
        return PoolConfig(
            max_size=self.pool_size,
            sample_latest_ratio=self.sample_latest_ratio,
            sample_best_ratio=self.sample_best_ratio,
            sample_random_ratio=self.sample_random_ratio,
            k_factor=self.elo_k_factor,
            elo_decay_rate=self.elo_decay_rate,
            elo_max_decay=self.elo_max_decay,
            initial_elo=self.initial_elo,
        )


class SelfPlayTrainer:
    """
    自博弈训练器

    训练智能体通过与自己或其他历史版本对战来提升

    训练模式:
    - SELF_PLAY_ONLY: 纯自博弈训练
    - RULE_BASED_ONLY: 纯规则AI训练
    - MIXED: 自博弈与规则AI混合(随机选择)
    - ALTERNATING: 自博弈与规则AI交替训练
    """

    def __init__(self, config: SelfPlayConfig = None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for self-play training")

        self.config = config or SelfPlayConfig()

        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建策略池(使用新配置)
        self.policy_pool = PolicyPool(
            pool_dir=str(self.log_dir / "policy_pool"),
            config=self.config.get_pool_config(),
        )

        self.match_history = MatchHistory()

        self.env: Optional[SelfPlayEnv] = None
        self.model: Optional[Any] = None
        self.current_policy_version: int = 0

        # 规则AI实例
        self.rule_ai = RuleAI(
            RuleAIConfig(
                aggressiveness=self.config.rule_ai_aggressiveness,
                defensiveness=self.config.rule_ai_defensiveness,
            )
        )

        self._step_count: int = 0
        self._episode_count: int = 0
        self._win_count: int = 0
        self._self_play_win_count: int = 0
        self._rule_based_win_count: int = 0
        self._total_reward: float = 0.0

        # 训练模式跟踪
        self._current_mode: str = "self_play"
        self._alternating_counter: int = 0
        self._self_play_episodes: int = 0
        self._rule_based_episodes: int = 0

        self._training_metrics: List[Dict] = []
        self._elo_tracking: List[Dict] = []

    def setup(self):
        sgs_config = SGSConfig(
            player_num=self.config.player_num,
            max_rounds=self.config.max_rounds,
        )

        self.env = make_multi_agent_env(
            player_num=self.config.player_num,
            training_agent_idx=0,
        )

        if MASKABLE_PPO_AVAILABLE:
            self.model = MaskablePPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                clip_range=self.config.clip_range,
                seed=self.config.seed,
                tensorboard_log=str(self.log_dir / "tensorboard"),
                verbose=1,
            )
        else:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                clip_range=self.config.clip_range,
                seed=self.config.seed,
                tensorboard_log=str(self.log_dir / "tensorboard"),
                verbose=1,
            )

    def _determine_training_mode(self) -> str:
        """
        确定当前训练模式

        Returns:
                "self_play" 或 "rule_based"
        """
        mode = self.config.training_mode

        if mode == TrainingMode.SELF_PLAY_ONLY:
            return "self_play"
        elif mode == TrainingMode.RULE_BASED_ONLY:
            return "rule_based"
        elif mode == TrainingMode.ALTERNATING:
            # 交替训练: 每alternating_epochs轮切换
            epoch_cycle = self._alternating_counter // self.config.alternating_epochs
            self._alternating_counter += 1
            if epoch_cycle % 2 == 0:
                return "self_play"
            else:
                return "rule_based"
        else:  # MIXED
            # 随机选择，按比例
            r = random.random()
            if r < self.config.self_play_ratio:
                return "self_play"
            else:
                return "rule_based"

    def _get_opponent_policy(self) -> Optional[Any]:
        """
        获取对手策略

        根据训练模式和采样策略返回对手
        """
        current_mode = self._determine_training_mode()
        self._current_mode = current_mode

        if current_mode == "rule_based":
            # 规则AI模式: 返回None，使用规则AI
            return None
        else:
            # 自博弈模式: 从策略池采样
            sampled = self.policy_pool.sample_policy()
            if sampled:
                try:
                    # 加载采样的策略模型
                    if MASKABLE_PPO_AVAILABLE:
                        return MaskablePPO.load(sampled.path)
                    else:
                        return PPO.load(sampled.path)
                except Exception as e:
                    print(f"Failed to load sampled policy: {e}")
                    return None
            return None

    def train(self):
        if self.model is None:
            self.setup()

        print(
            f"Starting self-play training for {self.config.total_timesteps} timesteps"
        )
        print(f"Log directory: {self.log_dir}")
        print(f"Training mode: {self.config.training_mode.value}")
        print(f"Self-play ratio: {self.config.self_play_ratio:.0%}")
        print(
            f"Opponent sampling: {self.config.sample_latest_ratio:.0%} latest, "
            f"{self.config.sample_best_ratio:.0%} best, "
            f"{self.config.sample_random_ratio:.0%} random"
        )

        start_time = time.time()

        while self._step_count < self.config.total_timesteps:
            obs, _ = self.env.reset(seed=self.config.seed + self._episode_count)

            # 获取对手策略
            opponent = self._get_opponent_policy()

            # 设置对手到环境
            if self._current_mode == "self_play" and opponent:
                self.env.set_opponent_pool([opponent])

            episode_reward = 0.0
            episode_length = 0
            done = False
            opponent_version = 0  # 用于ELO更新

            while not done:
                action = self._select_action(obs)
                next_obs, reward, terminated, truncated, info = (
                    self.env.step_with_policy(self.model)
                )

                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                self._step_count += 1

                if self._step_count % self.config.save_freq == 0:
                    self._save_checkpoint()

                if self._step_count % self.config.eval_freq == 0:
                    self._evaluate()

                if self._step_count % self.config.update_opponent_freq == 0:
                    self._update_opponents()

                obs = next_obs

            self._episode_count += 1
            self._total_reward += episode_reward

            winner = info.get("winner")
            training_identity = self.env.players[0].identity if self.env.players else ""

            won = winner and self._check_win(training_identity, winner)
            if won:
                self._win_count += 1
                if self._current_mode == "self_play":
                    self._self_play_win_count += 1
                    self._self_play_episodes += 1
                else:
                    self._rule_based_win_count += 1
                    self._rule_based_episodes += 1

            # 记录比赛结果
            if self._current_mode == "self_play" and opponent:
                self._record_match(opponent_version, won, training_identity)

            self._log_episode(episode_reward, episode_length, info)

        total_time = time.time() - start_time

        self._print_final_summary(total_time)
        self._save_final_model()

    def _record_match(self, opponent_version: int, won: bool, identity: str):
        """记录自博弈比赛结果"""
        current_version = self.current_policy_version

        self.match_history.add_match(
            policy_a_version=current_version,
            policy_b_version=opponent_version,
            winner_version=current_version if won else opponent_version,
            game_length=self.env.round_count if self.env else 0,
            policy_a_identity=identity,
        )

        # 更新ELO评分
        if won:
            self.policy_pool.update_elo(
                winner_version=current_version,
                loser_version=opponent_version,
                winner_identity=identity,
            )
        else:
            self.policy_pool.update_elo(
                winner_version=opponent_version,
                loser_version=current_version,
                loser_identity=identity,
            )

    def _print_final_summary(self, total_time: float):
        """打印最终训练摘要"""
        print(f"\n{'=' * 50}")
        print(f"Training completed!")
        print(f"{'=' * 50}")
        print(f"Total timesteps: {self._step_count}")
        print(f"Total episodes: {self._episode_count}")

        overall_win_rate = (
            self._win_count / self._episode_count if self._episode_count > 0 else 0
        )
        print(f"Overall win rate: {overall_win_rate:.2%}")

        if self._self_play_episodes > 0:
            sp_win_rate = self._self_play_win_count / self._self_play_episodes
            print(
                f"Self-play win rate: {sp_win_rate:.2%} ({self._self_play_episodes} episodes)"
            )

        if self._rule_based_episodes > 0:
            rb_win_rate = self._rule_based_win_count / self._rule_based_episodes
            print(
                f"Rule-based win rate: {rb_win_rate:.2%} ({self._rule_based_episodes} episodes)"
            )

        print(f"Average reward: {self._total_reward / self._episode_count:.2f}")
        print(f"Total time: {total_time / 3600:.2f} hours")

        # 打印策略池统计
        pool_stats = self.policy_pool.get_stats()
        print(f"\nPolicy Pool Statistics:")
        print(f"  Total policies: {pool_stats['total_policies']}")
        print(f"  Best ELO: {pool_stats['best_elo']:.0f}")
        print(f"  Average ELO: {pool_stats['avg_elo']:.0f}")

        # 打印身份胜率
        identity_rates = pool_stats.get("identity_win_rates", {})
        if identity_rates:
            print(f"  Identity win rates:")
            for identity, rate in identity_rates.items():
                if rate > 0:
                    print(f"    {identity}: {rate:.2%}")

    def _select_action(self, obs: Dict) -> int:
        legal_actions = self.env.get_legal_actions(0)

        if not legal_actions:
            return 0

        action_mask = self.env.get_action_mask(0)

        if MASKABLE_PPO_AVAILABLE:
            action, _ = self.model.predict(
                obs, action_masks=action_mask, deterministic=False
            )
        else:
            action, _ = self.model.predict(obs, deterministic=False)

        if action not in legal_actions:
            action = np.random.choice(legal_actions)

        return action

    def _check_win(self, training_identity: str, winner: str) -> bool:
        """检查训练玩家是否获胜"""
        if training_identity in ["主公", "忠臣"] and winner == "主公":
            return True
        if training_identity == "反贼" and winner == "反贼":
            return True
        if training_identity == "内奸" and winner == "内奸":
            return True
        return False

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = (
            self.log_dir / "checkpoints" / f"model_step_{self._step_count}"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(checkpoint_path))

        self.current_policy_version += 1

        # 添加到策略池(包含训练步数元数据)
        self.policy_pool.add_policy(
            str(checkpoint_path),
            parent_version=self.current_policy_version - 1,
            metadata={
                "training_steps": self._step_count,
                "episode_count": self._episode_count,
                "win_rate": self._win_count / max(self._episode_count, 1),
                "training_mode": self._current_mode,
            },
        )

        print(
            f"Saved checkpoint at step {self._step_count} (version {self.current_policy_version})"
        )

        # 追踪ELO变化
        stats = self.policy_pool.get_stats()
        self._elo_tracking.append(
            {
                "step": self._step_count,
                "version": self.current_policy_version,
                "best_elo": stats.get("best_elo"),
                "avg_elo": stats.get("avg_elo"),
            }
        )

        self._save_metrics()

    def _evaluate(self):
        """周期性评估"""
        print(f"\nEvaluating at step {self._step_count}...")

        # 分别评估自博弈和规则AI
        eval_results = {
            "self_play": self._evaluate_mode("self_play"),
            "rule_based": self._evaluate_mode("rule_based"),
        }

        for mode, result in eval_results.items():
            print(
                f"  {mode}: win_rate={result['win_rate']:.2%}, avg_reward={result['avg_reward']:.2f}"
            )

        self._training_metrics.append(
            {
                "step": self._step_count,
                "episode": self._episode_count,
                "overall_win_rate": self._win_count / max(self._episode_count, 1),
                "self_play_win_rate": eval_results["self_play"]["win_rate"],
                "rule_based_win_rate": eval_results["rule_based"]["win_rate"],
                "self_play_episodes": self._self_play_episodes,
                "rule_based_episodes": self._rule_based_episodes,
                "pool_stats": self.policy_pool.get_stats(),
            }
        )

    def _evaluate_mode(self, mode: str, n_games: int = None) -> Dict:
        """
        评估特定模式

        Args:
                mode: "self_play" 或 "rule_based"
                n_games: 评估局数(默认使用配置)

        Returns:
                评估结果字典
        """
        n_games = n_games or self.config.n_eval_games

        wins = 0
        total_reward = 0.0
        identity_wins = {"主公": 0, "忠臣": 0, "反贼": 0, "内奸": 0}
        identity_totals = {"主公": 0, "忠臣": 0, "反贼": 0, "内奸": 0}

        for _ in range(n_games):
            obs, _ = self.env.reset()

            # 设置对手
            if mode == "self_play":
                opponent = self.policy_pool.sample_policy(
                    strategy=SamplingStrategy.BEST_ELO
                )
                if opponent:
                    try:
                        loaded_policy = (
                            MaskablePPO.load(opponent.path)
                            if MASKABLE_PPO_AVAILABLE
                            else None
                        )
                        self.env.set_opponent_pool(
                            [loaded_policy] if loaded_policy else []
                        )
                    except:
                        pass

            done = False
            episode_reward = 0.0

            while not done:
                action_mask = self.env.get_action_mask(0)
                if MASKABLE_PPO_AVAILABLE:
                    action, _ = self.model.predict(
                        obs, action_masks=action_mask, deterministic=True
                    )
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = self.env.step_with_policy(
                    self.model
                )
                done = terminated or truncated
                episode_reward += reward

            total_reward += episode_reward

            winner = info.get("winner")
            training_identity = self.env.players[0].identity if self.env.players else ""

            identity_totals[training_identity] = (
                identity_totals.get(training_identity, 0) + 1
            )

            if winner and self._check_win(training_identity, winner):
                wins += 1
                identity_wins[training_identity] = (
                    identity_wins.get(training_identity, 0) + 1
                )

        win_rate = wins / n_games
        avg_reward = total_reward / n_games

        identity_rates = {}
        for identity in identity_totals:
            if identity_totals[identity] > 0:
                identity_rates[identity] = (
                    identity_wins[identity] / identity_totals[identity]
                )

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "identity_win_rates": identity_rates,
            "games": n_games,
        }

    def _update_opponents(self):
        """更新对手配置"""
        print(f"Updating opponents at step {self._step_count}")

        # 应用ELO衰减
        for policy in self.policy_pool:
            policy.apply_elo_decay(
                self.config.elo_decay_rate, self.config.elo_max_decay
            )
        self.policy_pool._save_pool()

        if len(self.policy_pool) > 0:
            sampled_policy = self.policy_pool.sample_policy()
            if sampled_policy:
                print(
                    f"  Sampled policy v{sampled_policy.version} "
                    f"(ELO: {sampled_policy.get_effective_elo():.0f}, "
                    f"decay: {sampled_policy.elo_decay_factor:.2f})"
                )

                # 打印采样分布统计
                sampling_dist = self.policy_pool.get_sampling_distribution()
                print(
                    f"  Sampling distribution: "
                    f"{sampling_dist['latest_ratio']:.0%} latest, "
                    f"{sampling_dist['best_ratio']:.0%} best, "
                    f"{sampling_dist['random_ratio']:.0%} random"
                )

    def _log_episode(self, episode_reward: float, episode_length: int, info: Dict):
        if self._episode_count % 100 == 0:
            win_rate = self._win_count / max(self._episode_count, 1)
            avg_reward = self._total_reward / max(self._episode_count, 1)

            mode_str = self._current_mode
            sp_rate = (
                self._self_play_win_count / max(self._self_play_episodes, 1)
                if self._self_play_episodes > 0
                else 0
            )
            rb_rate = (
                self._rule_based_win_count / max(self._rule_based_episodes, 1)
                if self._rule_based_episodes > 0
                else 0
            )

            print(
                f"Episode {self._episode_count} | "
                f"Step {self._step_count} | "
                f"Mode: {mode_str} | "
                f"Win rate: {win_rate:.2%} | "
                f"SP: {sp_rate:.2%} | "
                f"RB: {rb_rate:.2%} | "
                f"Avg reward: {avg_reward:.2f}"
            )

    def _save_metrics(self):
        metrics_path = self.log_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._training_metrics, f, indent=2)

        # 保存ELO追踪
        elo_path = self.log_dir / "elo_tracking.json"
        with open(elo_path, "w") as f:
            json.dump(self._elo_tracking, f, indent=2)

        # 保存比赛历史
        self.match_history.save(str(self.log_dir / "match_history.json"))

    def _save_final_model(self):
        final_path = self.log_dir / "final_model"
        self.model.save(str(final_path))

        self.policy_pool.add_policy(
            str(final_path),
            metadata={
                "training_steps": self._step_count,
                "episode_count": self._episode_count,
                "final_win_rate": self._win_count / max(self._episode_count, 1),
            },
        )

        print(f"Final model saved to {final_path}")

        # 保存配置摘要
        config_summary = {
            "total_timesteps": self._step_count,
            "total_episodes": self._episode_count,
            "final_win_rate": self._win_count / self._episode_count,
            "training_mode": self.config.training_mode.value,
            "self_play_ratio": self.config.self_play_ratio,
            "sampling_distribution": {
                "latest": self.config.sample_latest_ratio,
                "best": self.config.sample_best_ratio,
                "random": self.config.sample_random_ratio,
            },
            "elo_config": {
                "k_factor": self.config.elo_k_factor,
                "decay_rate": self.config.elo_decay_rate,
                "max_decay": self.config.elo_max_decay,
            },
            "policy_pool_stats": self.policy_pool.get_stats(),
        }

        with open(self.log_dir / "training_summary.json", "w") as f:
            json.dump(config_summary, f, indent=2)

    def load_model(self, path: str):
        if MASKABLE_PPO_AVAILABLE:
            try:
                self.model = MaskablePPO.load(path, env=self.env)
                print(f"Loaded MaskablePPO model from {path}")
                return
            except Exception:
                pass

        self.model = PPO.load(path, env=self.env)
        print(f"Loaded PPO model from {path}")


def run_self_play(
    total_timesteps: int = 10_000_000,
    player_num: int = 5,
    log_dir: str = None,
    training_mode: str = "mixed",
    self_play_ratio: float = 0.5,
    **kwargs,
):
    """
    运行自博弈训练

    Args:
            total_timesteps: 总训练步数
            player_num: 玩家数量
            log_dir: 日志目录
            training_mode: 训练模式 ("self_play_only", "rule_based_only", "mixed", "alternating")
            self_play_ratio: 自博弈比例(0-1)
            **kwargs: 其他配置参数
    """
    # 解析训练模式
    mode_map = {
        "self_play_only": TrainingMode.SELF_PLAY_ONLY,
        "rule_based_only": TrainingMode.RULE_BASED_ONLY,
        "mixed": TrainingMode.MIXED,
        "alternating": TrainingMode.ALTERNATING,
    }

    config = SelfPlayConfig(
        total_timesteps=total_timesteps,
        player_num=player_num,
        log_dir=log_dir,
        training_mode=mode_map.get(training_mode, TrainingMode.MIXED),
        self_play_ratio=self_play_ratio,
        rule_based_ratio=1.0 - self_play_ratio,
        **kwargs,
    )

    trainer = SelfPlayTrainer(config)
    trainer.train()

    return trainer.model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-play training for SGS")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--player-num", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)

    # 新增参数
    parser.add_argument(
        "--training-mode",
        type=str,
        default="mixed",
        choices=["self_play_only", "rule_based_only", "mixed", "alternating"],
        help="Training mode: self_play_only, rule_based_only, mixed, or alternating",
    )
    parser.add_argument(
        "--self-play-ratio",
        type=float,
        default=0.5,
        help="Ratio of self-play training (0.0-1.0)",
    )
    parser.add_argument(
        "--sample-latest-ratio",
        type=float,
        default=0.3,
        help="Ratio for sampling latest policies",
    )
    parser.add_argument(
        "--sample-best-ratio",
        type=float,
        default=0.3,
        help="Ratio for sampling best ELO policies",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=float,
        default=32.0,
        help="K factor for ELO rating updates",
    )
    parser.add_argument(
        "--elo-decay-rate", type=float, default=0.01, help="ELO decay rate per hour"
    )

    args = parser.parse_args()

    run_self_play(
        total_timesteps=args.timesteps,
        player_num=args.player_num,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        seed=args.seed,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        training_mode=args.training_mode,
        self_play_ratio=args.self_play_ratio,
        sample_latest_ratio=args.sample_latest_ratio,
        sample_best_ratio=args.sample_best_ratio,
        elo_k_factor=args.elo_k_factor,
        elo_decay_rate=args.elo_decay_rate,
    )
