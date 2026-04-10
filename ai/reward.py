"""
奖励函数 - 身份感知的奖励系统

奖励设计:
├── 终局奖励: ±100 (胜利/失败)
├── 伤害奖励: ±1~2 / 点
├── 击杀奖励: ±15~50
├── 救援奖励: +5
└── 身份特定奖励

Phase 2 优化 (Reward Refinement):
- 增加终端奖励主导性 (~10x intermediate rewards)
- 添加更多中间奖励减少稀疏性
- 潜势能塑形参数可配置
- 奖励日志记录和分析
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Identity(Enum):
    """身份枚举"""

    LORD = "主公"
    LOYALIST = "忠臣"
    REBEL = "反贼"
    SPY = "内奸"


# ============================================================================
# REWARD LOGGING SYSTEM
# ============================================================================


@dataclass
class RewardLogConfig:
    """奖励日志配置"""

    log_dir: str = "./logs/rewards"
    log_frequency: int = 100  # Log every N episodes
    save_detailed_logs: bool = True
    track_sparse_rewards: bool = True


class RewardLogger:
    """
    奖励日志记录器

    跟踪奖励分布、频率和幅度，用于分析和优化奖励函数

    功能:
    - 每回合奖励统计
    - 非零奖励频率追踪
    - 奖励幅度分布
    - 步数间隔分析
    - 稀疏奖励检测
    """

    def __init__(self, config: RewardLogConfig = None):
        self.config = config or RewardLogConfig()

        # Episode-level tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_reward_records: List[List[RewardRecord]] = []

        # Step-level tracking
        self.step_rewards: List[float] = []
        self.non_zero_reward_steps: List[int] = []
        self.steps_between_rewards: List[int] = []
        self._last_reward_step: int = -1

        # Reward distribution tracking
        self.reward_by_type: Dict[str, List[float]] = defaultdict(list)
        self.reward_magnitudes: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0, "sum": 0.0}
        )

        # Episode counter
        self._episode_count: int = 0
        self._current_episode_rewards: List[RewardRecord] = []
        self._current_episode_step: int = 0

        # Sparse reward metrics
        self.sparse_reward_ratio: float = 0.0
        self.avg_steps_between_rewards: float = 0.0

        # Terminal vs intermediate reward tracking
        self.terminal_rewards: List[float] = []
        self.intermediate_rewards: List[float] = []

    def log_reward(self, record: RewardRecord, step: int = 0):
        """记录单个奖励"""
        self._current_episode_rewards.append(record)
        self._current_episode_step = step

        # Track step-level rewards
        self.step_rewards.append(record.final_reward)

        if record.final_reward != 0.0:
            self.non_zero_reward_steps.append(step)

            # Calculate steps between rewards
            if self._last_reward_step >= 0:
                gap = step - self._last_reward_step
                self.steps_between_rewards.append(gap)
            self._last_reward_step = step

        # Track by event type
        self.reward_by_type[record.event_type].append(record.final_reward)

        # Update magnitude statistics
        stats = self.reward_magnitudes[record.event_type]
        stats["count"] += 1
        stats["sum"] += record.final_reward
        stats["min"] = min(stats["min"], record.final_reward)
        stats["max"] = max(stats["max"], record.final_reward)
        stats["mean"] = stats["sum"] / stats["count"]

        # Track terminal vs intermediate
        if record.event_type == "game_over":
            self.terminal_rewards.append(record.final_reward)
        else:
            self.intermediate_rewards.append(record.final_reward)

    def end_episode(self, episode_reward: float, episode_length: int):
        """结束回合并记录统计"""
        self._episode_count += 1

        # Store episode data
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_reward_records.append(self._current_episode_rewards.copy())

        # Reset for next episode
        self._current_episode_rewards = []
        self._current_episode_step = 0

        # Update sparse reward metrics
        self._update_sparse_metrics()

        # Periodic logging
        if self._episode_count % self.config.log_frequency == 0:
            self._log_summary()

    def _update_sparse_metrics(self):
        """更新稀疏奖励指标"""
        total_steps = sum(self.episode_lengths)
        non_zero_count = len(self.non_zero_reward_steps)

        if total_steps > 0:
            self.sparse_reward_ratio = non_zero_count / total_steps

        if self.steps_between_rewards:
            self.avg_steps_between_rewards = np.mean(self.steps_between_rewards)

    def _log_summary(self):
        """输出摘要日志"""
        if not self.episode_rewards:
            return

        avg_reward = np.mean(self.episode_rewards[-self.config.log_frequency :])
        avg_length = np.mean(self.episode_lengths[-self.config.log_frequency :])

        logger.info(
            f"[RewardLogger] Episode {self._episode_count}: "
            f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}, "
            f"sparse_ratio={self.sparse_reward_ratio:.3f}, "
            f"avg_steps_between_rewards={self.avg_steps_between_rewards:.1f}"
        )

        # Log terminal vs intermediate ratio
        if self.terminal_rewards and self.intermediate_rewards:
            terminal_avg = np.mean(self.terminal_rewards)
            intermediate_avg = np.mean(
                [r for r in self.intermediate_rewards if r != 0.0] or [0.0]
            )
            if intermediate_avg != 0:
                terminal_ratio = abs(terminal_avg / intermediate_avg)
                logger.info(
                    f"[RewardLogger] Terminal/Intermediate ratio: {terminal_ratio:.1f}x "
                    f"(terminal_avg={terminal_avg:.2f}, intermediate_avg={intermediate_avg:.2f})"
                )

    def get_analysis(self) -> Dict:
        """获取奖励分析数据"""
        analysis = {
            "episode_count": self._episode_count,
            "avg_episode_reward": np.mean(self.episode_rewards)
            if self.episode_rewards
            else 0.0,
            "std_episode_reward": np.std(self.episode_rewards)
            if self.episode_rewards
            else 0.0,
            "avg_episode_length": np.mean(self.episode_lengths)
            if self.episode_lengths
            else 0.0,
            "sparse_reward_ratio": self.sparse_reward_ratio,
            "avg_steps_between_rewards": self.avg_steps_between_rewards,
            "total_steps": sum(self.episode_lengths),
            "non_zero_reward_count": len(self.non_zero_reward_steps),
            "reward_by_type": {
                k: {
                    "count": len(v),
                    "mean": np.mean(v),
                    "std": np.std(v),
                    "min": np.min(v),
                    "max": np.max(v),
                }
                for k, v in self.reward_by_type.items()
            },
            "terminal_rewards": {
                "count": len(self.terminal_rewards),
                "mean": np.mean(self.terminal_rewards)
                if self.terminal_rewards
                else 0.0,
                "std": np.std(self.terminal_rewards) if self.terminal_rewards else 0.0,
            },
            "intermediate_rewards": {
                "count": len(self.intermediate_rewards),
                "non_zero_count": len(
                    [r for r in self.intermediate_rewards if r != 0.0]
                ),
                "mean": np.mean(self.intermediate_rewards)
                if self.intermediate_rewards
                else 0.0,
            },
        }

        # Calculate terminal dominance ratio
        if analysis["intermediate_rewards"]["mean"] != 0:
            analysis["terminal_dominance_ratio"] = abs(
                analysis["terminal_rewards"]["mean"]
                / analysis["intermediate_rewards"]["mean"]
            )
        else:
            analysis["terminal_dominance_ratio"] = float("inf")

        return analysis

    def save_logs(self, filename: str = None):
        """保存日志到文件"""
        if not self.config.save_detailed_logs:
            return

        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"reward_log_episode_{self._episode_count}.json"

        filepath = log_dir / filename

        analysis = self.get_analysis()

        # Add detailed records
        analysis["detailed_episode_rewards"] = [
            {
                "episode": i,
                "total_reward": r,
                "length": self.episode_lengths[i],
                "records": [
                    {
                        "event_type": rec.event_type,
                        "base_reward": rec.base_reward,
                        "final_reward": rec.final_reward,
                    }
                    for rec in records
                ],
            }
            for i, (r, records) in enumerate(
                zip(self.episode_rewards, self.episode_reward_records)
            )
        ]

        with open(filepath, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"[RewardLogger] Saved reward logs to {filepath}")

    def reset(self):
        """重置所有日志数据"""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_reward_records.clear()
        self.step_rewards.clear()
        self.non_zero_reward_steps.clear()
        self.steps_between_rewards.clear()
        self._last_reward_step = -1
        self.reward_by_type.clear()
        self.reward_magnitudes.clear()
        self._episode_count = 0
        self._current_episode_rewards.clear()
        self._current_episode_step = 0
        self.terminal_rewards.clear()
        self.intermediate_rewards.clear()


# ============================================================================
# REWARD CONFIGURATION
# ============================================================================


@dataclass
class RewardConfig:
    """
    奖励配置 - Phase 2 优化

    设计原则:
    1. 终端奖励主导性: terminal rewards ~10x intermediate
    2. 中间奖励密度: 减少稀疏性，提供更多学习信号
    3. 奖励塑形: 潜势能函数γ可配置
    4. 平衡性: 激励攻击但不忽视生存
    """

    # Terminal rewards (should be ~10x intermediate)
    victory: float = 100.0  # Phase 2: Increased from 50.0
    defeat: float = -100.0  # Phase 2: Increased from -50.0

    # Damage rewards (intermediate)
    damage_dealt: float = 3.0  # Phase 2: Adjusted balance
    damage_taken: float = -1.5  # Phase 2: Adjusted balance

    # Kill rewards (intermediate, but significant)
    kill_enemy: float = 15.0
    kill_ally: float = -20.0
    lord_kill_loyalist: float = -30.0

    # Survival rewards (dense intermediate)
    survive_per_turn: float = 0.5

    # Spy-specific rewards
    spy_last_survive: float = 10.0  # Phase 2: Increased

    # Card usage rewards (dense intermediate)
    use_sha_reward: float = 0.3  # Phase 2: Increased for denser rewards
    use_tao_reward: float = 0.5
    use_taoyuan_reward: float = 1.0
    use_nanman_reward: float = 1.5

    # Phase 2 新增: 进度奖励
    progress_reward: float = 0.3
    defense_reward: float = 0.2

    # Phase 2 新增: 技能激活奖励 (增加中间奖励密度)
    skill_activation_reward: float = 0.2
    effective_card_use_reward: float = 0.5  # 有效卡牌使用奖励

    # Phase 2 新增: 游戏进度奖励
    turn_progress_reward: float = 0.1  # 每回合进度
    round_progress_reward: float = 0.2  # 每轮进度

    # Phase 2 新增: 队友保护奖励
    protect_lord_reward: float = 2.0  # 保护主公
    assist_ally_reward: float = 1.0  # 协助队友

    # Reward clipping
    clip_reward: float = 50.0  # Phase 2: Adjusted

    # Phase 2 新增: 潜势能塑形参数
    shaping_gamma: float = 0.99  # 控制潜势能影响强度

    # Logging configuration
    log_rewards: bool = False
    log_config: Optional[RewardLogConfig] = None

    def get_terminal_to_intermediate_ratio(self) -> float:
        """计算终端/中间奖励比例"""
        # 使用平均中间奖励 (非零)
        avg_intermediate = abs(self.damage_dealt)  # 使用damage_dealt作为基准
        return abs(self.victory) / avg_intermediate

    def validate_config(self) -> Dict[str, str]:
        """验证配置合理性"""
        issues = {}

        ratio = self.get_terminal_to_intermediate_ratio()
        if ratio < 5:
            issues["terminal_ratio"] = (
                f"Terminal ratio {ratio:.1f}x is low (< 5x). Consider increasing victory/defeat."
            )
        elif ratio > 20:
            issues["terminal_ratio"] = (
                f"Terminal ratio {ratio:.1f}x is very high (> 20x). May cause sparse reward issues."
            )

        if abs(self.victory) < abs(self.defeat):
            issues["victory_defeat_balance"] = (
                "Victory reward less than defeat penalty may discourage winning."
            )

        if self.damage_dealt > self.kill_enemy / 3:
            issues["damage_kill_balance"] = (
                "Damage reward too high relative to kill reward."
            )

        return issues


@dataclass
class RewardRecord:
    """奖励记录"""

    event_type: str
    base_reward: float
    shaped_reward: float
    final_reward: float
    context: Dict = field(default_factory=dict)


class IdentityRelationship:
    """身份关系判断"""

    @staticmethod
    def get_relationship(identity_a: str, identity_b: str) -> str:
        """
        判断两个玩家的身份关系

        Returns:
                "ally": 队友
                "enemy": 敌人
                "neutral": 中立
        """
        # 主公-忠臣: 队友
        if {identity_a, identity_b} == {"主公", "忠臣"}:
            return "ally"

        # 主公-主公: 不可能
        if identity_a == "主公" and identity_b == "主公":
            return "ally"

        # 忠臣-忠臣: 队友
        if identity_a == "忠臣" and identity_b == "忠臣":
            return "ally"

        # 主公/忠臣 vs 反贼: 敌人
        if (identity_a in ["主公", "忠臣"] and identity_b == "反贼") or (
            identity_b in ["主公", "忠臣"] and identity_a == "反贼"
        ):
            return "enemy"

        # 反贼之间: 队友
        if identity_a == "反贼" and identity_b == "反贼":
            return "ally"

        # 内奸: 与所有人敌对
        if identity_a == "内奸" or identity_b == "内奸":
            if identity_a == "内奸" and identity_b == "内奸":
                return "ally"
            return "enemy"

        return "neutral"

    @staticmethod
    def is_victory(identity: str, winner: str) -> bool:
        """判断玩家是否胜利"""
        if winner == "主公":
            return identity in ["主公", "忠臣"]
        elif winner == "反贼":
            return identity == "反贼"
        elif winner == "内奸":
            return identity == "内奸"
        return False


class RewardCalculator:
    """奖励计算器"""

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.records: List[RewardRecord] = []

    def calculate_reward(
        self,
        event_type: str,
        source_identity: str,
        target_identity: str,
        current_identity: str,
        is_source: bool = False,
        is_target: bool = False,
        value: float = 1.0,
        context: Dict = None,
    ) -> float:
        """
        计算奖励

        Args:
                event_type: 事件类型
                source_identity: 事件来源身份
                target_identity: 事件目标身份
                current_identity: 当前AI玩家身份
                is_source: 当前玩家是否是事件来源
                is_target: 当前玩家是否是事件目标
                value: 事件数值 (伤害量等)
                context: 额外上下文

        Returns:
                奖励值
        """
        context = context or {}
        base_reward = 0.0

        # 获取身份关系
        if is_source:
            relationship = IdentityRelationship.get_relationship(
                current_identity, target_identity
            )
        elif is_target:
            relationship = IdentityRelationship.get_relationship(
                source_identity, current_identity
            )
        else:
            relationship = "neutral"

        # 根据事件类型计算奖励
        if event_type == "damage_dealt":
            if is_source:
                base_reward = self.config.damage_dealt * value

        elif event_type == "damage_taken":
            if is_target:
                base_reward = self.config.damage_taken * value

        elif event_type == "player_killed":
            if is_source:
                relationship = IdentityRelationship.get_relationship(
                    current_identity, target_identity
                )
                if relationship == "enemy":
                    base_reward = self.config.kill_enemy
                elif relationship == "ally":
                    base_reward = self.config.kill_ally
                    # 主公杀忠臣额外惩罚
                    if current_identity == "主公" and target_identity == "忠臣":
                        base_reward += self.config.lord_kill_loyalist

        elif event_type == "turn_survive":
            if is_target:
                base_reward = self.config.survive_per_turn

        elif event_type == "game_over":
            winner = context.get("winner")
            if winner and IdentityRelationship.is_victory(current_identity, winner):
                base_reward = self.config.victory
            else:
                base_reward = self.config.defeat

            if current_identity == "内奸":
                survivors = context.get("survivors", [])
                if winner == "内奸":
                    base_reward = self.config.victory * 1.5
                elif len(survivors) == 1:
                    survivor = survivors[0]
                    survivor_identity = (
                        survivor.identity
                        if hasattr(survivor, "identity")
                        else survivor.get("identity", "")
                    )
                    if survivor_identity == "内奸":
                        base_reward = self.config.spy_last_survive
            return base_reward

        elif event_type == "skill_activation":
            if is_source:
                base_reward = self.config.skill_activation_reward

        elif event_type == "skill_decision_yes_no":
            if is_source:
                activated = context.get("activated", True)
                is_valuable = context.get("is_valuable", True)
                if activated and is_valuable:
                    base_reward = (
                        self.config.skill_activation_reward * 2.5
                    )  # 0.5 for activating valuable skill
                elif not activated and is_valuable:
                    base_reward = (
                        -self.config.skill_activation_reward
                    )  # Penalty for declining valuable skill
                elif activated and not is_valuable:
                    base_reward = (
                        self.config.skill_activation_reward * 0.5
                    )  # Small reward for activating neutral skill

        elif event_type == "skill_decision_select_order":
            if is_source:
                quality_score = context.get("quality_score", 0.5)
                # Reward proportional to quality (0.2 to 2.0 range)
                base_reward = self.config.skill_activation_reward + (
                    quality_score * 1.5
                )

        elif event_type == "skill_decision_distribute":
            if is_source:
                ally_target = context.get("ally_target", True)
                cards_count = value if value else 1
                if ally_target:
                    base_reward = (
                        self.config.skill_activation_reward * 2 * cards_count
                    )  # Reward for giving to allies
                else:
                    base_reward = (
                        -self.config.skill_activation_reward
                    )  # Penalty for giving to enemies

        # 归一化
        final_reward = self._normalize(base_reward)

        # 记录
        self.records.append(
            RewardRecord(
                event_type=event_type,
                base_reward=base_reward,
                shaped_reward=base_reward,  # 后续可添加塑形
                final_reward=final_reward,
                context=context,
            )
        )

        return final_reward

    def _normalize(self, reward: float) -> float:
        """归一化奖励 - 使用配置的裁剪范围"""
        return np.clip(reward, -self.config.clip_reward, self.config.clip_reward)

    def reset(self):
        """重置记录"""
        self.records.clear()

    def get_total_reward(self) -> float:
        """获取累计奖励"""
        return sum(r.final_reward for r in self.records)

    def get_recent_rewards(self, n: int = 10) -> List[float]:
        """获取最近N条奖励"""
        return [r.final_reward for r in self.records[-n:]]


class SpyRewardCalculator(RewardCalculator):
    """内奸专用奖励计算器"""

    def calculate_reward(
        self,
        event_type: str,
        source_identity: str,
        target_identity: str,
        current_identity: str,
        is_source: bool = False,
        is_target: bool = False,
        value: float = 1.0,
        context: Dict = None,
    ) -> float:
        """内奸特殊奖励计算"""

        # 先用基础计算
        base_reward = super().calculate_reward(
            event_type,
            source_identity,
            target_identity,
            current_identity,
            is_source,
            is_target,
            value,
            context,
        )

        context = context or {}

        # 内奸特殊逻辑
        if current_identity != "内奸":
            return base_reward

        if event_type == "game_over":
            winner = context.get("winner")
            if winner == "内奸":
                return self._normalize(self.config.victory * 1.5)
            elif winner == "反贼":
                return self._normalize(self.config.defeat * 1.5)

        elif event_type == "player_killed":
            if is_source:
                # 内奸击杀奖励根据局势调整
                alive_count = context.get("alive_count", 5)
                rebels = context.get("rebels", 0)
                loyalists = context.get("loyalists", 0)

                # 前期：鼓励平衡
                if alive_count > 3:
                    if rebels > loyalists and target_identity == "反贼":
                        base_reward *= 1.5
                    elif loyalists > rebels and target_identity == "忠臣":
                        base_reward *= 1.5

                # 后期：击杀任何人都是正向
                else:
                    if base_reward < 0:
                        base_reward = abs(base_reward) * 0.5

        return self._normalize(base_reward)


class PotentialBasedReward:
    """基于势能的奖励塑形"""

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma
        self.prev_potential = 0.0

    def calculate_potential(self, state: Dict, player_idx: int) -> float:
        potential = 0.0

        players = state.get("players", [])
        if player_idx >= len(players):
            return 0.0

        player = players[player_idx]

        hp_ratio = player.get("current_hp", 0) / max(player.get("max_hp", 1), 1)
        potential += hp_ratio * 5.0

        equipment = player.get("equipment", {})
        equip_count = sum(1 for v in equipment.values() if v)
        potential += equip_count * 1.0

        return potential

    def _count_allies(self, players: List, identity: str) -> int:
        return 0

    def _count_enemies(self, players: List, identity: str) -> int:
        return 0

    def get_shaped_reward(
        self,
        base_reward: float,
        state: Dict,
        player_idx: int,
    ) -> float:
        """
        获取塑形后的奖励
        shaped_reward = base_reward + gamma * potential(s') - potential(s)
        """
        current_potential = self.calculate_potential(state, player_idx)
        shaped_reward = (
            base_reward + self.gamma * current_potential - self.prev_potential
        )
        self.prev_potential = current_potential
        return shaped_reward

    def reset(self):
        """重置势能"""
        self.prev_potential = 0.0


class RewardSystem:
    """
    完整奖励系统 - Phase 2优化

    整合奖励计算、塑形和日志记录
    """

    def __init__(
        self,
        config: RewardConfig = None,
        use_shaping: bool = True,
        use_logging: bool = False,
    ):
        self.config = config or RewardConfig()
        self.calculator = RewardCalculator(self.config)

        # Initialize shaping with configurable gamma
        shaping_gamma = getattr(self.config, "shaping_gamma", 0.99)
        self.shaping = (
            PotentialBasedReward(gamma=shaping_gamma) if use_shaping else None
        )

        # Initialize logger if configured
        self.logger = None
        if use_logging or self.config.log_rewards:
            log_config = self.config.log_config or RewardLogConfig()
            self.logger = RewardLogger(log_config)

        self._step_counter: int = 0

    def get_reward(
        self,
        event_type: str,
        source_identity: str,
        target_identity: str,
        current_identity: str,
        is_source: bool = False,
        is_target: bool = False,
        value: float = 1.0,
        state: Dict = None,
        player_idx: int = 0,
        context: Dict = None,
    ) -> float:
        """获取最终奖励"""
        # 基础奖励
        base_reward = self.calculator.calculate_reward(
            event_type,
            source_identity,
            target_identity,
            current_identity,
            is_source,
            is_target,
            value,
            context,
        )

        # 奖励塑形
        if self.shaping and state:
            reward = self.shaping.get_shaped_reward(base_reward, state, player_idx)
        else:
            reward = base_reward

        # 记录日志
        if self.logger:
            record = RewardRecord(
                event_type=event_type,
                base_reward=base_reward,
                shaped_reward=reward if self.shaping else base_reward,
                final_reward=reward,
                context=context or {},
            )
            self.logger.log_reward(record, self._step_counter)
            self._step_counter += 1

        return reward

    def end_episode(self, episode_reward: float, episode_length: int):
        """结束回合，记录统计"""
        if self.logger:
            self.logger.end_episode(episode_reward, episode_length)
        self._step_counter = 0

    def reset(self):
        """重置"""
        self.calculator.reset()
        if self.shaping:
            self.shaping.reset()
        self._step_counter = 0

    def get_total_reward(self) -> float:
        """获取累计奖励"""
        return self.calculator.get_total_reward()

    def get_records(self) -> List[RewardRecord]:
        """获取奖励记录"""
        return self.calculator.records.copy()

    def get_reward_analysis(self) -> Dict:
        """获取奖励分析报告"""
        if self.logger:
            return self.logger.get_analysis()
        return {}

    def save_reward_logs(self, filename: str = None):
        """保存奖励日志"""
        if self.logger:
            self.logger.save_logs(filename)

    def validate_config(self) -> Dict[str, str]:
        """验证当前配置"""
        return self.config.validate_config()


# 预定义的奖励事件
class RewardEvent:
    """奖励事件定义"""

    @staticmethod
    def damage_dealt(source, target, damage: int) -> Dict:
        return {
            "event_type": "damage_dealt",
            "source_identity": getattr(source, "identity", ""),
            "target_identity": getattr(target, "identity", ""),
            "value": damage,
        }

    @staticmethod
    def damage_taken(source, target, damage: int) -> Dict:
        return {
            "event_type": "damage_taken",
            "source_identity": getattr(source, "identity", ""),
            "target_identity": getattr(target, "identity", ""),
            "value": damage,
        }

    @staticmethod
    def player_killed(killer, victim) -> Dict:
        return {
            "event_type": "player_killed",
            "source_identity": getattr(killer, "identity", "") if killer else "",
            "target_identity": getattr(victim, "identity", ""),
        }

    @staticmethod
    def player_saved(saver, saved) -> Dict:
        return {
            "event_type": "player_saved",
            "source_identity": getattr(saver, "identity", ""),
            "target_identity": getattr(saved, "identity", ""),
        }

    @staticmethod
    def heal(player, amount: int) -> Dict:
        return {
            "event_type": "heal",
            "target_identity": getattr(player, "identity", ""),
            "value": amount,
        }

    @staticmethod
    def use_card(player) -> Dict:
        return {
            "event_type": "use_card",
            "source_identity": getattr(player, "identity", ""),
        }

    @staticmethod
    def turn_survive(player, lord_alive: bool = True) -> Dict:
        return {
            "event_type": "turn_survive",
            "target_identity": getattr(player, "identity", ""),
            "context": {"lord_alive": lord_alive},
        }

    @staticmethod
    def game_over(winner: str, survivors: List) -> Dict:
        return {
            "event_type": "game_over",
            "context": {
                "winner": winner,
                "survivors": [
                    {"identity": getattr(s, "identity", "")} for s in survivors
                ],
            },
        }
