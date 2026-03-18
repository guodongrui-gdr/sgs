"""
奖励函数 - 身份感知的奖励系统

奖励设计:
├── 终局奖励: ±100 (胜利/失败)
├── 伤害奖励: ±1~2 / 点
├── 击杀奖励: ±15~50
├── 救援奖励: +5
└── 身份特定奖励
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class Identity(Enum):
    """身份枚举"""

    LORD = "主公"
    LOYALIST = "忠臣"
    REBEL = "反贼"
    SPY = "内奸"


@dataclass
class RewardConfig:
    """奖励配置"""

    # 终局奖励
    victory: float = 10.0
    defeat: float = -10.0

    # 伤害奖励 (不使用身份)
    damage_dealt: float = 0.1
    damage_taken: float = -0.1

    # 击杀奖励 (使用目标的公开身份)
    kill_enemy: float = 2.0
    kill_ally: float = -3.0
    lord_kill_loyalist: float = -5.0

    # 存活奖励
    survive_per_turn: float = 0.01

    # 内奸最后存活
    spy_last_survive: float = 5.0


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
        """归一化奖励 - 保持小的中间奖励不变"""
        if abs(reward) < 1.0:
            return reward
        return np.clip(reward, -5.0, 5.0)

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

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.prev_potential = 0.0

    def calculate_potential(self, state: Dict, player_idx: int) -> float:
        """
        计算当前状态的势能 - 不使用隐藏身份信息
        """
        potential = 0.0

        players = state.get("players", [])
        if player_idx >= len(players):
            return 0.0

        player = players[player_idx]

        # 1. 体力势能
        hp_ratio = player.get("current_hp", 0) / max(player.get("max_hp", 1), 1)
        potential += hp_ratio * 10

        # 2. 手牌势能
        hand_count = len(player.get("hand_cards", []))
        hand_limit = player.get("current_hp", 0)
        hand_advantage = hand_count - hand_limit
        potential += hand_advantage * 2

        # 3. 装备势能
        equipment = player.get("equipment", {})
        equip_count = sum(1 for v in equipment.values() if v)
        potential += equip_count * 3

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
    """完整奖励系统"""

    def __init__(
        self,
        config: RewardConfig = None,
        use_shaping: bool = True,
    ):
        self.config = config or RewardConfig()
        self.calculator = RewardCalculator(self.config)
        self.shaping = PotentialBasedReward() if use_shaping else None

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

        return reward

    def reset(self):
        """重置"""
        self.calculator.reset()
        if self.shaping:
            self.shaping.reset()

    def get_total_reward(self) -> float:
        """获取累计奖励"""
        return self.calculator.get_total_reward()

    def get_records(self) -> List[RewardRecord]:
        """获取奖励记录"""
        return self.calculator.records.copy()


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
