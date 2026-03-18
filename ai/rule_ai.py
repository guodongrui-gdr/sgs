"""
规则AI - 使用启发式规则进行决策

用于:
1. 早期训练提供有意义的对手
2. 评估强化学习模型的性能
3. 提供基准对比
"""

import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from engine.game_engine import GameEngine
from player.player import Player
from cards.card import Card
from ai.action_encoder import ActionEncoder, ActionConfig, HierarchicalAction
from ai.action_encoder import ActionType


@dataclass
class RuleAIConfig:
    aggressiveness: float = 0.7
    defensiveness: float = 0.5
    smart_targeting: bool = True
    save_cards: bool = True


class RuleAI:
    """
    基于规则的AI

    启发式决策:
    - 根据身份选择目标
    - 优先使用强力卡牌
    - 保留防御牌
    - 智能响应
    """

    def __init__(self, config: RuleAIConfig = None):
        self.config = config or RuleAIConfig()
        self.action_encoder = ActionEncoder(ActionConfig())

    def select_action(
        self,
        player: Player,
        game_state: Dict,
        action_masks: Tuple,
    ) -> int:
        """
        选择动作

        Args:
            player: 当前玩家
            game_state: 游戏状态字典
            action_masks: (type_mask, card_mask, target_mask)

        Returns:
            动作索引
        """
        type_mask, card_mask, target_mask = action_masks

        valid_types = [i for i, m in enumerate(type_mask) if m > 0]

        if not valid_types:
            return 0

        if ActionType.END_TURN in valid_types and self._should_end_turn(
            player, game_state
        ):
            return ActionType.END_TURN

        if ActionType.USE_CARD in valid_types and self._can_use_card(
            player, game_state, card_mask
        ):
            return ActionType.USE_CARD

        if ActionType.USE_SKILL in valid_types and self._can_use_skill(
            player, game_state
        ):
            return ActionType.USE_SKILL

        if ActionType.EQUIP in valid_types and self._has_equipment(player):
            return ActionType.EQUIP

        if ActionType.END_TURN in valid_types:
            return ActionType.END_TURN

        return valid_types[0]

    def select_card(
        self,
        player: Player,
        game_state: Dict,
        card_mask: List,
    ) -> int:
        """选择要使用的卡牌"""
        valid_cards = [i for i, m in enumerate(card_mask) if m > 0]

        if not valid_cards:
            return 0

        hand_cards = player.hand_cards
        card_scores = []

        for idx in valid_cards:
            if idx >= len(hand_cards):
                continue

            card = hand_cards[idx]
            score = self._evaluate_card(card, player, game_state)
            card_scores.append((idx, score))

        if card_scores:
            card_scores.sort(key=lambda x: x[1], reverse=True)
            return card_scores[0][0]

        return valid_cards[0]

    def select_target(
        self,
        player: Player,
        game_state: Dict,
        target_mask: List,
        card: Card = None,
    ) -> int:
        """选择目标"""
        valid_targets = [i for i, m in enumerate(target_mask) if m > 0]

        if not valid_targets:
            return 0

        if len(valid_targets) == 1:
            return valid_targets[0]

        target_scores = []
        for idx in valid_targets:
            target = (
                game_state.get("players", [])[idx]
                if idx < len(game_state.get("players", []))
                else None
            )
            if target:
                score = self._evaluate_target(player, target, card, game_state)
                target_scores.append((idx, score))

        if target_scores:
            target_scores.sort(key=lambda x: x[1], reverse=True)
            return target_scores[0][0]

        return valid_targets[0]

    def _should_end_turn(self, player: Player, game_state: Dict) -> bool:
        """判断是否应该结束回合"""
        hand_count = len(player.hand_cards)
        hp = player.current_hp

        if hand_count <= hp:
            return True

        if hand_count == 0:
            return True

        return random.random() < 0.3

    def _can_use_card(self, player: Player, game_state: Dict, card_mask: List) -> bool:
        """判断是否能使用卡牌"""
        return any(m > 0 for m in card_mask)

    def _can_use_skill(self, player: Player, game_state: Dict) -> bool:
        """判断是否能使用技能"""
        return len(player.skills) > 0 and random.random() < 0.3

    def _has_equipment(self, player: Player) -> bool:
        """判断是否有装备牌"""
        for card in player.hand_cards:
            if hasattr(card, "card_type") and "Weapon" in str(type(card)):
                return True
        return False

    def _evaluate_card(self, card: Card, player: Player, game_state: Dict) -> float:
        """评估卡牌价值"""
        score = 0.0
        card_name = card.name if hasattr(card, "name") else str(card)

        attack_cards = ["杀", "火杀", "雷杀", "决斗", "火攻"]
        defense_cards = ["闪", "桃", "酒"]
        utility_cards = ["无中生有", "过河拆桥", "顺手牵羊", "五谷丰登"]
        aoe_cards = ["南蛮入侵", "万箭齐发", "桃园结义"]

        if card_name in attack_cards:
            score = 10.0 * self.config.aggressiveness
            if player.identity in ["反贼", "内奸"]:
                score *= 1.2
        elif card_name in defense_cards:
            hp_ratio = player.current_hp / player.max_hp
            score = 8.0 * (1 - hp_ratio) * self.config.defensiveness
            if card_name == "桃" and hp_ratio < 0.5:
                score *= 1.5
        elif card_name in utility_cards:
            score = 7.0
            if card_name == "无中生有":
                score *= 1.3
        elif card_name in aoe_cards:
            alive_count = sum(
                1 for p in game_state.get("players", []) if p.get("is_alive", True)
            )
            score = 6.0 + alive_count * 0.5
        else:
            score = 5.0

        return score

    def _evaluate_target(
        self,
        player: Player,
        target: Dict,
        card: Card,
        game_state: Dict,
    ) -> float:
        """评估目标价值"""
        if not target:
            return 0.0

        score = 0.0
        player_identity = player.identity
        target_identity = target.get("identity", "")

        is_enemy = self._is_enemy(player_identity, target_identity)
        is_ally = self._is_ally(player_identity, target_identity)

        card_name = card.name if card and hasattr(card, "name") else ""
        is_attack = card_name in ["杀", "火杀", "雷杀", "决斗", "火攻"]
        is_heal = card_name in ["桃"]

        if is_attack:
            if is_enemy:
                score = 10.0
                hp_ratio = target.get("current_hp", 4) / target.get("max_hp", 4)
                if hp_ratio < 0.3:
                    score *= 1.5
            elif is_ally:
                score = -10.0
            else:
                score = 5.0
        elif is_heal:
            if is_ally or target_identity == player_identity:
                score = 10.0
                hp_ratio = target.get("current_hp", 4) / target.get("max_hp", 4)
                if hp_ratio < 0.5:
                    score *= 1.5
            else:
                score = -5.0
        else:
            if is_enemy:
                score = 7.0
            elif is_ally:
                score = 3.0
            else:
                score = 5.0

        return score

    def _is_enemy(self, identity_a: str, identity_b: str) -> bool:
        """判断是否是敌人"""
        if identity_a in ["主公", "忠臣"] and identity_b == "反贼":
            return True
        if identity_a == "反贼" and identity_b in ["主公", "忠臣"]:
            return True
        if identity_a == "内奸" or identity_b == "内奸":
            return identity_a != identity_b
        return False

    def _is_ally(self, identity_a: str, identity_b: str) -> bool:
        """判断是否是队友"""
        if identity_a == identity_b:
            return True
        if identity_a in ["主公", "忠臣"] and identity_b in ["主公", "忠臣"]:
            return True
        return False


class HeuristicAI(RuleAI):
    """
    更智能的启发式AI

    增加功能:
    - 手牌管理
    - 仇恨值系统
    - 局势判断
    """

    def __init__(self, config: RuleAIConfig = None):
        super().__init__(config)
        self.hate_values: Dict[int, float] = {}

    def update_hate(self, attacker_idx: int, damage: int, target_idx: int):
        """更新仇恨值"""
        if target_idx not in self.hate_values:
            self.hate_values[target_idx] = {}
        self.hate_values[target_idx][attacker_idx] = (
            self.hate_values[target_idx].get(attacker_idx, 0) + damage * 2
        )

    def get_most_hated_target(
        self, player_idx: int, valid_targets: List[int]
    ) -> Optional[int]:
        """获取仇恨最高的目标"""
        if player_idx not in self.hate_values:
            return None

        hate_dict = self.hate_values[player_idx]
        valid_hate = [(t, hate_dict.get(t, 0)) for t in valid_targets if t in hate_dict]

        if valid_hate:
            valid_hate.sort(key=lambda x: x[1], reverse=True)
            return valid_hate[0][0]

        return None


def create_rule_ai(config: RuleAIConfig = None) -> RuleAI:
    """创建规则AI实例"""
    return RuleAI(config)


def create_heuristic_ai(config: RuleAIConfig = None) -> HeuristicAI:
    """创建启发式AI实例"""
    return HeuristicAI(config)
