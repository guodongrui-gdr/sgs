"""
动作编码器 - 分层动作空间编码

动作空间结构:
├── Step 0: 选择动作类型 - Discrete(12)
├── Step 1: 选择卡牌/技能 - Discrete(20)
└── Step 2: 选择目标 - Discrete(8)

支持动作掩码过滤非法动作
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import IntEnum


class ActionType(IntEnum):
    """动作类型枚举"""

    USE_CARD = 0  # 使用手牌
    END_TURN = 1  # 结束回合
    DISCARD = 2  # 弃置手牌
    RESPOND_SHAN = 3  # 出闪响应
    RESPOND_SHA = 4  # 出杀响应
    RESPOND_TAO = 5  # 出桃救人
    RESPOND_WUXIE = 6  # 无懈可击
    USE_SKILL = 7  # 发动技能
    PASS = 8  # 跳过/不响应
    SELECT_TARGET = 9  # 选择目标 (内部使用)
    JUDGE_MODIFY = 10  # 改判
    CONFIRM = 11  # 确认操作


@dataclass
class ActionConfig:
    """动作配置"""

    num_action_types: int = 12
    max_hand_size: int = 20
    max_players: int = 8
    max_skills: int = 36


@dataclass
class HierarchicalAction:
    """分层动作"""

    action_type: int
    card_idx: Optional[int] = None
    target_idx: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type,
            "card_idx": self.card_idx,
            "target_idx": self.target_idx,
        }


class ActionEncoder:
    """动作编码器"""

    def __init__(self, config: ActionConfig = None):
        self.config = config or ActionConfig()

        # 动作空间维度
        self.action_type_dim = self.config.num_action_types
        self.card_dim = self.config.max_hand_size
        self.target_dim = self.config.max_players

    def get_action_space_dim(self) -> int:
        """获取动作空间维度 (取三个维度的最大值)"""
        return max(self.action_type_dim, self.card_dim, self.target_dim)

    def encode_flat(self, action: HierarchicalAction) -> int:
        """
        将分层动作编码为单一索引 (用于兼容某些RL算法)

        注意: 这种编码方式会浪费很多空间，推荐使用分层编码
        """
        return (
            action.action_type * self.card_dim * self.target_dim
            + (action.card_idx or 0) * self.target_dim
            + (action.target_idx or 0)
        )

    def decode_flat(self, action_idx: int) -> HierarchicalAction:
        """从单一索引解码为分层动作"""
        action_type = action_idx // (self.card_dim * self.target_dim)
        remainder = action_idx % (self.card_dim * self.target_dim)
        card_idx = remainder // self.target_dim
        target_idx = remainder % self.target_dim

        return HierarchicalAction(
            action_type=action_type,
            card_idx=card_idx
            if action_type
            in [
                ActionType.USE_CARD,
                ActionType.DISCARD,
                ActionType.RESPOND_SHAN,
                ActionType.RESPOND_SHA,
                ActionType.RESPOND_TAO,
                ActionType.RESPOND_WUXIE,
                ActionType.USE_SKILL,
                ActionType.JUDGE_MODIFY,
            ]
            else None,
            target_idx=target_idx
            if action_type
            in [
                ActionType.USE_CARD,
                ActionType.RESPOND_TAO,
                ActionType.USE_SKILL,
                ActionType.SELECT_TARGET,
            ]
            else None,
        )

    def needs_card(self, action_type: int) -> bool:
        """判断动作类型是否需要选择卡牌"""
        return action_type in [
            ActionType.USE_CARD,
            ActionType.DISCARD,
            ActionType.RESPOND_SHAN,
            ActionType.RESPOND_SHA,
            ActionType.RESPOND_TAO,
            ActionType.RESPOND_WUXIE,
            ActionType.USE_SKILL,
            ActionType.JUDGE_MODIFY,
        ]

    def needs_target(self, action_type: int, card=None) -> bool:
        """判断动作是否需要选择目标"""
        if action_type in [ActionType.RESPOND_TAO, ActionType.USE_SKILL]:
            return True

        if action_type == ActionType.USE_CARD and card:
            return self._card_needs_target(card)

        return False

    def _card_needs_target(self, card) -> bool:
        if hasattr(card, "target_types"):
            target_types = card.target_types
            if not target_types:
                return False
            no_selection_types = ["self", "all_players", "all_other_players"]
            if all(t in no_selection_types for t in target_types):
                return False
            return True

        card_name = card.name if hasattr(card, "name") else card.get("name", "")
        no_target_cards = ["无中生有", "桃园结义", "五谷丰登", "南蛮入侵", "万箭齐发"]

        if card_name in no_target_cards:
            return False

        return True


class ActionMaskGenerator:
    """动作掩码生成器"""

    def __init__(self, encoder: ActionEncoder):
        self.encoder = encoder

    def _get_attr(self, obj, attr: str, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    def generate_masks(
        self,
        game_state: Dict,
        player,
        current_step: int = 0,
        pending_action: HierarchicalAction = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成三层动作掩码

        Args:
            game_state: 游戏状态
            player: 当前玩家
            current_step: 当前步骤 (0=选类型, 1=选卡牌, 2=选目标)
            pending_action: 待完成的动作

        Returns:
            (masks_type, masks_card, masks_target)
        """
        if current_step == 0:
            masks_type = self._get_valid_action_types(game_state, player)
            masks_card = np.zeros(self.encoder.card_dim, dtype=np.float32)
            masks_target = np.zeros(self.encoder.target_dim, dtype=np.float32)

        elif current_step == 1:
            masks_type = np.zeros(self.encoder.action_type_dim, dtype=np.float32)
            masks_card = self._get_valid_cards(game_state, player, pending_action)
            masks_target = np.zeros(self.encoder.target_dim, dtype=np.float32)

        elif current_step == 2:
            masks_type = np.zeros(self.encoder.action_type_dim, dtype=np.float32)
            masks_card = np.zeros(self.encoder.card_dim, dtype=np.float32)
            masks_target = self._get_valid_targets(game_state, player, pending_action)

        else:
            masks_type = np.zeros(self.encoder.action_type_dim, dtype=np.float32)
            masks_card = np.zeros(self.encoder.card_dim, dtype=np.float32)
            masks_target = np.zeros(self.encoder.target_dim, dtype=np.float32)

        return masks_type, masks_card, masks_target

    def _get_valid_action_types(self, game_state: Dict, player) -> np.ndarray:
        """获取当前合法的动作类型"""
        masks = np.zeros(self.encoder.action_type_dim, dtype=np.float32)

        phase = game_state.get("phase", "waiting")
        if hasattr(phase, "value"):
            phase = phase.value

        if phase == "play_phase":
            masks[ActionType.END_TURN] = 1.0
            if self._has_usable_cards(player, game_state):
                masks[ActionType.USE_CARD] = 1.0
            if self._has_usable_skills(player):
                masks[ActionType.USE_SKILL] = 1.0

        elif phase == "discard_phase":
            # 弃牌阶段
            hand_count = len(player.hand_cards) if hasattr(player, "hand_cards") else 0
            hand_limit = (
                player.hand_limit
                if hasattr(player, "hand_limit")
                else player.current_hp
            )
            if hand_count > hand_limit:
                masks[ActionType.DISCARD] = 1.0

        elif phase == "respond_shan":
            # 需要出闪
            if self._has_card_type(player, "闪"):
                masks[ActionType.RESPOND_SHAN] = 1.0
            masks[ActionType.PASS] = 1.0

        elif phase == "respond_sha":
            # 需要出杀
            if self._has_card_type(player, "杀"):
                masks[ActionType.RESPOND_SHA] = 1.0
            masks[ActionType.PASS] = 1.0

        elif phase == "respond_tao":
            # 需要出桃
            if self._has_card_type(player, "桃"):
                masks[ActionType.RESPOND_TAO] = 1.0
            masks[ActionType.PASS] = 1.0

        elif phase == "respond_wuxie":
            # 无懈可击响应
            if self._has_card_type(player, "无懈可击"):
                masks[ActionType.RESPOND_WUXIE] = 1.0
            masks[ActionType.PASS] = 1.0

        elif phase == "judge_modify":
            # 判定改判
            if self._can_modify_judge(player):
                masks[ActionType.JUDGE_MODIFY] = 1.0
            masks[ActionType.PASS] = 1.0

        # 如果没有合法动作，允许 pass
        if masks.sum() == 0:
            masks[ActionType.PASS] = 1.0

        return masks

    def _get_valid_cards(
        self,
        game_state: Dict,
        player,
        pending_action: HierarchicalAction,
    ) -> np.ndarray:
        """获取当前可用的卡牌/技能"""
        masks = np.zeros(self.encoder.card_dim, dtype=np.float32)

        if pending_action is None:
            return masks

        action_type = pending_action.action_type

        if action_type == ActionType.USE_CARD:
            hand_cards = self._get_attr(player, "hand_cards", [])
            for i, card in enumerate(hand_cards[: self.encoder.card_dim]):
                if self._can_use_card(player, card, game_state):
                    if self.encoder._card_needs_target(card):
                        targets = self._get_card_targets(
                            player, card, game_state.get("players", [])
                        )
                        if len(targets) > 0:
                            masks[i] = 1.0
                    else:
                        masks[i] = 1.0

        elif action_type == ActionType.DISCARD:
            # 可弃置的手牌
            hand_count = len(player.hand_cards) if hasattr(player, "hand_cards") else 0
            for i in range(min(hand_count, self.encoder.card_dim)):
                masks[i] = 1.0

        elif action_type == ActionType.RESPOND_SHAN:
            hand_cards = self._get_attr(player, "hand_cards", [])
            for i, card in enumerate(hand_cards[: self.encoder.card_dim]):
                if self._get_attr(card, "name") == "闪":
                    masks[i] = 1.0

        elif action_type == ActionType.RESPOND_SHA:
            hand_cards = self._get_attr(player, "hand_cards", [])
            for i, card in enumerate(hand_cards[: self.encoder.card_dim]):
                card_name = self._get_attr(card, "name", "")
                if "杀" in card_name:
                    masks[i] = 1.0

        elif action_type == ActionType.RESPOND_TAO:
            hand_cards = self._get_attr(player, "hand_cards", [])
            for i, card in enumerate(hand_cards[: self.encoder.card_dim]):
                if self._get_attr(card, "name") == "桃":
                    masks[i] = 1.0

        elif action_type == ActionType.RESPOND_WUXIE:
            hand_cards = self._get_attr(player, "hand_cards", [])
            for i, card in enumerate(hand_cards[: self.encoder.card_dim]):
                if self._get_attr(card, "name") == "无懈可击":
                    masks[i] = 1.0

        elif action_type == ActionType.USE_SKILL:
            skills = self._get_attr(player, "skills", [])
            for i, skill in enumerate(skills[: self.encoder.card_dim]):
                targets = self._get_skill_targets(
                    player, skill, game_state.get("players", [])
                )
                if len(targets) > 0:
                    masks[i] = 1.0

        elif action_type == ActionType.JUDGE_MODIFY:
            hand_cards = self._get_attr(player, "hand_cards", [])
            for i, card in enumerate(hand_cards[: self.encoder.card_dim]):
                masks[i] = 1.0

        return masks

    def _get_valid_targets(
        self,
        game_state: Dict,
        player,
        pending_action: HierarchicalAction,
    ) -> np.ndarray:
        """获取当前合法的目标"""
        masks = np.zeros(self.encoder.target_dim, dtype=np.float32)

        if pending_action is None:
            return masks

        action_type = pending_action.action_type
        card_idx = pending_action.card_idx

        players = game_state.get("players", [])

        if action_type == ActionType.USE_CARD:
            hand_cards = self._get_attr(player, "hand_cards", [])
            if card_idx is not None and 0 <= card_idx < len(hand_cards):
                card = hand_cards[card_idx]
                targets = self._get_card_targets(player, card, players)
                for t in targets:
                    t_idx = self._get_attr(t, "idx") or self._get_attr(t, "player_id")
                    if t_idx and 0 < t_idx <= self.encoder.target_dim:
                        masks[t_idx - 1] = 1.0

        elif action_type == ActionType.RESPOND_TAO:
            for p in players:
                if (
                    self._get_attr(p, "is_alive", True)
                    and self._get_attr(p, "current_hp", 0) <= 0
                ):
                    t_idx = self._get_attr(p, "idx") or self._get_attr(p, "player_id")
                    if t_idx and 0 < t_idx <= self.encoder.target_dim:
                        masks[t_idx - 1] = 1.0
            if self._get_attr(player, "current_hp", 0) < self._get_attr(
                player, "max_hp", 4
            ):
                masks[0] = 1.0

        elif action_type == ActionType.USE_SKILL:
            skills = self._get_attr(player, "skills", [])
            if card_idx is not None and 0 <= card_idx < len(skills):
                skill = skills[card_idx]
                targets = self._get_skill_targets(player, skill, players)
                for t in targets:
                    t_idx = self._get_attr(t, "idx") or self._get_attr(t, "player_id")
                    if t_idx and 0 < t_idx <= self.encoder.target_dim:
                        masks[t_idx - 1] = 1.0

        elif action_type == ActionType.SELECT_TARGET:
            for p in players:
                if self._get_attr(p, "is_alive", True) and p != player:
                    t_idx = self._get_attr(p, "idx") or self._get_attr(p, "player_id")
                    if t_idx and 0 < t_idx <= self.encoder.target_dim:
                        masks[t_idx - 1] = 1.0

        return masks

    def _has_card_type(self, player, card_name: str) -> bool:
        hand_cards = self._get_attr(player, "hand_cards", [])
        for card in hand_cards:
            card_name_attr = self._get_attr(card, "name", "")
            if card_name in card_name_attr:
                return True
        return False

    def _has_usable_cards(self, player, game_state: Dict) -> bool:
        hand_cards = self._get_attr(player, "hand_cards", [])
        for card in hand_cards:
            if self._can_use_card(player, card, game_state):
                if self.encoder._card_needs_target(card):
                    targets = self._get_card_targets(
                        player, card, game_state.get("players", [])
                    )
                    if len(targets) > 0:
                        return True
                else:
                    return True
        return False

    def _has_usable_skills(self, player) -> bool:
        skills = self._get_attr(player, "skills", [])
        return len(skills) > 0

    def _can_modify_judge(self, player) -> bool:
        """检查玩家是否可以改判"""
        # 有手牌就可以改判
        hand_count = len(player.hand_cards) if hasattr(player, "hand_cards") else 0
        return hand_count > 0

    def _can_use_card(self, player, card, game_state: Dict) -> bool:
        """检查玩家是否可以使用某张牌"""
        if not hasattr(card, "name"):
            return False

        card_name = card.name

        # 响应牌不能主动使用
        response_cards = ["闪"]
        if card_name in response_cards:
            return False

        # 杀的限制
        if card_name == "杀":
            if hasattr(player, "can_use_sha"):
                return player.can_use_sha()
            sha_count = player.sha_count if hasattr(player, "sha_count") else 0
            unlimited = (
                player.unlimited_sha if hasattr(player, "unlimited_sha") else False
            )
            return unlimited or sha_count < 1

        # 酒的限制
        if card_name == "酒":
            jiu_count = player.jiu_count if hasattr(player, "jiu_count") else 0
            return jiu_count < 1

        # 桃的限制
        if card_name == "桃":
            current_hp = player.current_hp if hasattr(player, "current_hp") else 0
            max_hp = player.max_hp if hasattr(player, "max_hp") else 4
            return current_hp < max_hp

        # 距离检查
        if card_name == "顺手牵羊":
            return self._has_target_in_range(player, game_state, 1)

        return True

    def _has_target_in_range(self, player, game_state: Dict, max_range: int) -> bool:
        players = game_state.get("players", [])
        for p in players:
            if p != player and self._get_attr(p, "is_alive", True):
                return True
        return False

    def _get_card_targets(self, player, card, players: List) -> List:
        targets = []

        card_name = self._get_attr(card, "name")
        if not card_name:
            return targets

        no_target_cards = ["无中生有", "桃园结义", "五谷丰登", "南蛮入侵", "万箭齐发"]
        if card_name in no_target_cards:
            return targets

        self_target_cards = ["桃", "酒"]
        if card_name in self_target_cards:
            return [player]

        for p in players:
            if p != player and self._get_attr(p, "is_alive", True):
                if card_name == "杀" or "杀" in card_name:
                    if self._is_in_range(player, p):
                        targets.append(p)
                elif card_name == "顺手牵羊":
                    if self._is_in_range(player, p, 1):
                        targets.append(p)
                else:
                    targets.append(p)

        return targets

    def _get_skill_targets(self, player, skill, players: List) -> List:
        targets = []

        skill_name = self._get_attr(skill, "name", "")

        if skill_name == "青囊":
            for p in players:
                if p != player and self._get_attr(p, "is_alive", True):
                    if self._get_attr(p, "current_hp", 0) < self._get_attr(
                        p, "max_hp", 4
                    ):
                        targets.append(p)

        elif skill_name == "结姻":
            for p in players:
                if p != player and self._get_attr(p, "is_alive", True):
                    gender = self._get_attr(p, "gender", "male")
                    if gender == "male":
                        targets.append(p)

        else:
            for p in players:
                if p != player and self._get_attr(p, "is_alive", True):
                    targets.append(p)

        return targets

    def _is_in_range(self, source, target, max_range: int = None) -> bool:
        if max_range is None:
            attack_range = self._get_attr(source, "attack_range", 1)
        else:
            attack_range = max_range

        distance = 1
        src_idx = self._get_attr(source, "idx") or self._get_attr(source, "player_id")
        tgt_idx = self._get_attr(target, "idx") or self._get_attr(target, "player_id")
        if src_idx and tgt_idx:
            distance = abs(src_idx - tgt_idx)
            total_players = 5
            distance = min(distance, total_players - distance)

        equipment = self._get_attr(source, "equipment", {})
        if equipment and equipment.get("进攻坐骑"):
            distance = max(1, distance - 1)

        tgt_equipment = self._get_attr(target, "equipment", {})
        if tgt_equipment and tgt_equipment.get("防御坐骑"):
            distance += 1

        return distance <= attack_range


class ActionDecoder:
    """动作解码器"""

    def __init__(self, encoder: ActionEncoder):
        self.encoder = encoder

    def decode(
        self,
        action: int,
        current_step: int,
        pending_action: HierarchicalAction = None,
    ) -> HierarchicalAction:
        """
        解码动作

        Args:
            action: 动作值
            current_step: 当前步骤
            pending_action: 待完成的动作

        Returns:
            更新后的分层动作
        """
        if current_step == 0:
            return HierarchicalAction(action_type=action)

        elif current_step == 1:
            if pending_action:
                return HierarchicalAction(
                    action_type=pending_action.action_type,
                    card_idx=action,
                )
            return HierarchicalAction(action_type=0, card_idx=action)

        elif current_step == 2:
            if pending_action:
                return HierarchicalAction(
                    action_type=pending_action.action_type,
                    card_idx=pending_action.card_idx,
                    target_idx=action,
                )
            return HierarchicalAction(action_type=0, target_idx=action)

        return HierarchicalAction(action_type=action)

    def get_action_description(self, action: HierarchicalAction) -> str:
        """获取动作描述"""
        action_names = {
            ActionType.USE_CARD: "使用卡牌",
            ActionType.END_TURN: "结束回合",
            ActionType.DISCARD: "弃置手牌",
            ActionType.RESPOND_SHAN: "出闪",
            ActionType.RESPOND_SHA: "出杀",
            ActionType.RESPOND_TAO: "出桃",
            ActionType.RESPOND_WUXIE: "无懈可击",
            ActionType.USE_SKILL: "发动技能",
            ActionType.PASS: "跳过",
            ActionType.SELECT_TARGET: "选择目标",
            ActionType.JUDGE_MODIFY: "改判",
            ActionType.CONFIRM: "确认",
        }

        desc = action_names.get(action.action_type, "未知动作")

        if action.card_idx is not None:
            desc += f" [牌{action.card_idx}]"

        if action.target_idx is not None:
            desc += f" -> 玩家{action.target_idx}"

        return desc
