"""
动作编码器 - 分层动作空间编码

动作空间结构:
├── Step 0: 选择动作类型 - Discrete(12)
├── Step 1: 选择卡牌/技能 - Discrete(20)
└── Step 2: 选择目标 - Discrete(8)

支持动作掩码过滤非法动作
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from card.base import is_sha_card


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
    # 技能决策动作类型 - Phase 1关键修复
    SKILL_DECISION_YES_NO = 12  # 技能是否发动决策
    SKILL_DECISION_SELECT_CARD = 13  # 技能选择卡牌决策
    SKILL_DECISION_SELECT_TARGET = 14  # 技能选择目标决策
    SKILL_DECISION_SELECT_ORDER = 15  # 技能选择顺序决策
    SKILL_DECISION_DISTRIBUTE = 16  # 技能分配决策


@dataclass
class ActionConfig:
    """动作配置"""

    num_action_types: int = 17  # Phase 1: 从12增加到17（新增5个技能决策动作）
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
        if action_type == ActionType.RESPOND_TAO:
            return True

        if action_type == ActionType.USE_SKILL:
            # 检查技能是否需要目标
            if card and hasattr(card, "target_types"):
                target_types = card.target_types
                if not target_types:
                    return False
                no_selection_types = ["self", "all_players", "all_other_players"]
                if all(t in no_selection_types for t in target_types):
                    return False
                return True
            # 默认需要目标
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
    """动作掩码生成器

    支持基于身份信念的目标选择:
    - 有益动作(治疗、增益)偏好高P(忠臣)目标
    - 攻击动作偏好高P(反贼)目标
    - 使用软偏好而非硬约束，允许智能体从错误中学习
    """

    def __init__(self, encoder: ActionEncoder, state_encoder=None):
        self.encoder = encoder
        self.state_encoder = state_encoder  # For accessing belief states

    # Identity-aware masking constants
    BENEFIT_CARD_NAMES = {"桃", "酒", "无中生有", "桃园结义", "五谷丰登"}
    ATTACK_CARD_NAMES = {"杀", "火杀", "雷杀", "决斗", "火攻", "南蛮入侵", "万箭齐发"}
    DEBUFF_CARD_NAMES = {"过河拆桥", "顺手牵羊", "借刀杀人", "乐不思蜀", "兵粮寸断"}
    HEAL_SKILL_NAMES = {"急救", "青囊"}
    ATTACK_SKILL_NAMES = {"突袭", "反间", "离间"}

    # Soft preference weights (not hard constraints)
    ALLY_BENEFIT_BONUS = 0.3  # Bonus for healing/buffing believed allies
    ENEMY_ATTACK_BONUS = 0.3  # Bonus for attacking believed enemies

    def _get_attr(self, obj, attr: str, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    def generate_masks(
        self,
        game_state: Dict,
        player,
        engine=None,
        current_step: int = 0,
        pending_action: HierarchicalAction = None,
        observer_idx: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成三层动作掩码

        Args:
                game_state: 游戏状态
                player: 当前玩家
                current_step: 当前步骤 (0=选类型, 1=选卡牌, 2=选目标)
                pending_action: 待完成的动作
                observer_idx: 观察者索引(用于身份感知的目标选择)

        Returns:
                (masks_type, masks_card, masks_target)
        """
        # Determine observer_idx from player if not provided
        if observer_idx is None and player is not None:
            observer_idx = self._get_attr(player, "idx") or self._get_attr(
                player, "player_id", 0
            )

        if current_step == 0:
            masks_type = self._get_valid_action_types(game_state, player)
            masks_card = np.zeros(self.encoder.card_dim, dtype=np.float32)
            masks_target = np.zeros(self.encoder.target_dim, dtype=np.float32)

        elif current_step == 1:
            masks_type = np.zeros(self.encoder.action_type_dim, dtype=np.float32)
            masks_card = self._get_valid_cards(
                game_state, player, pending_action, engine
            )
            masks_target = np.zeros(self.encoder.target_dim, dtype=np.float32)

        elif current_step == 2:
            masks_type = np.zeros(self.encoder.action_type_dim, dtype=np.float32)
            masks_card = np.zeros(self.encoder.card_dim, dtype=np.float32)
            masks_target = self._get_valid_targets(
                game_state, player, pending_action, observer_idx
            )

        else:
            masks_type = np.zeros(self.encoder.action_type_dim, dtype=np.float32)
            masks_card = np.zeros(self.encoder.card_dim, dtype=np.float32)
            masks_target = np.zeros(self.encoder.target_dim, dtype=np.float32)

        return masks_type, masks_card, masks_target

    def _get_valid_action_types(
        self, game_state: Dict, player, engine=None
    ) -> np.ndarray:
        """获取当前合法的动作类型"""
        masks = np.zeros(self.encoder.action_type_dim, dtype=np.float32)

        phase = game_state.get("phase", "waiting")
        if hasattr(phase, "value"):
            phase = phase.value

        if phase == "play_phase":
            masks[ActionType.END_TURN] = 1.0
            if self._has_usable_cards(player, game_state):
                masks[ActionType.USE_CARD] = 1.0
            if self._has_usable_skills(player, game_state, engine):
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
        engine=None,
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
            from skills.base import ActiveSkill

            skills = self._get_attr(player, "skills", [])
            hand_cards = self._get_attr(player, "hand_cards", [])
            hand_count = len(hand_cards)

            # 获取当前阶段和玩家索引
            phase = game_state.get("phase", "")
            current_player_idx = game_state.get("current_player_idx", 0)
            player_idx = self._get_attr(player, "idx") or self._get_attr(
                player, "player_id", 0
            )
            is_my_turn = current_player_idx == player_idx

            for i, skill in enumerate(skills[: self.encoder.card_dim]):
                if not isinstance(skill, ActiveSkill):
                    continue

                # 调用 is_available 检查技能是否可用
                if engine is not None:
                    try:
                        if not skill.is_available(engine):
                            continue
                    except Exception:
                        # 如果 is_available 抛出异常，回退到基础检查
                        pass

                skill_name = self._get_attr(skill, "name", "")

                # 急救：只能在回合外使用
                if skill_name == "急救":
                    if is_my_turn and phase == "play_phase":
                        continue

                # 特定技能的额外检查（兼容旧代码）
                if skill_name == "仁德" and hand_count == 0:
                    continue
                if skill_name == "遗计" and hand_count == 0:
                    continue

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
        observer_idx: int = None,
    ) -> np.ndarray:
        """获取当前合法的目标

        Args:
            game_state: 游戏状态
            player: 当前玩家
            pending_action: 待完成的动作
            observer_idx: 观察者索引(用于获取身份信念)，默认从player获取
        """
        masks = np.zeros(self.encoder.target_dim, dtype=np.float32)

        if pending_action is None:
            return masks

        action_type = pending_action.action_type
        card_idx = pending_action.card_idx

        players = game_state.get("players", [])

        # Determine if this is a beneficial or attack action
        is_beneficial, is_attack = self._classify_action(
            action_type, player, card_idx, game_state
        )

        if action_type == ActionType.USE_CARD:
            hand_cards = self._get_attr(player, "hand_cards", [])
            if card_idx is not None and 0 <= card_idx < len(hand_cards):
                card = hand_cards[card_idx]
                targets = self._get_card_targets(player, card, players)
                for t in targets:
                    t_idx = self._get_attr(t, "idx") or self._get_attr(t, "player_id")
                    if t_idx and 0 < t_idx <= self.encoder.target_dim:
                        base_mask = 1.0
                        # Apply identity-aware soft preferences
                        if self.state_encoder is not None and observer_idx is not None:
                            base_mask = self._apply_identity_preference(
                                observer_idx, t_idx, is_beneficial, is_attack, base_mask
                            )
                        masks[t_idx - 1] = base_mask

        elif action_type == ActionType.RESPOND_TAO:
            for p in players:
                if (
                    self._get_attr(p, "is_alive", True)
                    and self._get_attr(p, "current_hp", 0) <= 0
                ):
                    t_idx = self._get_attr(p, "idx") or self._get_attr(p, "player_id")
                    if t_idx and 0 < t_idx <= self.encoder.target_dim:
                        base_mask = 1.0
                        # 桃是beneficial，偏好盟友
                        if self.state_encoder is not None and observer_idx is not None:
                            base_mask = self._apply_identity_preference(
                                observer_idx, t_idx, True, False, base_mask
                            )
                        masks[t_idx - 1] = base_mask
            if self._get_attr(player, "current_hp", 0) < self._get_attr(
                player, "max_hp", 4
            ):
                masks[0] = 1.0

        elif action_type == ActionType.USE_SKILL:
            from skills.base import ActiveSkill

            skills = self._get_attr(player, "skills", [])
            if card_idx is not None and 0 <= card_idx < len(skills):
                skill = skills[card_idx]
                if not isinstance(skill, ActiveSkill):
                    return masks
                # Check if skill is beneficial or attack
                skill_name = self._get_attr(skill, "name", "")
                is_skill_beneficial = skill_name in self.HEAL_SKILL_NAMES
                is_skill_attack = skill_name in self.ATTACK_SKILL_NAMES
                targets = self._get_skill_targets(player, skill, players)
                for t in targets:
                    t_idx = self._get_attr(t, "idx") or self._get_attr(t, "player_id")
                    if t_idx and 0 < t_idx <= self.encoder.target_dim:
                        base_mask = 1.0
                        if self.state_encoder is not None and observer_idx is not None:
                            base_mask = self._apply_identity_preference(
                                observer_idx,
                                t_idx,
                                is_skill_beneficial,
                                is_skill_attack,
                                base_mask,
                            )
                        masks[t_idx - 1] = base_mask

        elif action_type == ActionType.SELECT_TARGET:
            for p in players:
                if self._get_attr(p, "is_alive", True) and p != player:
                    t_idx = self._get_attr(p, "idx") or self._get_attr(p, "player_id")
                    if t_idx and 0 < t_idx <= self.encoder.target_dim:
                        masks[t_idx - 1] = 1.0

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
            from skills.base import ActiveSkill

            skills = self._get_attr(player, "skills", [])
            if card_idx is not None and 0 <= card_idx < len(skills):
                skill = skills[card_idx]
                if not isinstance(skill, ActiveSkill):
                    return masks
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

    def _classify_action(
        self, action_type: int, player, card_idx: Optional[int], game_state: Dict
    ) -> Tuple[bool, bool]:
        """Classify action as beneficial and/or attack

        Returns:
            (is_beneficial, is_attack): Tuple of boolean flags
        """
        is_beneficial = False
        is_attack = False

        if action_type == ActionType.USE_CARD and card_idx is not None:
            hand_cards = self._get_attr(player, "hand_cards", [])
            if 0 <= card_idx < len(hand_cards):
                card = hand_cards[card_idx]
                card_name = self._get_attr(card, "name", "")
                if card_name in self.BENEFIT_CARD_NAMES:
                    is_beneficial = True
                if card_name in self.ATTACK_CARD_NAMES:
                    is_attack = True
                if card_name in self.DEBUFF_CARD_NAMES:
                    is_attack = True

        return is_beneficial, is_attack

    def _apply_identity_preference(
        self,
        observer_idx: int,
        target_idx: int,
        is_beneficial: bool,
        is_attack: bool,
        base_mask: float,
    ) -> float:
        """Apply identity-based soft preference to mask value

        - Beneficial actions prefer high P(忠臣) targets
        - Attack actions prefer high P(反贼) targets
        - Soft preference only - doesn't prevent "wrong" choices

        Args:
            observer_idx: Index of the observing player (RL agent)
            target_idx: Index of the target player
            is_beneficial: Whether this is a beneficial action (heal/buff)
            is_attack: Whether this is an attack action
            base_mask: Base mask value (1.0 for valid targets)

        Returns:
            Modified mask value with soft preference applied
        """
        if self.state_encoder is None:
            return base_mask

        # Get belief for target: [P(忠臣), P(反贼), P(内奸), P(unknown)]
        belief = self.state_encoder.get_belief(observer_idx, target_idx)
        p_loyalist = belief[0]  # P(忠臣)
        p_rebel = belief[1]  # P(反贼)

        modified_mask = base_mask

        if is_beneficial and not is_attack:
            # Beneficial actions: bonus for believed allies (high P(忠臣))
            # Scale bonus by belief strength
            modified_mask = base_mask + self.ALLY_BENEFIT_BONUS * p_loyalist

        elif is_attack and not is_beneficial:
            # Attack actions: bonus for believed enemies (high P(反贼))
            # Scale bonus by belief strength
            modified_mask = base_mask + self.ENEMY_ATTACK_BONUS * p_rebel

        # For actions that are both/neither (rare), keep base mask
        return modified_mask

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

    def _has_usable_skills(self, player, game_state: Dict = None, engine=None) -> bool:
        from skills.base import ActiveSkill

        skills = self._get_attr(player, "skills", [])
        hand_cards = self._get_attr(player, "hand_cards", [])
        hand_count = len(hand_cards)

        for skill in skills:
            if not isinstance(skill, ActiveSkill):
                continue

            skill_name = self._get_attr(skill, "name", "")

            if skill_name == "仁德" and hand_count == 0:
                continue
            if skill_name == "仁德" or skill_name == "遗计":
                if hand_count == 0:
                    continue

            if engine is not None:
                try:
                    if not skill.is_available(engine):
                        continue
                except Exception:
                    pass

            if game_state is not None:
                targets = self._get_skill_targets(
                    player, skill, game_state.get("players", [])
                )
                if len(targets) == 0:
                    continue

            return True
        return False

    def _can_modify_judge(self, player) -> bool:
        """检查玩家是否可以改判"""
        # 有手牌就可以改判
        hand_count = len(player.hand_cards) if hasattr(player, "hand_cards") else 0
        return hand_count > 0

    def _can_use_card(self, player, card, game_state: Dict) -> bool:
        if not hasattr(card, "name"):
            return False

        card_name = card.name

        response_cards = ["闪"]
        if card_name in response_cards:
            return False

        if is_sha_card(card):
            unlimited = (
                player.unlimited_sha if hasattr(player, "unlimited_sha") else False
            )
            sha_count = player.sha_count if hasattr(player, "sha_count") else 0
            return unlimited or sha_count < 1

        if card_name == "酒":
            jiu_count = player.jiu_count if hasattr(player, "jiu_count") else 0
            return jiu_count < 1

        if card_name == "桃":
            current_hp = player.current_hp if hasattr(player, "current_hp") else 0
            max_hp = player.max_hp if hasattr(player, "max_hp") else 4
            return current_hp < max_hp

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
                if is_sha_card(card):
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

    def _get_guanxing_mask(self, request, current_step: int) -> np.ndarray:
        """
        生成观星技能决策的掩码

        观星: 选择5张牌的排序（从牌堆顶观看的牌）
        使用顺序选择方法：N步选择N张牌的位置
        - 第1步：为第0张牌选择位置（5个选项：位置0-4）
        - 第2步：为第1张牌选择位置（4个剩余选项）
        - ...以此类推

        Args:
            request: SkillDecisionRequest，包含options（卡牌列表）和_selections（已选位置）
            current_step: 当前步骤（0到N-1，N为卡牌数量）

        Returns:
            动作掩码，大小为max_hand_size（20），有效位置标记为1.0
        """
        masks = np.zeros(self.encoder.card_dim, dtype=np.float32)

        # 获取卡牌数量（可能少于5张）
        num_cards = len(request.options) if request.options else 0
        if num_cards == 0:
            return masks

        # 获取已选择的位置
        selected_positions = (
            request._selections if hasattr(request, "_selections") else []
        )

        # 当前步骤应小于卡牌数量
        if current_step >= num_cards:
            return masks

        # 生成可用位置掩码
        # 观星的位置范围是0到num_cards-1
        for pos in range(num_cards):
            if pos not in selected_positions:
                # 位置映射到mask索引（位置0-4对应mask索引0-4）
                if pos < self.encoder.card_dim:
                    masks[pos] = 1.0

        return masks


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
            # Phase 1新增：技能决策动作描述
            ActionType.SKILL_DECISION_YES_NO: "技能决策(是否)",
            ActionType.SKILL_DECISION_SELECT_CARD: "技能决策(选牌)",
            ActionType.SKILL_DECISION_SELECT_TARGET: "技能决策(选目标)",
            ActionType.SKILL_DECISION_SELECT_ORDER: "技能决策(选顺序)",
            ActionType.SKILL_DECISION_DISTRIBUTE: "技能决策(分配)",
        }

        desc = action_names.get(action.action_type, "未知动作")

        if action.card_idx is not None:
            desc += f" [牌{action.card_idx}]"

        if action.target_idx is not None:
            desc += f" -> 玩家{action.target_idx}"

        return desc
