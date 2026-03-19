"""
状态编码器 - 将游戏状态编码为神经网络可处理的向量

状态空间结构 (~3000维):
├── 全局状态: 34维
├── 当前玩家状态: 8维
├── 手牌编码: 1520维 (20张 × 76维)
├── 装备状态: 25维
├── 判定区: 3维
├── 武将/技能: 98维
├── 其他玩家: 511维 (7人 × 73维)
└── 历史动作: 780维 (10条 × 78维)
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class EncodingConfig:
    """编码配置"""

    max_hand_size: int = 20
    max_players: int = 8
    max_history_actions: int = 10

    num_phases: int = 8
    num_card_types: int = 10
    num_card_names: int = 60
    num_colors: int = 4
    num_nations: int = 4
    num_identities: int = 4
    num_commanders: int = 22
    num_skills: int = 36
    num_weapons: int = 15
    num_armours: int = 7
    num_action_types: int = 10


class CardNameEncoder:
    """卡牌名称编码器"""

    CARD_NAMES = [
        # 基本牌
        "杀",
        "闪",
        "桃",
        "酒",
        "火杀",
        "雷杀",
        # 即时锦囊
        "决斗",
        "无中生有",
        "过河拆桥",
        "顺手牵羊",
        "借刀杀人",
        "无懈可击",
        "南蛮入侵",
        "万箭齐发",
        "火攻",
        "铁索连环",
        "五谷丰登",
        "桃园结义",
        # 延时锦囊
        "乐不思蜀",
        "兵粮寸断",
        "闪电",
        # 武器
        "诸葛连弩",
        "青釭剑",
        "青龙偃月刀",
        "丈八蛇矛",
        "贯石斧",
        "方天画戟",
        "麒麟弓",
        "朱雀羽扇",
        "古锭刀",
        "吴六剑",
        "三尖两刃刀",
        "太平要术",
        # 防具
        "八卦阵",
        "仁王盾",
        "白银狮子",
        "藤甲",
        # 坐骑
        "赤兔",
        "大宛",
        "紫骍",
        "的卢",
        "绝影",
        "爪黄飞电",
        # 宝物
        "木牛流马",
    ]

    def __init__(self):
        self.name_to_idx = {name: i for i, name in enumerate(self.CARD_NAMES)}

    def encode(self, card_name: str) -> np.ndarray:
        """将卡牌名称编码为one-hot向量"""
        encoding = np.zeros(len(self.CARD_NAMES), dtype=np.float32)
        if card_name in self.name_to_idx:
            encoding[self.name_to_idx[card_name]] = 1.0
        return encoding

    def get_num_cards(self) -> int:
        return len(self.CARD_NAMES)


class CardTypeEncoder:
    """卡牌类型编码器"""

    CARD_TYPES = [
        "BasicCard",
        "Sha",
        "Shan",
        "Tao",
        "Jiu",
        "FireSha",
        "ThunderSha",
        "CommonJinnangCard",
        "YanshiJinnangCard",
        "WeaponCard",
        "ArmourCard",
        "AttackHorseCard",
        "DefenseHorseCard",
        "TreasureCard",
    ]

    def __init__(self):
        self.type_to_idx = {t: i for i, t in enumerate(self.CARD_TYPES)}

    def encode(self, card_type: str) -> np.ndarray:
        """将卡牌类型编码为one-hot向量"""
        encoding = np.zeros(len(self.CARD_TYPES), dtype=np.float32)
        if card_type in self.type_to_idx:
            encoding[self.type_to_idx[card_type]] = 1.0
        return encoding


class CommanderEncoder:
    """武将编码器"""

    COMMANDERS = [
        # 魏
        "曹操",
        "司马懿",
        "夏侯惇",
        "张辽",
        "许褚",
        "郭嘉",
        "甄姬",
        # 蜀
        "刘备",
        "关羽",
        "张飞",
        "诸葛亮",
        "赵云",
        "马超",
        "黄月英",
        # 吴
        "孙权",
        "甘宁",
        "吕蒙",
        "黄盖",
        "周瑜",
        "大乔",
        "陆逊",
        "孙尚香",
        # 群
        "华佗",
        "吕布",
        "貂蝉",
    ]

    def __init__(self):
        self.name_to_idx = {name: i for i, name in enumerate(self.COMMANDERS)}

    def encode(self, commander_name: str) -> np.ndarray:
        encoding = np.zeros(len(self.COMMANDERS), dtype=np.float32)
        if commander_name in self.name_to_idx:
            encoding[self.name_to_idx[commander_name]] = 1.0
        return encoding


class SkillEncoder:
    """技能编码器"""

    SKILLS = [
        # 魏国
        "奸雄",
        "鬼才",
        "反馈",
        "刚烈",
        "突袭",
        "裸衣",
        "天妒",
        "遗计",
        "洛神",
        "倾国",
        # 蜀国
        "仁德",
        "武圣",
        "咆哮",
        "观星",
        "空城",
        "龙胆",
        "马术",
        "铁骑",
        "集智",
        # 吴国
        "制衡",
        "奇袭",
        "克己",
        "苦肉",
        "英姿",
        "反间",
        "国色",
        "流离",
        "谦逊",
        "连营",
        "结姻",
        "枭姬",
        # 群雄
        "急救",
        "青囊",
        "无双",
        "离间",
        "闭月",
    ]

    def __init__(self):
        self.name_to_idx = {name: i for i, name in enumerate(self.SKILLS)}

    def encode(self, skill_names: List[str]) -> np.ndarray:
        """将技能列表编码为multi-hot向量"""
        encoding = np.zeros(len(self.SKILLS), dtype=np.float32)
        for name in skill_names:
            if name in self.name_to_idx:
                encoding[self.name_to_idx[name]] = 1.0
        return encoding


class StateEncoder:
    """状态编码器"""

    def __init__(self, config: EncodingConfig = None):
        self.config = config or EncodingConfig()

        self.card_name_encoder = CardNameEncoder()
        self.card_type_encoder = CardTypeEncoder()
        self.commander_encoder = CommanderEncoder()
        self.skill_encoder = SkillEncoder()

        self._update_config()

    def _update_config(self):
        """更新配置以匹配实际编码器"""
        self.config.num_card_types = len(CardTypeEncoder.CARD_TYPES)
        self.config.num_card_names = self.card_name_encoder.get_num_cards()
        self.config.num_commanders = len(CommanderEncoder.COMMANDERS)
        self.config.num_skills = len(SkillEncoder.SKILLS)

    def encode(self, game_state: Dict, player_idx: int) -> np.ndarray:
        """
        编码完整游戏状态

        Args:
            game_state: 游戏状态字典
            player_idx: 当前AI控制的玩家索引 (0-based)

        Returns:
            编码后的状态向量
        """
        parts = []

        # 1. 全局状态
        parts.append(self._encode_global_state(game_state, player_idx))

        # 2. 当前玩家状态
        player = game_state["players"][player_idx]
        parts.append(self._encode_player_basic(player))

        # 3. 手牌编码
        parts.append(self._encode_hand_cards(player))

        # 4. 装备状态
        parts.append(self._encode_equipment(player))

        # 5. 判定区
        parts.append(self._encode_judge_area(player))

        # 6. 武将/技能
        parts.append(self._encode_character(player))

        # 7. 其他玩家
        parts.append(self._encode_other_players(game_state, player_idx))

        # 8. 历史动作
        parts.append(self._encode_action_history(game_state))

        return np.concatenate(parts)

    def _encode_global_state(self, state: Dict, player_idx: int) -> np.ndarray:
        """编码全局状态 (34维)"""
        parts = []

        # Phase one-hot (8维)
        phase_encoding = np.zeros(self.config.num_phases, dtype=np.float32)
        phase_map = {
            "waiting": 0,
            "turn_start": 1,
            "judge_phase": 2,
            "draw_phase": 3,
            "play_phase": 4,
            "discard_phase": 5,
            "turn_end": 6,
            "game_over": 7,
        }
        phase_str = state.get("phase", "waiting")
        if isinstance(phase_str, str):
            phase_encoding[phase_map.get(phase_str, 0)] = 1.0
        parts.append(phase_encoding)

        # Round number (1维)
        round_num = state.get("round_num", 1)
        parts.append(np.array([min(round_num / 50.0, 1.0)], dtype=np.float32))

        # Current player idx one-hot (8维)
        current_encoding = np.zeros(self.config.max_players, dtype=np.float32)
        current_idx = state.get("current_player_idx", 0)
        if 0 <= current_idx < self.config.max_players:
            current_encoding[current_idx] = 1.0
        parts.append(current_encoding)

        # My idx one-hot (8维)
        my_encoding = np.zeros(self.config.max_players, dtype=np.float32)
        if 0 <= player_idx < self.config.max_players:
            my_encoding[player_idx] = 1.0
        parts.append(my_encoding)

        # Deck count (1维)
        deck_count = state.get("deck_count", 0)
        parts.append(np.array([min(deck_count / 160.0, 1.0)], dtype=np.float32))

        # Discard pile count (1维)
        discard_count = state.get("discard_pile_count", 0)
        parts.append(np.array([min(discard_count / 160.0, 1.0)], dtype=np.float32))

        # Player num one-hot (7维, 2-8人)
        player_num_encoding = np.zeros(7, dtype=np.float32)
        player_num = len(state.get("players", []))
        if 2 <= player_num <= 8:
            player_num_encoding[player_num - 2] = 1.0
        parts.append(player_num_encoding)

        return np.concatenate(parts)

    def _encode_player_basic(self, player: Dict) -> np.ndarray:
        """编码玩家基础状态 (8维)"""
        parts = []

        # HP ratio (1维)
        max_hp = player.get("max_hp", 4)
        current_hp = player.get("current_hp", 4)
        parts.append(np.array([current_hp / max_hp], dtype=np.float32))

        # Max HP normalized (1维)
        parts.append(np.array([max_hp / 5.0], dtype=np.float32))

        # Is alive (1维)
        parts.append(
            np.array([1.0 if player.get("is_alive", True) else 0.0], dtype=np.float32)
        )

        # Is chained (1维)
        parts.append(
            np.array(
                [1.0 if player.get("is_chained", False) else 0.0], dtype=np.float32
            )
        )

        # Sha count (1维)
        sha_count = player.get("sha_count", 0)
        parts.append(np.array([min(sha_count / 10.0, 1.0)], dtype=np.float32))

        # Jiu count (1维)
        jiu_count = player.get("jiu_count", 0)
        parts.append(np.array([min(jiu_count / 2.0, 1.0)], dtype=np.float32))

        # Jiu effect (1维)
        parts.append(
            np.array(
                [1.0 if player.get("jiu_effect", 0) > 0 else 0.0], dtype=np.float32
            )
        )

        # Hand count (1维)
        hand_count = len(player.get("hand_cards", []))
        parts.append(np.array([min(hand_count / 20.0, 1.0)], dtype=np.float32))

        return np.concatenate(parts)

    def _encode_hand_cards(self, player: Dict) -> np.ndarray:
        """编码手牌"""
        hand_cards = player.get("hand_cards", [])
        max_hand = self.config.max_hand_size

        # 每张牌的编码维度 - 使用实际编码器长度
        card_dim = (
            len(CardTypeEncoder.CARD_TYPES)
            + len(CardNameEncoder.CARD_NAMES)
            + 4  # color
            + 1  # point
            + 1  # is_red
            + 1  # is_black
        )

        encoded_cards = np.zeros((max_hand, card_dim), dtype=np.float32)

        for i, card in enumerate(hand_cards[:max_hand]):
            if isinstance(card, dict):
                encoded_cards[i] = self._encode_single_card(card)
            else:
                encoded_cards[i] = self._encode_single_card_from_object(card)

        return encoded_cards.flatten()

    def _encode_single_card(self, card: Dict) -> np.ndarray:
        """编码单张卡牌"""
        parts = []

        # Card type one-hot (15维)
        card_type = card.get("card_type", "BasicCard")
        parts.append(self.card_type_encoder.encode(card_type))

        # Card name one-hot
        card_name = card.get("name", "")
        parts.append(self.card_name_encoder.encode(card_name))

        # Color one-hot (4维)
        color_encoding = np.zeros(4, dtype=np.float32)
        color_map = {"黑桃": 0, "红桃": 1, "梅花": 2, "方块": 3}
        color = card.get("color", "")
        if color in color_map:
            color_encoding[color_map[color]] = 1.0
        parts.append(color_encoding)

        # Point normalized (1维)
        point = card.get("point", 0)
        parts.append(np.array([point / 13.0], dtype=np.float32))

        # Is red (1维)
        parts.append(
            np.array([1.0 if color in ["红桃", "方块"] else 0.0], dtype=np.float32)
        )

        # Is black (1维)
        parts.append(
            np.array([1.0 if color in ["黑桃", "梅花"] else 0.0], dtype=np.float32)
        )

        return np.concatenate(parts)

    def _encode_single_card_from_object(self, card) -> np.ndarray:
        """从Card对象编码"""
        parts = []

        parts.append(
            self.card_type_encoder.encode(getattr(card, "card_type", "BasicCard"))
        )
        parts.append(self.card_name_encoder.encode(getattr(card, "name", "")))

        color_encoding = np.zeros(4, dtype=np.float32)
        color_map = {"黑桃": 0, "红桃": 1, "梅花": 2, "方块": 3}
        color = getattr(card, "color", "")
        if color in color_map:
            color_encoding[color_map[color]] = 1.0
        parts.append(color_encoding)

        point = getattr(card, "point", 0)
        parts.append(np.array([point / 13.0], dtype=np.float32))

        is_red = 1.0 if hasattr(card, "is_red") and card.is_red() else 0.0
        parts.append(np.array([is_red], dtype=np.float32))

        is_black = 1.0 if hasattr(card, "is_black") and card.is_black() else 0.0
        parts.append(np.array([is_black], dtype=np.float32))

        return np.concatenate(parts)

    def _encode_equipment(self, player: Dict) -> np.ndarray:
        """编码装备状态 (25维)"""
        parts = []

        equipment = player.get("equipment", {})

        # Weapon one-hot (15维)
        weapon = equipment.get("武器")
        weapon_name = weapon.get("name", "") if weapon else ""
        parts.append(self._encode_weapon(weapon_name))

        # Armour one-hot (7维)
        armour = equipment.get("防具")
        armour_name = armour.get("name", "") if armour else ""
        parts.append(self._encode_armour(armour_name))

        # Attack horse (1维)
        parts.append(
            np.array([1.0 if equipment.get("进攻坐骑") else 0.0], dtype=np.float32)
        )

        # Defense horse (1维)
        parts.append(
            np.array([1.0 if equipment.get("防御坐骑") else 0.0], dtype=np.float32)
        )

        # Treasure (1维)
        parts.append(
            np.array([1.0 if equipment.get("宝物") else 0.0], dtype=np.float32)
        )

        return np.concatenate(parts)

    def _encode_weapon(self, weapon_name: str) -> np.ndarray:
        """编码武器"""
        weapons = [
            "",
            "诸葛连弩",
            "青釭剑",
            "青龙偃月刀",
            "丈八蛇矛",
            "贯石斧",
            "方天画戟",
            "麒麟弓",
            "朱雀羽扇",
            "古锭刀",
            "吴六剑",
            "三尖两刃刀",
            "太平要术",
            "雌雄双股剑",
            "寒冰剑",
        ]
        encoding = np.zeros(len(weapons), dtype=np.float32)
        if weapon_name in weapons:
            encoding[weapons.index(weapon_name)] = 1.0
        return encoding

    def _encode_armour(self, armour_name: str) -> np.ndarray:
        """编码防具"""
        armours = ["", "八卦阵", "仁王盾", "白银狮子", "藤甲", "黄金甲", "国风玉袍"]
        encoding = np.zeros(len(armours), dtype=np.float32)
        if armour_name in armours:
            encoding[armours.index(armour_name)] = 1.0
        return encoding

    def _encode_judge_area(self, player: Dict) -> np.ndarray:
        """编码判定区 (3维)"""
        judge_area = player.get("judge_area", [])

        has_le = any(c.get("name") == "乐不思蜀" for c in judge_area)
        has_bing = any(c.get("name") == "兵粮寸断" for c in judge_area)
        has_lightning = any(c.get("name") == "闪电" for c in judge_area)

        return np.array(
            [
                1.0 if has_le else 0.0,
                1.0 if has_bing else 0.0,
                1.0 if has_lightning else 0.0,
            ],
            dtype=np.float32,
        )

    def _encode_character(self, player: Dict) -> np.ndarray:
        """编码武将和技能 (98维)"""
        parts = []

        # Commander one-hot (22维)
        commander_name = player.get("commander_name", "")
        parts.append(self.commander_encoder.encode(commander_name))

        # Nation one-hot (4维)
        nation_encoding = np.zeros(4, dtype=np.float32)
        nation_map = {"魏": 0, "蜀": 1, "吴": 2, "群": 3}
        nation = player.get("nation", "")
        if nation in nation_map:
            nation_encoding[nation_map[nation]] = 1.0
        parts.append(nation_encoding)

        # Skills multi-hot (36维)
        skills = player.get("skills", [])
        if skills and isinstance(skills[0], str):
            parts.append(self.skill_encoder.encode(skills))
        else:
            skill_names = [s.name for s in skills if hasattr(s, "name")]
            parts.append(self.skill_encoder.encode(skill_names))

        # Skill used multi-hot (36维)
        parts.append(np.zeros(len(SkillEncoder.SKILLS), dtype=np.float32))

        return np.concatenate(parts)

    def _encode_other_players(self, state: Dict, player_idx: int) -> np.ndarray:
        """编码其他玩家状态 (7人 × 73维 = 511维)"""
        players = state.get("players", [])
        encoded_players = []

        for i, player in enumerate(players):
            if i == player_idx:
                continue

            parts = []

            # Player idx one-hot (8维)
            idx_encoding = np.zeros(self.config.max_players, dtype=np.float32)
            idx_encoding[i] = 1.0
            parts.append(idx_encoding)

            # HP ratio (1维)
            max_hp = player.get("max_hp", 4)
            current_hp = player.get("current_hp", 4)
            parts.append(np.array([current_hp / max_hp], dtype=np.float32))

            # Max HP (1维)
            parts.append(np.array([max_hp / 5.0], dtype=np.float32))

            # Is alive (1维)
            parts.append(
                np.array(
                    [1.0 if player.get("is_alive", True) else 0.0], dtype=np.float32
                )
            )

            # Is chained (1维)
            parts.append(
                np.array(
                    [1.0 if player.get("is_chained", False) else 0.0], dtype=np.float32
                )
            )

            # Hand count (1维)
            hand_count = len(player.get("hand_cards", []))
            parts.append(np.array([min(hand_count / 20.0, 1.0)], dtype=np.float32))

            # Identity (4维) - 主公公开，其他隐藏
            identity_encoding = np.zeros(4, dtype=np.float32)
            identity = player.get("identity", "")
            if identity == "主公":
                identity_encoding[0] = 1.0
            parts.append(identity_encoding)

            # Identity known (1维)
            identity_known = 1.0 if identity == "主公" else 0.0
            parts.append(np.array([identity_known], dtype=np.float32))

            # Nation (4维)
            nation_encoding = np.zeros(4, dtype=np.float32)
            nation_map = {"魏": 0, "蜀": 1, "吴": 2, "群": 3}
            nation = player.get("nation", "")
            if nation in nation_map:
                nation_encoding[nation_map[nation]] = 1.0
            parts.append(nation_encoding)

            # Commander (22维)
            commander_name = player.get("commander_name", "")
            parts.append(self.commander_encoder.encode(commander_name))

            # Equipment (25维)
            parts.append(self._encode_equipment(player))

            # Judge area (3维)
            parts.append(self._encode_judge_area(player))

            # Distance (1维) - 估算
            distance = abs(i - player_idx)
            distance = min(distance, len(players) - distance)
            parts.append(np.array([distance / 7.0], dtype=np.float32))

            # Threat level (1维) - 简单估算
            threat = self._estimate_threat(player)
            parts.append(np.array([threat], dtype=np.float32))

            encoded_players.append(np.concatenate(parts))

        # Padding to 7 players
        while len(encoded_players) < 7:
            encoded_players.append(np.zeros(73, dtype=np.float32))

        return np.concatenate(encoded_players[:7])

    def _estimate_threat(self, player: Dict) -> float:
        """估算玩家威胁度"""
        threat = 0.0

        # HP factor
        max_hp = player.get("max_hp", 4)
        current_hp = player.get("current_hp", 4)
        hp_ratio = current_hp / max_hp if max_hp > 0 else 0
        threat += hp_ratio * 0.3

        # Hand count factor
        hand_count = len(player.get("hand_cards", []))
        threat += min(hand_count / 10.0, 1.0) * 0.3

        # Equipment factor
        equipment = player.get("equipment", {})
        equip_count = sum(1 for v in equipment.values() if v)
        threat += min(equip_count / 4.0, 1.0) * 0.4

        return min(threat, 1.0)

    def _encode_action_history(self, state: Dict) -> np.ndarray:
        """编码历史动作 (10条 × 78维 = 780维)"""
        history = state.get("action_history", [])
        max_history = self.config.max_history_actions

        encoded_history = []

        for action in history[-max_history:]:
            parts = []

            # Action type one-hot (10维)
            action_type_encoding = np.zeros(
                self.config.num_action_types, dtype=np.float32
            )
            action_type = action.get("action_type", 0)
            if 0 <= action_type < self.config.num_action_types:
                action_type_encoding[action_type] = 1.0
            parts.append(action_type_encoding)

            # Card name one-hot (60维)
            card_name = action.get("card_name", "")
            parts.append(self.card_name_encoder.encode(card_name))

            # Target idx one-hot (8维)
            target_encoding = np.zeros(self.config.max_players, dtype=np.float32)
            target_idx = action.get("target_idx")
            if target_idx is not None and 0 <= target_idx < self.config.max_players:
                target_encoding[target_idx] = 1.0
            parts.append(target_encoding)

            encoded_history.append(np.concatenate(parts))

        # Padding
        action_dim = (
            self.config.num_action_types
            + self.config.num_card_names
            + self.config.max_players
        )
        while len(encoded_history) < max_history:
            encoded_history.append(np.zeros(action_dim, dtype=np.float32))

        return np.concatenate(encoded_history)

    def get_state_dim(self, player_num: int = 5) -> int:
        """获取状态向量维度"""
        return len(
            self.encode(
                {
                    "players": [
                        {
                            "hand_cards": [],
                            "equipment": {},
                            "judge_area": [],
                            "skills": [],
                        }
                    ]
                    * player_num
                },
                0,
            )
        )

    def get_observation_spec(self) -> Dict[str, Any]:
        """获取观察空间规格"""
        state_dim = self.get_state_dim()
        return {
            "state": (state_dim,),
            "dtype": np.float32,
        }
