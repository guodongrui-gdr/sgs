from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from player.player import Player


def is_sha_card(card) -> bool:
    """检查卡牌是否是杀（包括普通杀、火杀、雷杀）"""
    return isinstance(card, ShaCard)


class Card:
    def __init__(
        self,
        name: str,
        color: str,
        point: int,
        card_type: str = "Card",
        target_types: List[str] = None,
        distance: int = 0,
    ):
        self.name = name
        self.color = color
        self.point = point
        self.card_type = card_type
        self.target_types = target_types or []
        self.distance = distance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "color": self.color,
            "point": self.point,
            "card_type": self.card_type,
            "target_types": self.target_types,
            "distance": self.distance,
        }

    def is_red(self) -> bool:
        return self.color in ["红桃", "方块"]

    def is_black(self) -> bool:
        return self.color in ["黑桃", "梅花"]

    def __repr__(self) -> str:
        return f"{self.color}{self.point}{self.name}"


class BasicCard(Card):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "BasicCard")
        super().__init__(name, color, point, **kwargs)

        if name == "闪":
            self.target_types = []
        elif name == "桃":
            self.target_types = ["self", "dying_player"]
        elif name == "酒":
            self.target_types = ["self"]


class ShaCard(BasicCard):
    """杀类卡牌的基类"""

    def __init__(self, name: str = "杀", color: str = "", point: int = 0, **kwargs):
        super().__init__(name, color, point, **kwargs)
        self.target_types = ["another_player"]
        self.distance = 1
        self.is_elemental = False

    def is_sha(self) -> bool:
        return True


class FireSha(ShaCard):
    """火杀"""

    def __init__(self, name: str = "火杀", color: str = "", point: int = 0, **kwargs):
        super().__init__(name, color, point, **kwargs)
        self.is_elemental = True
        self.is_fire = True


class ThunderSha(ShaCard):
    """雷杀"""

    def __init__(self, name: str = "雷杀", color: str = "", point: int = 0, **kwargs):
        super().__init__(name, color, point, **kwargs)
        self.is_elemental = True
        self.is_thunder = True


class JinnangCard(Card):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "JinnangCard")
        super().__init__(name, color, point, **kwargs)


class CommonJinnangCard(JinnangCard):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "CommonJinnangCard")
        super().__init__(name, color, point, **kwargs)

        if name == "决斗":
            self.target_types = ["another_player"]
        elif name == "无中生有":
            self.target_types = ["self"]
        elif name == "过河拆桥":
            self.target_types = ["another_player_with_cards"]
        elif name == "顺手牵羊":
            self.target_types = ["another_player_with_cards"]
            self.distance = 1
        elif name == "借刀杀人":
            self.target_types = ["player_with_weapon"]
        elif name == "南蛮入侵":
            self.target_types = ["all_other_players"]
        elif name == "万箭齐发":
            self.target_types = ["all_other_players"]
        elif name == "桃园结义":
            self.target_types = ["all_players"]
        elif name == "五谷丰登":
            self.target_types = ["all_players"]
        elif name == "无懈可击":
            self.target_types = []
        elif name == "火攻":
            self.target_types = ["player_with_hand_cards"]
        elif name == "铁索连环":
            self.target_types = ["one_or_two_players"]


class YanshiJinnangCard(JinnangCard):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "YanshiJinnangCard")
        super().__init__(name, color, point, **kwargs)

        if name == "乐不思蜀":
            self.target_types = ["another_player"]
            self.distance = 1
        elif name == "兵粮寸断":
            self.target_types = ["another_player"]
            self.distance = 1
        elif name == "闪电":
            self.target_types = ["self"]


class EquipmentCard(Card):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "EquipmentCard")
        kwargs.setdefault("target_types", ["self"])
        super().__init__(name, color, point, **kwargs)


class WeaponCard(EquipmentCard):
    def __init__(self, name: str, color: str, point: int, dis: int = 1, **kwargs):
        kwargs.setdefault("card_type", "WeaponCard")
        super().__init__(name, color, point, **kwargs)
        self.attack_range = dis
        self.distance = dis


class ArmourCard(EquipmentCard):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "ArmourCard")
        super().__init__(name, color, point, **kwargs)


class AttackHorseCard(EquipmentCard):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "AttackHorseCard")
        super().__init__(name, color, point, **kwargs)


class DefenseHorseCard(EquipmentCard):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "DefenseHorseCard")
        super().__init__(name, color, point, **kwargs)


class TreasureCard(EquipmentCard):
    def __init__(self, name: str, color: str, point: int, **kwargs):
        kwargs.setdefault("card_type", "TreasureCard")
        super().__init__(name, color, point, **kwargs)
