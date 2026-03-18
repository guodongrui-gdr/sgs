from dataclasses import dataclass, field
from typing import List, Optional, Dict, TYPE_CHECKING

from card.base import (
    Card,
    WeaponCard,
    ArmourCard,
    AttackHorseCard,
    DefenseHorseCard,
    TreasureCard,
)
from skills.base import Skill

if TYPE_CHECKING:
    pass


@dataclass
class Player:
    idx: int = 0
    commander_id: str = ""
    commander_name: str = ""
    nation: str = ""
    identity: str = ""
    gender: str = "male"

    max_hp: int = 4
    current_hp: int = 4

    hand_cards: List[Card] = field(default_factory=list)
    equipment: Dict[str, Optional[Card]] = field(default_factory=dict)
    judge_area: List[Card] = field(default_factory=list)

    skills: List[Skill] = field(default_factory=list)

    is_alive: bool = True
    is_chained: bool = False

    sha_count: int = 0
    jiu_count: int = 0
    jiu_effect: int = 0
    unlimited_sha: bool = False

    next_player: Optional["Player"] = None
    prev_player: Optional["Player"] = None

    is_human: bool = False

    has_mashu: bool = False
    cards_to_draw: int = 2
    luoyi_active: bool = False
    keji_active: bool = False
    zhiheng_used: bool = False
    fanjian_used: bool = False
    jieyin_used: bool = False
    qingnang_used: bool = False
    lijian_used: bool = False

    def __post_init__(self):
        if not self.equipment:
            self.equipment = {
                "武器": None,
                "防具": None,
                "进攻坐骑": None,
                "防御坐骑": None,
                "宝物": None,
            }

    @property
    def hand_limit(self) -> int:
        return self.current_hp

    @property
    def attack_range(self) -> int:
        if self.equipment.get("武器"):
            return self.equipment["武器"].distance
        return 1

    def can_use_sha(self) -> bool:
        if self.unlimited_sha:
            return True
        return self.sha_count < 1

    def equip(self, card: Card):
        if isinstance(card, WeaponCard):
            if self.equipment["武器"]:
                pass
            self.equipment["武器"] = card
        elif isinstance(card, ArmourCard):
            if self.equipment["防具"]:
                pass
            self.equipment["防具"] = card
        elif isinstance(card, AttackHorseCard):
            if self.equipment["进攻坐骑"]:
                pass
            self.equipment["进攻坐骑"] = card
        elif isinstance(card, DefenseHorseCard):
            if self.equipment["防御坐骑"]:
                pass
            self.equipment["防御坐骑"] = card
        elif isinstance(card, TreasureCard):
            if self.equipment["宝物"]:
                pass
            self.equipment["宝物"] = card

    def take_damage(self, damage: int):
        self.current_hp -= damage
        if self.current_hp <= 0:
            self.current_hp = 0

    def heal(self, amount: int):
        self.current_hp = min(self.current_hp + amount, self.max_hp)

    def reset_turn_state(self):
        self.sha_count = 0
        self.jiu_count = 0
        self.jiu_effect = 0
        self.unlimited_sha = False
        self.has_mashu = False
        self.cards_to_draw = 2
        self.luoyi_active = False
        self.keji_active = False
        self.zhiheng_used = False
        self.fanjian_used = False
        self.jieyin_used = False
        self.qingnang_used = False
        self.lijian_used = False

    def to_dict(self) -> Dict:
        return {
            "idx": self.idx,
            "commander_name": self.commander_name,
            "nation": self.nation,
            "identity": self.identity,
            "max_hp": self.max_hp,
            "current_hp": self.current_hp,
            "hand_count": len(self.hand_cards),
            "equipment": {k: v.name if v else None for k, v in self.equipment.items()},
            "is_alive": self.is_alive,
            "is_chained": self.is_chained,
            "skills": [s.name for s in self.skills],
        }
