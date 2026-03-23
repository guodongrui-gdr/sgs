import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import (
    Card,
    BasicCard,
    ShaCard,
    FireSha,
    ThunderSha,
    JinnangCard,
    CommonJinnangCard,
    YanshiJinnangCard,
    EquipmentCard,
    WeaponCard,
    ArmourCard,
    AttackHorseCard,
    DefenseHorseCard,
    TreasureCard,
)


class CardFactory:
    _type_mapping = {
        "BasicCard": BasicCard,
        "ShaCard": ShaCard,
        "FireSha": FireSha,
        "ThunderSha": ThunderSha,
        "CommonJinnangCard": CommonJinnangCard,
        "YanshiJinnangCard": YanshiJinnangCard,
        "WeaponCard": WeaponCard,
        "ArmourCard": ArmourCard,
        "AttackHorseCard": AttackHorseCard,
        "DefenseHorseCard": DefenseHorseCard,
        "TreasureCard": TreasureCard,
    }

    _name_type_mapping = {
        "杀": ShaCard,
        "火杀": FireSha,
        "雷杀": ThunderSha,
    }

    @classmethod
    def create(cls, config: Dict[str, Any]) -> List[Card]:
        config = config.copy()
        card_type = config.pop("type", "BasicCard")
        count = config.pop("count", 1)
        card_name = config.get("name", "")

        if card_type == "BasicCard" and card_name in cls._name_type_mapping:
            card_class = cls._name_type_mapping[card_name]
        else:
            card_class = cls._type_mapping.get(card_type, BasicCard)

        cards = []
        for _ in range(count):
            card = card_class(**config)
            cards.append(card)

        return cards

    @classmethod
    def load_from_config(cls, config_path: Path) -> List[Card]:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        all_cards = []
        for card_config in data.get("cards", []):
            cards = cls.create(card_config)
            all_cards.extend(cards)

        return all_cards

    @classmethod
    def register_type(cls, type_name: str, card_class: type):
        cls._type_mapping[type_name] = card_class


def create_empty_card() -> Card:
    return Card(name="", color="", point=0)
