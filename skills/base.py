from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from player.player import Player
    from engine.event import Event, EventType
    from engine.game_engine import GameEngine


@dataclass
class Skill(ABC):
    name: str
    trigger_events: List["EventType"]
    description: str = ""

    player: Optional["Player"] = None

    def bind_player(self, player: "Player"):
        self.player = player

    @abstractmethod
    def can_activate(self, event: "Event", engine: "GameEngine") -> bool:
        pass

    @abstractmethod
    def execute(self, event: "Event", engine: "GameEngine") -> Optional["Event"]:
        pass

    def on_event(self, event: "Event") -> Optional["Event"]:
        from engine.game_engine import GameEngine

        engine = getattr(event, "engine", None)
        if engine is None:
            return None

        if self.can_activate(event, engine):
            return self.execute(event, engine)
        return None

    def ask_player(self, message: str) -> bool:
        if self.player and self.player.is_human:
            response = input(f"{message} (y/n): ")
            return response.lower() == "y"
        return True


class ActiveSkill(Skill):
    def can_activate(self, event: "Event", engine: "GameEngine") -> bool:
        return False

    def execute(self, event: "Event", engine: "GameEngine") -> Optional["Event"]:
        return None


class PassiveSkill(Skill):
    def can_activate(self, event: "Event", engine: "GameEngine") -> bool:
        return False

    def execute(self, event: "Event", engine: "GameEngine") -> Optional["Event"]:
        return None


class TriggerSkill(Skill):
    def can_activate(self, event: "Event", engine: "GameEngine") -> bool:
        return False

    def execute(self, event: "Event", engine: "GameEngine") -> Optional["Event"]:
        return None
