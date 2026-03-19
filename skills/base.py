from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

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

    max_uses_per_turn: int = field(default=0, repr=False)
    used_this_turn: int = field(default=0, repr=False)

    def bind_player(self, player: "Player"):
        self.player = player

    def can_use(self) -> bool:
        """检查技能本回合是否还能使用"""
        if self.max_uses_per_turn == 0:
            return True
        return self.used_this_turn < self.max_uses_per_turn

    def use(self) -> None:
        """标记技能已使用一次"""
        self.used_this_turn += 1

    def reset_turn_state(self) -> None:
        """重置回合状态"""
        self.used_this_turn = 0

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
    """主动技能基类"""

    def can_activate(self, event: "Event", engine: "GameEngine") -> bool:
        """事件触发时检查（被动触发用）"""
        return False

    def execute(self, event: "Event", engine: "GameEngine") -> Optional["Event"]:
        return None

    def is_available(self, engine: "GameEngine") -> bool:
        """
        检查技能当前是否可用（供AI决策用）

        子类应覆盖此方法，检查技能的使用条件：
        - 是否有足够的手牌
        - 是否满足发动条件
        - 是否有合法目标等

        Returns:
            True if skill can be used right now
        """
        if self.player is None:
            return False
        if not self.can_use():
            return False
        return True


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
