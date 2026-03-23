from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import logging

if TYPE_CHECKING:
    from player.player import Player
    from engine.event import Event, EventType
    from engine.game_engine import GameEngine

logger = logging.getLogger(__name__)

_current_env = None


def set_current_env(env):
    global _current_env
    _current_env = env


def get_current_env():
    return _current_env


def clear_current_env():
    global _current_env
    _current_env = None


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

        from ai.skill_decision import has_cached_decision, get_cached_decision

        if has_cached_decision(self.name):
            result = get_cached_decision(self.name)
            return result if isinstance(result, bool) else True

        env = get_current_env()
        if env is not None and hasattr(env, "request_skill_decision"):
            from ai.skill_decision import create_yes_no_request

            request = create_yes_no_request(self.name, message)
            env.request_skill_decision(request)

        return True

    def ask_decision(
        self,
        decision_type,
        options,
        min_selections=1,
        max_selections=1,
        context=None,
        default=None,
    ):
        """发起技能决策请求 - 返回None时使用default"""
        from ai.skill_decision import has_cached_decision, get_cached_decision

        if has_cached_decision(self.name):
            return get_cached_decision(self.name)

        env = get_current_env()
        if env is not None and hasattr(env, "request_skill_decision"):
            from ai.skill_decision import SkillDecisionRequest

            request = SkillDecisionRequest(
                decision_type=decision_type,
                skill_name=self.name,
                options=options,
                min_selections=min_selections,
                max_selections=max_selections,
                context=context or {},
            )
            env.request_skill_decision(request)

        return default

    def ask_select_order(self, items, min_selections=None, default=None):
        """请求选择排列顺序"""
        from ai.skill_decision import SkillDecisionType

        return self.ask_decision(
            SkillDecisionType.SELECT_ORDER,
            items,
            min_selections=min_selections or len(items),
            max_selections=len(items),
            default=default,
        )

    def ask_select_pair(self, options, default=None):
        """请求选择一对"""
        from ai.skill_decision import SkillDecisionType

        return self.ask_decision(
            SkillDecisionType.SELECT_PAIR,
            options,
            min_selections=2,
            max_selections=2,
            default=default,
        )

    def ask_distribute(self, items, targets, default=None):
        """请求分配物品给目标"""
        from ai.skill_decision import SkillDecisionType

        return self.ask_decision(
            SkillDecisionType.DISTRIBUTE,
            targets,
            min_selections=len(items),
            max_selections=len(items),
            context={"items": items},
            default=default,
        )


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
