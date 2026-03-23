"""
技能决策系统 - 支持RL参与技能内部决策

决策流程 (异步模式):
1. 技能执行时检查是否有缓存的结果
2. 如果没有结果，发起决策请求并返回 PAUSE 信号
3. 环境检测到决策请求，返回特殊observation
4. RL通过多次step输出决策结果
5. 环境缓存结果，重新调用技能
6. 技能使用缓存的结果继续执行
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from card.base import Card
    from player.player import Player


class SkillDecisionType(IntEnum):
    """技能决策类型"""

    YES_NO = 0
    SELECT_CARDS = 1
    SELECT_TARGETS = 2
    SELECT_ORDER = 3
    DISTRIBUTE = 4
    SELECT_PAIR = 5
    SELECT_SINGLE = 6


class SkillExecutionStatus(IntEnum):
    """技能执行状态"""

    CONTINUE = 0
    PAUSE = 1
    DONE = 2


@dataclass
class SkillDecisionRequest:
    """技能决策请求"""

    decision_type: SkillDecisionType
    skill_name: str
    prompt: str = ""

    options: List[Any] = field(default_factory=list)
    min_selections: int = 1
    max_selections: int = 1

    context: Dict = field(default_factory=dict)

    result: Optional[Any] = None
    is_resolved: bool = False

    _current_step: int = 0
    _selections: List[int] = field(default_factory=list)

    def get_remaining_options(self) -> List[int]:
        """获取剩余可选选项的索引"""
        return [i for i in range(len(self.options)) if i not in self._selections]

    def add_selection(self, idx: int) -> bool:
        """添加选择"""
        if idx in self._selections:
            return False
        if idx < 0 or idx >= len(self.options):
            return False
        if len(self._selections) >= self.max_selections:
            return False
        self._selections.append(idx)
        return True

    def is_complete(self) -> bool:
        """检查决策是否完成"""
        if self.decision_type == SkillDecisionType.YES_NO:
            return self.result is not None
        elif self.decision_type == SkillDecisionType.SELECT_ORDER:
            return len(self._selections) >= self.min_selections
        elif self.decision_type == SkillDecisionType.SELECT_PAIR:
            return len(self._selections) >= 2
        elif self.decision_type in (
            SkillDecisionType.SELECT_CARDS,
            SkillDecisionType.SELECT_TARGETS,
        ):
            return len(self._selections) >= self.min_selections
        elif self.decision_type == SkillDecisionType.DISTRIBUTE:
            return self.result is not None
        return self.result is not None

    def get_result(self) -> Any:
        """获取决策结果"""
        if self.decision_type == SkillDecisionType.YES_NO:
            return self.result
        elif self.decision_type == SkillDecisionType.SELECT_ORDER:
            return self._selections.copy()
        elif self.decision_type == SkillDecisionType.SELECT_PAIR:
            if len(self._selections) >= 2:
                return (self._selections[0], self._selections[1])
            return None
        elif self.decision_type in (
            SkillDecisionType.SELECT_CARDS,
            SkillDecisionType.SELECT_TARGETS,
        ):
            return self._selections.copy()
        return self.result


@dataclass
class SkillDecisionContext:
    """技能决策上下文 - 存储在环境中"""

    active_request: Optional[SkillDecisionRequest] = None
    cached_result: Optional[Any] = None

    pending_skill: Optional[Any] = None
    pending_event: Optional[Any] = None
    pending_engine: Optional[Any] = None

    def has_pending_decision(self) -> bool:
        return self.active_request is not None and not self.active_request.is_resolved

    def clear(self):
        self.active_request = None
        self.cached_result = None
        self.pending_skill = None
        self.pending_event = None
        self.pending_engine = None


_cached_decisions: Dict[str, Any] = {}


def cache_decision(skill_name: str, result: Any):
    """缓存决策结果"""
    _cached_decisions[skill_name] = result


def get_cached_decision(skill_name: str) -> Optional[Any]:
    """获取缓存的决策结果"""
    return _cached_decisions.get(skill_name)


def clear_cached_decision(skill_name: str):
    """清除缓存的决策结果"""
    if skill_name in _cached_decisions:
        del _cached_decisions[skill_name]


def has_cached_decision(skill_name: str) -> bool:
    """检查是否有缓存的决策结果"""
    return skill_name in _cached_decisions


def create_yes_no_request(skill_name: str, prompt: str = "") -> SkillDecisionRequest:
    """创建是否发动决策请求"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.YES_NO,
        skill_name=skill_name,
        prompt=prompt,
        options=["否", "是"],
        min_selections=1,
        max_selections=1,
    )


def create_select_order_request(
    skill_name: str,
    items: List[Any],
    prompt: str = "",
    min_selections: int = None,
) -> SkillDecisionRequest:
    """创建排列顺序决策请求 (观星)"""
    n = len(items)
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.SELECT_ORDER,
        skill_name=skill_name,
        prompt=prompt,
        options=items,
        min_selections=min_selections or n,
        max_selections=n,
    )


def create_select_pair_request(
    skill_name: str,
    options: List[Any],
    prompt: str = "",
) -> SkillDecisionRequest:
    """创建选择一对决策请求 (离间)"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.SELECT_PAIR,
        skill_name=skill_name,
        prompt=prompt,
        options=options,
        min_selections=2,
        max_selections=2,
    )


def create_distribute_request(
    skill_name: str,
    items: List[Any],
    targets: List[Any],
    prompt: str = "",
) -> SkillDecisionRequest:
    """创建分配决策请求 (遗计)"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.DISTRIBUTE,
        skill_name=skill_name,
        prompt=prompt,
        options=targets,
        context={"items": items},
        min_selections=len(items),
        max_selections=len(items),
    )


def create_select_cards_request(
    skill_name: str,
    cards: List[Any],
    prompt: str = "",
    min_selections: int = 1,
    max_selections: int = 1,
) -> SkillDecisionRequest:
    """创建选择卡牌决策请求"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.SELECT_CARDS,
        skill_name=skill_name,
        prompt=prompt,
        options=cards,
        min_selections=min_selections,
        max_selections=max_selections,
    )


def create_select_targets_request(
    skill_name: str,
    targets: List[Any],
    prompt: str = "",
    min_selections: int = 1,
    max_selections: int = 1,
) -> SkillDecisionRequest:
    """创建选择目标决策请求"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.SELECT_TARGETS,
        skill_name=skill_name,
        prompt=prompt,
        options=targets,
        min_selections=min_selections,
        max_selections=max_selections,
    )


def create_select_order_request(
    skill_name: str,
    items: List[Any],
    prompt: str = "",
    min_selections: int = None,
) -> SkillDecisionRequest:
    """创建排列顺序决策请求 (观星)"""
    n = len(items)
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.SELECT_ORDER,
        skill_name=skill_name,
        prompt=prompt,
        options=items,
        min_selections=min_selections or n,
        max_selections=n,
    )


def create_select_pair_request(
    skill_name: str,
    options: List[Any],
    prompt: str = "",
) -> SkillDecisionRequest:
    """创建选择一对决策请求 (离间)"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.SELECT_PAIR,
        skill_name=skill_name,
        prompt=prompt,
        options=options,
        min_selections=2,
        max_selections=2,
    )


def create_distribute_request(
    skill_name: str,
    items: List[Any],
    targets: List[Any],
    prompt: str = "",
) -> SkillDecisionRequest:
    """创建分配决策请求 (遗计)"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.DISTRIBUTE,
        skill_name=skill_name,
        prompt=prompt,
        options=targets,
        context={"items": items},
        min_selections=len(items),
        max_selections=len(items),
    )


def create_select_cards_request(
    skill_name: str,
    cards: List[Any],
    prompt: str = "",
    min_selections: int = 1,
    max_selections: int = 1,
) -> SkillDecisionRequest:
    """创建选择卡牌决策请求"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.SELECT_CARDS,
        skill_name=skill_name,
        prompt=prompt,
        options=cards,
        min_selections=min_selections,
        max_selections=max_selections,
    )


def create_select_targets_request(
    skill_name: str,
    targets: List[Any],
    prompt: str = "",
    min_selections: int = 1,
    max_selections: int = 1,
) -> SkillDecisionRequest:
    """创建选择目标决策请求"""
    return SkillDecisionRequest(
        decision_type=SkillDecisionType.SELECT_TARGETS,
        skill_name=skill_name,
        prompt=prompt,
        options=targets,
        min_selections=min_selections,
        max_selections=max_selections,
    )
