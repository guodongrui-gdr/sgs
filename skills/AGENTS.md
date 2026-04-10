# Skills Module

Event-driven skill system with 43+ skills organized by faction.

## Overview

Skills subscribe to game events via EventBus. Passive skills respond automatically, active skills require player input.

## Structure

```
skills/
├── base.py       # Abstract classes: Skill, ActiveSkill, TriggerSkill, PassiveSkill
├── registry.py   # SkillRegistry singleton + @skill_decorator
├── wei.py        # 9 Wei skills: 奸雄, 鬼才, 反馈, 刚烈, 突袭, 裸衣, 天妒, 遗计, 洛神
├── shu.py        # 10 Shu skills: 仁德, 武圣, 咆哮, 观星, 龙胆, 马术, 铁骑, 集智, 奇才, 激将
├── wu.py         # 13 Wu skills: 制衡, 奇袭, 克己, 苦肉, 英姿, 反间, 国色, 流离, 谦逊, 连营, 结姻, 枭姬, 救援
└── qun.py        # 5 Qun skills: 急救, 青囊, 无双, 离间, 闭月
```

## Where to Look

| Task | File |
|------|------|
| Add new skill | `skills/{faction}.py` → add to `__all__` in `__init__.py` |
| Change skill base class | `skills/base.py` |
| Skill registration | `skills/registry.py` |
| RL skill decisions | `ai/skill_decision.py` |
| Event types | `engine/event.py` |

## Conventions

**Subclass Structure:**
```python
@SkillRegistry.register
class MySkill(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="技能名",
            trigger_events=[EventType.DAMAGE_TAKEN],
            description="描述"
        )

    def can_activate(self, event, engine) -> bool:
        return event.target == self.player

    def execute(self, event, engine) -> Optional[Event]:
        return event
```

**Turn Limits:** Active skills set `self.max_uses_per_turn` and call `self.use()`.

**Player Association:** `skill.bind_player(player)` called by engine.

**Human vs AI:** Use `ask_decision()` for RL when `not self.player.is_human`.

## Anti-Patterns

**Don't check skills by name string:**
```python
# Wrong: if "反馈" in player.skills
# Right: SkillRegistry.has_skill("反馈")
```

**Don't modify event.card without null check:** Always guard with `if not event.card: return event`.

**Don't forget to handle max_uses_per_turn:** Check `self.can_use()` before activation.

**Don't import game engine at module level:** Use `TYPE_CHECKING` for type hints.
