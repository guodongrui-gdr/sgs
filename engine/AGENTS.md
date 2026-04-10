# Game Engine Module

**Generated:** 2026-04-09

## Overview

Event-driven game engine with 40+ event types, pub-sub skill triggering, and turn-based state machine.

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Game orchestrator | `game_engine.py` | 883 lines, 35 methods |
| Event types | `event.py` | EventType enum (40+ types) |
| Event bus | `event_bus.py` | Pub-sub with priority |
| Game state | `state.py` | GameState, PlayerState, GamePhase |
| Card effects | `response.py` | CardResolver, weapon/armour effects |
| Judgment | `judge.py` | Delayed tricks (乐不思蜀/兵粮寸断/闪电) |
| Player responses | `response.py` | ResponseSystem (闪/杀/桃/无懈可击) |

## Conventions

**Event Flow**: EventBus.emit() → Skill.on_event() → GameEngine state update

**State Machine**: GamePhase enum (WAITING → TURN_START → PLAY_PHASE → ...)

**Linked List Seating**: Player.next_player/prev_player for circular seating

**Damage Propagation**: `_propagate_chain_damage()` preserves elemental attributes

## Anti-Patterns

**Never use string matching for card types:**
```python
# WRONG
if "杀" in card.name: ...

# CORRECT
from card.base import is_sha_card
if is_sha_card(card): ...
```

**Always check distance bidirectionally:**
```python
dist = min(dist_forward, dist_backward)
```

**Never skip event emission**: Skills depend on events being emitted at correct times

## Key Classes

- `GameEngine`: Central orchestrator (883 lines)
- `EventBus`: Pub-sub event dispatcher
- `CardResolver`: Card effect resolution with weapon/armour
- `JudgeSystem`: Delayed trick judgment
- `ResponseSystem`: Player response requests

## Notes

- 40+ event types for fine-grained skill triggering
- Weapon effects in CardResolver.resolve_sha()
- Armor checks before damage calculation