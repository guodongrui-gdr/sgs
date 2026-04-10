# 三国杀 (SGS) RL Training Project

**Generated:** 2026-04-09
**Commit:** 171cf24
**Branch:** autoresearch/apr8

## Overview

Three Kingdoms card game (三国杀) with reinforcement learning training. Python 3.10+ game engine with PyGame GUI, Stable-Baselines3 RL training, and transformer-based policy networks.

## Structure

```
sgs/
├── main.py              # Entry: CLI + GUI modes
├── config.py            # Global config, paths, constants
├── engine/              # Core game engine (event-driven)
├── ai/                  # RL agents, Gym environment, policies
├── gui/                 # PyGame GUI (rendering, animations)
├── skills/              # 43 skills by faction (Wei/Shu/Wu/Qun)
├── train/               # Training scripts, curriculum, evaluation
├── card/                # Card types and factory
├── player/              # Player dataclass
├── data/                # JSON configs (cards.json, commanders.json)
└── tests/               # pytest test suites
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Add new skill | `skills/{faction}.py` → register in `skills/__init__.py` | Use `@skill_decorator` |
| Add new card | `card/base.py` + `data/cards.json` | Extend Card hierarchy |
| Modify game rules | `engine/game_engine.py` + `engine/response.py` | Core logic |
| Train RL agent | `train/train_sb3.py` or `train/final_training.py` | Use MaskablePPO |
| GUI changes | `gui/main_window.py` (state) + `gui/game_renderer.py` (render) | PyGame |
| Add AI behavior | `ai/rl_ai.py` (RL) or `ai/rule_ai.py` (heuristic) | Implement AIInterface |
| Debug skills | `skills/base.py` → set `VERBOSE=True` in config.py | verbose_print |
| View training logs | `train/logs/{timestamp}/` | TensorBoard in `MaskablePPO_1/` |

## Conventions

**Type Safety**: All functions annotated. Use `TYPE_CHECKING` for circular imports.
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from player.player import Player
```

**Dataclasses**: Configs, states, DTOs use `@dataclass` with type annotations.
```python
@dataclass
class SGSConfig:
    player_num: int = 5
    max_rounds: int = 15
```

**Enums**: EventTypes, ActionTypes, GamePhases use `Enum` or `IntEnum`.
```python
class EventType(Enum):
    GAME_START = auto()
    TURN_START = auto()
```

**Chinese Identifiers**: Game concepts use Chinese (主公, 忠臣, 反贼, 内奸) in config and UI.

## Anti-Patterns (CRITICAL)

### Card Type Checking

**NEVER use string matching for 杀 (Sha) cards:**
```python
# WRONG - misses 火杀, 雷杀
if "杀" in card.name: ...
if card.name == "杀": ...

# CORRECT - use type checking
from card.base import is_sha_card
if is_sha_card(card): ...
```

### Card Attributes

**ALWAYS use getattr with defaults:**
```python
# WRONG - may raise AttributeError
card.is_elemental

# CORRECT
getattr(card, "is_elemental", False)
getattr(card, "is_fire", False)
```

### Distance Calculation

Distance must be **bidirectional minimum**:
```python
dist = min(dist_forward, dist_backward)
```

### Damage Propagation

When propagating chain damage (铁索连环), **preserve elemental attributes**:
```python
self._propagate_chain_damage(source, target, card, damage, is_fire, is_thunder)
```

## Unique Styles

**Event-Driven Architecture**: EventBus → Skills → GameEngine. Skills subscribe to EventTypes.

**Registry Pattern**: `SkillRegistry` auto-discovers skills via `@skill_decorator`.

**Factory Pattern**: `CardFactory` creates cards from JSON config.

**Skill Decision System**: RL-driven skill decisions via `ai/skill_decision.py` - skills can request RL input for complex decisions (观星牌序, 遗计分配).

**Training Artifact Isolation**: Each run creates timestamped dir: `train/logs/final_training_{YYYYMMDD}_{HHMMSS}/`

## Commands

```bash
# GUI mode
python main.py --gui
./run_gui.sh

# CLI mode
python main.py --player-num 5

# Train RL agent
.venv/bin/python train/train_sb3.py --n-steps 4096 --timesteps 100000 --n-envs 4

# Quick training test
.venv/bin/python train/train_sb3.py --n-steps 256 --timesteps 500 --n-envs 1

# Run tests
pytest tests/
python -m unittest discover tests/

# Load trained model
python main.py --ai-type rl --model-path train/logs/{run}/final_model.zip
```

## Notes

- **No requirements.txt**: Dependencies managed via `.venv/`. Run `.venv/bin/pip freeze > requirements.txt` to create.
- **GPU Training**: Requires CUDA 12.8. Models use torch + stable-baselines3.
- **Missing pyproject.toml**: Project lacks formal packaging. Install dependencies directly in venv.
- **LSP Errors**: 405 diagnostics in train/*.py mostly from optional imports (SB3 availability checks).
- **Large Files**: `ai/gym_wrapper.py` (1290 lines), `gui/main_window.py` (1029 lines) - complex, may need refactoring.
- **Bilingual Docs**: BUG_FIXES.md, PROJECT_STATUS.md, RULES.md in Chinese.
- **Autoresearch Mode**: See `train/program.md` for automated experiment workflow.

## Dependencies

Core: `torch`, `stable_baselines3`, `gymnasium`, `numpy`, `pygame`
ML: `sb3_contrib` (MaskablePPO), `tensorboard`
Dev: `pytest`, `matplotlib`, `rich`