# 三国杀 (SGS) RL Training Project

**Generated:** 2026-04-16
**Branch:** autoresearch/apr8

## Overview

Three Kingdoms card game (三国杀) with reinforcement learning training. Python 3.10+ game engine with PyGame GUI, Stable-Baselines3 RL training (MaskablePPO, IPPO, MAPPO), and transformer-based policy networks.

## Quick Start

```bash
# GUI mode (recommended for human play)
./run_gui.sh
source .venv/bin/activate && python main.py --gui

# CLI mode
python main.py --player-num 5

# Quick training test
.venv/bin/python train/train_self_play.py --test-mode

# Run tests
pytest tests/
```

## Environment Setup

### Virtual Environment
```bash
source .venv/bin/activate
# Python 3.10+ required, CUDA 12.8 for GPU training
```

### Dependencies
Core: `torch`, `stable_baselines3`, `sb3_contrib`, `gymnasium`, `numpy`, `pygame`
Dev: `pytest`, `tensorboard`, `matplotlib`, `rich`

**No requirements.txt** - generate with `.venv/bin/pip freeze > requirements.txt`

## Structure

```
sgs/
├── main.py              # Entry: CLI + GUI modes, RL model loading
├── config.py            # Global config, paths, VERBOSE flag
├── engine/              # Core game engine (event-driven, 40+ event types)
├── ai/                  # RL agents, Gym environment, state/action encoding
├── gui/                 # PyGame GUI (1029-line main_window.py)
├── skills/              # 43+ skills by faction (Wei/Shu/Wu/Qun)
├── train/               # 8 training scripts + evaluation
├── card/                # Card types and factory
├── player/              # Player dataclass with linked list seating
├── data/                # JSON configs (cards.json, commanders.json)
├── config/              # YAML configs (world_model_config.yaml)
├── tests/               # pytest + unittest test suites
├── docs/                # Architecture and training documentation
└── 素材/                # GUI assets (backgrounds, buttons)
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Add new skill | `skills/{faction}.py` → `@SkillRegistry.register` | Use TriggerSkill/ActiveSkill base |
| Add new card | `card/base.py` + `data/cards.json` | Extend Card hierarchy |
| Modify game rules | `engine/game_engine.py` + `engine/response.py` | 883 lines, 35 methods |
| Self-play training | `train/train_self_play.py` | MaskablePPO, agent pool |
| MAPPO training | `train/train_mappo.py` | Centralized critic |
| World model | `train/train_dynamics.py` + `train/train_mixed.py` | RSSM imagination |
| GUI changes | `gui/main_window.py` (state) + `gui/game_renderer.py` | 1029 lines, 46 methods |
| Debug skills | `config.py` → `VERBOSE = True` | verbose_print output |
| View training logs | `train/logs/{timestamp}/` | TensorBoard in `tensorboard/` |

## Training Commands

### Self-Play Training (Primary)
```bash
# Quick test (1000 steps, single env)
.venv/bin/python train/train_self_play.py --test-mode

# Standard training (2M steps, 8 envs)
.venv/bin/python train/train_self_play.py --timesteps 2000000 --n-envs 8

# Resume from checkpoint
.venv/bin/python train/train_self_play.py --resume train/logs/xxx/step_1600000.zip
```

### MAPPO Training
```bash
# Quick test
.venv/bin/python train/train_mappo.py --steps 100 --n-envs 1

# Full training
.venv/bin/python train/train_mappo.py --steps 50000 --n-envs 4 --device cuda
```

### IPPO Global Baseline
```bash
.venv/bin/python train/train_ippo_global.py --steps 10000 --n-envs 4
```

### World Model Training
```bash
# Dynamics model
.venv/bin/python train/train_dynamics.py --steps 50000

# Mixed real/imagined
.venv/bin/python train/train_mixed.py --steps 50000 --warmup 10000
```

### Model Evaluation
```bash
.venv/bin/python train/evaluate_model.py --model-type ippo --model-path path/to/model.zip --num-episodes 100
```

### Training Arguments (train_self_play.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--timesteps` | 2M | Total training steps |
| `--n-envs` | 8 | Parallel environments |
| `--pool-size` | 10 | Agent pool for self-play |
| `--checkpoint-freq` | 100K | Checkpoint interval |
| `--test-mode` | False | Quick 1000-step test |

### Shell Scripts
```bash
./scripts/start_training.sh       # Self-play launcher (1M steps)
./scripts/continue_training.sh    # Resume from checkpoint
./scripts/monitor_training.sh     # Progress monitor
./scripts/rollback_to_ippo.sh     # Disable World Model/MAPPO
```

## Testing

```bash
# Run full suite
pytest tests/

# Run specific file
pytest tests/test_curriculum.py

# Run specific test
pytest tests/test_curriculum.py::TestCurriculumStage::test_stage_ordering

# Pattern match
pytest tests/ -k "belief"

# unittest alternative
python -m unittest discover tests/

# Special standalone script
python tests/validation_ab_test.py --steps 1000
```

### Test Categories

| Category | Files | Focus |
|----------|-------|-------|
| Core Training | `test_curriculum.py`, `test_extended_training.py` | Curriculum, checkpoints |
| RL Agents | `test_mappo.py`, `test_ippo_global_env.py` | MAPPO, IPPO |
| World Model | `test_imagination_env.py`, `world_model/` | RSSM, dynamics |
| Self-Play | `test_self_play.py`, `test_agent_pool_manager.py` | Policy pool, ELO |
| Belief | `test_belief_update.py`, `test_identity_belief_encoding.py` | Identity tracking |

## Conventions

### TYPE_CHECKING Pattern (Universal)
All modules avoid circular imports:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from player.player import Player
    from engine.game_engine import GameEngine
```

### Conditional Imports for Optional Dependencies
ML libraries guarded with availability flags:
```python
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
```

### Dataclasses with ABC
Core classes combine both:
```python
@dataclass
class Skill(ABC):
    name: str
    trigger_events: List["EventType"]
    
    @abstractmethod
    def can_activate(self, event, engine) -> bool:
        pass
```

### Enums for Types
```python
class EventType(Enum):
    GAME_START = auto()
    DAMAGE_TAKEN = auto()  # 40+ event types
```

### Registry Pattern
Skills auto-register via decorator:
```python
@SkillRegistry.register
class JianXiong(TriggerSkill):
    def __init__(self):
        super().__init__(name="奸雄", trigger_events=[EventType.DAMAGE_TAKEN])
```

### Factory Pattern
```python
class CardFactory:
    @classmethod
    def create(cls, config: Dict) -> List[Card]:
        card_class = cls._type_mapping.get(card_type, BasicCard)
```

### Chinese Identifiers
Game concepts use Chinese in config and UI:
```python
IDENTITY_CONFIG = {5: ["主公", "忠臣", "反贼", "反贼", "内奸"]}
player.equipment = {"武器": None, "防具": None}
```

### Linked List Seating
Players connected circularly:
```python
player.next_player: Optional["Player"]
player.prev_player: Optional["Player"]
```

### VERBOSE Debug Flag
```python
# config.py
VERBOSE = False  # Set True for skill debug output
from config import verbose_print as print
```

## Anti-Patterns (CRITICAL)

### Card Type Checking
```python
# WRONG - misses 火杀, 雷杀
if "杀" in card.name: ...
if card.name == "杀": ...

# CORRECT
from card.base import is_sha_card
if is_sha_card(card): ...
```

### Card Attributes
```python
# WRONG - may raise AttributeError
card.is_elemental

# CORRECT
getattr(card, "is_elemental", False)
getattr(card, "is_fire", False)
```

### Distance Calculation
Must be bidirectional minimum:
```python
dist = min(dist_forward, dist_backward)
```

### Event Card Null Check
```python
# WRONG - crashes if event.card is None
if event.card.name == "杀": ...

# CORRECT
if not event.card: return event
if is_sha_card(event.card): ...
```

### Skill Name Checking
```python
# WRONG
if "反馈" in player.skills: ...

# CORRECT
SkillRegistry.has_skill("反馈")
```

## Data Files

| File | Contents |
|------|----------|
| `data/cards.json` | 160+ cards (basic, tricks, equipment) |
| `data/commanders.json` | 22 commanders (Wei/Shu/Wu/Qun) |
| `素材/` | GUI assets (背景.jpg, 结束.jpg, buttons) |
| `config/world_model_config.yaml` | World model architecture settings |

## Training Outputs

Each run creates timestamped directory: `train/logs/{script}_{YYYYMMDD}_{HHMMSS}/`

Contains:
- `checkpoints/step_XXXXXX.zip` - Model checkpoints
- `final_model.zip` - Final model
- `vecnormalize.pkl` - VecNormalize statistics (required for inference)
- `evaluation_report.json` - Final win rates
- `tensorboard/` - TensorBoard logs

View logs:
```bash
tensorboard --logdir train/logs/
```

## Model Loading

```bash
# CLI with trained model
python main.py --ai-type rl --model-path train/logs/xxx/final_model.zip

# MAPPO model (.pt file)
python main.py --ai-type mappo --model-path train/logs/xxx/mappo_checkpoint.pt
```

## Architecture Notes

- **Event-Driven**: EventBus → Skills → GameEngine. 40+ event types for fine-grained skill triggering.
- **Hierarchical Actions**: Three-step decoding (action_type → card_idx → target_idx)
- **Action Masks**: Always use masks to filter invalid actions. Pass to `model.predict(obs, action_masks=masks)`
- **State Encoding**: Fixed ~3000-dim vectors (padded, not variable length)
- **Skill Decision System**: Skills can request RL decisions via `ai/skill_decision.py` (观星牌序, 遗计分配)

## Notes

- **GPU Optional**: CPU fallback available, GPU (CUDA 12.8) recommended for full training
- **No pyproject.toml**: Project lacks formal packaging
- **Large Files**: `ai/gym_wrapper.py` (1290 lines), `gui/main_window.py` (1029 lines)
- **Bilingual Docs**: BUG_FIXES.md, RULES.md, PROJECT_STATUS.md in Chinese
- **VecNormalize Required**: Load stats with model checkpoints for correct inference

## Key Documentation

- `docs/WORLD_MODEL_MAPPO.md` - World model integration phases
- `docs/training_pipeline.md` - Extended training workflow
- `docs/FINAL_REPORT.md` - Training results summary
- `BUG_FIXES.md` - Rule fixes log (白银狮子, 藤甲, 铁索连环)
- `RULES.md` - Official game rules reference