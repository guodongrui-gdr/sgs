# GUI Module (PyGame)

**Generated:** 2026-04-09

## Overview

PyGame-based graphical interface with event-driven state machine, 1029-line main window orchestrator.

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Main game loop | `main_window.py` | GameWindow class, 46 methods |
| Game rendering | `game_renderer.py` | Table, players, cards |
| Card drawing | `card_renderer.py` | Card sprites, colors by type |
| Player area | `player_renderer.py` | HP, equipment, status |
| Input handling | `input_handler.py` | Mouse/keyboard events |
| UI components | `ui_elements.py` | Buttons, dialogs, panels |
| Animations | `animations.py` | Damage, heal, card move effects |
| Sound effects | `audio.py` | BGM, card sounds |
| Response UI | `response_manager.py` | 闪/杀/桃/无懈可击 prompts |
| Skill dialogs | `skill_ui.py` | 观星/遗计/离间 decisions |
| Game log | `skill_ui.py` | GameLogPanel class |
| Resource loading | `assets.py` | Images, fonts, colors |

## Conventions

**State Machine**: GameWindow manages game states (menu, playing, paused, game_over)

**Separation of Concerns**: Rendering (renderers) vs state (main_window) vs input (input_handler)

**Animation Base Class**: All animations inherit from Animation with update()/render()

## Anti-Patterns

**God Class**: GameWindow has 46 methods - handles both state and coordination

**Tight Coupling**: main_window.py couples UI rendering with game logic

**Large Function**: `_use_card_with_logic()` is 183 lines - consider extraction

## Key Files

- `main_window.py` (1029 lines) - Main game window and state management
- `game_renderer.py` - Primary rendering logic
- `animations.py` - Animation base class and effects

## Notes

- PyGame event loop at 60 FPS
- Chinese UI text (主公, 忠臣, 反贼, 内奸)
- Assets loaded from `素材/` directory
- Color scheme in `assets.py`