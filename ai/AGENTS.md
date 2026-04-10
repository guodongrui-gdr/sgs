# AI Module

Reinforcement learning stack with Gymnasium environment, state/action encoding, reward shaping, and transformer policy networks.

## Structure

```
ai/
├── __init__.py              # 40+ exports
├── interface.py             # AIInterface abstract base
├── state_encoder.py         # Game state → neural vectors (~3000 dim)
├── action_encoder.py        # Hierarchical action space with masking
├── reward.py                # Identity-aware reward system
├── skill_decision.py        # RL-driven skill internal decisions
├── gym_wrapper.py           # SGSEnv (Gymnasium environment)
├── rl_ai.py                 # RLAI (trained model inference)
├── rule_ai.py               # RuleAI (heuristic baseline)
├── models/
│   └── transformer_policy.py  # Transformer encoder policy
├── multi_agent_env.py       # Multi-agent training support
├── self_play.py             # Self-play training
└── policy_pool.py           # Policy pool for opponent diversity
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Train RL agent | `gym_wrapper.py` → `SGSEnv` | Gymnasium env, 1290 lines |
| Load trained model | `rl_ai.py` → `RLAI` | Supports MaskablePPO/PPO |
| Rule-based baseline | `rule_ai.py` → `RuleAI` | Heuristic for comparison |
| Encode game state | `state_encoder.py` → `StateEncoder` | ~3000 dim vector |
| Encode actions | `action_encoder.py` → `ActionEncoder` | Hierarchical: type → card → target |
| Action masks | `action_encoder.py` → `ActionMaskGenerator` | Filters illegal actions |
| Reward shaping | `reward.py` → `RewardSystem` | Identity-aware rewards |
| Skill decisions | `skill_decision.py` → `SkillDecisionRequest` | 观星, 遗计, etc. |
| Transformer policy | `models/transformer_policy.py` | Custom SB3 policy |
| Self-play training | `self_play.py` → `SelfPlayTrainer` | Curriculum + policy pool |

## Conventions

**Conditional Imports**: SB3/Gymnasium are optional. Check availability flags:
```python
if SB3_AVAILABLE:  # stable_baselines3 import guard
    from stable_baselines3 import PPO
```

**Hierarchical Actions**: Three-step decoding (action_type, card_idx, target_idx)
```python
class HierarchicalAction:
    action_type: int      # USE_CARD, END_TURN, USE_SKILL, etc.
    card_idx: Optional[int]
    target_idx: Optional[int]
```

**Action Masks**: Always use masks to filter invalid actions
```python
type_mask, card_mask, target_mask = action_masks
valid_types = [i for i, m in enumerate(type_mask) if m > 0]
```

**State Encoding**: Fixed-size vectors (padded, not variable length)
- Global state: 34 dims
- Hand cards: 20 × 76 = 1520 dims
- Other players: 7 × 73 = 511 dims
- History: 10 × 78 = 780 dims

## Anti-Patterns

**Never assume SB3 is available** - use conditional imports with availability flags

**Never ignore action masks** - always pass masks to `model.predict(obs, action_masks=masks)`

**Never skip VecNormalize stats** - save/load with model checkpoints
