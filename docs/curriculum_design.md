# Curriculum Learning Design for SGS RL Training

## Overview

This document describes the curriculum learning system implemented for the Three Kingdoms Kill (SGS) card game AI training. Curriculum learning progressively increases task difficulty, enabling faster and more stable convergence compared to direct training on the full complexity task.

## Problem Context

- **Current Performance**: ~20% win rate (random baseline)
- **Goal**: 50% win rate
- **Challenge**: Large state/action space, complex game mechanics, delayed rewards

## Curriculum Strategy

### Design Philosophy

The curriculum follows the **"Easy-to-Hard"** paradigm:

1. **Start Simple**: Short games against weak opponents to learn basic mechanics
2. **Gradual Complexity**: Increase game length and opponent strength
3. **Self-Play Finale**: Train against past versions for robust strategies

### Why Curriculum Learning?

| Challenge | Curriculum Solution |
|-----------|-------------------|
| Sparse rewards | Shorter games = more frequent terminal signals |
| Large action space | Simpler opponents = fewer meaningful choices |
| Long credit assignment | Shorter episodes = clearer cause-effect |
| Opponent exploitation | Progressive difficulty = generalizable skills |

## Stage Definitions

### Stage 1: Easy (Foundation Building)

**Configuration:**
- `max_rounds`: 5 (shorter games)
- `opponent_policy`: "random" (unpredictable but weak)
- `reward_multiplier`: 1.5x (amplified learning signals)

**Advancement Criteria:**
- Win rate ≥ 40%
- Minimum 100 episodes completed
- 2 consecutive evaluations above threshold

**Rationale:**
- Short games provide rapid feedback
- Random opponents teach basic card usage
- 1.5x reward multiplier strengthens value function gradients
- Higher entropy coefficient (0.08) encourages exploration

**Expected Learning:**
- Basic card mechanics
- Turn structure understanding
- Simple attack patterns
- Card value assessment

### Stage 2: Medium (Strategy Development)

**Configuration:**
- `max_rounds`: 10 (medium-length games)
- `opponent_policy`: "rule" (deterministic AI)
- `reward_multiplier`: 1.0x (normal signals)

**Advancement Criteria:**
- Win rate ≥ 45%
- Minimum 200 episodes completed
- 2 consecutive evaluations above threshold

**Rationale:**
- Rule AI provides consistent challenge
- Medium game length introduces multi-turn strategy
- Normal rewards prevent overfitting to early patterns
- Standard entropy (0.05) balances exploration/exploitation

**Expected Learning:**
- Strategic card sequencing
- Resource management (HP, cards)
- Identity-specific play patterns
- Counter-strategies to common rule AI moves

### Stage 3: Hard (Self-Play Mastery)

**Configuration:**
- `max_rounds`: 15 (full-length games)
- `opponent_policy`: "self_play" (policy pool sampling)
- `reward_multiplier`: 1.0x (standard signals)

**Completion Criteria:**
- Win rate ≥ 50% (target achieved)
- Minimum 500 episodes completed

**Rationale:**
- Full complexity matches deployment conditions
- Self-play creates diverse opponent strategies
- Policy pool maintains variety (avoids collapse)
- Lower entropy (0.03) focuses on exploitation

**Expected Learning:**
- Complex multi-turn strategies
- Adaptation to various play styles
- Identity-specific optimal policies
- Endgame tactics

## Progression Logic

### Win Rate Calculation

```python
# Rolling window average (default: last 100 episodes)
recent_win_rate = sum(recent_wins) / len(recent_wins)

# Threshold check with patience
if win_rate >= min_win_rate:
    consecutive_checks += 1
    if consecutive_checks >= patience:
        advance_stage()
else:
    consecutive_checks = 0
```

### Stage Transition Triggers

| Trigger | Action |
|---------|--------|
| Win rate threshold met + patience | Advance stage |
| Timesteps limit reached | Advance stage |
| Performance drop > 15% | Regress stage |
| Manual CLI override | Set specific stage |

### Regression Handling

If performance drops significantly during harder stages, the system can regress to easier stages:

```python
if overall_win_rate - recent_win_rate > regression_threshold:
    regress_to_previous_stage()
```

This prevents catastrophic forgetting and allows the agent to relearn fundamentals.

## Reward Structure

### Base Rewards (from `ai/reward.py`)

| Event | Base Reward |
|-------|-------------|
| Victory | +100 |
| Defeat | -100 |
| Kill enemy | +15 |
| Damage dealt | +3/damage |
| Damage taken | -1.5/damage |
| Turn survive | +0.5 |

### Stage-Specific Modification

| Stage | Multiplier | Victory | Defeat |
|-------|------------|---------|--------|
| Easy | 1.5x | +150 | -150 |
| Medium | 1.0x | +100 | -100 |
| Hard | 1.0x | +100 | -100 |

**Why Stronger Signals in Stage 1?**
- Amplifies sparse rewards for faster early learning
- Creates stronger value function gradients
- Helps overcome initial random exploration phase

## Implementation Details

### File Structure

```
train/
├── curriculum.py              # Core curriculum logic
│   ├── CurriculumStage        # Stage enum
│   ├── StageConfig            # Per-stage configuration
│   ├── CurriculumConfig       # Full curriculum setup
│   ├── StageProgress          # Progress tracking
│   └── CurriculumManager      # Main controller
│
├── train_with_curriculum.py   # Training script
│   ├── CurriculumSGSEnv       # Environment wrapper
│   ├── CurriculumProgressCallback
│   ├── CurriculumSelfPlayCallback
│   └── visualize_curriculum_progress()
│
└── logs/curriculum_*/
    ├── curriculum_state_*.json
    ├── stage_transitions.json
    ├── win_rate_history.json
    └── curriculum_progress.png
```

### Usage

```bash
# Start curriculum training from Stage 1
python train/train_with_curriculum.py --timesteps 5000000

# Start at specific stage
python train/train_with_curriculum.py --stage medium

# Resume from saved state
python train/train_with_curriculum.py --resume --curriculum-state path/to/state.json

# Visualize completed training
python train/train_with_curriculum.py --visualize logs/curriculum_20250409/
```

### Integration with Existing Pipeline

The curriculum system integrates seamlessly with the existing training infrastructure:

1. **Environment Wrapper**: `CurriculumSGSEnv` extends `SGSEnv`
2. **Callbacks**: Integrate with SB3 callback system
3. **Checkpointing**: Stage-specific model saves
4. **Logging**: Compatible with TensorBoard

## Expected Timeline

### Estimated Training Time

Based on 1500 steps/sec performance:

| Stage | Timesteps | Episodes | Estimated Time |
|-------|-----------|----------|----------------|
| Easy | ~1.5M | ~3000 | ~17 minutes |
| Medium | ~1.5M | ~2000 | ~17 minutes |
| Hard | ~2M | ~1000 | ~22 minutes |
| **Total** | ~5M | ~6000 | **~56 minutes** |

### Milestone Win Rates

| Training Progress | Expected Win Rate |
|-------------------|-------------------|
| 0% (start) | 20% |
| 20% (end Stage 1) | 40-45% |
| 50% (end Stage 2) | 45-48% |
| 100% (complete) | 50%+ |

## Comparison: Curriculum vs Non-Curriculum

### Expected Improvements

| Metric | Non-Curriculum | Curriculum | Improvement |
|--------|---------------|------------|-------------|
| Time to 30% win rate | ~2M steps | ~0.5M steps | 4x faster |
| Time to 40% win rate | ~5M steps | ~1.5M steps | 3.3x faster |
| Final win rate (5M steps) | ~35-40% | ~50%+ | +10-15% |
| Training stability | High variance | Stable | Better |

### Why Curriculum Wins

1. **Dense Learning Signals**: Shorter games → more terminal rewards per step
2. **Progressive Complexity**: Skills build incrementally
3. **Better Exploration**: Early high entropy finds diverse strategies
4. **Avoidance of Local Optima**: Self-play prevents opponent overfitting

## Hyperparameter Schedule

### Learning Rate

- **Schedule**: Cosine annealing
- **Range**: 5e-4 → 5e-5
- **No stage-specific changes** (continues across stages)

### Entropy Coefficient

| Stage | Entropy Coef | Rationale |
|-------|--------------|-----------|
| Easy | 0.08 | High exploration for diverse strategies |
| Medium | 0.05 | Balanced exploration/exploitation |
| Hard | 0.03 | Exploitation for refinement |

### Other Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| n_envs | 8 | Parallel environments |
| n_steps | 2048 | Steps per rollout |
| batch_size | 256 | Training batch |
| n_epochs | 10 | PPO epochs |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.98 | GAE parameter |

## Monitoring and Debugging

### Key Metrics to Watch

1. **Win Rate Progression**: Should increase monotonically within stages
2. **Episode Length**: Should increase with stage difficulty
3. **Value Loss**: Should decrease steadily
4. **Entropy**: Should decrease over time, but jump at stage transitions

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Stuck in Stage 1 | Win rate plateaus at 35% | Increase reward multiplier to 2.0x |
| Fast advancement | Skips learning | Increase patience threshold |
| Stage regression loop | Oscillates between stages | Increase min_episodes |
| Self-play collapse | Win rate drops in Stage 3 | Increase policy pool diversity |

### Debug Commands

```bash
# Check current state
cat logs/curriculum_*/curriculum_state_*.json | jq '.current_stage'

# View transitions
cat logs/curriculum_*/stage_transitions.json | jq '.'

# Plot progress
python train/train_with_curriculum.py --visualize logs/curriculum_*/
```

## Future Improvements

### Potential Enhancements

1. **Adaptive Thresholds**: Adjust win rate thresholds based on learning curve
2. **Identity-Specific Curriculum**: Different progression for each player role
3. **Opponent Curriculum**: Train opponent AI alongside main agent
4. **Multi-Stage Parallel Training**: Pre-train on multiple stages simultaneously

### Research Directions

- **Meta-Learning**: Learn the curriculum schedule itself
- **Automatic Stage Discovery**: Let the agent decide difficulty levels
- **Curriculum for Skill Discovery**: Separate curricula for different skill trees

## References

1. **Curriculum Learning** (Bengio et al., 2009): Original curriculum learning paper
2. **Self-Play Training** (Silver et al., 2017): AlphaZero methodology
3. **Progressive Neural Networks** (Rusu et al., 2016): Transfer learning across tasks
4. **PPO** (Schulman et al., 2017): Core RL algorithm

## Appendix: Configuration Examples

### Minimal Configuration

```python
from train.curriculum import CurriculumConfig, StageConfig, CurriculumStage

config = CurriculumConfig()  # Use defaults
manager = CurriculumManager(config)
```

### Custom Configuration

```python
from train.curriculum import CurriculumConfig, StageConfig, CurriculumStage

custom_stages = [
    StageConfig(
        stage=CurriculumStage.EASY,
        max_rounds=3,  # Even shorter
        opponent_policy="random",
        reward_multiplier=2.0,  # Stronger boost
        min_win_rate=0.30,  # Lower threshold
        min_episodes=50,
        target_win_rate=0.40,
        ent_coef=0.10,  # More exploration
    ),
    StageConfig(
        stage=CurriculumStage.MEDIUM,
        max_rounds=8,
        opponent_policy="rule",
        reward_multiplier=1.5,
        min_win_rate=0.40,
        min_episodes=100,
        target_win_rate=0.50,
    ),
    StageConfig(
        stage=CurriculumStage.HARD,
        max_rounds=15,
        opponent_policy="self_play",
        reward_multiplier=1.0,
        min_win_rate=0.50,
        min_episodes=200,
        target_win_rate=0.55,
    ),
]

config = CurriculumConfig(
    stages=custom_stages,
    advancement_patience=3,
    total_timesteps_per_stage=1_000_000,
)
```

---

**Document Version**: 1.0
**Last Updated**: 2025-04-09
**Author**: SGS AI Training Pipeline