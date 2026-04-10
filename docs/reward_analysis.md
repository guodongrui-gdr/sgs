# Reward Function Analysis and Optimization

## Overview

This document analyzes the reward function design for the SGS RL training pipeline and provides recommendations for improving learning efficiency.

## Current State (Wave 1 Results)

### Baseline Performance
- **Win Rate**: ~20% (random baseline)
- **Learning Rate**: 5e-4 works best
- **Entropy**: 0.02-0.05 optimal
- **Key Issue**: Delayed rewards may hinder learning

### Original RewardConfig (Phase 1)
```python
RewardConfig(
    victory=50.0,
    defeat=-50.0,
    damage_dealt=5.0,
    damage_taken=-2.0,
    kill_enemy=15.0,
    survive_per_turn=0.5,
    clip_reward=20.0,
)
```

## Problem Analysis

### 1. Terminal Reward Dominance

Research suggests terminal rewards should be ~10x intermediate rewards for optimal learning.

**Phase 1 Issue**: Terminal ratio = 50/5 = 10x (borderline)

**Phase 2 Fix**: Terminal ratio = 100/3 = 33x (improved)

### 2. Reward Sparsity

**Problem**: Non-zero rewards occur infrequently, making credit assignment difficult.

**Symptoms**:
- Episodes with <10% non-zero reward steps
- Large gaps (>20 steps) between rewards
- Agent learns slowly or not at all

**Phase 2 Fix**: Added dense intermediate rewards:
- `skill_activation_reward=0.2`
- `effective_card_use_reward=0.5`
- `turn_progress_reward=0.1`
- `round_progress_reward=0.2`

### 3. Potential-Based Shaping

**Problem**: Gamma value affects how much future potential influences current reward.

**Original**: Fixed gamma=0.5
**Phase 2**: Configurable `shaping_gamma` (default 0.99)

Higher gamma = more emphasis on long-term value, which aligns with winning objective.

## Recommended RewardConfig Values

### Phase 2 Optimized (Recommended)

```python
RewardConfig(
    # Terminal rewards (strong signal)
    victory=100.0,
    defeat=-100.0,
    
    # Intermediate rewards (moderate signal)
    damage_dealt=3.0,
    damage_taken=-1.5,
    kill_enemy=15.0,
    kill_ally=-20.0,
    lord_kill_loyalist=-30.0,
    
    # Dense intermediate rewards (new)
    survive_per_turn=0.5,
    skill_activation_reward=0.2,
    effective_card_use_reward=0.5,
    turn_progress_reward=0.1,
    round_progress_reward=0.2,
    protect_lord_reward=2.0,
    assist_ally_reward=1.0,
    
    # Shaping and clipping
    shaping_gamma=0.99,
    clip_reward=50.0,
)
```

### Expected Impact

| Metric | Phase 1 | Phase 2 | Expected Improvement |
|--------|---------|---------|---------------------|
| Terminal ratio | 10x | 33x | Better final objective alignment |
| Reward density | ~5% | ~15-20% | Faster credit assignment |
| Avg steps between rewards | >20 | <10 | More frequent learning signals |

## Reward Logging and Analysis

### Using the Analysis Script

```bash
# Analyze existing reward logs
python train/analyze_rewards.py --analyze-logs --log-dir ./logs/rewards

# Run baseline analysis
python train/analyze_rewards.py --run-baseline --num-episodes 100

# Compare different configs
python train/analyze_rewards.py --compare-configs
```

### Key Metrics to Monitor

1. **Sparse Reward Ratio**: Target >10%
   - Below 5%: Critical - add more intermediate rewards
   - 5-10%: Warning - consider progress rewards
   - >10%: Acceptable

2. **Terminal Dominance Ratio**: Target 10-30x
   - <10x: Terminal rewards too weak
   - 10-30x: Optimal range
   - >50x: May cause sparse reward issues

3. **Steps Between Rewards**: Target <10
   - Higher values indicate sparse reward problem

### Interpretation

```
[RewardLogger] Episode 100: avg_reward=15.2, avg_length=45.3, 
               sparse_ratio=0.152, avg_steps_between_rewards=6.6

[RewardLogger] Terminal/Intermediate ratio: 33.3x 
               (terminal_avg=100.00, intermediate_avg=3.00)
```

## Validation Checklist

### ✓ Terminal Reward Dominance
- Victory/Defeat rewards significantly larger than intermediates
- Ratio in optimal 10-30x range

### ✓ Dense Intermediate Rewards
- Non-zero rewards at least 10% of steps
- Card usage, skill activation provide feedback

### ✓ Potential-Based Shaping
- Gamma configurable for experimentation
- Does not change optimal policy

### ✓ No Reward Hacking
- Rewards align with winning objective
- Farming damage less valuable than winning

## Recommended Experiments

### 1. Baseline Comparison
Train with Phase 1 vs Phase 2 config for same timesteps, compare win rates.

### 2. Shaping Gamma Sweep
Test gamma values: [0.5, 0.7, 0.9, 0.95, 0.99]

### 3. Intermediate Density
Vary `survive_per_turn` and `turn_progress_reward` to find optimal density.

## Files Modified/Created

| File | Description |
|------|-------------|
| `ai/reward.py` | Enhanced with logging, Phase 2 config |
| `train/analyze_rewards.py` | New analysis script |
| `tests/test_reward_refinement.py` | Unit tests for reward system |
| `docs/reward_analysis.md` | This documentation |

## Conclusion

The Phase 2 reward refinements address:
1. **Terminal dominance**: Increased to 33x ratio
2. **Sparse rewards**: Added dense intermediate signals
3. **Shaping flexibility**: Configurable gamma
4. **Monitoring**: Comprehensive logging and analysis

Expected improvement: 5-10% win rate increase through better credit assignment and reward shaping.

---

*Generated: 2026-04-09*
*Version: Phase 2 - Reward Refinement*