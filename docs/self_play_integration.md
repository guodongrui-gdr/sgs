# Self-Play Integration for SGS RL Training

## Overview

This document describes the self-play training system for the Three Kingdoms Kill (三国杀) card game AI. Self-play is a crucial technique for improving AI performance by training against previous versions of itself, allowing for continuous improvement beyond what's possible with fixed opponents.

## Background

### Current State
- **Project Goal**: Achieve 50% win rate (current baseline ~20% against random opponents)
- **Current Training**: Against rule-based AI only
- **Challenge**: Rule-based AI provides limited strategic diversity

### Why Self-Play?
1. **Continuous Improvement**: Agents can improve beyond the skill level of fixed opponents
2. **Strategic Diversity**: Training against various versions prevents overfitting to specific strategies
3. **Scalability**: Quality improves with more training without needing expert demonstrations
4. **Proven Success**: Used in AlphaGo, AlphaZero, and modern game AI systems

## Self-Play Strategy

### Opponent Sampling Distribution

The system uses a three-tier opponent sampling strategy:

| Sampling Type | Ratio | Purpose |
|--------------|-------|---------|
| **Latest 30%** | 30% | Train against most recent policies for curriculum learning |
| **Best 30%** | 30% | Train against strongest historical policies |
| **Random 40%** | 40% | Ensure diversity and prevent strategy collapse |

#### Why This Distribution?

1. **Latest Policies (30%)**:
   - Ensures smooth curriculum progression
   - Prevents the agent from developing strategies that only work against weak opponents
   - Recent policies are closer to current skill level

2. **Best ELO Policies (30%)**:
   - Trains against the strongest known opponents
   - Challenges the agent to improve against high-quality play
   - Prevents "gaming" weak opponents

3. **Random Sampling (40%)**:
   - Ensures exposure to diverse strategies
   - Prevents overfitting to specific opponent styles
   - Maintains exploration in strategy space

### ELO Rating System

The system uses an ELO rating system adapted for self-play:

```
ELO_new = ELO_old + K × (actual_score - expected_score)
```

**Parameters:**
- **K-factor**: 32.0 (adjustable, controls rating volatility)
- **Initial ELO**: 1000.0
- **Expected score**: Based on rating difference using standard ELO formula

#### ELO Decay

Old policies have their ELO decayed over time:

```
effective_elo = base_elo × (1 - decay_rate × age_hours)
```

**Default Parameters:**
- Decay rate: 1% per hour
- Maximum decay: 30%

This ensures that:
- Recent policies are prioritized
- Old strategies that may be outdated don't dominate sampling
- The pool stays relevant to current meta

### Identity-Aware Win Rate Tracking

The system tracks win rates per player identity:

| Identity | Baseline Win Rate | Target Win Rate |
|----------|------------------|-----------------|
| 主公 (Lord) | 20% | 35% |
| 忠臣 (Loyalist) | 20% | 35% |
| 反贼 (Rebel) | 20% | 35% |
| 内奸 (Spy) | 20% | 35% |

This helps identify:
- Identity-specific strategy weaknesses
- Balance issues in the game
- Areas needing targeted training

## Training Modes

### 1. Self-Play Only (`self_play_only`)
- 100% training against policy pool
- Best for advanced training with established policy pool
- Requires initial seeding with rule-based training

```bash
python train/self_play_training.py --mode self_play_only --timesteps 1000000
```

### 2. Rule-Based Only (`rule_based_only`)
- 100% training against rule AI
- Best for initial training or baseline establishment
- Provides stable learning signal

```bash
python train/self_play_training.py --mode rule_based_only --timesteps 1000000
```

### 3. Mixed Training (`mixed`) - **Recommended Default**
- Randomly mixes self-play and rule-based opponents
- Ratio configurable via `--self-play-ratio`
- Provides diversity while maintaining stable learning

```bash
python train/self_play_training.py --mode mixed --self-play-ratio 0.5
```

### 4. Alternating Training (`alternating`)
- Alternates between self-play and rule-based epochs
- Configurable epoch length via `--alternating-epochs`
- Provides structured curriculum

```bash
python train/self_play_training.py --mode alternating --alternating-epochs 5
```

## Configuration Options

### SelfPlayConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 10,000,000 | Total training timesteps |
| `self_play_ratio` | 0.5 | Fraction of self-play training |
| `sample_latest_ratio` | 0.3 | Ratio for sampling latest policies |
| `sample_best_ratio` | 0.3 | Ratio for sampling best ELO policies |
| `sample_random_ratio` | 0.4 | Ratio for random sampling |
| `pool_size` | 10 | Maximum policies in pool |
| `elo_k_factor` | 32.0 | ELO update factor |
| `elo_decay_rate` | 0.01 | ELO decay per hour |
| `elo_max_decay` | 0.3 | Maximum ELO decay |
| `alternating_epochs` | 5 | Epochs before mode switch |

### PoolConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size` | 10 | Maximum policies to keep |
| `keep_best_count` | 2 | Always keep top N ELO policies |
| `keep_latest_count` | 2 | Always keep latest N policies |
| `min_elo` | 800 | Minimum ELO to keep policy |

## Expected Impact on Win Rate

### Projected Improvements

Based on similar self-play systems and our training analysis:

| Phase | Training | Expected Win Rate | Time |
|-------|----------|-------------------|------|
| Baseline | Random | ~20% | - |
| Phase 1 | Rule-based only | ~25-30% | 1M steps |
| Phase 2 | Mixed (50% SP) | ~30-35% | 2M steps |
| Phase 3 | Self-play heavy (70% SP) | ~35-40% | 3M steps |
| Phase 4 | Self-play focused | ~40-50% | 5M+ steps |

### Why These Numbers?

1. **Phase 1**: Rule-based training provides stable foundation
   - Learns basic game mechanics and rules
   - Develops fundamental strategies

2. **Phase 2**: Mixed training adds diversity
   - Exposed to various strategies
   - Learns to adapt to different opponents

3. **Phase 3**: Self-play heavy enables continuous improvement
   - Trains against progressively stronger opponents
   - Develops advanced strategies

4. **Phase 4**: Self-play focused for final optimization
   - Fine-tunes against best historical policies
   - Converges to optimal play

## Usage Examples

### Quick Start

```bash
# Basic mixed training (recommended)
python train/self_play_training.py --mode mixed --timesteps 1000000

# With custom sampling distribution
python train/self_play_training.py \
    --mode mixed \
    --sample-latest-ratio 0.4 \
    --sample-best-ratio 0.4 \
    --sample-random-ratio 0.2

# Alternating training with 10-epoch cycles
python train/self_play_training.py \
    --mode alternating \
    --alternating-epochs 10 \
    --timesteps 2000000
```

### Advanced Configuration

```bash
# High self-play ratio with custom ELO settings
python train/self_play_training.py \
    --mode mixed \
    --self-play-ratio 0.7 \
    --elo-k-factor 24 \
    --elo-decay-rate 0.005 \
    --pool-size 20 \
    --timesteps 5000000
```

### Monitoring Training

Training produces several monitoring outputs:

1. **TensorBoard logs**: `logs/<run>/tensorboard/`
   - Learning curves
   - ELO progression
   - Win rate tracking

2. **Training metrics**: `logs/<run>/training_metrics.json`
   - Per-episode statistics
   - Evaluation results

3. **Policy pool stats**: `logs/<run>/policy_pool/manifest.json`
   - ELO ratings
   - Win rates per identity
   - Pool composition

4. **Match history**: `logs/<run>/match_history.json`
   - Detailed game records
   - Head-to-head statistics

## Integration with Existing Training

### Wave 1 Results (MLP Architecture)
- Optimal: `ent_coef=0.05`, `lr=5e-4`
- These parameters are used as defaults in self-play training

### Wave 2 Results (Dense Rewards)
- Training speed: ~1500 steps/sec
- Self-play adds minimal overhead (~5-10% slowdown)
- Recommended: Use same hyperparameters

### Combining with Curriculum Learning

Self-play can be combined with curriculum learning:

```python
from ai.self_play import SelfPlayConfig, TrainingMode
from train.self_play_training import SelfPlayTrainingConfig

# Start with rule-based curriculum
config1 = SelfPlayTrainingConfig(
    self_play_ratio=0.2,  # Mostly rule-based
    sample_latest_ratio=0.5,  # Focus on recent
)

# Transition to mixed
config2 = SelfPlayTrainingConfig(
    self_play_ratio=0.5,
    sample_best_ratio=0.4,  # More best policies
)

# Final phase: self-play heavy
config3 = SelfPlayTrainingConfig(
    self_play_ratio=0.8,
    sample_random_ratio=0.5,  # More diversity
)
```

## Troubleshooting

### Common Issues

1. **Policy pool not growing**
   - Check `save_freq` is reasonable
   - Verify disk space for checkpoints
   - Ensure model saves successfully

2. **Win rate not improving**
   - Increase `sample_best_ratio` to train against stronger opponents
   - Check ELO decay isn't too aggressive
   - Verify training mode is appropriate

3. **Training instability**
   - Reduce `elo_k_factor` for smoother ELO changes
   - Increase `alternating_epochs` for more stable transitions
   - Check for reward function issues

### Performance Tips

1. **Memory Management**
   - Reduce `pool_size` if memory constrained
   - Use checkpoint cleanup for old policies

2. **Training Speed**
   - Reduce `n_eval_episodes` for faster iterations
   - Increase `save_freq` to reduce I/O overhead

3. **Strategy Quality**
   - Monitor ELO distribution in pool
   - Check identity-specific win rates
   - Review match history for patterns

## API Reference

### Key Classes

- `SelfPlayConfig`: Configuration for self-play training
- `SelfPlayTrainer`: Main trainer class
- `PolicyPool`: Manages policy versions and ELO
- `PolicyRecord`: Individual policy metadata
- `MatchHistory`: Records game history

### Key Functions

- `run_self_play()`: Main entry point for self-play training
- `run_integrated_training()`: Integrated training with multiple modes
- `sample_policy()`: Sample opponent from pool

## Future Improvements

1. **Population-Based Training**
   - Multiple concurrent agents
   - Hyperparameter evolution

2. **League Training**
   - Specialized strategies
   - Main exploiter and main players

3. **Prioritized Sampling**
   - Weight by recency of last match
   - Balance match counts across policies

4. **ELO Monte Carlo**
   - More robust rating estimation
   - Handle draw scenarios

## Conclusion

The self-play integration provides a robust framework for continuous improvement of the SGS AI. By combining multiple training modes, sophisticated opponent sampling, and comprehensive tracking, the system is designed to push win rates from the current ~20% baseline toward the 50% target.

The recommended approach is to:
1. Start with mixed training at 50% self-play ratio
2. Monitor ELO progression and identity win rates
3. Gradually increase self-play ratio as the policy pool matures
4. Fine-tune sampling distribution based on observed results

With proper configuration and sufficient training time, self-play should provide a significant boost to AI performance, helping achieve the project's win rate goals.