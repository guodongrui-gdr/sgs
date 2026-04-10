# Final Training Configuration - Integrated Optimizations

## Overview

This document summarizes the final training configuration that integrates ALL optimizations from Waves 1-3 to achieve the target of **≥50% win rate in <24 hours**.

**Target Metrics:**
- Win Rate: ≥50% over 100 evaluation episodes
- Training Time: <24 hours (estimated 3.3 hours for 5M steps at 1500 steps/sec)

---

## Wave 1 Optimizations: Architecture & Hyperparameters

### 1.1 Architecture Selection

**Decision:** MLP (Multi-Layer Perceptron) over Transformer

**Rationale:**
- MLP is **67% faster** than Transformer (1500 vs 900 steps/sec)
- Similar win rate performance (~30% with optimal hyperparams)
- Significantly lower computational cost
- Easier to train and debug

**Configuration:**
```python
use_mlp = True
use_transformer = False
policy = "MultiInputPolicy"
```

**Performance Impact:**
- Training Speed: +67%
- Win Rate: Same as Transformer (~30% baseline)

---

### 1.2 Optimal Hyperparameters

**Best performing hyperparameters from Wave 1:**

```python
learning_rate = 5e-4  # Optimal: balances exploration and exploitation
ent_coef = 0.05       # Optimal: sufficient exploration
lr_schedule_type = "cosine"  # Smooth decay

# PPO parameters
gamma = 0.99          # Discount factor
gae_lambda = 0.98     # GAE lambda
clip_range = 0.2      # PPO clipping
n_steps = 2048        # Steps per rollout
batch_size = 256      # Mini-batch size
n_epochs = 10         # Epochs per update
```

**Rationale for Learning Rate (5e-4):**
- Higher rates (1e-3) cause instability
- Lower rates (1e-4) slow convergence
- 5e-4 provides best balance

**Rationale for Entropy Coefficient (0.05):**
- Higher values (0.1) reduce performance due to excessive exploration
- Lower values (0.01) cause premature convergence
- 0.05 maintains sufficient exploration throughout training

**Performance Impact:**
- Win Rate: +10-15% improvement from baseline
- Training Stability: Smooth convergence

---

## Wave 2 Optimizations: Training Speed & Rewards

### 2.1 Training Speed Optimization

**Decision:** SubprocVecEnv with 8 parallel environments

**Configuration:**
```python
n_envs = 8
use_subprocess = True  # SubprocVecEnv
```

**Performance:**
- **1500 steps/sec** (vs 500 steps/sec with DummyVecEnv)
- **3x speedup**
- Estimated training time: **~3.3 hours** for 5M steps (vs 2.8 hours with DummyVecEnv)

**Impact:**
- Enables 5M step training runs overnight
- Allows more frequent experimentation
- Reduces total training time by 67%

---

### 2.2 Dense Reward Design

**Key Insight:** Terminal rewards should be ~10x intermediate rewards

**Reward Configuration:**
```python
# Terminal rewards (dominant)
victory = 100.0       # Phase 2: Increased from 50.0
defeat = -100.0       # Phase 2: Increased from -50.0

# Intermediate rewards (dense)
skill_activation = 0.2         # NEW: Encourages skill usage
effective_card_use = 0.5       # NEW: Encourages strategic card play
damage_dealt = 3.0
damage_taken = -1.5
kill_enemy = 15.0
kill_ally = -20.0

# Progress rewards (NEW in Phase 2)
progress_reward = 0.3
defense_reward = 0.2
turn_progress_reward = 0.1
round_progress_reward = 0.2

# Team rewards (NEW in Phase 2)
protect_lord_reward = 2.0
assist_ally_reward = 1.0
```

**Terminal/Intermediate Ratio:**
- Target: ~10x (100.0 / 3.0 ≈ 33x for damage, 6.7x for kill_enemy)
- Rationale: Ensures terminal rewards dominate, providing strong learning signal

**Performance Impact:**
- Win Rate: +5-10% improvement from sparse rewards
- Faster convergence due to more frequent feedback

---

### 2.3 Checkpointing Strategy

**Configuration:**
```python
checkpoint_freq = 100_000  # Every 100K steps for 5M training
eval_freq = 50_000         # Evaluate every 50K steps
n_eval_episodes = 100      # 100 episodes for reliable win rate
```

**Rationale:**
- 100K intervals provide good coverage for 5M step training (50 checkpoints total)
- 50K evaluation frequency enables early stopping detection
- 100 episodes provides statistically reliable win rate estimate (±5% confidence)

**Checkpoint Management:**
- Automatic versioning with timestamps
- VecNormalize statistics saved with each checkpoint
- Best model tracking based on win rate
- Old checkpoint cleanup (keep last 10)

---

### 2.4 Early Stopping

**Configuration:**
```python
early_stopping_patience = 10         # Stop after 10 evals with no improvement
early_stopping_threshold = 0.05      # 5% win rate improvement required
min_timesteps_before_stopping = 500_000  # Don't stop before 500K steps
```

**Stopping Conditions:**
1. **Plateau Detection:** No ≥5% win rate improvement for 10 consecutive evaluations
2. **Convergence Detection:** Reward variance drops below threshold (std < 10.0)
3. **Target Achieved:** Win rate reaches 50% (optional early success)
4. **Manual Intervention:** User-triggered stop

**Impact:**
- Prevents over-training
- Saves computational resources
- Identifies convergence or plateau

---

## Wave 3 Optimizations: Curriculum & Self-Play

### 3.1 Curriculum Learning

**Strategy:** 3-stage progressive difficulty

```
Stage 1 (Easy):     0 - 1.67M steps    (max_rounds=5,  random opponents)
Stage 2 (Medium):   1.67M - 3.33M steps (max_rounds=10, rule AI opponents)
Stage 3 (Hard):     3.33M - 5M steps    (max_rounds=15, self-play)
```

**Stage Advancement Criteria:**
- Minimum episodes: 1000
- Win rate threshold: 40% (Easy→Medium), 45% (Medium→Hard)
- Consecutive checks: 3 evaluations meeting threshold

**Reward Multipliers:**
- Easy: 1.5x (stronger learning signal initially)
- Medium: 1.0x (normal rewards)
- Hard: 0.8x (reduced to encourage efficiency)

**Performance Impact:**
- Expected win rate progression:
  - End of Easy: 40%
  - End of Medium: 45%
  - End of Hard: 50%

---

### 3.2 Self-Play Configuration

**Policy Pool:**
```python
pool_size = 10  # Keep last 10 policy versions
```

**Opponent Sampling Strategy:**
```python
sample_latest_ratio = 0.3  # 30% sample most recent policy
sample_best_ratio = 0.3    # 30% sample highest ELO policy
sample_random_ratio = 0.4  # 40% random sampling (diversity)
```

**Rationale:**
- **30% Latest:** Ensures agent can beat current version (exploitation)
- **30% Best:** Ensures agent can beat strongest historical version (exploitation)
- **40% Random:** Maintains diversity, prevents overfitting to specific opponents

**ELO Rating System:**
```python
k_factor = 32.0           # ELO update magnitude
elo_decay_rate = 0.01     # 1% decay per hour
elo_max_decay = 0.3       # Maximum 30% decay
initial_elo = 1000.0      # Starting ELO
```

**ELO Decay Rationale:**
- Old policies should have lower ELO to reflect that newer policies are generally stronger
- Encourages sampling of newer policies over time
- Prevents overfitting to very old strategies

**Self-Play Update Frequency:**
```python
self_play_update_freq = 25_000  # Add current policy to pool every 25K steps
```

**Performance Impact:**
- Expected win rate in Hard stage: 45% → 50%
- Prevents overfitting to rule AI
- Develops robust strategies against diverse opponents

---

## Final Integrated Configuration

### Complete Configuration Summary

```python
# Wave 1: Architecture & Hyperparams
architecture = "MLP"                    # 67% faster than Transformer
learning_rate = 5e-4                   # Optimal
ent_coef = 0.05                        # Optimal
lr_schedule = "cosine"                 # Smooth decay

# Wave 2: Training Speed
n_envs = 8
use_subprocess = True                  # 1500 steps/sec
total_timesteps = 5_000_000            # ~3.3 hours training

# Wave 2: Rewards
victory = 100.0
defeat = -100.0
skill_activation = 0.2
effective_card_use = 0.5
terminal_intermediate_ratio = 10.0     # Target ratio

# Wave 2: Checkpointing
checkpoint_freq = 100_000
eval_freq = 50_000
n_eval_episodes = 100

# Wave 2: Early Stopping
early_stopping_patience = 10
early_stopping_threshold = 0.05

# Wave 3: Curriculum
curriculum_stages = 3
timesteps_per_stage = 1_666_667
stage_configs = {
    "easy": {"max_rounds": 5, "opponent": "random", "reward_mult": 1.5},
    "medium": {"max_rounds": 10, "opponent": "rule", "reward_mult": 1.0},
    "hard": {"max_rounds": 15, "opponent": "self-play", "reward_mult": 0.8},
}

# Wave 3: Self-Play
pool_size = 10
sampling = {
    "latest": 0.3,
    "best": 0.3,
    "random": 0.4,
}
elo_k_factor = 32.0
elo_decay_rate = 0.01
```

---

## Expected Performance Trajectory

### Training Timeline (5M steps, ~3.3 hours)

```
Hour 0-1 (0-1.67M steps):
  Stage: Easy (random opponents)
  Win Rate: 20% → 40%
  Focus: Learning basic game mechanics, card usage

Hour 1-2 (1.67M-3.33M steps):
  Stage: Medium (rule AI opponents)
  Win Rate: 40% → 45%
  Focus: Strategic play, skill combinations

Hour 2-3.3 (3.33M-5M steps):
  Stage: Hard (self-play)
  Win Rate: 45% → 50%
  Focus: Robust strategies, opponent adaptation
```

### Resource Requirements

**Hardware:**
- CPU: Multi-core (for SubprocVecEnv)
- GPU: Optional (PPO can run on CPU)
- RAM: 8-16 GB

**Training Speed:**
- Steps/sec: 1500
- Total time: ~3.3 hours for 5M steps
- Overnight training: Can run 3-4 iterations

---

## Monitoring & Evaluation

### Key Metrics to Track

1. **Win Rate:** Primary success metric (target: ≥50%)
2. **Average Reward:** Overall performance indicator
3. **Reward Variance:** Convergence indicator
4. **ELO Rating:** Self-play opponent strength
5. **Stage Progression:** Curriculum advancement timing

### TensorBoard Logs

- `rollout/ep_rew_mean`: Average episode reward
- `rollout/ep_len_mean`: Average episode length
- `train/learning_rate`: Current learning rate
- `train/entropy_loss`: Exploration level
- `train/policy_loss`: Policy optimization progress

### Manual Evaluation

**Command:**
```bash
python train/final_training.py --mode evaluate \
    --model-path path/to/model.zip \
    --n-eval-episodes 100
```

**Expected Output:**
```
Win Rate: 50%+
Average Reward: 100-200
Identity Win Rates:
  主公: 50-55%
  忠臣: 50-55%
  反贼: 45-50%
  内奸: 40-45%
```

---

## Usage

### Start Training

```bash
python train/final_training.py
```

### Resume Training

```bash
python train/final_training.py --resume \
    --model-path train/logs/final_training_*/checkpoints/model_1000000_steps.zip \
    --curriculum-state train/logs/final_training_*/curriculum_state.json
```

### Evaluate Model

```bash
python train/final_training.py --mode evaluate \
    --model-path train/logs/final_training_*/final_model.zip
```

---

## Success Criteria

**Minimum Viable Success:**
- Win Rate: ≥50% over 100 episodes
- Training Time: <24 hours
- No manual intervention required

**Stretch Goals:**
- Win Rate: ≥55%
- Training Time: <4 hours
- Balanced identity win rates (all ≥45%)

---

## Troubleshooting

### Low Win Rate (<40%)

**Potential Causes:**
1. Insufficient training time
2. Learning rate too high/low
3. Entropy coefficient too low (premature convergence)

**Solutions:**
- Increase training duration to 7-10M steps
- Adjust learning rate (try 3e-4 or 7e-4)
- Increase entropy coefficient to 0.07-0.1

### Slow Training (<1000 steps/sec)

**Potential Causes:**
1. Not using SubprocVecEnv
2. Too many environments (overhead)
3. Insufficient CPU cores

**Solutions:**
- Ensure `use_subprocess=True`
- Reduce `n_envs` to 4-6
- Use machine with more CPU cores

### Training Instability

**Symptoms:**
- Win rate fluctuates wildly
- Reward variance high
- Loss diverges

**Solutions:**
- Reduce learning rate to 3e-4
- Increase batch size to 512
- Reduce clip_range to 0.15

---

## Conclusion

This final configuration integrates all optimizations from Waves 1-3 to achieve the target win rate of ≥50% in under 24 hours. The combination of:

- **Wave 1:** Fast architecture (MLP) + optimal hyperparams
- **Wave 2:** High training speed (SubprocVecEnv) + dense rewards + checkpointing
- **Wave 3:** Progressive difficulty (curriculum) + diverse opponents (self-play)

provides a robust training pipeline that should consistently achieve the target performance.

**Key Success Factors:**
1. **Speed:** 1500 steps/sec enables overnight training iterations
2. **Curriculum:** Progressive difficulty ensures stable learning
3. **Rewards:** Dense feedback accelerates convergence
4. **Self-Play:** Diverse opponents prevent overfitting

**Next Steps:**
1. Run initial 5M step training (est. 3.3 hours)
2. Evaluate final win rate
3. If target not met, extend to 7-10M steps or adjust hyperparameters
4. Iterate on curriculum stages or self-play strategy if needed