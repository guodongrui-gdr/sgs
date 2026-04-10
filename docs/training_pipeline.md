# Extended Training Pipeline Documentation

## Overview

The extended training pipeline enables training runs of 5-10M steps with proper checkpointing, early stopping, and real-time monitoring. Based on Wave 1 findings, the pipeline is optimized for speed and configured to achieve the 50% win rate target.

## Key Components

### 1. CheckpointManager (`train/checkpoint_manager.py`)

Manages checkpoint lifecycle for extended training runs.

**Features:**
- Save checkpoints every 100K steps (default for 5M training)
- Save VecNormalize statistics with checkpoints
- Version checkpoints with timestamps
- Automatic recovery from interruption
- Track best checkpoint based on win rate
- Cleanup old checkpoints to save disk space

**Usage:**
```python
from train.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    log_dir="train/logs/sgs_run",
    checkpoint_freq=100_000,  # 100K steps
    max_checkpoints=10,  # Keep last 10 checkpoints
)

# Save checkpoint
manager.save_checkpoint(model, env, timestep=100_000, metrics={"win_rate": 0.25})

# Find latest checkpoint for recovery
latest = manager.find_latest_checkpoint()
if latest:
    model, env = manager.load_checkpoint(latest)

# Create recovery script
manager.create_recovery_script()
```

### 2. EarlyStoppingCallback (`train/early_stopping.py`)

Detects training plateaus and convergence to enable early stopping.

**Features:**
- Win rate plateau detection (no improvement for N evaluations)
- Reward variance threshold (convergence detection)
- Target achievement detection (50% win rate)
- Configurable patience and thresholds
- Logging of stopping reasons for analysis

**Usage:**
```python
from train.early_stopping import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    eval_freq=50_000,  # Evaluate every 50K steps
    n_eval_episodes=100,  # 100 episodes for reliable win rate
    plateau_patience=10,  # Stop after 10 evals with no improvement
    plateau_threshold=0.05,  # 5% win rate improvement required
    convergence_threshold=10.0,  # Reward variance threshold
    min_timesteps=500_000,  # Don't stop before 500K
)

callback = early_stopping.create_callback()
model.learn(total_timesteps=5_000_000, callback=[callback])
```

**Stopping Conditions:**

| Condition | Trigger | Threshold |
|-----------|---------|-----------|
| Plateau | Win rate unchanged for N evals | 5% improvement required |
| Convergence | Reward variance drops below threshold | std < 10.0 |
| Target Achieved | Win rate reaches 50% | 0.50 |
| Max Steps | Training reaches target timesteps | total_timesteps |

### 3. TrainingMonitor (`train/monitor_dashboard.py`)

Real-time monitoring dashboard for training progress.

**Features:**
- Real-time training progress display
- Win rate progression tracking
- Reward mean and variance tracking
- Steps per second and estimated completion time
- Progress bar with rich formatting
- Training report generation

**Usage:**
```python
from train.monitor_dashboard import TrainingMonitor

monitor = TrainingMonitor(
    total_timesteps=5_000_000,
    log_dir="train/logs/sgs_run",
    update_freq=1000,  # Update every 1000 steps
)

callback = monitor.create_callback()
model.learn(total_timesteps=5_000_000, callback=[callback])

# Save final report
monitor.save_report()
```

**Output Example:**
```
============================================================
SGS Training Pipeline - 20260409 12:00:00
============================================================
Progress: 20.0% (1,000,000 / 5,000,000)
Speed: 1500.0 steps/sec
Elapsed: 0:11:07 | Remaining: 0:44:28
Avg Reward: -10.5 ± 25.2
Win Rate: 25.0%
============================================================
```

### 4. TrainingSpeedOptimizer

Analyzes and optimizes training speed configuration.

**Recommended Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_envs | 8 | 8 parallel environments maximize CPU utilization |
| use_subprocess | True | SubprocVecEnv enables true parallelism |
| max_rounds | 50 | Extended for complete game cycles |
| n_steps | 2048 | Sufficient rollout diversity |
| batch_size | 256 | Balance between speed and stability |
| n_epochs | 10 | Gradient updates per rollout |

**Expected Performance:**
- Estimated speed: 1500+ steps/sec
- Training time for 5M steps: ~3.3 hours
- Training time for 10M steps: ~6.7 hours

## Training Configuration

Based on Wave 1 findings, the optimal configuration for 5M step training:

```python
from train.train_sb3 import TrainingConfig

config = TrainingConfig(
    # Training duration
    total_timesteps=5_000_000,  # 5M steps for overnight training
    
    # Algorithm (MLP recommended, 67% faster than Transformer)
    algorithm="ppo",
    use_masking=True,
    use_transformer=False,
    
    # Hyperparameters (optimal from Wave 1)
    learning_rate=5e-4,
    lr_schedule_type="cosine",
    ent_coef=0.05,
    batch_size=256,
    n_epochs=10,
    
    # Environment optimization
    n_envs=8,
    use_subprocess=True,
    max_rounds=50,
    
    # Checkpointing
    checkpoint_freq=100_000,  # 100K steps
    
    # Evaluation
    eval_freq=50_000,
    n_eval_episodes=100,
    
    # Early stopping
    early_stopping_patience=10,
    early_stopping_threshold=0.05,
    
    # Monitoring
    enable_monitoring=True,
)
```

## Command-Line Usage

### Basic Training (5M steps)

```bash
python train/train_sb3.py --mode train --timesteps 5000000
```

### Extended Training (10M steps)

```bash
python train/train_sb3.py --mode train --timesteps 10000000 \
    --n-envs 8 \
    --use-subprocess \
    --max-rounds 50 \
    --lr 5e-4 \
    --ent-coef 0.05 \
    --lr-schedule cosine \
    --checkpoint-freq 100000 \
    --eval-freq 50000 \
    --n-eval-episodes 100
```

### Resume Training from Checkpoint

```bash
python train/train_sb3.py --mode train \
    --resume \
    --model-path train/logs/sgs_run/checkpoints/checkpoint_100000_steps.zip \
    --timesteps 5000000
```

### View Recommended Configuration

```bash
python train/train_sb3.py --mode benchmark --timesteps 5000000
```

### Evaluate Model

```bash
python train/train_sb3.py --mode evaluate \
    --model-path train/logs/sgs_run/final_model.zip \
    --n-eval-episodes 100
```

## Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Initialize Environment (n_envs=8, SubprocVecEnv)        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Create Model (MaskablePPO, MLP policy)                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Initialize Callbacks                                    │
│     - CheckpointManager (100K intervals)                    │
│     - EarlyStoppingCallback (plateau detection)             │
│     - TrainingMonitor (real-time dashboard)                 │
│     - SelfPlayCallback (policy pool)                        │
│     - AsyncEvalCallback (background evaluation)             │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Training Loop                                           │
│     for each timestep:                                      │
│       - Collect rollout (n_steps=2048)                      │
│       - Update policy (n_epochs=10)                         │
│       - Update monitoring display                           │
│       - Check checkpoint freq (save at 100K)                │
│       - Check eval freq (run eval at 50K)                   │
│       - Check early stopping conditions                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Training End                                            │
│     - Save final model                                      │
│     - Save VecNormalize stats                               │
│     - Generate training report                              │
│     - Log stopping reason (if early stopped)                │
└─────────────────────────────────────────────────────────────┘
```

## Output Files

After training, the following files are generated in the log directory:

```
train/logs/sgs_run/
├── checkpoints/
│   ├── checkpoint_100000_steps_20260409_100000.zip
│   ├── checkpoint_100000_steps_20260409_100000_vecnormalize.pkl
│   ├── checkpoint_200000_steps_20260409_110000.zip
│   ├── ...
│   ├── best_model.zip
│   ├── best_model_vecnormalize.pkl
│   ├── metadata/
│   │   └── checkpoint_history.json
│   └── recover_training.sh  # Recovery script
├── final_model.zip
├── vec_normalize.pkl
├── policy_pool/
│   └── policy_v*.zip
├── best_model/
│   └── best_model.zip
├── eval/
│   └── eval_results.txt
├── training_report.json
├── training_metrics.json
├── early_stopping_config.json
├── evaluation_history.json
├── stopping_condition.json  # If early stopped
├── monitor_config.json
├── events.out.tfevents.*    # TensorBoard logs
└── PPO_*/
    └── events.out.tfevents.*
```

## Monitoring with TensorBoard

```bash
tensorboard --logdir train/logs/sgs_run
```

Key metrics to monitor:
- `rollout/ep_rew_mean`: Episode reward mean
- `rollout/ep_len_mean`: Episode length mean
- `train/entropy_loss`: Policy entropy
- `train/learning_rate`: Current learning rate
- `train/loss`: Total loss
- `train/policy_gradient_loss`: Policy gradient loss
- `train/value_loss`: Value function loss

## Recovery from Interruption

If training is interrupted (power loss, crash, manual stop):

1. Find latest checkpoint:
```python
from train.checkpoint_manager import CheckpointManager

manager = CheckpointManager(log_dir="train/logs/sgs_run")
latest = manager.find_latest_checkpoint()
print(f"Latest checkpoint: {latest.timestep} steps")
```

2. Resume training:
```bash
# Using recovery script
bash train/logs/sgs_run/checkpoints/recover_training.sh

# Or manually
python train/train_sb3.py --mode train \
    --resume \
    --model-path train/logs/sgs_run/checkpoints/checkpoint_200000_steps_*.zip \
    --timesteps 5000000
```

## Expected Training Timeline

Based on Wave 1 findings and speed optimization:

| Training Steps | Expected Win Rate | Estimated Time |
|----------------|------------------|----------------|
| 100K | ~5-10% | ~7 minutes |
| 500K | ~15-20% | ~33 minutes |
| 1M | ~25-30% | ~1.1 hours |
| 2M | ~35-40% | ~2.2 hours |
| 3M | ~40-45% | ~3.3 hours |
| 5M | ~50% (target) | ~5.5 hours |
| 10M | ~60%+ | ~11 hours |

**Note:** Actual results may vary based on hyperparameter tuning and environment complexity.

## Troubleshooting

### Slow Training Speed (<500 steps/sec)

**Possible causes:**
- Not using SubprocVecEnv
- n_envs too low
- max_rounds too high causing long episodes
- CPU bottleneck

**Solutions:**
- Enable `--use-subprocess`
- Increase `--n-envs` to 8
- Check CPU utilization with `htop`

### Early Stopping Too Early

**Possible causes:**
- Patience too low
- Threshold too high
- Not enough evaluation episodes

**Solutions:**
- Increase `--early-stopping-patience` to 10+
- Lower `--early-stopping-threshold` to 0.05
- Increase `--n-eval-episodes` to 100+

### Win Rate Not Improving

**Possible causes:**
- Learning rate too low
- Entropy coefficient too high/low
- Training not long enough

**Solutions:**
- Use `--lr 5e-4` (optimal from Wave 1)
- Use `--ent-coef 0.05` (optimal from Wave 1)
- Train longer (at least 1M steps)

### Memory Issues

**Possible causes:**
- Too many parallel environments
- Large batch size

**Solutions:**
- Reduce `--n-envs` to 4
- Reduce `--batch-size` to 128
- Check memory usage with `nvidia-smi` or `free -h`

## Success Criteria Checklist

Before running 5M step training, verify:

- [ ] Training speed >1000 steps/sec (benchmark first)
- [ ] CheckpointManager configured (100K intervals)
- [ ] EarlyStoppingCallback configured (patience=10)
- [ ] TrainingMonitor enabled (real-time display)
- [ ] Recovery script created (`recover_training.sh`)
- [ ] TensorBoard running for monitoring
- [ ] Sufficient disk space for checkpoints (~500MB per checkpoint)
- [ ] Estimated training time <24 hours

---

*Generated: 2026-04-09*
*Framework: stable-baselines3 v2.8.0, sb3-contrib v2.8.0*
*Based on: Wave 1 Hyperparameter and Architecture Analysis*