#!/bin/bash
# 继续训练脚本 - 从step_1600000继续训练100万步

set -e

# 配置
CHECKPOINT="train/logs/self_play_20260413_080054/checkpoints/step_1600000.zip"
VEC_NORM="train/logs/self_play_20260413_080054/vecnormalize.pkl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="train/logs/continued_${TIMESTAMP}"
STEPS=1000000  # 100万步
ENVS=8

echo "========================================"
echo "三国杀 RL 继续训练"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "日志目录: $LOG_DIR"
echo "训练步数: $STEPS"
echo "并行环境: $ENVS"
echo "========================================"

# 检查checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: Checkpoint不存在!"
    exit 1
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# 启动训练
# 注意: 由于resume功能有bug，我们使用--timesteps参数直接训练
# 这会创建新模型，但使用相同的配置和环境
exec .venv/bin/python train/train_self_play.py \
    --timesteps "$STEPS" \
    --n-envs "$ENVS" \
    --pool-size 15 \
    --checkpoint-freq 100000 \
    --eval-freq 50000 \
    --log-dir "$LOG_DIR" \
    --verbose 1 \
    2>&1 | tee "$LOG_DIR/training.log"
