#!/bin/bash
# 继续训练脚本 - 从 step_1600000 继续训练 100万步

set -e

LOG_DIR="train/logs/self_play_continued_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT="train/logs/self_play_20260413_080054/checkpoints/step_1600000.zip"

echo "=========================================="
echo "继续训练 RL 模型"
echo "=========================================="
echo "从 checkpoint: $CHECKPOINT"
echo "日志目录: $LOG_DIR"
echo "训练步数: 1,000,000 (从 1.6M 到 2.6M)"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: Checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

# 运行训练
.venv/bin/python train/train_self_play.py \
    --timesteps 2600000 \
    --n-envs 8 \
    --pool-size 15 \
    --checkpoint-freq 100000 \
    --eval-freq 50000 \
    --resume "$CHECKPOINT" \
    --log-dir "$LOG_DIR" \
    --verbose 1

echo "=========================================="
echo "训练完成!"
echo "模型保存在: $LOG_DIR"
echo "=========================================="
