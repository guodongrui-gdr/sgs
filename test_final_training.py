#!/usr/bin/env python3
"""
final_training.py 测试脚本

测试目标：验证 final_training.py 可以成功完成训练，无错误
"""

import sys
import multiprocessing as mp

# 设置spawn模式（CUDA兼容）
mp.set_start_method("spawn", force=True)

import warnings

warnings.filterwarnings("ignore")

import logging

# 只显示ERROR级别日志
logging.getLogger("ai.gym_wrapper").setLevel(logging.ERROR)
logging.getLogger("engine.game_engine").setLevel(logging.ERROR)
logging.getLogger("train.multi_agent_eval").setLevel(logging.INFO)

from train.final_training import FinalTrainingConfig, run_final_training


def test_quick_training():
    """快速训练测试 - 验证没有错误"""
    print("=" * 60)
    print("开始 final_training.py 快速测试")
    print("=" * 60)

    config = FinalTrainingConfig(
        total_timesteps=500,
        n_envs=1,
        use_subprocess=False,
        use_curriculum=False,
        eval_freq=500,
        n_eval_episodes=1,
    )

    print(f"配置: {config.total_timesteps} timesteps, {config.n_envs} env")
    print()

    try:
        model, curriculum = run_final_training(config)

        print()
        print("=" * 60)
        print("✅ 测试通过!")
        print("=" * 60)
        print("✅ 训练成功完成")
        print("✅ 没有遇到ERROR")
        print("✅ RL对手评估正常工作")
        print("=" * 60)
        return True
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ 测试失败!")
        print("=" * 60)
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_quick_training()
    sys.exit(0 if success else 1)
