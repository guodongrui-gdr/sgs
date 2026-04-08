"""
train.py - 可编辑的训练脚本

agent 可以修改此文件中的：
- HyperparametersConfig: 超参数配置
- RewardConfig: 奖励函数参数
- ModelConfig: 模型架构选择
- train_main(): 训练循环逻辑

agent 不能修改：
- prepare.py中的函数和常量
- 固定时间预算
- evaluate_model()评估逻辑
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare import (
    make_env,
    make_vec_env,
    evaluate_model,
    print_results,
    TimeBudgetTracker,
    get_peak_memory_mb,
    FIXED_TIME_BUDGET,
    EVAL_EPISODES,
    verify_dependencies,
    SB3_AVAILABLE,
    MASKABLE_PPO_AVAILABLE,
)

import time
import numpy as np

# ============ 可修改部分开始 - agent 可自由修改此区域 ============


class HyperparametersConfig:
    """
    超参数配置 - agent 可修改

    这些参数直接影响训练效果，agent 可以尝试不同的值
    """

    learning_rate = 5e-4  # 学习率（可尝试：1e-4, 1e-3, 3e-4等）
    gamma = 0.996  # 折扣因子（可尝试：0.95, 0.99, 0.995等）
    gae_lambda = 0.96  # GAE lambda（可尝试：0.9, 0.95, 0.98等）
    clip_range = 0.25  # PPO clip范围
    ent_coef = 0.05  # 熵系数（可尝试：0.01, 0.05, 0.1等）
    vf_coef = 0.5  # 价值函数系数
    n_steps = 2048  # 每次rollout步数（可尝试：1024, 2048, 4096等）
    batch_size = 256  # 批大小（可尝试：128, 256, 512等）
    n_epochs = 20  # 每次更新epoch数（可尝试：4, 10, 20等）
    max_grad_norm = 0.5  # 梯度裁剪（可尝试：0.5, 1.0等）
    n_envs = 4  # 环境数量（可尝试：1, 4, 8等）


class RewardConfig:
    """
    奖励函数配置 - agent 可修改

    这些参数影响奖励函数的计算方式，agent 可以调整奖励策略
    """

    # 主要奖励
    win_reward = 150.0  # 胜利奖励（可尝试：50, 100, 200等）
    lose_penalty = -30.0  # 失败惩罚（可尝试：-50, -100, -200等）

    # 伤害奖励
    damage_dealt_scale = 3.0  # 造成伤害奖励倍数（增加，鼓励进攻）
    damage_taken_penalty = -0.5  # 受到伤害惩罚倍数（可尝试：-0.5, -1.0, -2.0等）

    # 技能奖励
    skill_use_reward = 0.5  # 使用技能奖励（可尝试：0.1, 0.5, 1.0等）
    skill_success_reward = 2.0  # 技能成功奖励（可尝试：1.0, 2.0, 5.0等）

    # 卡牌奖励
    card_use_reward = 0.1  # 使用卡牌奖励（可尝试：0.05, 0.1, 0.2等）

    # 回合奖励
    round_alive_reward = 1.0  # 每回合存活奖励（可尝试：0.5, 1.0, 2.0等）


class ModelConfig:
    """
    模型架构配置 - agent 可修改

    agent 可以选择不同的模型架构和网络结构
    """

    use_transformer = False  # 是否使用Transformer（可尝试：True/False）
    use_masking = True  # 是否使用action masking（推荐True）
    policy_type = "mlp"  # 策略类型（可尝试："mlp", "transformer")

    # MLP 配置
    mlp_hidden_sizes = [
        1024,
        1024,
    ]  # MLP隐藏层大小
    n_epochs = 20  # 每次更新epoch数

    # Transformer 配置（如果use_transformer=True）
    transformer_depth = 4  # Transformer深度（可尝试：2, 4, 6, 8）
    transformer_heads = 4  # 注意力头数（可尝试：2, 4, 8）
    transformer_embed_dim = 256  # 嵌入维度（可尝试：128, 256, 512）

    # 其他配置
    activation_fn = "relu"  # 激活函数（可尝试："relu", "tanh", "elu"）


def create_model(env, hp_config: HyperparametersConfig, model_config: ModelConfig):
    """
    创建模型 - agent 可修改模型创建逻辑

    Args:
        env: 环境
        hp_config: 超参数配置
        model_config: 模型配置

    Returns:
        创建的模型
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 not installed")

    policy = "MultiInputPolicy"
    policy_kwargs = None

    # agent 可修改：添加自定义策略网络
    if model_config.use_transformer:
        try:
            from ai.models.transformer_policy import (
                TransformerFeaturesExtractor,
                TransformerConfig,
            )

            transformer_config = TransformerConfig(
                depth=model_config.transformer_depth,
                heads=model_config.transformer_heads,
                embed_dim=model_config.transformer_embed_dim,
            )

            policy_kwargs = dict(
                features_extractor_class=TransformerFeaturesExtractor,
                features_extractor_kwargs={
                    "config": transformer_config,
                    "state_dim": env.observation_space["state"].shape[0]
                    if hasattr(env.observation_space, "spaces")
                    else 3000,
                },
            )
        except ImportError:
            print("Warning: Transformer not available, using MLP")

    # agent 可修改：选择不同的算法
    use_masking = model_config.use_masking and MASKABLE_PPO_AVAILABLE

    if use_masking:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO(
            policy,
            env,
            learning_rate=hp_config.learning_rate,
            n_steps=hp_config.n_steps,
            batch_size=hp_config.batch_size,
            n_epochs=hp_config.n_epochs,
            gamma=hp_config.gamma,
            gae_lambda=hp_config.gae_lambda,
            clip_range=hp_config.clip_range,
            ent_coef=hp_config.ent_coef,
            vf_coef=hp_config.vf_coef,
            max_grad_norm=hp_config.max_grad_norm,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    else:
        from stable_baselines3 import PPO

        model = PPO(
            policy,
            env,
            learning_rate=hp_config.learning_rate,
            n_steps=hp_config.n_steps,
            batch_size=hp_config.batch_size,
            n_epochs=hp_config.n_epochs,
            gamma=hp_config.gamma,
            gae_lambda=hp_config.gae_lambda,
            clip_range=hp_config.clip_range,
            ent_coef=hp_config.ent_coef,
            vf_coef=hp_config.vf_coef,
            max_grad_norm=hp_config.max_grad_norm,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )

    return model


def inject_reward_config(reward_config: RewardConfig):
    """
    注入奖励配置到 ai/reward.py - agent 可修改注入逻辑

    这个函数将 RewardConfig 中的参数注入到奖励计算中
    """
    try:
        from ai.reward import RewardCalculator

        # agent 可修改：如何注入奖励参数
        RewardCalculator.win_reward = reward_config.win_reward
        RewardCalculator.lose_penalty = reward_config.lose_penalty
        RewardCalculator.damage_dealt_scale = reward_config.damage_dealt_scale
        RewardCalculator.damage_taken_penalty = reward_config.damage_taken_penalty
        RewardCalculator.skill_use_reward = reward_config.skill_use_reward
        RewardCalculator.skill_success_reward = reward_config.skill_success_reward
        RewardCalculator.card_use_reward = reward_config.card_use_reward
        RewardCalculator.round_alive_reward = reward_config.round_alive_reward

    except ImportError:
        print("Warning: Could not inject reward config")


def train_main():
    """
    训练主函数 - agent 可修改训练逻辑

    这是 agent 可以自由修改的训练循环
    agent 可以尝试不同的训练策略、学习率调度等
    """
    # 验证依赖
    if not verify_dependencies():
        return

    # 注入奖励配置（agent 可修改）
    inject_reward_config(RewardConfig())

    # 创建环境（agent 不能修改环境创建）
    env = make_vec_env(n_envs=HyperparametersConfig.n_envs, seed=42)

    # 创建模型（agent 可修改模型创建）
    model = create_model(env, HyperparametersConfig(), ModelConfig())

    # 时间预算跟踪（agent 不能修改时间预算）
    tracker = TimeBudgetTracker(FIXED_TIME_BUDGET)
    tracker.start()

    # 训练循环（agent 可修改）
    total_steps = 0
    steps_per_update = 5000  # agent 可修改：每次更新的步数

    print(f"Starting training with {FIXED_TIME_BUDGET}s time budget...")
    print(
        f"Hyperparameters: LR={HyperparametersConfig.learning_rate}, "
        f"gamma={HyperparametersConfig.gamma}, "
        f"n_steps={HyperparametersConfig.n_steps}"
    )

    # agent 可修改：训练策略
    # 例如：可以添加学习率调度、早停、课程学习等

    while not tracker.should_stop():
        remaining_time = tracker.remaining()

        # agent 可修改：根据剩余时间调整训练步数
        if remaining_time < 90:
            break  # 剩余时间不足，停止训练（留出评估时间）

        # 训练一个batch
        model.learn(total_timesteps=steps_per_update, progress_bar=False)
        total_steps += steps_per_update

        # agent 可修改：添加额外的训练逻辑
        # 例如：打印中间结果、调整学习率等

        # 每50K步打印一次进度（agent 可修改）
        if total_steps % 50000 == 0:
            elapsed = tracker.elapsed()
            fps = total_steps / elapsed
            print(f"Step {total_steps}: elapsed={elapsed:.1f}s, fps={fps:.1f}")

    # 获取峰值内存
    peak_memory = get_peak_memory_mb()

    # 计算FPS
    training_seconds = tracker.elapsed()
    fps = total_steps / training_seconds if training_seconds > 0 else 0

    # 评估模型（agent 不能修改评估逻辑）
    print("\nEvaluating model...")
    win_rate, mean_reward, std_reward, eval_info = evaluate_model(
        model,
        n_episodes=10,  # 增加评估回合数到10减少随机性
        deterministic=False,
        verbose=True,
    )

    # 输出结果（agent 不能修改输出格式）
    print_results(
        win_rate=win_rate,
        mean_reward=mean_reward,
        std_reward=std_reward,
        training_seconds=training_seconds,
        total_steps=total_steps,
        peak_memory_mb=peak_memory,
        fps=fps,
        model_params={
            "n_envs": HyperparametersConfig.n_envs,
            "use_masking": ModelConfig.use_masking,
            "use_transformer": ModelConfig.use_transformer,
        },
    )

    env.close()


# ============ 可修改部分结束 ============

if __name__ == "__main__":
    train_main()
