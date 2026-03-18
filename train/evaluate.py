"""
模型评估脚本 - 验证训练效果

功能:
- 加载训练好的模型
- 与随机/规则策略对战
- 统计胜率和奖励
- 各身份详细统计
- 可视化对局
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.gym_wrapper import SGSEnv, SGSConfig
from ai.reward import IdentityRelationship


def _get_action_masks_from_obs(obs):
    """从观察中提取动作掩码"""
    if isinstance(obs, dict):
        type_mask = obs.get("action_mask_type", np.ones(12))
        card_mask = obs.get("action_mask_card", np.ones(20))
        target_mask = obs.get("action_mask_target", np.ones(8))
        return np.concatenate([type_mask, card_mask, target_mask])
    return np.ones(40)


def evaluate_model(
    model_path: str,
    n_episodes: int = 100,
    player_num: int = 5,
    max_rounds: int = 100,
    render: bool = False,
    verbose: bool = True,
    save_results: bool = False,
):
    """评估模型"""

    try:
        from sb3_contrib import MaskablePPO

        use_masking = True
    except ImportError:
        from stable_baselines3 import PPO as MaskablePPO

        use_masking = False

    config = SGSConfig(
        player_num=player_num,
        max_rounds=max_rounds,
    )

    env = SGSEnv(config)

    model = MaskablePPO.load(model_path, env=env)

    results = {
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "total_episodes": n_episodes,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_reward": 0.0,
        "identity_wins": {
            "主公": 0,
            "忠臣": 0,
            "反贼": 0,
            "内奸": 0,
        },
        "identity_games": {
            "主公": 0,
            "忠臣": 0,
            "反贼": 0,
            "内奸": 0,
        },
        "game_lengths": [],
        "episode_rewards": [],
        "win_by_round": {},
        "damage_dealt": [],
        "damage_taken": [],
    }

    print(f"\n{'=' * 60}")
    print(f"评估模型: {model_path}")
    print(f"对局数: {n_episodes}")
    print(f"{'=' * 60}\n")

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        step_count = 0
        episode_reward = 0.0

        while not done:
            if use_masking:
                if hasattr(env, "action_masks"):
                    action_masks = env.action_masks()
                elif hasattr(env, "get_attr"):
                    action_masks = env.get_attr("action_masks")[0]
                else:
                    action_masks = _get_action_masks_from_obs(obs)
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1

            if render and step_count % 10 == 0:
                env.render()

        results["game_lengths"].append(step_count)
        results["total_reward"] += episode_reward
        results["episode_rewards"].append(episode_reward)

        winner = info.get("winner", "unknown")
        player_identity = info.get("player_identity", "")

        results["identity_games"][player_identity] = (
            results["identity_games"].get(player_identity, 0) + 1
        )

        is_win = False
        if winner != "unknown" and player_identity:
            is_win = IdentityRelationship.is_victory(player_identity, winner)

        if is_win:
            results["wins"] += 1
            results["identity_wins"][player_identity] = (
                results["identity_wins"].get(player_identity, 0) + 1
            )
        else:
            results["losses"] += 1

        round_num = info.get("round_num", 0)
        if is_win and round_num > 0:
            results["win_by_round"][round_num] = (
                results["win_by_round"].get(round_num, 0) + 1
            )

        if verbose and (episode + 1) % 10 == 0:
            win_rate = results["wins"] / (episode + 1)
            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"Win rate: {win_rate:.1%} | "
                f"Identity: {player_identity} | "
                f"Winner: {winner} | "
                f"Result: {'WIN' if is_win else 'LOSE'}"
            )

    results["win_rate"] = results["wins"] / n_episodes
    results["avg_reward"] = results["total_reward"] / n_episodes
    results["avg_game_length"] = np.mean(results["game_lengths"])
    results["std_reward"] = np.std(results["episode_rewards"])

    print(f"\n{'=' * 60}")
    print("评估结果")
    print(f"{'=' * 60}")
    print(f"总对局: {results['total_episodes']}")
    print(f"胜利: {results['wins']}")
    print(f"失败: {results['losses']}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"平均奖励: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"平均对局长度: {results['avg_game_length']:.1f} 步")

    print(f"\n各身份胜率:")
    identity_stats = []
    for identity in ["主公", "忠臣", "反贼", "内奸"]:
        games = results["identity_games"].get(identity, 0)
        wins = results["identity_wins"].get(identity, 0)
        if games > 0:
            win_rate = wins / games
            print(f"  {identity}: {wins}/{games} ({win_rate:.1%})")
            identity_stats.append(
                {
                    "identity": identity,
                    "games": games,
                    "wins": wins,
                    "win_rate": win_rate,
                }
            )

    results["identity_stats"] = identity_stats

    if save_results:
        results_path = (
            Path(model_path).parent
            / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        serializable_results = {
            k: v
            for k, v in results.items()
            if isinstance(v, (str, int, float, list, dict))
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {results_path}")

    env.close()
    return results


def compare_with_random(
    model_path: str,
    n_episodes: int = 50,
    player_num: int = 5,
):
    """模型 vs 随机策略对比"""

    print(f"\n{'=' * 60}")
    print("模型 vs 随机策略对比")
    print(f"{'=' * 60}")

    model_results = evaluate_model(model_path, n_episodes, player_num, verbose=False)

    print(f"\n模型胜率: {model_results['win_rate']:.2%}")
    print(f"随机策略期望胜率: ~{100 / player_num:.1f}% (5人局)")

    expected_win_rate = 1.0 / player_num
    improvement = (
        (model_results["win_rate"] - expected_win_rate) / expected_win_rate * 100
    )

    if model_results["win_rate"] > expected_win_rate:
        print(f"模型比随机策略高 {improvement:.1f}%")
    else:
        print(f"模型比随机策略低 {-improvement:.1f}%")

    return model_results


def test_specific_scenario(
    model_path: str,
    identity: str = "主公",
    n_episodes: int = 20,
):
    """测试特定身份场景"""

    print(f"\n{'=' * 60}")
    print(f"测试身份: {identity}")
    print(f"{'=' * 60}")

    results = evaluate_model(model_path, n_episodes, verbose=False)

    games = results["identity_games"].get(identity, 0)
    wins = results["identity_wins"].get(identity, 0)

    if games > 0:
        win_rate = wins / games
        print(f"{identity} 身份胜率: {win_rate:.2%} ({wins}/{games})")
    else:
        print(f"没有测试到 {identity} 身份的对局")

    return results


def main():
    parser = argparse.ArgumentParser(description="评估SGS RL模型")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--n-episodes", type=int, default=100, help="评估对局数")
    parser.add_argument("--player-num", type=int, default=5, help="玩家数量")
    parser.add_argument("--max-rounds", type=int, default=100, help="最大回合数")
    parser.add_argument("--render", action="store_true", help="渲染对局")
    parser.add_argument("--compare", action="store_true", help="与随机策略对比")
    parser.add_argument("--identity", type=str, default=None, help="测试特定身份")
    parser.add_argument("--save", action="store_true", help="保存评估结果")

    args = parser.parse_args()

    if args.compare:
        compare_with_random(
            args.model_path,
            args.n_episodes,
            args.player_num,
        )
    elif args.identity:
        test_specific_scenario(
            args.model_path,
            args.identity,
            args.n_episodes,
        )
    else:
        evaluate_model(
            args.model_path,
            args.n_episodes,
            args.player_num,
            args.max_rounds,
            args.render,
            save_results=args.save,
        )


if __name__ == "__main__":
    main()
