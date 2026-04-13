"""
模型评估模块 - 支持RL模型与对手池对战评估

功能:
- 运行N场游戏与对手池对战
- 追踪身份胜率（主公、忠臣、反贼、内奸）
- 计算整体胜率和ELO评分
- 生成JSON格式评估报告
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from train.statistics import (
    WinRateStats,
    IdentityTracker,
    wilson_score_interval,
    compare_to_baseline,
    get_baseline_win_rate,
)
from ai.rl_ai import RLAI, RLAIConfig
from ai.rule_ai import RuleAI, RuleAIConfig
from engine.game_engine import GameEngine
from player.player import Player
from skills.registry import SkillRegistry
from config import COMMANDERS_CONFIG

logger = logging.getLogger(__name__)


def load_commanders() -> Dict:
    """加载武将配置"""
    with open(COMMANDERS_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def select_commanders(player_num: int) -> List[str]:
    """随机选择武将"""
    commander_configs = load_commanders()
    available_ids = list(commander_configs.keys())
    return random.sample(available_ids, player_num)


def create_player(commander_id: str, commander_config: dict) -> Player:
    """创建玩家"""
    skill_names = commander_config.get("skills", [])
    skills = [
        SkillRegistry.get(name) for name in skill_names if SkillRegistry.get(name)
    ]

    return Player(
        commander_id=commander_id,
        commander_name=commander_config["name"],
        nation=commander_config["nation"],
        max_hp=commander_config["max_hp"],
        current_hp=commander_config["max_hp"],
        skills=skills,
        is_human=False,
    )


@dataclass
class EvaluationConfig:
    """评估配置"""

    num_games: int = 100  # 评估游戏数量
    player_num: int = 5  # 每局玩家数量
    max_rounds: int = 100  # 每局最大回合数
    use_masking: bool = True  # 是否使用动作掩码
    deterministic: bool = True  # 是否确定性策略
    verbose: bool = False  # 是否输出详细日志
    seed: Optional[int] = None  # 随机种子
    opponent_types: List[str] = field(default_factory=lambda: ["rule", "random"])
    # 对手类型: "rule"=规则AI, "random"=随机AI, "pool"=从池采样


@dataclass
class EvaluationResult:
    """评估结果"""

    model_path: str
    total_games: int
    wins: int
    win_rate: float
    identity_win_rates: Dict[str, float]
    elo_rating: float
    confidence_interval: Tuple[float, float]
    comparison_to_baseline: Dict
    game_details: List[Dict]
    metadata: Dict

    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化）"""
        return {
            "model_path": self.model_path,
            "total_games": self.total_games,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "identity_win_rates": self.identity_win_rates,
            "elo_rating": self.elo_rating,
            "confidence_interval": {
                "low": self.confidence_interval[0],
                "high": self.confidence_interval[1],
            },
            "comparison_to_baseline": self.comparison_to_baseline,
            "game_details": self.game_details,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> None:
        """保存到文件"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def evaluate_model(
    model_path: str,
    num_games: int = 100,
    player_num: int = 5,
    opponent_pool: Optional[List[Any]] = None,
    config: Optional[EvaluationConfig] = None,
    vec_normalize_path: Optional[str] = None,
) -> EvaluationResult:
    """
    评估模型

    运行N场游戏，评估模型相对于对手池的表现。

    Args:
        model_path: 模型文件路径
        num_games: 游戏数量
        player_num: 每局玩家数
        opponent_pool: 对手池（可选，默认使用规则AI）
        config: 评估配置（可选）
        vec_normalize_path: VecNormalize统计文件路径（可选）

    Returns:
        EvaluationResult 评估结果
    """
    if config is None:
        config = EvaluationConfig(num_games=num_games, player_num=player_num)

    # 设置随机种子
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    logger.info(f"Starting evaluation: {num_games} games, {player_num} players")

    # 初始化统计
    overall_stats = WinRateStats(name="Overall")
    identity_tracker = IdentityTracker()
    game_details = []

    # 加载被评估模型
    eval_ai = RLAI(
        RLAIConfig(
            model_path=model_path,
            use_masking=config.use_masking,
            deterministic=config.deterministic,
            vec_normalize_path=vec_normalize_path,
            player_num=player_num,
            max_rounds=config.max_rounds,
        )
    )

    # 运行游戏
    for game_idx in range(num_games):
        game_result = run_single_evaluation_game(
            eval_ai=eval_ai,
            player_num=player_num,
            max_rounds=config.max_rounds,
            opponent_pool=opponent_pool,
            game_idx=game_idx,
        )

        # 更新统计
        won = game_result["won"]
        identity = game_result["identity"]

        overall_stats.add_result(won)
        identity_tracker.add_result(identity, won)

        game_details.append(game_result)

        if config.verbose:
            logger.info(
                f"Game {game_idx + 1}/{num_games}: "
                f"{'Win' if won else 'Loss'} as {identity}"
            )

    # 计算结果
    win_rate = overall_stats.win_rate
    ci_low, ci_high = overall_stats.get_confidence_interval()

    # 计算身份胜率
    identity_win_rates = {}
    for identity in ["主公", "忠臣", "反贼", "内奸"]:
        stats = identity_tracker.get_stats(identity)
        if stats.total > 0:
            identity_win_rates[identity] = stats.win_rate

    # 与基线比较
    baseline_rate = get_baseline_win_rate(player_num, strategy="rule")
    comparison = compare_to_baseline(
        overall_stats.wins, overall_stats.total, baseline_rate=baseline_rate
    )

    # 计算ELO（基于胜率）
    elo_rating = calculate_elo_from_win_rate(win_rate, baseline_rate)

    result = EvaluationResult(
        model_path=model_path,
        total_games=overall_stats.total,
        wins=overall_stats.wins,
        win_rate=win_rate,
        identity_win_rates=identity_win_rates,
        elo_rating=elo_rating,
        confidence_interval=(ci_low, ci_high),
        comparison_to_baseline=comparison,
        game_details=game_details,
        metadata={
            "player_num": player_num,
            "max_rounds": config.max_rounds,
            "deterministic": config.deterministic,
            "seed": config.seed,
        },
    )

    logger.info(f"Evaluation complete: {win_rate:.2%} win rate")
    return result


def run_single_evaluation_game(
    eval_ai: RLAI,
    player_num: int,
    max_rounds: int,
    opponent_pool: Optional[List[Any]],
    game_idx: int,
) -> Dict:
    """
    运行单局评估游戏

    Args:
        eval_ai: 被评估的AI
        player_num: 玩家数量
        max_rounds: 最大回合数
        opponent_pool: 对手池
        game_idx: 游戏索引

    Returns:
        游戏结果字典
    """
    # 创建游戏引擎
    commander_ids = select_commanders(player_num)
    engine = GameEngine(player_num=player_num, commander_ids=commander_ids)

    # 创建玩家
    commander_configs = load_commanders()
    players = []
    for i, cid in enumerate(commander_ids):
        player = create_player(cid, commander_configs[cid])
        players.append(player)

    engine.setup_game(players)

    # 身份已在setup_game中分配，获取被评估玩家的身份
    identities = [p.identity for p in engine.players]

    # 确定被评估AI的位置（随机）
    eval_player_idx = random.randint(0, player_num - 1)
    eval_identity = identities[eval_player_idx]

    # 创建对手AI
    opponent_ais = {}
    for i in range(player_num):
        if i != eval_player_idx:
            if opponent_pool:
                # 从池采样对手
                opponent = random.choice(opponent_pool)
                if hasattr(opponent, "path") and opponent.path:
                    # 是PolicyRecord，创建RLAI
                    try:
                        opponent_ais[i] = RLAI(
                            RLAIConfig(
                                model_path=opponent.path,
                                use_masking=True,
                                deterministic=False,
                                player_num=player_num,
                            )
                        )
                    except:
                        # 失败则回退到规则AI
                        opponent_ais[i] = RuleAI(RuleAIConfig())
                else:
                    # 默认使用规则AI
                    opponent_ais[i] = RuleAI(RuleAIConfig())
            else:
                # 使用规则AI作为对手
                opponent_ais[i] = RuleAI(RuleAIConfig())

    # 运行游戏
    winner_idx = run_game_loop(
        engine=engine,
        eval_ai=eval_ai,
        eval_player_idx=eval_player_idx,
        opponent_ais=opponent_ais,
        max_rounds=max_rounds,
    )

    # 判断胜负
    won = winner_idx == eval_player_idx

    return {
        "game_idx": game_idx,
        "won": won,
        "identity": eval_identity,
        "eval_player_idx": eval_player_idx,
        "winner_idx": winner_idx,
        "rounds_played": engine.round_count if hasattr(engine, "round_count") else 0,
    }


def run_game_loop(
    engine: GameEngine,
    eval_ai: RLAI,
    eval_player_idx: int,
    opponent_ais: Dict[int, Any],
    max_rounds: int,
) -> Optional[int]:
    """
    运行游戏主循环

    Args:
        engine: 游戏引擎
        eval_ai: 被评估AI
        eval_player_idx: 被评估玩家的索引
        opponent_ais: 对手AI字典
        max_rounds: 最大回合数

    Returns:
        获胜者索引，None表示平局或无胜者
    """
    try:
        for _ in range(max_rounds * engine.player_num * 10):
            current_player = engine.players[engine.current_player_idx]
            if not current_player.is_alive:
                engine.next_turn()
                continue

            player_idx = engine.current_player_idx

            if player_idx == eval_player_idx:
                ai = eval_ai
            else:
                ai = opponent_ais.get(player_idx)
                if ai is None:
                    engine.next_turn()
                    continue

            try:
                card, target = ai.select_action(engine, current_player)
                if card is None:
                    engine.next_turn()
                else:
                    engine.use_card(current_player, card, target)
            except Exception as e:
                logger.debug(f"Action error: {e}")
                engine.next_turn()

            if engine._winner:
                winner = engine._winner
                return (
                    engine.players.index(winner) if winner in engine.players else None
                )

        alive_players = [p for p in engine.players if p.is_alive]
        if len(alive_players) == 1:
            return engine.players.index(alive_players[0])

        return None

    except Exception as e:
        logger.error(f"Game loop error: {e}")
        return None


def assign_random_identities(player_num: int) -> List[str]:
    """
    随机分配身份

    Args:
        player_num: 玩家数量

    Returns:
        身份列表
    """
    if player_num == 5:
        identities = ["主公", "忠臣", "反贼", "反贼", "内奸"]
    elif player_num == 6:
        identities = ["主公", "忠臣", "反贼", "反贼", "反贼", "内奸"]
    elif player_num == 8:
        identities = ["主公", "忠臣", "忠臣", "反贼", "反贼", "反贼", "反贼", "内奸"]
    else:
        # 默认5人配置
        identities = ["主公", "忠臣", "反贼", "反贼", "内奸"]

    random.shuffle(identities)
    return identities[:player_num]


def calculate_elo_from_win_rate(win_rate: float, baseline_rate: float = 0.2) -> float:
    """
    从胜率计算ELO评分

    基于期望胜率公式反推ELO。

    Args:
        win_rate: 观察到的胜率
        baseline_rate: 基线胜率（用于归一化）

    Returns:
        ELO评分
    """
    if win_rate <= 0:
        return 800.0
    if win_rate >= 1:
        return 2400.0

    # ELO期望公式: E = 1 / (1 + 10^((Rb-Ra)/400))
    # 反推: Ra = Rb - 400 * log10((1/E) - 1)
    # 假设基线对手ELO = 1000
    baseline_elo = 1000.0

    try:
        elo = baseline_elo - 400 * math.log10((1 / win_rate) - 1)
    except (ValueError, ZeroDivisionError):
        elo = 1000.0

    # 限制范围
    return max(400.0, min(2800.0, elo))


def run_evaluation_games(
    model_path: str,
    num_games: int,
    player_num: int = 5,
    opponent_pool: Optional[List[Any]] = None,
    config: Optional[EvaluationConfig] = None,
    vec_normalize_path: Optional[str] = None,
) -> EvaluationResult:
    """
    运行评估游戏（便捷函数）

    与 evaluate_model 相同，提供更直观的命名。

    Args:
        model_path: 模型路径
        num_games: 游戏数量
        player_num: 玩家数量
        opponent_pool: 对手池
        config: 评估配置
        vec_normalize_path: VecNormalize路径

    Returns:
        EvaluationResult
    """
    return evaluate_model(
        model_path=model_path,
        num_games=num_games,
        player_num=player_num,
        opponent_pool=opponent_pool,
        config=config,
        vec_normalize_path=vec_normalize_path,
    )


# 导入math模块用于ELO计算
import math
