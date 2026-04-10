"""
统计工具模块 - 提供胜率统计、置信区间计算等功能

功能:
- Wilson score interval 置信区间计算
- Clopper-Pearson 置信区间
- 标准误差计算
- Z-score 和 P-value 计算
- 基线比较
- 身份追踪统计
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from scipy import stats
import numpy as np


# 有效的身份列表
VALID_IDENTITIES = ["主公", "忠臣", "反贼", "内奸"]


def wilson_score_interval(
    wins: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    计算 Wilson score 置信区间

    Wilson score interval 是比正态近似更准确的二项分布置信区间，
    特别是在极端值（接近0或1）和小样本情况下。

    Args:
        wins: 成功次数
        total: 总次数
        confidence: 置信水平（默认0.95）

    Returns:
        (ci_low, ci_high) - 置信区间上下限

    Raises:
        ValueError: 如果 wins > total 或 wins < 0
    """
    if total == 0:
        return (0.0, 0.0)

    if wins < 0:
        raise ValueError(f"wins must be non-negative, got {wins}")

    if wins > total:
        raise ValueError(f"wins ({wins}) cannot exceed total ({total})")

    # z-score for the confidence level
    z = stats.norm.ppf((1 + confidence) / 2)

    # Sample proportion
    p = wins / total

    # Wilson score interval formula
    z2 = z * z
    n = total

    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denominator

    ci_low = max(0.0, center - margin)
    ci_high = min(1.0, center + margin)

    return (ci_low, ci_high)


def clopper_pearson_interval(
    wins: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    计算 Clopper-Pearson (精确) 置信区间

    CP区间是基于二项分布的精确置信区间，比Wilson区间更保守（更宽）。

    Args:
        wins: 成功次数
        total: 总次数
        confidence: 置信水平（默认0.95）

    Returns:
        (ci_low, ci_high) - 置信区间上下限
    """
    if total == 0:
        return (0.0, 0.0)

    if wins < 0 or wins > total:
        return (0.0, 0.0)

    alpha = 1 - confidence

    # Handle edge cases
    if wins == 0:
        ci_low = 0.0
        ci_high = 1 - (alpha / 2) ** (1 / total)
    elif wins == total:
        ci_low = (alpha / 2) ** (1 / total)
        ci_high = 1.0
    else:
        # Use beta distribution for exact binomial CI
        ci_low = stats.beta.ppf(alpha / 2, wins, total - wins + 1)
        ci_high = stats.beta.ppf(1 - alpha / 2, wins + 1, total - wins)

    return (ci_low, ci_high)


def calculate_standard_error(wins: int, total: int) -> float:
    """
    计算标准误差 (Standard Error)

    用于衡量样本比例的估计精度。

    Args:
        wins: 成功次数
        total: 总次数

    Returns:
        标准误差值
    """
    if total == 0:
        return 0.0

    p = wins / total
    return math.sqrt(p * (1 - p) / total)


def calculate_z_score(
    observed_rate: float, baseline_rate: float, sample_size: int
) -> float:
    """
    计算 Z-score (标准分数)

    用于统计检验，衡量观察值与期望值之间的差异。

    Args:
        observed_rate: 观察到的胜率
        baseline_rate: 基线胜率
        sample_size: 样本大小

    Returns:
        Z-score 值
    """
    if sample_size == 0:
        return 0.0

    p = observed_rate
    p0 = baseline_rate
    se = math.sqrt(p0 * (1 - p0) / sample_size)

    if se == 0:
        return 0.0

    return (p - p0) / se


def calculate_p_value(z_score: float, two_tailed: bool = True) -> float:
    """
    计算 P-value

    用于假设检验，表示观察结果出现的概率。

    Args:
        z_score: Z-score 值
        two_tailed: 是否双尾检验（默认True）

    Returns:
        P-value
    """
    if two_tailed:
        return 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        return 1 - stats.norm.cdf(z_score)


def compare_to_baseline(
    wins: int, total: int, baseline_rate: float = 0.5, confidence: float = 0.95
) -> Dict:
    """
    与基线比较

    比较观察到的胜率是否显著优于基线胜率。

    Args:
        wins: 获胜次数
        total: 总次数
        baseline_rate: 基线胜率（默认0.5）
        confidence: 置信水平

    Returns:
        包含比较结果的字典
    """
    if total == 0:
        return {
            "observed_rate": 0.0,
            "baseline_rate": baseline_rate,
            "z_score": 0.0,
            "p_value": 1.0,
            "significant": False,
            "better_than_baseline": False,
            "relative_improvement": 0.0,
        }

    observed_rate = wins / total
    z = calculate_z_score(observed_rate, baseline_rate, total)
    p_value = calculate_p_value(z, two_tailed=True)

    # Significance threshold
    alpha = 1 - confidence
    significant = p_value < alpha

    # Better than baseline?
    better_than_baseline = observed_rate > baseline_rate

    # Relative improvement (percentage)
    if baseline_rate > 0:
        relative_improvement = ((observed_rate - baseline_rate) / baseline_rate) * 100
    else:
        relative_improvement = 0.0

    return {
        "observed_rate": observed_rate,
        "baseline_rate": baseline_rate,
        "z_score": z,
        "p_value": p_value,
        "significant": significant,
        "better_than_baseline": better_than_baseline,
        "relative_improvement": relative_improvement,
    }


def format_confidence_interval(
    rate: float,
    ci_low: float,
    ci_high: float,
    as_percentage: bool = True,
    decimals: int = 1,
) -> str:
    """
    格式化置信区间为字符串

    Args:
        rate: 中心值（胜率）
        ci_low: 置信区间下限
        ci_high: 置信区间上限
        as_percentage: 是否以百分比形式显示
        decimals: 小数位数

    Returns:
        格式化后的字符串
    """
    if as_percentage:
        fmt = f"{{:.{decimals}f}}%"
        return f"{fmt.format(rate * 100)} [{fmt.format(ci_low * 100)}, {fmt.format(ci_high * 100)}]"
    else:
        fmt = f"{{:.{decimals}f}}"
        return f"{fmt.format(rate)} [{fmt.format(ci_low)}, {fmt.format(ci_high)}]"


def required_sample_size(
    expected_rate: float, margin_of_error: float = 0.05, confidence: float = 0.95
) -> int:
    """
    计算所需的样本量

    用于确定达到指定误差范围和置信水平所需的样本大小。

    Args:
        expected_rate: 预期胜率
        margin_of_error: 允许的误差范围（默认0.05）
        confidence: 置信水平（默认0.95）

    Returns:
        所需的样本量
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    p = expected_rate
    e = margin_of_error

    n = (z**2 * p * (1 - p)) / (e**2)
    return math.ceil(n)


def cohen_h(p1: float, p2: float) -> float:
    """
    计算 Cohen's h (效应量)

    Cohen's h 是用于比较两个比例差异的效应量度量。

    Args:
        p1: 第一个比例
        p2: 第二个比例

    Returns:
        Cohen's h 值
    """
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def get_baseline_win_rate(player_num: int, strategy: str = "random") -> float:
    """
    获取基线胜率

    根据不同策略和玩家数量返回预期的基线胜率。

    Args:
        player_num: 玩家数量
        strategy: 策略类型（"random" 或 "rule"）

    Returns:
        基线胜率
    """
    if strategy == "random":
        # Random strategy: 1 / player_num
        return 1.0 / player_num
    elif strategy == "rule":
        # Rule-based heuristic: typically ~35% for 5-player games
        return 0.35
    else:
        # Default: 1 / player_num
        return 1.0 / player_num


@dataclass
class WinRateStats:
    """
    胜率统计类

    用于追踪单个策略或玩家的胜负统计。
    """

    name: str
    wins: int = 0
    total: int = 0

    def add_result(self, won: bool) -> None:
        """
        添加比赛结果

        Args:
            won: 是否获胜
        """
        self.total += 1
        if won:
            self.wins += 1

    @property
    def win_rate(self) -> float:
        """计算胜率"""
        if self.total == 0:
            return 0.0
        return self.wins / self.total

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        获取置信区间

        Args:
            confidence: 置信水平

        Returns:
            (ci_low, ci_high)
        """
        return wilson_score_interval(self.wins, self.total, confidence)

    def compare_to(self, baseline_rate: float, confidence: float = 0.95) -> Dict:
        """
        与基线比较

        Args:
            baseline_rate: 基线胜率
            confidence: 置信水平

        Returns:
            比较结果字典
        """
        return compare_to_baseline(self.wins, self.total, baseline_rate, confidence)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,
            "wins": self.wins,
            "total": self.total,
            "win_rate": self.win_rate,
        }


@dataclass
class IdentityTracker:
    """
    身份追踪器

    追踪不同身份（主公、忠臣、反贼、内奸）的胜负统计。
    """

    _stats: Dict[str, WinRateStats] = field(default_factory=dict)

    def __post_init__(self):
        """初始化时创建所有身份的统计对象"""
        for identity in VALID_IDENTITIES:
            if identity not in self._stats:
                self._stats[identity] = WinRateStats(name=identity)

    def add_result(self, identity: str, won: bool) -> None:
        """
        添加身份相关的比赛结果

        Args:
            identity: 身份（主公/忠臣/反贼/内奸）
            won: 是否获胜
        """
        if identity not in VALID_IDENTITIES:
            return  # Ignore invalid identities

        if identity not in self._stats:
            self._stats[identity] = WinRateStats(name=identity)

        self._stats[identity].add_result(won)

    def get_stats(self, identity: str) -> WinRateStats:
        """
        获取特定身份的统计

        Args:
            identity: 身份

        Returns:
            WinRateStats 对象
        """
        if identity not in self._stats:
            self._stats[identity] = WinRateStats(name=identity)
        return self._stats[identity]

    def get_all_stats(self) -> Dict[str, WinRateStats]:
        """
        获取所有身份的统计

        Returns:
            字典：{身份: WinRateStats}
        """
        return {
            identity: self._stats.get(identity, WinRateStats(name=identity))
            for identity in VALID_IDENTITIES
            if identity in self._stats and self._stats[identity].total > 0
        }

    def get_formatted_report(self) -> str:
        """
        获取格式化的报告

        Returns:
            格式化的字符串报告
        """
        lines = ["Identity Win Rates (身份胜率):"]
        for identity in VALID_IDENTITIES:
            stats = self._stats.get(identity)
            if stats and stats.total > 0:
                ci_low, ci_high = stats.get_confidence_interval()
                ci_str = format_confidence_interval(
                    stats.win_rate, ci_low, ci_high, as_percentage=True
                )
                lines.append(f"  {identity}: {ci_str} (n={stats.total})")
        return "\n".join(lines)
