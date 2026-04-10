"""
策略池管理 - 管理多个版本的策略模型

功能:
- 保存训练过程中的策略快照
- 采样历史策略作为对手
- 支持策略淘汰和保留
- 支持ELO评分系统
- 支持ELO衰减(旧策略评分降低)
- 支持身份感知胜率统计
- 支持策略版本管理
"""

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np


class SamplingStrategy(Enum):
    """对手采样策略"""

    LATEST = "latest"  # 最新策略
    BEST_ELO = "best_elo"  # 最高ELO评分
    RANDOM = "random"  # 随机选择
    DIVERSITY = "diversity"  # 多样性采样(避免重复对手)


@dataclass
class IdentityStats:
    """身份特定胜率统计"""

    主公_wins: int = 0
    主公_losses: int = 0
    忠臣_wins: int = 0
    忠臣_losses: int = 0
    反贼_wins: int = 0
    反贼_losses: int = 0
    内奸_wins: int = 0
    内奸_losses: int = 0

    def add_result(self, identity: str, won: bool):
        """添加身份相关的胜负结果"""
        if identity == "主公":
            if won:
                self.主公_wins += 1
            else:
                self.主公_losses += 1
        elif identity == "忠臣":
            if won:
                self.忠臣_wins += 1
            else:
                self.忠臣_losses += 1
        elif identity == "反贼":
            if won:
                self.反贼_wins += 1
            else:
                self.反贼_losses += 1
        elif identity == "内奸":
            if won:
                self.内奸_wins += 1
            else:
                self.内奸_losses += 1

    def get_win_rate(self, identity: str) -> float:
        """获取特定身份的胜率"""
        if identity == "主公":
            total = self.主公_wins + self.主公_losses
            return self.主公_wins / total if total > 0 else 0.0
        elif identity == "忠臣":
            total = self.忠臣_wins + self.忠臣_losses
            return self.忠臣_wins / total if total > 0 else 0.0
        elif identity == "反贼":
            total = self.反贼_wins + self.反贼_losses
            return self.反贼_wins / total if total > 0 else 0.0
        elif identity == "内奸":
            total = self.内奸_wins + self.内奸_losses
            return self.内奸_wins / total if total > 0 else 0.0
        return 0.0

    def get_all_win_rates(self) -> Dict[str, float]:
        """获取所有身份的胜率"""
        return {
            "主公": self.get_win_rate("主公"),
            "忠臣": self.get_win_rate("忠臣"),
            "反贼": self.get_win_rate("反贼"),
            "内奸": self.get_win_rate("内奸"),
        }

    def to_dict(self) -> Dict:
        return {
            "主公_wins": self.主公_wins,
            "主公_losses": self.主公_losses,
            "忠臣_wins": self.忠臣_wins,
            "忠臣_losses": self.忠臣_losses,
            "反贼_wins": self.反贼_wins,
            "反贼_losses": self.反贼_losses,
            "内奸_wins": self.内奸_wins,
            "内奸_losses": self.内奸_losses,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "IdentityStats":
        return cls(**data)


@dataclass
class PolicyRecord:
    path: str
    version: int
    timestamp: str
    elo_rating: float = 1000.0
    base_elo: float = 1000.0  # 域始ELO(衰减前的基准)
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    parent_version: int = -1
    identity_stats: IdentityStats = field(default_factory=IdentityStats)
    last_match_time: str = ""
    elo_decay_factor: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.identity_stats is None:
            self.identity_stats = IdentityStats()
        if not isinstance(self.identity_stats, IdentityStats):
            if isinstance(self.identity_stats, dict):
                self.identity_stats = IdentityStats.from_dict(self.identity_stats)
            else:
                self.identity_stats = IdentityStats()

    def get_age_hours(self) -> float:
        """获取策略年龄(小时)"""
        try:
            ts = datetime.fromisoformat(self.timestamp)
            age = datetime.now() - ts
            return age.total_seconds() / 3600
        except:
            return 0.0

    def apply_elo_decay(self, decay_rate: float = 0.01, max_decay: float = 0.3):
        """
        应用ELO衰减

        Args:
                decay_rate: 每小时的衰减率(默认1%)
                max_decay: 最大衰减比例(默认30%)
        """
        age_hours = self.get_age_hours()
        decay = min(decay_rate * age_hours, max_decay)
        self.elo_decay_factor = 1.0 - decay
        # ELO衰减但不低于基准的70%
        self.elo_rating = self.base_elo * self.elo_decay_factor

    def get_effective_elo(self) -> float:
        """获取有效ELO(考虑衰减)"""
        return self.elo_rating

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "version": self.version,
            "timestamp": self.timestamp,
            "elo_rating": self.elo_rating,
            "base_elo": self.base_elo,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "parent_version": self.parent_version,
            "identity_stats": self.identity_stats.to_dict(),
            "last_match_time": self.last_match_time,
            "elo_decay_factor": self.elo_decay_factor,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PolicyRecord":
        # 处理旧版本数据兼容
        if "identity_stats" not in data:
            data["identity_stats"] = IdentityStats()
        if "base_elo" not in data:
            data["base_elo"] = data.get("elo_rating", 1000.0)
        if "last_match_time" not in data:
            data["last_match_time"] = data.get("timestamp", "")
        if "elo_decay_factor" not in data:
            data["elo_decay_factor"] = 1.0
        if "metadata" not in data:
            data["metadata"] = {}

        return cls(**data)


@dataclass
class PoolConfig:
    """策略池配置"""

    max_size: int = 10
    min_elo: float = 800.0
    elo_decay_rate: float = 0.01  # 每小时衰减1%
    elo_max_decay: float = 0.3  # 最大衰减30%
    sample_latest_ratio: float = 0.3  # 30%采样最新策略
    sample_best_ratio: float = 0.3  # 30%采样最高ELO策略
    sample_random_ratio: float = 0.4  # 40%随机采样
    k_factor: float = 32.0  # ELO更新的K因子
    initial_elo: float = 1000.0  # 初始ELO评分
    keep_best_count: int = 2  # 始终保留的最佳策略数量
    keep_latest_count: int = 2  # 始终保留的最新策略数量
    diversity_threshold: int = 3  # 多样性采样时避免重复对手


class PolicyPool:
    """
    策略池

    管理训练过程中的多个策略版本

    采样策略:
    - 30% 最新策略(最近的版本)
    - 30% 最高ELO策略(历史最强)
    - 40% 随机采样(多样性)
    """

    def __init__(
        self,
        pool_dir: str = None,
        config: PoolConfig = None,
        # 保留旧参数接口以兼容
        max_size: int = 10,
        min_elo: float = 800.0,
        sample_latest_prob: float = 0.3,
        sample_best_prob: float = 0.3,
    ):
        self.pool_dir = Path(pool_dir) if pool_dir else Path("policy_pool")

        # 使用新配置或从旧参数创建
        if config is None:
            self.config = PoolConfig(
                max_size=max_size,
                min_elo=min_elo,
                sample_latest_ratio=sample_latest_prob,
                sample_best_ratio=sample_best_prob,
                sample_random_ratio=1.0 - sample_latest_prob - sample_best_prob,
            )
        else:
            self.config = config

        self.policies: List[PolicyRecord] = []
        self.current_version: int = 0
        self._last_sampled_versions: List[int] = []  # 用于多样性采样
        self._total_matches: int = 0
        self._elo_history: List[Dict] = []  # ELO变化历史

        self._load_pool()
        self._apply_all_elo_decay()

    def _load_pool(self):
        if self.pool_dir.exists():
            manifest_path = self.pool_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    data = json.load(f)
                    self.current_version = data.get("current_version", 0)
                    self.policies = [
                        PolicyRecord.from_dict(p) for p in data.get("policies", [])
                    ]
                    self._total_matches = data.get("total_matches", 0)
                    self._elo_history = data.get("elo_history", [])

    def _apply_all_elo_decay(self):
        """对所有策略应用ELO衰减"""
        for policy in self.policies:
            policy.apply_elo_decay(
                self.config.elo_decay_rate, self.config.elo_max_decay
            )

    def _save_pool(self):
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.pool_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "current_version": self.current_version,
                    "policies": [p.to_dict() for p in self.policies],
                    "total_matches": self._total_matches,
                    "elo_history": self._elo_history[-100:],  # 只保存最近100条历史
                    "config": {
                        "max_size": self.config.max_size,
                        "elo_decay_rate": self.config.elo_decay_rate,
                        "elo_max_decay": self.config.elo_max_decay,
                        "sample_latest_ratio": self.config.sample_latest_ratio,
                        "sample_best_ratio": self.config.sample_best_ratio,
                        "sample_random_ratio": self.config.sample_random_ratio,
                    },
                },
                f,
                indent=2,
            )

    def add_policy(
        self,
        policy_path: str,
        elo_rating: float = None,
        parent_version: int = -1,
        metadata: Dict = None,
    ) -> PolicyRecord:
        """
        添加新策略到池中

        Args:
                policy_path: 策略模型路径
                elo_rating: 初始ELO评分(默认使用配置的初始值)
                parent_version: 父策略版本
                metadata: 元数据(训练步数等)

        Returns:
                新创建的策略记录
        """
        self.current_version += 1

        if elo_rating is None:
            elo_rating = self.config.initial_elo

        # 如果有父策略，继承其ELO的80%作为基准
        if parent_version >= 0:
            parent = self.get_policy_by_version(parent_version)
            if parent:
                elo_rating = max(elo_rating, parent.base_elo * 0.8)

        record = PolicyRecord(
            path=policy_path,
            version=self.current_version,
            timestamp=datetime.now().isoformat(),
            elo_rating=elo_rating,
            base_elo=elo_rating,
            parent_version=parent_version,
            metadata=metadata or {},
            last_match_time=datetime.now().isoformat(),
        )

        self.policies.append(record)
        self._prune_pool()
        self._save_pool()

        return record

    def _prune_pool(self):
        """
        修剪策略池，保留最佳和最新策略

        保留策略:
        - 始终保留最新N个策略
        - 始终保留ELO最高的N个策略
        - 按ELO评分保留其他策略
        """
        if len(self.policies) <= self.config.max_size:
            return

        keep_indices = set()

        # 保留最新策略
        sorted_by_version = sorted(self.policies, key=lambda p: p.version, reverse=True)
        for i in range(min(self.config.keep_latest_count, len(sorted_by_version))):
            policy = sorted_by_version[i]
            keep_indices.add(self.policies.index(policy))

        # 保留最高ELO策略
        sorted_by_elo = sorted(
            self.policies, key=lambda p: p.get_effective_elo(), reverse=True
        )
        for i in range(min(self.config.keep_best_count, len(sorted_by_elo))):
            policy = sorted_by_elo[i]
            keep_indices.add(self.policies.index(policy))

        # 按ELO评分添加其他策略
        for policy in sorted_by_elo:
            if len(keep_indices) >= self.config.max_size:
                break
            if policy.elo_rating >= self.config.min_elo:
                keep_indices.add(self.policies.index(policy))

        self.policies = [self.policies[i] for i in sorted(keep_indices)]

    def sample_policy(
        self,
        strategy: SamplingStrategy = None,
        exclude_versions: List[int] = None,
    ) -> Optional[PolicyRecord]:
        """
        采样策略

        采样分布:
        - 30% 最新策略
        - 30% 最高ELO策略
        - 40% 随机采样(多样性)

        Args:
                strategy: 指定采样策略(可选)
                exclude_versions: 要排除的版本列表

        Returns:
                采样的策略记录
        """
        if not self.policies:
            return None

        # 过滤排除的版本
        available_policies = [
            p
            for p in self.policies
            if exclude_versions is None or p.version not in exclude_versions
        ]

        if not available_policies:
            return None

        if strategy is not None:
            return self._sample_with_strategy(available_policies, strategy)

        # 默认混合采样策略
        r = random.random()
        latest_ratio = self.config.sample_latest_ratio
        best_ratio = self.config.sample_best_ratio

        if r < latest_ratio:
            # 采样最新策略(最近的30%版本)
            return self._sample_latest(available_policies)
        elif r < latest_ratio + best_ratio:
            # 采样最高ELO策略
            return self._sample_best_elo(available_policies)
        else:
            # 随机采样(考虑多样性)
            return self._sample_random(available_policies)

    def _sample_with_strategy(
        self, policies: List[PolicyRecord], strategy: SamplingStrategy
    ) -> PolicyRecord:
        """根据指定策略采样"""
        if strategy == SamplingStrategy.LATEST:
            return max(policies, key=lambda p: p.version)
        elif strategy == SamplingStrategy.BEST_ELO:
            return max(policies, key=lambda p: p.get_effective_elo())
        elif strategy == SamplingStrategy.RANDOM:
            return random.choice(policies)
        elif strategy == SamplingStrategy.DIVERSITY:
            return self._sample_diversity(policies)
        return random.choice(policies)

    def _sample_latest(self, policies: List[PolicyRecord]) -> PolicyRecord:
        """采样最新策略(最近30%版本)"""
        # 按版本排序
        sorted_policies = sorted(policies, key=lambda p: p.version, reverse=True)
        # 选择最近30%的策略
        latest_count = max(1, int(len(sorted_policies) * 0.3))
        latest_policies = sorted_policies[:latest_count]
        return random.choice(latest_policies)

    def _sample_best_elo(self, policies: List[PolicyRecord]) -> PolicyRecord:
        """采样最高ELO策略(前30%)"""
        # 按有效ELO排序
        sorted_policies = sorted(
            policies, key=lambda p: p.get_effective_elo(), reverse=True
        )
        # 选择ELO最高的30%策略
        best_count = max(1, int(len(sorted_policies) * 0.3))
        best_policies = sorted_policies[:best_count]
        return random.choice(best_policies)

    def _sample_random(self, policies: List[PolicyRecord]) -> PolicyRecord:
        """随机采样(可考虑多样性)"""
        # 使用加权随机，权重基于ELO
        elo_weights = np.array([p.get_effective_elo() for p in policies])
        elo_weights = elo_weights / elo_weights.sum()
        return np.random.choice(policies, p=elo_weights)

    def _sample_diversity(self, policies: List[PolicyRecord]) -> PolicyRecord:
        """多样性采样(避免最近采样的策略)"""
        # 排除最近采样的策略
        available = [
            p for p in policies if p.version not in self._last_sampled_versions
        ]

        if not available:
            # 如果所有策略都被排除了，清空历史并重新采样
            self._last_sampled_versions = []
            available = policies

        sampled = random.choice(available)

        # 更新采样历史
        self._last_sampled_versions.append(sampled.version)
        if len(self._last_sampled_versions) > self.config.diversity_threshold:
            self._last_sampled_versions = self._last_sampled_versions[
                -self.config.diversity_threshold :
            ]

        return sampled

    def get_policy_by_version(self, version: int) -> Optional[PolicyRecord]:
        for p in self.policies:
            if p.version == version:
                return p
        return None

    def get_best_policy(self) -> Optional[PolicyRecord]:
        """获取当前最高ELO策略"""
        if not self.policies:
            return None
        return max(self.policies, key=lambda p: p.get_effective_elo())

    def get_latest_policy(self) -> Optional[PolicyRecord]:
        """获取最新版本策略"""
        if not self.policies:
            return None
        return max(self.policies, key=lambda p: p.version)

    def get_policies_by_age(self, max_age_hours: float) -> List[PolicyRecord]:
        """获取指定年龄内的策略"""
        return [p for p in self.policies if p.get_age_hours() <= max_age_hours]

    def update_elo(
        self,
        winner_version: int,
        loser_version: int,
        k_factor: float = None,
        winner_identity: str = "",
        loser_identity: str = "",
    ):
        """
        更新ELO评分

        Args:
                winner_version: 获胜策略版本
                loser_version: 失败策略版本
                k_factor: K因子(可选，默认使用配置)
                winner_identity: 获胜者身份
                loser_identity: 失败者身份
        """
        winner = self.get_policy_by_version(winner_version)
        loser = self.get_policy_by_version(loser_version)

        if winner is None or loser is None:
            return

        if k_factor is None:
            k_factor = self.config.k_factor

        # 计算期望胜率
        expected_winner = 1.0 / (
            1.0 + 10 ** ((loser.get_effective_elo() - winner.get_effective_elo()) / 400)
        )
        expected_loser = 1.0 - expected_winner

        # 更新ELO
        elo_change_winner = k_factor * (1.0 - expected_winner)
        elo_change_loser = k_factor * (0.0 - expected_loser)

        winner.elo_rating += elo_change_winner
        winner.base_elo += elo_change_winner * 0.5  # 基准ELO只增加一半
        loser.elo_rating += elo_change_loser

        # 更新比赛统计
        winner.games_played += 1
        winner.wins += 1
        winner.win_rate = winner.wins / winner.games_played
        winner.last_match_time = datetime.now().isoformat()

        loser.games_played += 1
        loser.losses += 1
        loser.win_rate = (
            loser.wins / loser.games_played if loser.games_played > 0 else 0.0
        )
        loser.last_match_time = datetime.now().isoformat()

        # 更新身份统计
        if winner_identity:
            winner.identity_stats.add_result(winner_identity, True)
        if loser_identity:
            loser.identity_stats.add_result(loser_identity, False)

        # 记录ELO变化历史
        self._total_matches += 1
        self._elo_history.append(
            {
                "match_id": self._total_matches,
                "winner_version": winner_version,
                "loser_version": loser_version,
                "winner_elo_change": elo_change_winner,
                "loser_elo_change": elo_change_loser,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self._save_pool()

    def record_game(self, policy_version: int, won: bool, identity: str = ""):
        """
        记录游戏结果(不更新对手ELO)

        Args:
                policy_version: 策略版本
                won: 是否获胜
                identity: 玩家身份
        """
        policy = self.get_policy_by_version(policy_version)
        if policy is None:
            return

        policy.games_played += 1
        if won:
            policy.wins += 1
        else:
            policy.losses += 1
        policy.win_rate = policy.wins / policy.games_played

        if identity:
            policy.identity_stats.add_result(identity, won)

        self._save_pool()

    def get_stats(self) -> Dict:
        """获取策略池统计信息"""
        if not self.policies:
            return {
                "total_policies": 0,
                "best_elo": None,
                "avg_elo": None,
                "total_games": 0,
                "identity_win_rates": {},
            }

        # 计算平均身份胜率
        all_identity_stats = IdentityStats()
        for p in self.policies:
            all_identity_stats.主公_wins += p.identity_stats.主公_wins
            all_identity_stats.主公_losses += p.identity_stats.主公_losses
            all_identity_stats.忠臣_wins += p.identity_stats.忠臣_wins
            all_identity_stats.忠臣_losses += p.identity_stats.忠臣_losses
            all_identity_stats.反贼_wins += p.identity_stats.反贼_wins
            all_identity_stats.反贼_losses += p.identity_stats.反贼_losses
            all_identity_stats.内奸_wins += p.identity_stats.内奸_wins
            all_identity_stats.内奸_losses += p.identity_stats.内奸_losses

        return {
            "total_policies": len(self.policies),
            "best_elo": max(p.get_effective_elo() for p in self.policies),
            "avg_elo": np.mean([p.get_effective_elo() for p in self.policies]),
            "avg_base_elo": np.mean([p.base_elo for p in self.policies]),
            "total_games": sum(p.games_played for p in self.policies),
            "total_matches": self._total_matches,
            "versions": [
                p.version for p in sorted(self.policies, key=lambda p: p.version)
            ],
            "identity_win_rates": all_identity_stats.get_all_win_rates(),
            "oldest_policy_age_hours": max(p.get_age_hours() for p in self.policies),
            "elo_decay_status": {
                p.version: {
                    "base_elo": p.base_elo,
                    "effective_elo": p.get_effective_elo(),
                    "decay_factor": p.elo_decay_factor,
                }
                for p in self.policies
            },
        }

    def get_top_policies(self, n: int = 5) -> List[PolicyRecord]:
        """获取ELO最高的N个策略"""
        sorted_policies = sorted(
            self.policies, key=lambda p: p.get_effective_elo(), reverse=True
        )
        return sorted_policies[:n]

    def get_recent_policies(self, n: int = 5) -> List[PolicyRecord]:
        """获取最近N个策略"""
        sorted_policies = sorted(self.policies, key=lambda p: p.version, reverse=True)
        return sorted_policies[:n]

    def get_sampling_distribution(self) -> Dict[str, float]:
        """获取当前采样分布"""
        return {
            "latest_ratio": self.config.sample_latest_ratio,
            "best_ratio": self.config.sample_best_ratio,
            "random_ratio": self.config.sample_random_ratio,
        }

    def __len__(self) -> int:
        return len(self.policies)

    def __iter__(self):
        return iter(self.policies)


class MatchHistory:
    """对局历史记录"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.matches: List[Dict] = []

    def add_match(
        self,
        policy_a_version: int,
        policy_b_version: int,
        winner_version: int,
        game_length: int,
        final_state: Dict = None,
        policy_a_identity: str = "",
        policy_b_identity: str = "",
    ):
        """
        添加比赛记录

        Args:
                policy_a_version: 策略A版本
                policy_b_version: 策略B版本
                winner_version: 获胜策略版本
                game_length: 游戏回合数
                final_state: 最终状态
                policy_a_identity: 策略A身份
                policy_b_identity: 策略B身份
        """
        match = {
            "policy_a": policy_a_version,
            "policy_b": policy_b_version,
            "winner": winner_version,
            "game_length": game_length,
            "timestamp": datetime.now().isoformat(),
            "final_state": final_state or {},
            "policy_a_identity": policy_a_identity,
            "policy_b_identity": policy_b_identity,
        }

        self.matches.append(match)

        if len(self.matches) > self.max_size:
            self.matches = self.matches[-self.max_size :]

    def get_head_to_head(self, version_a: int, version_b: int) -> Dict:
        wins_a = 0
        wins_b = 0
        total = 0

        for match in self.matches:
            if (match["policy_a"] == version_a and match["policy_b"] == version_b) or (
                match["policy_a"] == version_b and match["policy_b"] == version_a
            ):
                total += 1
                if match["winner"] == version_a:
                    wins_a += 1
                else:
                    wins_b += 1

        return {
            "total_games": total,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "win_rate_a": wins_a / total if total > 0 else 0.0,
        }

    def get_recent_performance(self, version: int, n_games: int = 10) -> Dict:
        recent_matches = [
            m
            for m in reversed(self.matches)
            if m["policy_a"] == version or m["policy_b"] == version
        ][:n_games]

        wins = sum(1 for m in recent_matches if m["winner"] == version)
        total = len(recent_matches)

        return {
            "recent_games": total,
            "recent_wins": wins,
            "recent_win_rate": wins / total if total > 0 else 0.0,
        }

    def get_identity_performance(self, identity: str, n_games: int = 50) -> Dict:
        """获取特定身份的胜率"""
        recent_matches = [
            m
            for m in reversed(self.matches)
            if m.get("policy_a_identity") == identity
            or m.get("policy_b_identity") == identity
        ][:n_games]

        wins = 0
        total = 0
        for match in recent_matches:
            if match.get("policy_a_identity") == identity:
                if match["winner"] == match["policy_a"]:
                    wins += 1
                total += 1
            if match.get("policy_b_identity") == identity:
                if match["winner"] == match["policy_b"]:
                    wins += 1
                total += 1

        return {
            "identity": identity,
            "games": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0.0,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.matches, f, indent=2)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.matches = json.load(f)


def create_policy_pool(pool_dir: str = "policy_pool", **kwargs) -> PolicyPool:
    return PolicyPool(pool_dir=pool_dir, **kwargs)


def create_pool_config(**kwargs) -> PoolConfig:
    """创建策略池配置"""
    return PoolConfig(**kwargs)
