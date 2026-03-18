"""
策略池管理 - 管理多个版本的策略模型

功能:
- 保存训练过程中的策略快照
- 采样历史策略作为对手
- 支持策略淘汰和保留
- 支持ELO评分系统
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class PolicyRecord:
    path: str
    version: int
    timestamp: str
    elo_rating: float = 1000.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    parent_version: int = -1

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "version": self.version,
            "timestamp": self.timestamp,
            "elo_rating": self.elo_rating,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PolicyRecord":
        return cls(**data)


class PolicyPool:
    """
    策略池

    管理训练过程中的多个策略版本
    """

    def __init__(
        self,
        pool_dir: str = None,
        max_size: int = 10,
        min_elo: float = 800.0,
        sample_latest_prob: float = 0.3,
        sample_best_prob: float = 0.3,
    ):
        self.pool_dir = Path(pool_dir) if pool_dir else Path("policy_pool")
        self.max_size = max_size
        self.min_elo = min_elo
        self.sample_latest_prob = sample_latest_prob
        self.sample_best_prob = sample_best_prob

        self.policies: List[PolicyRecord] = []
        self.current_version: int = 0

        self._load_pool()

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

    def _save_pool(self):
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.pool_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "current_version": self.current_version,
                    "policies": [p.to_dict() for p in self.policies],
                },
                f,
                indent=2,
            )

    def add_policy(
        self,
        policy_path: str,
        elo_rating: float = 1000.0,
        parent_version: int = -1,
    ) -> PolicyRecord:
        self.current_version += 1

        record = PolicyRecord(
            path=policy_path,
            version=self.current_version,
            timestamp=datetime.now().isoformat(),
            elo_rating=elo_rating,
            parent_version=parent_version,
        )

        self.policies.append(record)

        self._prune_pool()
        self._save_pool()

        return record

    def _prune_pool(self):
        if len(self.policies) <= self.max_size:
            return

        sorted_policies = sorted(
            self.policies, key=lambda p: p.elo_rating, reverse=True
        )

        keep_indices = set()

        keep_indices.add(0)

        latest = max(self.policies, key=lambda p: p.version)
        keep_indices.add(self.policies.index(latest))

        best = max(self.policies, key=lambda p: p.elo_rating)
        keep_indices.add(self.policies.index(best))

        for i in sorted_policies:
            if len(keep_indices) >= self.max_size:
                break
            if i.elo_rating >= self.min_elo:
                keep_indices.add(self.policies.index(i))

        self.policies = [self.policies[i] for i in sorted(keep_indices)]

    def sample_policy(self) -> Optional[PolicyRecord]:
        if not self.policies:
            return None

        r = random.random()

        if r < self.sample_latest_prob:
            return max(self.policies, key=lambda p: p.version)
        elif r < self.sample_latest_prob + self.sample_best_prob:
            return max(self.policies, key=lambda p: p.elo_rating)
        else:
            weights = np.array([p.elo_rating for p in self.policies])
            weights = weights / weights.sum()
            return np.random.choice(self.policies, p=weights)

    def get_policy_by_version(self, version: int) -> Optional[PolicyRecord]:
        for p in self.policies:
            if p.version == version:
                return p
        return None

    def get_best_policy(self) -> Optional[PolicyRecord]:
        if not self.policies:
            return None
        return max(self.policies, key=lambda p: p.elo_rating)

    def get_latest_policy(self) -> Optional[PolicyRecord]:
        if not self.policies:
            return None
        return max(self.policies, key=lambda p: p.version)

    def update_elo(
        self,
        winner_version: int,
        loser_version: int,
        k_factor: float = 32.0,
    ):
        winner = self.get_policy_by_version(winner_version)
        loser = self.get_policy_by_version(loser_version)

        if winner is None or loser is None:
            return

        expected_winner = 1.0 / (
            1.0 + 10 ** ((loser.elo_rating - winner.elo_rating) / 400)
        )
        expected_loser = 1.0 - expected_winner

        winner.elo_rating += k_factor * (1.0 - expected_winner)
        loser.elo_rating += k_factor * (0.0 - expected_loser)

        winner.games_played += 1
        winner.wins += 1
        winner.win_rate = winner.wins / winner.games_played

        loser.games_played += 1
        loser.losses += 1
        loser.win_rate = (
            loser.wins / loser.games_played if loser.games_played > 0 else 0.0
        )

        self._save_pool()

    def record_game(self, policy_version: int, won: bool):
        policy = self.get_policy_by_version(policy_version)
        if policy is None:
            return

        policy.games_played += 1
        if won:
            policy.wins += 1
        else:
            policy.losses += 1
        policy.win_rate = policy.wins / policy.games_played

        self._save_pool()

    def get_stats(self) -> Dict:
        if not self.policies:
            return {
                "total_policies": 0,
                "best_elo": None,
                "avg_elo": None,
                "total_games": 0,
            }

        return {
            "total_policies": len(self.policies),
            "best_elo": max(p.elo_rating for p in self.policies),
            "avg_elo": np.mean([p.elo_rating for p in self.policies]),
            "total_games": sum(p.games_played for p in self.policies),
            "versions": [
                p.version for p in sorted(self.policies, key=lambda p: p.version)
            ],
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
    ):
        match = {
            "policy_a": policy_a_version,
            "policy_b": policy_b_version,
            "winner": winner_version,
            "game_length": game_length,
            "timestamp": datetime.now().isoformat(),
            "final_state": final_state or {},
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

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.matches, f, indent=2)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.matches = json.load(f)


def create_policy_pool(pool_dir: str = "policy_pool", **kwargs) -> PolicyPool:
    return PolicyPool(pool_dir=pool_dir, **kwargs)
