"""
AgentPoolManager - 代理池管理器

包装 ai/policy_pool.py:PolicyPool，为训练系统提供代理池管理功能。
"""

import tempfile
import shutil
from typing import Dict, List, Optional, Iterator
from pathlib import Path

from ai.policy_pool import (
    PolicyPool,
    PolicyRecord,
    PoolConfig,
    IdentityStats,
    MatchHistory,
)


class AgentPoolManager:
    """
    代理池管理器

    包装 PolicyPool，提供简化的接口用于管理 RL 代理。
    支持 ELO 评分、身份统计、对手采样等功能。
    """

    def __init__(self, pool_size: int = 10):
        """
        初始化代理池管理器

        Args:
            pool_size: 池大小限制（默认10）
        """
        self.pool_size = pool_size
        self.config = PoolConfig(max_size=pool_size)
        # 使用临时目录创建 PolicyPool，避免磁盘持久化和测试隔离问题
        self._temp_dir = tempfile.mkdtemp()
        self.policy_pool = PolicyPool(pool_dir=self._temp_dir, config=self.config)
        self._match_history = MatchHistory()

    def add_agent(
        self,
        path: str,
        elo_rating: float = 1000.0,
        parent_version: int = -1,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        添加代理到池中

        Args:
            path: 模型路径
            elo_rating: 初始 ELO 评分
            parent_version: 父版本号（用于追踪进化关系）
            metadata: 元数据字典

        Returns:
            新版本号
        """
        record = self.policy_pool.add_policy(
            policy_path=path,
            elo_rating=elo_rating,
            parent_version=parent_version,
            metadata=metadata or {},
        )
        return record.version

    def remove_agent(self, version: int) -> None:
        """
        从池中移除代理

        Args:
            version: 要移除的版本号

        Raises:
            ValueError: 如果版本不存在
        """
        policy = self.policy_pool.get_policy_by_version(version)
        if policy is None:
            raise ValueError(f"Agent version {version} not found")

        # 从策略列表中移除
        self.policy_pool.policies = [
            p for p in self.policy_pool.policies if p.version != version
        ]
        self.policy_pool._save_pool()

    def sample_opponent(
        self, exclude_versions: Optional[List[int]] = None
    ) -> Optional[PolicyRecord]:
        """
        采样对手

        采样分布：30% 最新，30% 最佳 ELO，40% 随机

        Args:
            exclude_versions: 要排除的版本列表

        Returns:
            采样的策略记录，或 None（如果池为空）
        """
        return self.policy_pool.sample_policy(exclude_versions=exclude_versions)

    def sample_opponents(self, n: int) -> List[PolicyRecord]:
        """
        采样多个对手

        Args:
            n: 要采样的对手数量

        Returns:
            策略记录列表（可能包含重复）
        """
        opponents = []
        for _ in range(n):
            opponent = self.sample_opponent()
            if opponent is not None:
                opponents.append(opponent)
            else:
                # 如果池为空，创建一个默认的模拟对手
                opponents.append(self._create_default_opponent())
        return opponents

    def _create_default_opponent(self) -> PolicyRecord:
        """创建默认对手记录"""
        return PolicyRecord(
            version=0,
            path="",
            timestamp="",
            elo_rating=1000.0,
            wins=0,
            losses=0,
            parent_version=-1,
        )

    def update_elo(self, winner_version: int, loser_version: int) -> None:
        """
        更新 ELO 评分

        赢家获得分数，输家失去分数。根据双方 ELO 差距计算变化量。

        Args:
            winner_version: 获胜代理版本
            loser_version: 失败代理版本
        """
        self.policy_pool.update_elo(winner_version, loser_version)

        # 记录比赛历史
        self._match_history.add_match(
            policy_a_version=winner_version,
            policy_b_version=loser_version,
            winner_version=winner_version,
            game_length=0,  # 简化版本，不追踪回合数
        )

    def record_identity_result(self, version: int, identity: str, won: bool) -> None:
        """
        记录身份相关的比赛结果

        Args:
            version: 代理版本
            identity: 身份（主公/忠臣/反贼/内奸）
            won: 是否获胜
        """
        policy = self.policy_pool.get_policy_by_version(version)
        if policy:
            policy.identity_stats.add_result(identity, won)
            self.policy_pool._save_pool()

    def get_identity_win_rate(self, version: int, identity: str) -> float:
        """
        获取特定身份的胜率

        Args:
            version: 代理版本
            identity: 身份

        Returns:
            胜率（0.0-1.0）
        """
        policy = self.policy_pool.get_policy_by_version(version)
        if policy:
            return policy.identity_stats.get_win_rate(identity)
        return 0.0

    def get_all_identity_win_rates(self, version: int) -> Dict[str, float]:
        """
        获取所有身份的胜率

        Args:
            version: 代理版本

        Returns:
            字典：{身份: 胜率}
        """
        policy = self.policy_pool.get_policy_by_version(version)
        if policy:
            return policy.identity_stats.get_all_win_rates()
        return {}

    def get_agent(self, version: int) -> Optional[PolicyRecord]:
        """
        获取特定版本的代理

        Args:
            version: 版本号

        Returns:
            策略记录，或 None（如果不存在）
        """
        return self.policy_pool.get_policy_by_version(version)

    def get_best_agent(self) -> Optional[PolicyRecord]:
        """
        获取当前最佳代理（最高 ELO）

        Returns:
            最佳策略记录，或 None（如果池为空）
        """
        return self.policy_pool.get_best_policy()

    def get_latest_agent(self) -> Optional[PolicyRecord]:
        """
        获取最新代理（最高版本号）

        Returns:
            最新策略记录，或 None（如果池为空）
        """
        return self.policy_pool.get_latest_policy()

    def get_total_matches(self) -> int:
        """
        获取总比赛数

        Returns:
            比赛总数
        """
        return self.policy_pool._total_matches

    def __len__(self) -> int:
        """返回池中代理数量"""
        return len(self.policy_pool)

    def __iter__(self) -> Iterator[PolicyRecord]:
        """迭代所有代理"""
        return iter(self.policy_pool)
