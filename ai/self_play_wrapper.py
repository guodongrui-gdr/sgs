"""
SelfPlayWrapper - 自博弈训练环境包装器

功能:
- 继承自SGSEnv，保持相同接口
- 从AgentPoolManager采样对手
- 循环控制所有玩家(controlled_player_idx)
- 存储对手版本信息
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ai.gym_wrapper import SGSEnv, SGSConfig


class SelfPlayWrapper(SGSEnv):
    """
    自博弈训练环境包装器

    特点:
    - 继承SGSEnv的所有功能
    - 从AgentPoolManager采样对手策略
    - 在reset()时为每个对手位置采样一个对手
    - step()时循环controlled_player_idx到下一个存活玩家
    - 在info中存储对手版本信息
    """

    def __init__(self, config: Optional[SGSConfig] = None):
        """
        初始化SelfPlayWrapper

        Args:
            config: SGSConfig配置，如果为None则使用默认配置
        """
        super().__init__(config)

        # 导入并创建AgentPoolManager
        try:
            from train.agent_pool_manager import AgentPoolManager

            self.agent_pool_manager = AgentPoolManager(pool_size=10)
        except ImportError:
            # 如果AgentPoolManager不存在，创建一个mock对象
            self.agent_pool_manager = self._create_mock_pool_manager()

        # 存储对手信息
        self._opponents: List[Any] = []
        self._opponent_versions: List[int] = []
        self._opponent_observations: Dict[int, Dict] = {}

        # controlled_player_idx在SGSEnv中已定义，初始为0
        # 我们不需要覆盖它，而是使用父类的controlled_player_idx

    def _create_mock_pool_manager(self) -> Any:
        """创建模拟的AgentPoolManager用于测试"""

        class MockAgentPoolManager:
            def __init__(self):
                self._agents = []

            def sample_opponents(self, n: int) -> List[Any]:
                """采样n个对手，返回对手列表"""
                # 返回模拟的对手列表，每个对手有一个version属性
                opponents = []
                for i in range(n):

                    class MockOpponent:
                        def __init__(self, version):
                            self.version = version

                    opponents.append(MockOpponent(version=i + 1))
                return opponents

            def sample_opponent(
                self, exclude_versions: Optional[List[int]] = None
            ) -> Any:
                """采样单个对手"""

                class MockOpponent:
                    def __init__(self):
                        self.version = 1

                return MockOpponent()

        return MockAgentPoolManager()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        重置环境并从AgentPoolManager采样对手

        Args:
            seed: 随机种子
            options: 可选参数

        Returns:
            (obs, info)元组
        """
        # 调用父类的reset
        obs, info = super().reset(seed=seed, options=options)

        # 从AgentPoolManager采样对手 (player_num - 1个对手)
        num_opponents = self.config.player_num - 1
        self._opponents = self.agent_pool_manager.sample_opponents(num_opponents)

        # 提取对手版本信息
        self._opponent_versions = []
        for opponent in self._opponents:
            if hasattr(opponent, "version"):
                self._opponent_versions.append(opponent.version)
            else:
                self._opponent_versions.append(1)  # 默认版本

        # 存储对手版本到info
        info["opponent_versions"] = self._opponent_versions.copy()

        # 重置controlled_player_idx为0 (从第一个玩家开始)
        self.controlled_player_idx = 0

        # 初始化对手观察缓存
        self._opponent_observations = {}

        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行动作并推进controlled_player_idx到下一个存活玩家

        Args:
            action: 动作索引

        Returns:
            (obs, reward, done, truncated, info)元组
        """
        # 调用父类的step
        obs, reward, done, truncated, info = super().step(action)

        # 如果游戏结束，不需要更新controlled_player_idx
        if done or truncated:
            return obs, reward, done, truncated, info

        # 推进controlled_player_idx到下一个存活玩家
        self._advance_controlled_player_idx()

        # 更新info中的对手版本信息
        if hasattr(self, "_opponent_versions"):
            info["opponent_versions"] = self._opponent_versions.copy()

        return obs, reward, done, truncated, info

    def _advance_controlled_player_idx(self):
        """
        将controlled_player_idx推进到下一个存活玩家

        循环顺序: 0 -> 1 -> 2 -> 3 -> 4 -> 0 (对于5个玩家)
        跳过死亡的玩家
        """
        if not self.players:
            return

        num_players = len(self.players)
        current_idx = self.controlled_player_idx

        # 寻找下一个存活玩家
        for i in range(1, num_players + 1):
            next_idx = (current_idx + i) % num_players
            if next_idx < len(self.players) and self.players[next_idx].is_alive:
                self.controlled_player_idx = next_idx
                return

        # 如果没有找到其他存活玩家，保持当前值
        # (可能游戏即将结束)

    def _get_observation(self) -> Dict:
        """
        获取观察，包含对手信息

        Returns:
            观察字典
        """
        obs = super()._get_observation()

        # 存储当前观察用于对手
        self._opponent_observations[self.controlled_player_idx] = obs.copy()

        return obs
