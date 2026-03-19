"""
强化学习AI - 使用训练好的RL模型进行决策

支持:
- MaskablePPO / PPO 模型加载
- 状态归一化 (VecNormalize)
- 动作掩码
- 与游戏引擎集成
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    VecNormalize = None
    DummyVecEnv = None

try:
    from sb3_contrib import MaskablePPO

    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MASKABLE_PPO_AVAILABLE = False
    MaskablePPO = None

from ai.state_encoder import StateEncoder, EncodingConfig
from ai.action_encoder import (
    ActionEncoder,
    ActionConfig,
    ActionMaskGenerator,
    HierarchicalAction,
    ActionType,
)
from engine.game_engine import GameEngine
from player.player import Player


@dataclass
class RLAIConfig:
    model_path: str
    use_masking: bool = True
    deterministic: bool = True
    vec_normalize_path: Optional[str] = None

    state_config: Optional[EncodingConfig] = None
    action_config: Optional[ActionConfig] = None

    player_num: int = 5
    max_rounds: int = 100


class RLAI:
    """
    基于强化学习的AI

    加载训练好的模型，使用模型进行决策
    """

    def __init__(self, config: RLAIConfig):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for RLAI. "
                "Install with: pip install stable-baselines3"
            )

        self.config = config
        self.model_path = Path(config.model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.state_encoder = StateEncoder(config.state_config or EncodingConfig())
        self.action_encoder = ActionEncoder(config.action_config or ActionConfig())
        self.action_mask_generator = ActionMaskGenerator(self.action_encoder)

        self._load_model()

        logger.info(f"RLAI initialized with model: {self.model_path}")

    def _load_model(self):
        """加载训练好的模型"""
        model_file = str(self.model_path)

        if self.config.use_masking and MASKABLE_PPO_AVAILABLE:
            try:
                self.model = MaskablePPO.load(model_file)
                self.use_masking = True
                logger.info("Loaded MaskablePPO model with action masking")
            except Exception as e:
                logger.warning(f"Failed to load MaskablePPO: {e}, falling back to PPO")
                self.model = PPO.load(model_file)
                self.use_masking = False
        else:
            self.model = PPO.load(model_file)
            self.use_masking = False
            logger.info("Loaded PPO model without action masking")

        if self.config.vec_normalize_path:
            vec_path = Path(self.config.vec_normalize_path)
            if vec_path.exists():
                logger.info(f"Loading VecNormalize stats from {vec_path}")
                dummy_env = DummyVecEnv([lambda: None])
                self.vec_normalize = VecNormalize.load(str(vec_path), dummy_env)
            else:
                logger.warning(f"VecNormalize file not found: {vec_path}")
                self.vec_normalize = None
        else:
            self.vec_normalize = None

    def select_action(
        self,
        engine: GameEngine,
        player: Player,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        选择要执行的动作

        Args:
            engine: 游戏引擎
            player: 当前玩家

        Returns:
            (card, target) - 要使用的卡牌和目标，如果结束回合则返回 (None, None)
        """
        try:
            obs = self._encode_observation(engine, player)

            if self.use_masking and hasattr(self.model, "predict"):
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.config.deterministic,
                )
            else:
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.config.deterministic,
                )

            hierarchical_action = self._decode_action(action)

            return self._execute_action(hierarchical_action, player, engine)

        except Exception as e:
            logger.error(f"Error in RL action selection: {e}", exc_info=True)
            return None, None

    def _encode_observation(self, engine: GameEngine, player: Player) -> Dict:
        """编码游戏状态为观察"""
        game_state = engine.get_state()
        game_state_dict = game_state.to_dict()

        player_idx = player.idx - 1

        state = self.state_encoder.encode(game_state_dict, player_idx)

        type_mask, card_mask, target_mask = self.action_mask_generator.generate_masks(
            game_state=game_state_dict,
            player=player,
            engine=engine,
        )

        obs = {
            "state": state.astype(np.float32),
            "action_mask_type": type_mask.astype(np.float32),
            "action_mask_card": card_mask.astype(np.float32),
            "action_mask_target": target_mask.astype(np.float32),
            "current_step": 0,
        }

        return obs

    def _decode_action(
        self,
        action: int,
    ) -> HierarchicalAction:
        """解码动作为分层动作"""
        return self.action_encoder.decode_flat(action)

    def _execute_action(
        self,
        action: HierarchicalAction,
        player: Player,
        engine: GameEngine,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        将分层动作转换为游戏动作

        Returns:
            (card, target) 或 (None, None) 表示结束回合
        """
        if action.action_type == ActionType.END_TURN:
            return None, None

        if action.action_type == ActionType.USE_CARD:
            card_idx = action.card_idx

            if card_idx is None or card_idx >= len(player.hand_cards):
                logger.warning(f"Invalid card index: {card_idx}")
                return None, None

            card = player.hand_cards[card_idx]

            target = None
            if action.target_idx is not None:
                target = self._get_target_by_idx(action.target_idx, engine, player)

            return card, target

        if action.action_type == ActionType.DISCARD:
            card_idx = action.card_idx

            if card_idx is None or card_idx >= len(player.hand_cards):
                return None, None

            card = player.hand_cards[card_idx]
            return card, None

        logger.warning(f"Unsupported action type: {action.action_type}")
        return None, None

    def _get_target_by_idx(
        self,
        target_idx: int,
        engine: GameEngine,
        player: Player,
    ) -> Optional[Player]:
        """根据索引获取目标玩家"""
        if target_idx < 0 or target_idx >= len(engine.players):
            return None

        target = engine.players[target_idx]

        if target == player or not target.is_alive:
            return None

        return target

    def get_action_name(self, action: HierarchicalAction, player: Player) -> str:
        """获取动作的可读名称"""
        if action.action_type == ActionType.END_TURN:
            return "结束回合"

        if action.action_type == ActionType.USE_CARD:
            card_idx = action.card_idx
            if card_idx is not None and card_idx < len(player.hand_cards):
                card = player.hand_cards[card_idx]
                card_name = card.name
                if action.target_idx is not None:
                    return f"使用 {card_name}"
                return f"使用 {card_name}"
            return "使用卡牌"

        if action.action_type == ActionType.DISCARD:
            return "弃牌"

        return "未知动作"


def create_rl_ai(
    model_path: str,
    player_num: int = 5,
    use_masking: bool = True,
    deterministic: bool = True,
    vec_normalize_path: Optional[str] = None,
) -> RLAI:
    """
    创建RL AI实例

    Args:
        model_path: 模型文件路径
        player_num: 玩家数量
        use_masking: 是否使用动作掩码
        deterministic: 是否使用确定性策略
        vec_normalize_path: VecNormalize统计文件路径

    Returns:
        RLAI实例
    """
    config = RLAIConfig(
        model_path=model_path,
        use_masking=use_masking,
        deterministic=deterministic,
        vec_normalize_path=vec_normalize_path,
        player_num=player_num,
    )

    return RLAI(config)
