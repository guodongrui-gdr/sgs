"""
强化学习AI - 使用训练好的RL模型进行决策

支持:
- MaskablePPO / PPO 模型加载
- 状态归一化 (VecNormalize)
- 动作掩码
- 与游戏引擎集成
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np

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
                # Create a minimal environment for VecNormalize.load()
                # VecNormalize needs an env to infer observation space
                try:
                    from ai.gym_wrapper import SGSEnv, SGSConfig

                    dummy_env = DummyVecEnv(
                        [
                            lambda: SGSEnv(
                                SGSConfig(
                                    player_num=self.config.player_num,
                                    max_rounds=self.config.max_rounds,
                                )
                            )
                        ]
                    )
                    self.vec_normalize = VecNormalize.load(str(vec_path), dummy_env)
                    logger.info("VecNormalize stats loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to create dummy env for VecNormalize: {e}")
                    # Fallback: load stats directly without env (limited functionality)
                    import pickle

                    with open(vec_path, "rb") as f:
                        stats = pickle.load(f)
                    self.vec_normalize = None  # Can't use without proper env
                    logger.warning(
                        "VecNormalize stats loaded but cannot be applied without env"
                    )
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

        使用与训练相同的3-step hierarchical action selection:
        - Step 0: 选择动作类型 (action_type = model output directly)
        - Step 1: 选择卡牌/技能 (card_idx = model output directly)
        - Step 2: 选择目标 (target_idx = model output directly)

        Args:
                engine: 游戏引擎
                player: 当前玩家

        Returns:
                (card, target) - 要使用的卡牌和目标，如果结束回合则返回 (None, None)
        """
        try:
            # 获取游戏状态以确定当前阶段
            game_state_dict = self._get_game_state_dict(engine)
            phase = game_state_dict.get("phase", "waiting")
            if hasattr(phase, "value"):
                phase = phase.value

            # Step 0: 选择动作类型
            obs = self._encode_observation(engine, player)
            action_masks = self._get_action_masks_for_step(engine, player, step=0)

            if self.use_masking and hasattr(self.model, "predict"):
                action, _ = self.model.predict(
                    obs,
                    action_masks=action_masks,
                    deterministic=self.config.deterministic,
                )
            else:
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.config.deterministic,
                )

            action_type = int(action)  # 直接使用模型输出作为action_type

            # 如果是结束回合或跳过，直接返回
            if action_type == ActionType.END_TURN:
                return None, None
            if action_type == ActionType.PASS:
                return None, None

            # 检查是否需要选择卡牌
            needs_card = self.action_encoder.needs_card(action_type)
            if not needs_card:
                # 不需要卡牌，检查是否需要目标
                needs_target = self.action_encoder.needs_target(action_type)
                if needs_target:
                    # 需要目标，进入Step 2
                    target_masks = self._get_action_masks_for_step(
                        engine, player, step=2, pending_action_type=action_type
                    )
                    target_obs = self._encode_observation_for_step(
                        engine, player, step=2, pending_action_type=action_type
                    )
                    if self.use_masking:
                        target_action, _ = self.model.predict(
                            target_obs,
                            action_masks=target_masks,
                            deterministic=self.config.deterministic,
                        )
                    else:
                        target_action, _ = self.model.predict(
                            target_obs,
                            deterministic=self.config.deterministic,
                        )
                    target_idx = int(target_action)
                    return self._execute_simple_action(
                        action_type, target_idx, player, engine
                    )
                else:
                    # 不需要卡牌也不需要目标，直接执行
                    return self._execute_simple_action(
                        action_type, None, player, engine
                    )

            # Step 1: 选择卡牌/技能
            card_masks = self._get_action_masks_for_step(
                engine, player, step=1, pending_action_type=action_type
            )
            card_obs = self._encode_observation_for_step(
                engine, player, step=1, pending_action_type=action_type
            )

            if self.use_masking:
                card_action, _ = self.model.predict(
                    card_obs,
                    action_masks=card_masks,
                    deterministic=self.config.deterministic,
                )
            else:
                card_action, _ = self.model.predict(
                    card_obs,
                    deterministic=self.config.deterministic,
                )

            card_idx = int(card_action)  # 直接使用模型输出作为card_idx

            # 检查是否需要选择目标
            if action_type == ActionType.USE_SKILL:
                skills = getattr(player, "skills", [])
                skill = skills[card_idx] if card_idx < len(skills) else None
                needs_target = self.action_encoder.needs_target(action_type, skill)
            else:
                hand_cards = player.hand_cards
                card = hand_cards[card_idx] if card_idx < len(hand_cards) else None
                needs_target = self.action_encoder.needs_target(action_type, card)

            if not needs_target:
                # 不需要目标，直接执行
                return self._execute_hierarchical_action(
                    action_type, card_idx, None, player, engine
                )

            # Step 2: 选择目标
            target_masks = self._get_action_masks_for_step(
                engine,
                player,
                step=2,
                pending_action_type=action_type,
                pending_card_idx=card_idx,
            )
            target_obs = self._encode_observation_for_step(
                engine,
                player,
                step=2,
                pending_action_type=action_type,
                pending_card_idx=card_idx,
            )

            if self.use_masking:
                target_action, _ = self.model.predict(
                    target_obs,
                    action_masks=target_masks,
                    deterministic=self.config.deterministic,
                )
            else:
                target_action, _ = self.model.predict(
                    target_obs,
                    deterministic=self.config.deterministic,
                )

            target_idx = int(target_action)  # 直接使用模型输出作为target_idx

            return self._execute_hierarchical_action(
                action_type, card_idx, target_idx, player, engine
            )

        except Exception as e:
            logger.error(f"Error in RL action selection: {e}", exc_info=True)
            return None, None

    def _get_action_masks_for_step(
        self,
        engine: GameEngine,
        player: Player,
        step: int = 0,
        pending_action_type: Optional[int] = None,
        pending_card_idx: Optional[int] = None,
    ) -> np.ndarray:
        """获取指定步骤的动作掩码，格式与训练时的action_masks()一致

        Args:
            engine: 游戏引擎
            player: 当前玩家
            step: 当前步骤 (0=选类型, 1=选卡牌, 2=选目标)
            pending_action_type: 待完成的动作类型
            pending_card_idx: 待完成的卡牌索引

        Returns:
            action_dim大小的掩码数组，掩码内容放在开头位置
        """
        game_state_dict = self._get_game_state_dict(engine)

        # 创建pending_action用于Step 1和Step 2
        pending_action = None
        if pending_action_type is not None:
            pending_action = HierarchicalAction(
                action_type=pending_action_type,
                card_idx=pending_card_idx,
            )

        type_mask, card_mask, target_mask = self.action_mask_generator.generate_masks(
            game_state=game_state_dict,
            player=player,
            engine=engine,
            current_step=step,
            pending_action=pending_action,
        )

        # 创建action_dim大小的掩码，与gym_wrapper.py一致
        action_dim = self.action_encoder.get_action_space_dim()
        result = np.zeros(action_dim, dtype=np.float32)

        if step == 0:
            result[: len(type_mask)] = type_mask
            if type_mask.sum() == 0:
                result[0] = 1.0  # Fallback to END_TURN
        elif step == 1:
            result[: len(card_mask)] = card_mask
            if card_mask.sum() == 0:
                result[0] = 1.0  # Fallback
        elif step == 2:
            result[: len(target_mask)] = target_mask
            if target_mask.sum() == 0:
                result[0] = 1.0  # Fallback

        return result

    def _encode_observation_for_step(
        self,
        engine: GameEngine,
        player: Player,
        step: int = 0,
        pending_action_type: Optional[int] = None,
        pending_card_idx: Optional[int] = None,
    ) -> Dict:
        """编码指定步骤的游戏状态为观察

        Args:
            engine: 游戏引擎
            player: 当前玩家
            step: 当前步骤
            pending_action_type: 待完成的动作类型
            pending_card_idx: 待完成的卡牌索引

        Returns:
            包含state和各层掩码的观察字典
        """
        game_state_dict = self._get_game_state_dict(engine)
        player_idx = player.idx - 1

        state = self.state_encoder.encode(game_state_dict, player_idx)

        pending_action = None
        if pending_action_type is not None:
            pending_action = HierarchicalAction(
                action_type=pending_action_type,
                card_idx=pending_card_idx,
            )

        type_mask, card_mask, target_mask = self.action_mask_generator.generate_masks(
            game_state=game_state_dict,
            player=player,
            engine=engine,
            current_step=step,
            pending_action=pending_action,
        )

        obs = {
            "state": state.astype(np.float32),
            "action_mask_type": type_mask.astype(np.float32),
            "action_mask_card": card_mask.astype(np.float32),
            "action_mask_target": target_mask.astype(np.float32),
            "current_step": step,
        }

        return obs

    def _execute_simple_action(
        self,
        action_type: int,
        target_idx: Optional[int],
        player: Player,
        engine: GameEngine,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """执行不需要卡牌选择的动作"""
        target = None
        if target_idx is not None:
            target = self._get_target_by_idx(target_idx, engine, player)

        # 返回None表示结束回合（对于END_TURN/PASS）
        return None, target

    def _execute_hierarchical_action(
        self,
        action_type: int,
        card_idx: int,
        target_idx: Optional[int],
        player: Player,
        engine: GameEngine,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """执行分层动作（需要卡牌选择）"""
        if action_type == ActionType.USE_CARD:
            if card_idx < 0 or card_idx >= len(player.hand_cards):
                logger.warning(
                    f"Invalid card index: {card_idx}, hand size: {len(player.hand_cards)}"
                )
                return None, None

            card = player.hand_cards[card_idx]
            target = None
            if target_idx is not None:
                target = self._get_target_by_idx(target_idx, engine, player)

            return card, target

        elif action_type == ActionType.USE_SKILL:
            skills = getattr(player, "skills", [])
            if card_idx < 0 or card_idx >= len(skills):
                logger.warning(
                    f"Invalid skill index: {card_idx}, skills count: {len(skills)}"
                )
                return None, None

            # 技能执行需要通过engine，返回None表示技能已触发
            # 实际上返回技能索引让后续处理
            target = None
            if target_idx is not None:
                target = self._get_target_by_idx(target_idx, engine, player)

            # 对于技能，返回特殊标记让调用者知道需要触发技能
            return card_idx, target  # card_idx作为技能索引

        elif action_type == ActionType.DISCARD:
            if card_idx < 0 or card_idx >= len(player.hand_cards):
                logger.warning(f"Invalid discard index: {card_idx}")
                return None, None

            card = player.hand_cards[card_idx]
            return card, None

        else:
            logger.warning(f"Unsupported action type: {action_type}")
            return None, None

    def _get_game_state_dict(self, engine: GameEngine) -> Dict:
        """获取游戏状态字典"""
        state = engine.get_state()
        return state.to_dict()

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
