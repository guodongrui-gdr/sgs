"""
Gym环境包装器 - 将三国杀游戏包装为Gymnasium环境

支持:
- 单玩家训练视角
- 分层动作空间
- 动作掩码
- 身份感知奖励
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
    GYM_VERSION = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces

        GYM_AVAILABLE = True
        GYM_VERSION = "gym"
    except ImportError:
        GYM_AVAILABLE = False
        GYM_VERSION = None
        gym = None
        spaces = None

from ai.state_encoder import StateEncoder, EncodingConfig
from ai.action_encoder import (
    ActionEncoder,
    ActionConfig,
    ActionMaskGenerator,
    HierarchicalAction,
    ActionType,
)
from ai.reward import RewardSystem, RewardConfig
from engine.game_engine import GameEngine
from engine.state import GamePhase
from player.player import Player
from skills.registry import SkillRegistry


class SGSPhase(IntEnum):
    SELECT_ACTION_TYPE = 0
    SELECT_CARD = 1
    SELECT_TARGET = 2


@dataclass
class SGSConfig:
    player_num: int = 5
    max_rounds: int = 15
    use_action_mask: bool = True
    use_shaping: bool = True
    other_player_policy: str = "rule"

    state_config: Optional[EncodingConfig] = None
    action_config: Optional[ActionConfig] = None
    reward_config: Optional[RewardConfig] = None

    def __post_init__(self):
        if self.state_config is None:
            self.state_config = EncodingConfig()
        if self.action_config is None:
            self.action_config = ActionConfig()
        if self.reward_config is None:
            self.reward_config = RewardConfig()


if GYM_AVAILABLE:
    _BaseEnv = gym.Env
else:
    _BaseEnv = object


class SGSEnv(_BaseEnv):
    """
    三国杀Gym环境

    观察空间: Dict
        - state: Box(state_dim,) - 编码后的游戏状态
        - action_mask_type: Box(num_action_types,) - 动作类型掩码
        - action_mask_card: Box(max_hand_size,) - 卡牌掩码
        - action_mask_target: Box(max_players,) - 目标掩码
        - current_step: Discrete(3) - 当前动作步骤

    动作空间: Discrete(max_dim)
        - 分三步选择: 动作类型 -> 卡牌/技能 -> 目标
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, config: Optional[SGSConfig] = None):
        if not GYM_AVAILABLE:
            raise ImportError(
                "gymnasium or gym is required to use SGSEnv. "
                "Install with: pip install gymnasium"
            )

        super().__init__()

        self.config = config or SGSConfig()

        self.state_encoder = StateEncoder(self.config.state_config)
        self.action_encoder = ActionEncoder(self.config.action_config)
        self.action_mask_generator = ActionMaskGenerator(self.action_encoder)
        self.reward_system = RewardSystem(
            self.config.reward_config, self.config.use_shaping
        )

        self._setup_spaces()

        self.engine: Optional[GameEngine] = None
        self.players: List[Player] = []
        self.current_player_idx: int = 0
        self.current_step: int = 0
        self.pending_action: Optional[HierarchicalAction] = None
        self.round_count: int = 0
        self.action_history: List[Dict] = []
        self._total_steps: int = 0
        self._turn_steps: int = 0
        self._max_turn_steps: int = 100
        self._action_failures: int = 0
        self._max_action_failures: int = 10

        self._winner: Optional[str] = None

        self._prev_hp: Dict[int, int] = {}
        self._prev_alive: Dict[int, bool] = {}
        self._pending_rewards: float = 0.0

        self._rule_ai = None
        if self.config.other_player_policy == "rule":
            from ai.rule_ai import RuleAI, RuleAIConfig

            self._rule_ai = RuleAI(RuleAIConfig())

        self.controlled_player_idx: int = 0

        from ai.skill_decision import SkillDecisionContext

        self.skill_decision_context = SkillDecisionContext()
        self._skill_decision_step: int = 0

    def _setup_spaces(self):
        state_dim = self.state_encoder.get_state_dim(self.config.player_num)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                ),
                "action_mask_type": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.action_encoder.action_type_dim,),
                    dtype=np.float32,
                ),
                "action_mask_card": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.action_encoder.card_dim,),
                    dtype=np.float32,
                ),
                "action_mask_target": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.action_encoder.target_dim,),
                    dtype=np.float32,
                ),
                "current_step": spaces.Discrete(4),
                "skill_decision_type": spaces.Discrete(8),
                "skill_decision_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.action_encoder.card_dim,),
                    dtype=np.float32,
                ),
            }
        )

        action_dim = self.action_encoder.get_action_space_dim()
        self.action_space = spaces.Discrete(action_dim)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        self._setup_game()

        self.current_step = 0
        self.pending_action = None
        self.round_count = 0
        self.action_history = []
        self._winner = None
        self._pending_rewards = 0.0
        self._skill_decision_step = 0

        self.skill_decision_context.clear()

        self.reward_system.reset()

        self._save_player_states()

        self._run_until_player_turn()

        # 更新受控玩家索引为当前玩家
        self.controlled_player_idx = self.current_player_idx

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _save_player_states(self):
        """保存玩家状态用于奖励计算"""
        self._prev_hp = {}
        self._prev_alive = {}
        for i, p in enumerate(self.players):
            self._prev_hp[i] = p.current_hp
            self._prev_alive[i] = p.is_alive

    def _run_until_player_turn(self):
        """运行游戏直到受控玩家可以行动"""
        max_iterations = 200
        start_time = time.time()
        for iteration in range(max_iterations):
            if iteration > 0 and iteration % 50 == 0:
                elapsed = time.time() - start_time
                logger.warning(
                    f"_run_until_player_turn iteration {iteration}, elapsed={elapsed:.1f}s, phase={self.engine.phase}, player_idx={self.current_player_idx}"
                )

            logger.debug(f"_run_until_player_turn iteration {iteration}")
            if self.engine.phase == GamePhase.GAME_OVER:
                break

            current = self.players[self.current_player_idx]

            # 检查受控玩家是否存活，如果已死则更新
            controlled = (
                self.players[self.controlled_player_idx]
                if self.controlled_player_idx < len(self.players)
                else None
            )
            if controlled is None or not controlled.is_alive:
                # 受控玩家已死，更新为当前玩家
                self.controlled_player_idx = self.current_player_idx

            if not current.is_alive:
                self.engine.next_turn()
                self.current_player_idx = self.engine.current_player_idx
                continue

            self.engine.phase = GamePhase.JUDGE_PHASE
            judge_result = self.engine.judge_phase(current)

            if judge_result.get("lightning_damage", 0) > 0:
                logger.debug(
                    f"lightning_damage: {judge_result['lightning_damage']} to player {current.idx}"
                )
                self.engine.deal_damage(
                    None,
                    current,
                    None,
                    judge_result["lightning_damage"],
                    True,
                    False,
                    True,
                )

            if not current.is_alive:
                self.engine.next_turn()
                self.current_player_idx = self.engine.current_player_idx
                continue

            if not judge_result.get("skip_draw", False):
                drawn = self.engine.draw_cards(current, 2)
                current.hand_cards.extend(drawn)

            if judge_result.get("skip_play", False):
                self.engine.next_turn()
                self.current_player_idx = self.engine.current_player_idx
                self.round_count = self.engine.round_num
                continue

            if (
                self._rule_ai is not None
                and self.current_player_idx != self.controlled_player_idx
            ):
                self.engine.phase = GamePhase.PLAY_PHASE
                self._execute_ai_turn(current)
                self._do_end_turn(current)
                self.engine.next_turn()
                self.current_player_idx = self.engine.current_player_idx
                self.round_count = self.engine.round_num
                continue
            else:
                self.engine.phase = GamePhase.PLAY_PHASE
                break

        if iteration >= max_iterations - 1:
            logger.warning(
                f"_run_until_player_turn reached max_iterations={max_iterations}"
            )
            self.engine.phase = GamePhase.GAME_OVER

    def _execute_ai_turn(self, player: Player, max_actions: int = 20) -> None:
        """使用 AI 执行一个玩家的回合"""
        if self._rule_ai is None:
            self._do_end_turn(player)
            return

        game_state = self._get_game_state_dict()
        actions_taken = 0

        while actions_taken < max_actions:
            if self.engine.phase == GamePhase.GAME_OVER:
                break

            type_mask, card_mask, target_mask = (
                self.action_mask_generator.generate_masks(
                    game_state, player, self.engine, 0, None
                )
            )

            valid_types = [i for i, m in enumerate(type_mask) if m > 0]
            if not valid_types or ActionType.END_TURN in valid_types:
                self._do_end_turn(player)
                break

            action_type = self._rule_ai.select_action(
                player, game_state, (type_mask, card_mask, target_mask)
            )

            if action_type == ActionType.END_TURN:
                self._do_end_turn(player)
                break

            if action_type == ActionType.USE_CARD:
                card_idx = self._rule_ai.select_card(player, game_state, card_mask)
                if card_idx < 0:
                    self._do_end_turn(player)
                    break

                card = (
                    player.hand_cards[card_idx]
                    if card_idx < len(player.hand_cards)
                    else None
                )
                needs_target = self.action_encoder.needs_target(action_type, card)

                if needs_target:
                    _, _, new_target_mask = self.action_mask_generator.generate_masks(
                        game_state,
                        player,
                        self.engine,
                        1,
                        HierarchicalAction(action_type=action_type, card_idx=card_idx),
                    )
                    target_idx = self._rule_ai.select_target(
                        player, game_state, new_target_mask.tolist()
                    )
                    self._execute_card_action(player, card_idx, target_idx)
                else:
                    self._execute_card_action(player, card_idx, None)

                actions_taken += 1

            elif action_type == ActionType.USE_SKILL:
                skills = getattr(player, "skills", [])
                if not skills:
                    self._do_end_turn(player)
                    break

                # 重新生成 current_step=1 的技能掩码
                skill_mask, _, _ = self.action_mask_generator.generate_masks(
                    game_state,
                    player,
                    self.engine,
                    1,
                    HierarchicalAction(action_type=action_type),
                )

                # 找到第一个可用的技能
                skill_idx = -1
                for i, skill in enumerate(skills):
                    if i < len(skill_mask) and skill_mask[i] > 0:
                        skill_idx = i
                        break

                if skill_idx < 0:
                    # 没有可用技能，结束回合
                    self._do_end_turn(player)
                    break

                _, _, new_target_mask = self.action_mask_generator.generate_masks(
                    game_state,
                    player,
                    self.engine,
                    1,
                    HierarchicalAction(action_type=action_type, card_idx=skill_idx),
                )
                target_idx = self._rule_ai.select_target(
                    player, game_state, new_target_mask.tolist()
                )
                self._execute_skill_action(player, skill_idx, target_idx)
                actions_taken += 1

            else:
                self._do_end_turn(player)
                break

            game_state = self._get_game_state_dict()

    def _do_end_turn(self, player: Player) -> None:
        """执行回合结束（弃牌）"""
        hand_limit = max(0, getattr(player, "hand_limit", player.current_hp))
        while len(player.hand_cards) > hand_limit:
            if player.hand_cards:
                card = player.hand_cards.pop()
                self.engine.discard_pile.append(card)

    def _execute_card_action(
        self, player: Player, card_idx: int, target_idx: int
    ) -> bool:
        """执行卡牌动作"""
        if card_idx >= len(player.hand_cards):
            return False

        card = player.hand_cards[card_idx]
        target = None
        if target_idx is not None and 0 <= target_idx < len(self.players):
            target = self.players[target_idx]

        try:
            result = self.engine.use_card(player, card, target)
            return result
        except Exception as e:
            logger.debug(f"AI card action failed: {e}")
            return False

    def _execute_skill_action(
        self, player: Player, skill_idx: int, target_idx: int
    ) -> bool:
        """执行技能动作"""
        from skills.base import ActiveSkill
        from engine.event import Event, EventType

        skills = getattr(player, "skills", [])
        if skill_idx >= len(skills):
            return False

        skill = skills[skill_idx]
        if not isinstance(skill, ActiveSkill):
            logger.debug(f"Cannot actively use passive skill: {skill.name}")
            return False

        target = None
        if target_idx is not None and 0 <= target_idx < len(self.players):
            target = self.players[target_idx]

        try:
            event = Event(
                type=EventType.SKILL_TRIGGERED,
                source=player,
                target=target,
                engine=self.engine,
            )
            if skill.can_activate(event, self.engine):
                result = skill.execute(event, self.engine)
                return result is not None
            else:
                logger.debug(f"Skill {skill.name} cannot be activated")
                return False
        except Exception as e:
            logger.debug(f"AI skill action failed: {e}")
            return False

    def _setup_game(self):
        from card.factory import CardFactory
        from pathlib import Path

        self.engine = GameEngine(
            player_num=self.config.player_num,
            commander_ids=[],
            human_player_idx=-1,
        )

        self._create_players()

        self.engine.setup_game(self.players)

        self.current_player_idx = self.engine.current_player_idx

    def _create_players(self):
        self.players = []

        import json

        config_path = Path(__file__).parent.parent / "data" / "commanders.json"
        with open(config_path, encoding="utf-8") as f:
            commander_configs = json.load(f)

        commander_ids = list(commander_configs.keys())
        import random

        random.shuffle(commander_ids)

        for i in range(self.config.player_num):
            commander_id = commander_ids[i % len(commander_ids)]
            config = commander_configs[commander_id]

            player = Player(
                idx=i + 1,
                commander_id=commander_id,
                commander_name=config.get("name", commander_id),
                nation=config.get("nation", "群"),
                gender=config.get("gender", "male"),
                max_hp=config.get("hp", 4),
            )

            skills = SkillRegistry.create_skills_for_commander(commander_id, player)
            player.skills = skills

            self.players.append(player)

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        from skills.base import set_current_env, clear_current_env

        self._total_steps += 1

        # 自动处理技能决策（训练时使用默认策略）
        while self.skill_decision_context.has_pending_decision():
            request = self.skill_decision_context.active_request
            if request is None:
                break
            mask = self._get_skill_decision_mask()
            valid_options = np.where(mask > 0)[0]
            if len(valid_options) == 0:
                self.skill_decision_context.clear()
                break
            auto_action = valid_options[0]
            self._handle_skill_decision(int(auto_action))

        if self.skill_decision_context.has_pending_decision():
            return self._handle_skill_decision(action)

        set_current_env(self)
        self._turn_steps += 1

        if self._turn_steps > self._max_turn_steps:
            logger.warning(
                f"Turn exceeded max steps ({self._max_turn_steps}), forcing end turn"
            )
            self._execute_end_turn()
            self._turn_steps = 0

        if self._turn_steps % 50 == 0 and self._turn_steps > 0:
            pending_info = (
                self.pending_action.to_dict() if self.pending_action else "None"
            )
            logger.warning(
                f"Turn steps: {self._turn_steps}, current_step={self.current_step}, action={action}, player={self.current_player_idx}, pending={pending_info}"
            )

        if self._total_steps % 1000 == 0:
            logger.info(
                f"Env step {self._total_steps}, round={self.round_count}, player={self.current_player_idx}, phase={self.engine.phase if self.engine else 'None'}, turn_steps={self._turn_steps}"
            )

        action = int(action)
        truncated = False

        valid_action = self._validate_action(action)

        if not valid_action:
            obs = self._get_observation()
            clear_current_env()
            return obs, -0.1, False, False, {"error": "Invalid action"}

        last_action_type = (
            self.pending_action.action_type if self.pending_action else -1
        )

        self._process_action(action)

        done = self._check_done()

        reward = self._calculate_reward()

        if (
            self._turn_steps > 5
            and self.pending_action is not None
            and last_action_type == self.pending_action.action_type
        ):
            reward -= 0.01

        obs = self._get_observation()
        info = self._get_info()

        if done:
            info["winner"] = self._winner
            info["player_identity"] = self.players[self.current_player_idx].identity

        clear_current_env()
        return obs, reward, done, truncated, info

    def _validate_action(self, action: int) -> bool:
        action = int(action)
        if self.current_step == 0:
            if action >= self.action_encoder.action_type_dim:
                return False
            mask = self._get_action_type_mask()
            return mask[action] > 0
        elif self.current_step == 1:
            if action >= self.action_encoder.card_dim:
                return False
            mask = self._get_card_mask()
            if mask.sum() == 0:
                # 没有可用的卡牌/技能，应该结束回合而不是接受任意动作
                return False
            return mask[action] > 0
        elif self.current_step == 2:
            if action >= self.action_encoder.target_dim:
                return False
            mask = self._get_target_mask()
            if mask.sum() == 0:
                # 没有可用目标，动作无效
                return False
            return mask[action] > 0
        return False

    def _process_action(self, action: int):
        if self.current_step == 0:
            self.pending_action = HierarchicalAction(action_type=action)

            if action == ActionType.END_TURN:
                self._execute_end_turn()
                self.current_step = 0
                self.pending_action = None
            elif action == ActionType.PASS:
                self._execute_pass()
                self.current_step = 0
                self.pending_action = None
            else:
                needs_card = self.action_encoder.needs_card(action)
                if needs_card:
                    self.current_step = 1
                else:
                    needs_target = self.action_encoder.needs_target(action)
                    if needs_target:
                        self.current_step = 2
                    else:
                        self._execute_action(self.pending_action)
                        self.current_step = 0
                        self.pending_action = None

        elif self.current_step == 1:
            if self.pending_action:
                card_mask = self._get_card_mask()
                if card_mask.sum() == 0:
                    # 没有可用的卡牌/技能，强制结束回合
                    logger.warning(
                        f"No valid cards/skills available, forcing end turn. "
                        f"action_type={self.pending_action.action_type}"
                    )
                    self._execute_end_turn()
                    return

                # 检查选择的动作是否在掩码中
                if action >= len(card_mask) or card_mask[action] == 0:
                    logger.warning(
                        f"Invalid card/skill selection: action={action}, mask_sum={card_mask.sum()}"
                    )
                    # 选择第一个有效的
                    valid_indices = [i for i, m in enumerate(card_mask) if m > 0]
                    if valid_indices:
                        action = valid_indices[0]
                    else:
                        self._execute_end_turn()
                        return

                self.pending_action.card_idx = int(action)

                player = self.players[self.current_player_idx]

                # 区分 USE_CARD 和 USE_SKILL
                if self.pending_action.action_type == ActionType.USE_SKILL:
                    # 技能选择
                    skills = getattr(player, "skills", [])
                    skill = skills[int(action)] if int(action) < len(skills) else None

                    if skill is None:
                        logger.warning(
                            f"Step=1: action={action} but skill is None, skills_count={len(skills)}"
                        )

                    needs_target = self.action_encoder.needs_target(
                        self.pending_action.action_type, skill
                    )
                else:
                    # 卡牌选择
                    hand_cards = player.hand_cards
                    card = (
                        hand_cards[int(action)]
                        if int(action) < len(hand_cards)
                        else None
                    )

                    if card is None:
                        logger.warning(
                            f"Step=1: action={action} but card is None, hand_size={len(hand_cards)}, "
                            f"action_type={self.pending_action.action_type}"
                        )

                    needs_target = self.action_encoder.needs_target(
                        self.pending_action.action_type, card
                    )

                if needs_target:
                    self.current_step = 2
                else:
                    self._execute_action(self.pending_action)
                    self.current_step = 0
                    self.pending_action = None

        elif self.current_step == 2:
            if self.pending_action:
                self.pending_action.target_idx = action
                logger.info(
                    f"Step 2: executing skill {self.pending_action.card_idx} on target {action}"
                )
                action_executed = self._execute_action(self.pending_action)
                if action_executed:
                    self.current_step = 0
                    self.pending_action = None
                    self._action_failures = 0
                else:
                    self._action_failures += 1
                    if self._action_failures >= self._max_action_failures:
                        logger.warning(
                            f"Too many action failures ({self._action_failures}), forcing end turn"
                        )
                        self._execute_end_turn()
                    else:
                        self.current_step = 1
                        self.pending_action.target_idx = None
                        logger.warning(
                            f"Skill execution failed ({self._action_failures}/{self._max_action_failures}), returning to skill selection"
                        )

    def _execute_action(self, action: HierarchicalAction):
        player = self.players[self.current_player_idx]
        action_executed = False

        if action.action_type == ActionType.USE_CARD:
            if action.card_idx is not None and action.card_idx < len(player.hand_cards):
                card = player.hand_cards[action.card_idx]
                target = None
                if action.target_idx is not None:
                    target_idx = action.target_idx
                    if 0 <= target_idx < len(self.players):
                        target = self.players[target_idx]

                success = self.engine.use_card(player, card, target)
                if success:
                    self._record_action(action, card.name if card else "")
                    action_executed = True
                else:
                    logger.warning(
                        f"USE_CARD failed: card={card.name}, player={player.idx}, "
                        f"sha_count={player.sha_count}, hand_size={len(player.hand_cards)}, "
                        f"card_in_hand={card in player.hand_cards}"
                    )
            else:
                logger.warning(
                    f"USE_CARD failed: card_idx={action.card_idx}, hand_size={len(player.hand_cards)}"
                )

        elif action.action_type == ActionType.DISCARD:
            if action.card_idx is not None and action.card_idx < len(player.hand_cards):
                card = player.hand_cards.pop(action.card_idx)
                self.engine.discard_pile.append(card)
                self._record_action(action, "discard")

        elif action.action_type == ActionType.RESPOND_SHAN:
            if action.card_idx is not None:
                shan_cards = [c for c in player.hand_cards if c.name == "闪"]
                if shan_cards and action.card_idx < len(shan_cards):
                    card = shan_cards[action.card_idx]
                    player.hand_cards.remove(card)
                    self.engine.discard_pile.append(card)
                    self._record_action(action, "闪")

        elif action.action_type == ActionType.RESPOND_SHA:
            if action.card_idx is not None:
                sha_cards = [c for c in player.hand_cards if "杀" in c.name]
                if sha_cards and action.card_idx < len(sha_cards):
                    card = sha_cards[action.card_idx]
                    player.hand_cards.remove(card)
                    self.engine.discard_pile.append(card)
                    self._record_action(action, "杀")

        elif action.action_type == ActionType.RESPOND_TAO:
            if action.card_idx is not None:
                tao_cards = [c for c in player.hand_cards if c.name == "桃"]
                if tao_cards and action.card_idx < len(tao_cards):
                    card = tao_cards[action.card_idx]

                    target = None
                    if action.target_idx is not None:
                        target = self.players[action.target_idx]

                    if target is None:
                        target = player

                    success = self.engine.use_card(player, card, target)
                    if success:
                        self._record_action(action, "桃")
                        action_executed = True
                    else:
                        logger.warning(f"RESPOND_TAO failed: use_card returned False")
                else:
                    logger.warning(f"RESPOND_TAO failed: no tao cards or invalid index")

        elif action.action_type == ActionType.USE_SKILL:
            from skills.base import ActiveSkill

            skills = player.skills
            if action.card_idx is not None and 0 <= action.card_idx < len(skills):
                skill = skills[action.card_idx]
                if isinstance(skill, ActiveSkill):
                    target = None
                    if action.target_idx is not None and 0 <= action.target_idx < len(
                        self.players
                    ):
                        target = self.players[action.target_idx]

                    from engine.event import Event, EventType

                    event = Event(
                        type=EventType.SKILL_TRIGGERED,
                        source=player,
                        target=target,
                        engine=self.engine,
                    )
                    if skill.can_activate(event, self.engine):
                        result = skill.execute(event, self.engine)
                        if result is not None:
                            logger.info(
                                f"Using skill: {skill.name}"
                                + (f" on {target.commander_name}" if target else "")
                            )
                            action_executed = True
                        else:
                            logger.warning(
                                f"Skill {skill.name} execution returned None"
                            )
                    else:
                        logger.warning(f"Skill {skill.name} cannot be activated")
                else:
                    logger.warning(f"Cannot actively use passive skill: {skill.name}")
            else:
                logger.warning(
                    f"USE_SKILL failed: skill_idx={action.card_idx}, skills_count={len(skills)}"
                )

        return action_executed

    def _execute_end_turn(self):
        player = self.players[self.current_player_idx]

        hand_limit = max(0, player.hand_limit)
        while len(player.hand_cards) > hand_limit:
            card = player.hand_cards.pop()
            self.engine.discard_pile.append(card)

        survive_reward = self.reward_system.get_reward(
            event_type="turn_survive",
            source_identity="",
            target_identity=player.identity,
            current_identity=player.identity,
            is_target=True,
        )
        self._pending_rewards += survive_reward

        self.engine.next_turn()
        self.current_player_idx = self.engine.current_player_idx
        self.round_count = self.engine.round_num
        self.current_step = 0
        self._turn_steps = 0
        self._action_failures = 0
        self._record_action(HierarchicalAction(ActionType.END_TURN), "end_turn")

        if self.engine.phase != GamePhase.GAME_OVER:
            self._run_until_player_turn()

    def _execute_pass(self):
        self.current_step = 0
        self._record_action(HierarchicalAction(ActionType.PASS), "pass")

    def _record_action(self, action: HierarchicalAction, card_name: str = ""):
        self.action_history.append(
            {
                "action_type": action.action_type,
                "card_idx": action.card_idx,
                "target_idx": action.target_idx,
                "card_name": card_name,
                "player_idx": self.current_player_idx,
            }
        )

    def _check_done(self) -> bool:
        if self.engine.phase == GamePhase.GAME_OVER:
            self._winner = self._determine_winner()
            return True

        if self.engine.round_num >= self.config.max_rounds:
            self._winner = self._determine_winner()
            return True

        return False

    def _determine_winner(self) -> str:
        alive_identities = [p.identity for p in self.players if p.is_alive]

        if "主公" not in alive_identities:
            if len(alive_identities) == 1 and alive_identities[0] == "内奸":
                return "内奸"
            return "反贼"

        if "反贼" not in alive_identities and "内奸" not in alive_identities:
            return "主公"

        return "unknown"

    def _calculate_reward(self) -> float:
        player = self.players[self.current_player_idx]
        total_reward = self._pending_rewards
        self._pending_rewards = 0.0

        if self._winner:
            final_reward = self.reward_system.get_reward(
                event_type="game_over",
                source_identity="",
                target_identity="",
                current_identity=player.identity,
                context={
                    "winner": self._winner,
                    "survivors": [p for p in self.players if p.is_alive],
                },
            )
            return total_reward + final_reward

        for i, p in enumerate(self.players):
            if i == self.current_player_idx:
                continue

            prev_hp = self._prev_hp.get(i, p.current_hp)
            curr_hp = p.current_hp

            if curr_hp < prev_hp:
                damage = prev_hp - curr_hp
                if (
                    hasattr(player, "last_damage_source")
                    and player.last_damage_source == self.current_player_idx
                ):
                    total_reward += self.reward_system.get_reward(
                        event_type="damage_dealt",
                        source_identity=player.identity,
                        target_identity=p.identity,
                        current_identity=player.identity,
                        is_source=True,
                        value=damage,
                    )

                if i == self.current_player_idx:
                    total_reward += self.reward_system.get_reward(
                        event_type="damage_taken",
                        source_identity="",
                        target_identity=player.identity,
                        current_identity=player.identity,
                        is_target=True,
                        value=damage,
                    )

            prev_alive = self._prev_alive.get(i, True)
            curr_alive = p.is_alive

            if prev_alive and not curr_alive:
                if hasattr(player, "last_kill") and player.last_kill == i:
                    total_reward += self.reward_system.get_reward(
                        event_type="player_killed",
                        source_identity=player.identity,
                        target_identity=p.identity,
                        current_identity=player.identity,
                        is_source=True,
                    )

        self._save_player_states()

        return total_reward

    def _get_observation(self) -> Dict:
        state = self._get_game_state_dict()

        encoded_state = self.state_encoder.encode(state, self.current_player_idx)

        mask_type, mask_card, mask_target = self._get_action_masks()

        obs = {
            "state": encoded_state,
            "action_mask_type": mask_type,
            "action_mask_card": mask_card,
            "action_mask_target": mask_target,
            "current_step": self.current_step,
            "skill_decision_type": 0,
            "skill_decision_mask": np.zeros(
                self.action_encoder.card_dim, dtype=np.float32
            ),
        }

        if self.skill_decision_context.has_pending_decision():
            request = self.skill_decision_context.active_request
            obs["current_step"] = 3
            obs["skill_decision_type"] = int(request.decision_type)
            obs["skill_decision_mask"] = self._get_skill_decision_mask()

        return obs

    def _get_game_state_dict(self) -> Dict:
        if self.engine is None:
            return {"players": [], "phase": "waiting"}

        state = self.engine.get_state()
        state_dict = state.to_dict()

        state_dict["action_history"] = self.action_history[-10:]

        return state_dict

    def _get_action_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.use_action_mask:
            player = self.players[self.current_player_idx]
            game_state = self._get_game_state_dict()

            return self.action_mask_generator.generate_masks(
                game_state, player, self.engine, self.current_step, self.pending_action
            )
        else:
            ones_type = np.ones(self.action_encoder.action_type_dim, dtype=np.float32)
            ones_card = np.ones(self.action_encoder.card_dim, dtype=np.float32)
            ones_target = np.ones(self.action_encoder.target_dim, dtype=np.float32)
            return ones_type, ones_card, ones_target

    def _get_action_type_mask(self) -> np.ndarray:
        player = self.players[self.current_player_idx]
        game_state = self._get_game_state_dict()
        return self.action_mask_generator._get_valid_action_types(
            game_state, player, self.engine
        )

    def _get_card_mask(self) -> np.ndarray:
        player = self.players[self.current_player_idx]
        game_state = self._get_game_state_dict()
        return self.action_mask_generator._get_valid_cards(
            game_state, player, self.pending_action, self.engine
        )

    def _get_target_mask(self) -> np.ndarray:
        player = self.players[self.current_player_idx]
        game_state = self._get_game_state_dict()
        return self.action_mask_generator._get_valid_targets(
            game_state, player, self.pending_action
        )

    def _get_info(self) -> Dict:
        player = self.players[self.current_player_idx] if self.players else None

        return {
            "current_player_idx": self.current_player_idx,
            "player_identity": player.identity if player else "",
            "round_num": self.round_count,
            "phase": self.engine.phase.value if self.engine else "waiting",
            "action_history": self.action_history[-5:],
        }

    def render(self, mode: str = "human"):
        if mode == "ansi":
            return self._render_ansi()
        else:
            print(self._render_ansi())

    def _render_ansi(self) -> str:
        lines = []
        lines.append(f"=== Round {self.round_count} ===")
        lines.append(f"Phase: {self.engine.phase.value if self.engine else 'N/A'}")
        lines.append(f"Current Player: {self.current_player_idx}")
        lines.append("")

        for i, player in enumerate(self.players):
            status = "ALIVE" if player.is_alive else "DEAD"
            lines.append(
                f"Player {i}: {player.commander_name} ({player.identity}) "
                f"HP: {player.current_hp}/{player.max_hp} [{status}]"
            )
            lines.append(f"  Hand: {len(player.hand_cards)} cards")
            lines.append(
                f"  Equipment: {[(k, v.name if v else '-') for k, v in player.equipment.items()]}"
            )

        return "\n".join(lines)

    def _handle_skill_decision(
        self, action: int
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """处理技能决策的step"""
        request = self.skill_decision_context.active_request
        if request is None:
            obs = self._get_observation()
            return obs, 0.0, False, False, {"error": "No pending decision"}

        action = int(action)
        mask = self._get_skill_decision_mask()

        if action < 0 or action >= len(mask) or mask[action] == 0:
            obs = self._get_observation()
            return obs, -0.1, False, False, {"error": "Invalid skill decision"}

        from ai.skill_decision import SkillDecisionType

        if request.decision_type == SkillDecisionType.YES_NO:
            request.result = action == 1
            request.is_resolved = True

        elif request.decision_type == SkillDecisionType.SELECT_ORDER:
            request.add_selection(action)
            if request.is_complete():
                request.result = request.get_result()
                request.is_resolved = True

        elif request.decision_type == SkillDecisionType.SELECT_PAIR:
            request.add_selection(action)
            if request.is_complete():
                request.result = request.get_result()
                request.is_resolved = True

        elif request.decision_type in (
            SkillDecisionType.SELECT_CARDS,
            SkillDecisionType.SELECT_TARGETS,
        ):
            request.add_selection(action)
            if request.is_complete():
                request.result = request.get_result()
                request.is_resolved = True

        elif request.decision_type == SkillDecisionType.DISTRIBUTE:
            if request.result is None:
                request.result = {}
            item_idx = self._skill_decision_step
            if item_idx < len(request.context.get("items", [])):
                request.result[item_idx] = action
                self._skill_decision_step += 1
            if self._skill_decision_step >= len(request.context.get("items", [])):
                request.is_resolved = True

        if request.is_resolved:
            self.skill_decision_context.clear()
            self._skill_decision_step = 0

        obs = self._get_observation()
        info = self._get_info()
        info["skill_decision"] = request.skill_name
        info["skill_decision_type"] = request.decision_type.name
        info["skill_decision_complete"] = request.is_resolved

        return obs, 0.0, False, False, info

    def _get_skill_decision_mask(self) -> np.ndarray:
        """获取技能决策的mask"""
        request = self.skill_decision_context.active_request
        if request is None:
            return np.zeros(self.action_encoder.card_dim, dtype=np.float32)

        from ai.skill_decision import SkillDecisionType

        mask = np.zeros(self.action_encoder.card_dim, dtype=np.float32)

        if request.decision_type == SkillDecisionType.YES_NO:
            mask[0] = 1.0
            mask[1] = 1.0

        elif request.decision_type == SkillDecisionType.SELECT_ORDER:
            remaining = request.get_remaining_options()
            for idx in remaining:
                if idx < len(mask):
                    mask[idx] = 1.0

        elif request.decision_type == SkillDecisionType.SELECT_PAIR:
            remaining = request.get_remaining_options()
            for idx in remaining:
                if idx < len(mask):
                    mask[idx] = 1.0

        elif request.decision_type in (
            SkillDecisionType.SELECT_CARDS,
            SkillDecisionType.SELECT_TARGETS,
        ):
            remaining = request.get_remaining_options()
            for idx in remaining:
                if idx < len(mask):
                    mask[idx] = 1.0

        elif request.decision_type == SkillDecisionType.DISTRIBUTE:
            for i, target in enumerate(request.options):
                if i < len(mask):
                    mask[i] = 1.0

        return mask

    def request_skill_decision(self, request) -> bool:
        """外部调用此方法发起技能决策请求"""
        from ai.skill_decision import SkillDecisionContext

        self.skill_decision_context.active_request = request
        return True

    def get_skill_decision_result(self):
        """获取技能决策结果"""
        if self.skill_decision_context.active_request:
            return self.skill_decision_context.active_request.get_result()
        return None

    def close(self):
        self.engine = None
        self.players = []
        self.action_history = []

    def get_legal_actions(self) -> List[int]:
        if self.current_step == 0:
            mask = self._get_action_type_mask()
            return [i for i, v in enumerate(mask) if v > 0]
        elif self.current_step == 1:
            mask = self._get_card_mask()
            return [i for i, v in enumerate(mask) if v > 0]
        elif self.current_step == 2:
            mask = self._get_target_mask()
            return [i for i, v in enumerate(mask) if v > 0]
        return []

    def action_masks(self) -> np.ndarray:
        action_dim = self.action_space.n
        if self.current_step == 0:
            mask = self._get_action_type_mask()
            result = np.zeros(action_dim, dtype=np.float32)
            result[: len(mask)] = mask
            if mask.sum() == 0:
                result[0] = 1.0
            return result
        elif self.current_step == 1:
            mask = self._get_card_mask()
            result = np.zeros(action_dim, dtype=np.float32)
            result[: len(mask)] = mask
            if mask.sum() == 0:
                result[0] = 1.0
            return result
        elif self.current_step == 2:
            mask = self._get_target_mask()
            result = np.zeros(action_dim, dtype=np.float32)
            result[: len(mask)] = mask
            if mask.sum() == 0:
                result[0] = 1.0
            return result
        return np.ones(action_dim, dtype=np.float32)


def make_env(config: Optional[SGSConfig] = None) -> SGSEnv:
    return SGSEnv(config)
