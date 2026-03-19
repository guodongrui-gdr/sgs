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
    max_rounds: int = 100
    use_action_mask: bool = True
    use_shaping: bool = False

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
                "current_step": spaces.Discrete(3),
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

        self.reward_system.reset()

        self._save_player_states()

        self._run_until_player_turn()

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
        """运行游戏直到当前玩家可以行动"""
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
                    None, current, None, judge_result["lightning_damage"], True
                )

            if not current.is_alive:
                self.engine.next_turn()
                self.current_player_idx = self.engine.current_player_idx
                continue

            if not judge_result.get("skip_draw", False):
                drawn = self.engine.draw_cards(current, 2)
                current.hand_cards.extend(drawn)

            if not judge_result.get("skip_play", False):
                self.engine.phase = GamePhase.PLAY_PHASE
                break

            self.engine.next_turn()
            self.current_player_idx = self.engine.current_player_idx
            self.round_count = self.engine.round_num

        if iteration >= max_iterations - 1:
            logger.warning(
                f"_run_until_player_turn reached max_iterations={max_iterations}"
            )
            self.engine.phase = GamePhase.GAME_OVER

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
        self._total_steps += 1
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
            return obs, -0.1, False, True, {"error": "Invalid action"}

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
                return True
            return mask[action] > 0
        elif self.current_step == 2:
            if action >= self.action_encoder.target_dim:
                return False
            mask = self._get_target_mask()
            if mask.sum() == 0:
                return True
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
                    self._execute_end_turn()
                    return

                self.pending_action.card_idx = int(action)

                player = self.players[self.current_player_idx]
                hand_cards = player.hand_cards
                card = (
                    hand_cards[int(action)] if int(action) < len(hand_cards) else None
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
                    from engine.event import Event, EventType

                    event = Event(
                        type=EventType.SKILL_TRIGGERED,
                        source=player,
                        engine=self.engine,
                    )
                    if skill.can_activate(event, self.engine):
                        result = skill.execute(event, self.engine)
                        if result is not None:
                            logger.info(f"Using skill: {skill.name}")
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

        return {
            "state": encoded_state,
            "action_mask_type": mask_type,
            "action_mask_card": mask_card,
            "action_mask_target": mask_target,
            "current_step": self.current_step,
        }

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
