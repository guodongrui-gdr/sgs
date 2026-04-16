"""
MAPPO AI - Inference class for trained MAPPO models

Loads .pt checkpoints saved by train_mappo_world_model.py and provides
the same interface as RLAI for use in game engine.

Usage:
    from ai.mappo_inference import create_mappo_ai
    mappo_ai = create_mappo_ai(model_path="path/to/model.pt", player_num=5)
    card, target = mappo_ai.select_action(engine, player)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player

from ai.state_encoder import StateEncoder, EncodingConfig
from ai.action_encoder import (
    ActionEncoder,
    ActionConfig,
    ActionMaskGenerator,
    ActionType,
)

logger = logging.getLogger(__name__)


@dataclass
class MAPPOAIConfig:
    model_path: str
    player_num: int = 5
    deterministic: bool = True
    state_config: Optional[EncodingConfig] = None
    action_config: Optional[ActionConfig] = None


class MAPPOAI:
    """
    MAPPO AI for inference

    Loads trained MAPPOAgent from .pt checkpoint and uses the actors
    for decentralized action selection.
    """

    def __init__(self, config: MAPPOAIConfig):
        self.config = config
        self.model_path = Path(config.model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.state_encoder = StateEncoder(config.state_config or EncodingConfig())
        self.action_encoder = ActionEncoder(config.action_config or ActionConfig())
        self.action_mask_generator = ActionMaskGenerator(self.action_encoder)

        self._load_model()

        logger.info(f"MAPPOAI initialized with model: {self.model_path}")

    def _load_model(self):
        """Load MAPPOAgent from checkpoint"""
        from ai.mappo.mappo_agent import MAPPOAgent, MAPPOAgentConfig
        from ai.mappo.mappo_policy import MAPPOActorConfig
        from ai.mappo.centralized_critic import CentralizedCriticConfig

        default_agent_config = MAPPOAgentConfig(
            num_agents=self.config.player_num,
            actor_config=MAPPOActorConfig(
                local_state_dim=2670,
                action_dim=20,
            ),
            critic_config=CentralizedCriticConfig(
                local_state_dim=2670,
                num_agents=self.config.player_num,
                global_state_dim=2670 * self.config.player_num,
                action_dim=20,
            ),
        )

        try:
            import sys
            from types import ModuleType

            if hasattr(torch.serialization, "add_safe_globals"):
                from train.train_mappo_world_model import MAPPOWorldModelConfig

                m = ModuleType("__main__")
                m.MAPPOWorldModelConfig = MAPPOWorldModelConfig
                sys.modules["__main__"] = m
                torch.serialization.add_safe_globals([MAPPOWorldModelConfig])

            checkpoint = torch.load(
                self.model_path, weights_only=False, map_location="cpu"
            )

            if hasattr(sys.modules, "__main__") and sys.modules["__main__"] != m:
                pass
            else:
                sys.modules["__main__"] = sys.modules.get(
                    "__main__", sys.modules.get("__main__")
                )

            checkpoint_config = checkpoint.get("config")
            if checkpoint_config is not None and hasattr(
                checkpoint_config, "num_agents"
            ):
                num_agents = checkpoint_config.num_agents
                local_state_dim = getattr(checkpoint_config, "local_state_dim", 2670)
                global_state_dim = getattr(
                    checkpoint_config, "global_state_dim", 2670 * num_agents
                )
                action_dim = getattr(checkpoint_config, "action_dim", 20)
                lr = getattr(checkpoint_config, "learning_rate", 3e-4)
                gamma = getattr(checkpoint_config, "gamma", 0.99)
                gae_lambda = getattr(checkpoint_config, "gae_lambda", 0.95)

                agent_config = MAPPOAgentConfig(
                    num_agents=num_agents,
                    actor_config=MAPPOActorConfig(
                        local_state_dim=local_state_dim,
                        action_dim=action_dim,
                    ),
                    critic_config=CentralizedCriticConfig(
                        local_state_dim=local_state_dim,
                        num_agents=num_agents,
                        global_state_dim=global_state_dim,
                        action_dim=action_dim,
                    ),
                    learning_rate=lr,
                    ppo_gamma=gamma,
                    ppo_gae_lambda=gae_lambda,
                )
            else:
                saved_agent_config = checkpoint.get("agent_config")
                agent_config = (
                    saved_agent_config if saved_agent_config else default_agent_config
                )

            agent_state = checkpoint.get("agent_state", checkpoint)

        except Exception as e:
            logger.warning(f"Failed to load full checkpoint: {e}, using defaults")
            agent_config = default_agent_config
            agent_state = None

        self.agent = MAPPOAgent(agent_config)

        if agent_state is not None:
            if "actors_state_dict" in agent_state:
                for i, actor in enumerate(self.agent.actors):
                    if i < len(agent_state["actors_state_dict"]):
                        actor.load_state_dict(agent_state["actors_state_dict"][i])
            if "critic_state_dict" in agent_state:
                self.agent.shared_critic.load_state_dict(
                    agent_state["critic_state_dict"]
                )

        self.agent.eval()
        for actor in self.agent.actors:
            actor.eval()

        logger.info("MAPPO model loaded successfully")

    def select_action(
        self,
        engine: "GameEngine",
        player: "Player",
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Select action using 3-step hierarchical decision

        Args:
            engine: Game engine
            player: Current player

        Returns:
            (card, target) - card object and target player, or (None, None) for end turn
        """
        try:
            player_idx = player.idx - 1
            if player_idx < 0 or player_idx >= len(self.agent.actors):
                logger.warning(f"Invalid player idx: {player.idx}")
                return None, None

            actor = self.agent.actors[player_idx]

            game_state = engine.get_state()
            game_state_dict = game_state.to_dict()

            obs = self.state_encoder.encode(game_state_dict, player_idx)
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            phase = game_state_dict.get("phase", "waiting")
            if hasattr(phase, "value"):
                phase = phase.value

            step = 0
            pending_action_type = None
            pending_card_idx = None

            while True:
                if step == 0:
                    masks = self._get_mask(
                        step, game_state_dict, player, engine, None, None, None
                    )
                    mask_tensor = torch.from_numpy(masks).float().unsqueeze(0)

                    with torch.no_grad():
                        logits = actor(obs_tensor, mask_tensor)
                        probs = torch.softmax(logits, dim=-1)

                    if self.config.deterministic:
                        action = torch.argmax(probs, dim=-1)
                    else:
                        action = torch.multinomial(probs, 1)

                    action_type = int(action.item())

                    if (
                        action_type == ActionType.END_TURN
                        or action_type == ActionType.PASS
                    ):
                        return None, None

                    pending_action_type = action_type

                    needs_card = self.action_encoder.needs_card(action_type)
                    if not needs_card:
                        needs_target = self.action_encoder.needs_target(action_type)
                        if needs_target:
                            step = 2
                            continue
                        else:
                            return self._execute_simple_action(
                                action_type, None, player, engine
                            )
                    step = 1

                elif step == 1:
                    masks = self._get_mask(
                        step,
                        game_state_dict,
                        player,
                        engine,
                        pending_action_type,
                        None,
                        None,
                    )
                    mask_tensor = torch.from_numpy(masks).float().unsqueeze(0)

                    with torch.no_grad():
                        logits = actor(obs_tensor, mask_tensor)
                        probs = torch.softmax(logits, dim=-1)

                    if self.config.deterministic:
                        action = torch.argmax(probs, dim=-1)
                    else:
                        action = torch.multinomial(probs, 1)

                    card_idx = int(action.item())
                    pending_card_idx = card_idx

                    needs_target = self.action_encoder.needs_target(
                        pending_action_type,
                        player.hand_cards[card_idx]
                        if card_idx < len(player.hand_cards)
                        else None,
                    )

                    if not needs_target:
                        return self._execute_hierarchical_action(
                            pending_action_type, card_idx, None, player, engine
                        )
                    step = 2

                elif step == 2:
                    masks = self._get_mask(
                        step,
                        game_state_dict,
                        player,
                        engine,
                        pending_action_type,
                        pending_card_idx,
                        None,
                    )
                    mask_tensor = torch.from_numpy(masks).float().unsqueeze(0)

                    with torch.no_grad():
                        logits = actor(obs_tensor, mask_tensor)
                        probs = torch.softmax(logits, dim=-1)

                    if self.config.deterministic:
                        action = torch.argmax(probs, dim=-1)
                    else:
                        action = torch.multinomial(probs, 1)

                    target_idx = int(action.item())

                    return self._execute_hierarchical_action(
                        pending_action_type,
                        pending_card_idx,
                        target_idx,
                        player,
                        engine,
                    )

        except Exception as e:
            logger.error(f"Error in MAPPO action selection: {e}", exc_info=True)
            return None, None

    def _get_mask(
        self,
        step: int,
        game_state: Dict,
        player: "Player",
        engine: "GameEngine",
        pending_action_type: Optional[int],
        pending_card_idx: Optional[int],
        pending_target_idx: Optional[int],
    ) -> np.ndarray:
        """Get action mask for current step"""
        from ai.action_encoder import HierarchicalAction

        pending_action = None
        if pending_action_type is not None:
            pending_action = HierarchicalAction(
                action_type=pending_action_type,
                card_idx=pending_card_idx,
            )

        type_mask, card_mask, target_mask = self.action_mask_generator.generate_masks(
            game_state=game_state,
            player=player,
            engine=engine,
            current_step=step,
            pending_action=pending_action,
        )

        action_dim = self.action_encoder.get_action_space_dim()
        result = np.zeros(action_dim, dtype=np.float32)

        if step == 0:
            result[: len(type_mask)] = type_mask
            if type_mask.sum() == 0:
                result[0] = 1.0
        elif step == 1:
            result[: len(card_mask)] = card_mask
            if card_mask.sum() == 0:
                result[0] = 1.0
        elif step == 2:
            result[: len(target_mask)] = target_mask
            if target_mask.sum() == 0:
                result[0] = 1.0

        return result

    def _execute_simple_action(
        self,
        action_type: int,
        target_idx: Optional[int],
        player: "Player",
        engine: "GameEngine",
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Execute action that doesn't need card selection"""
        target = None
        if target_idx is not None:
            target = self._get_target_by_idx(target_idx, engine, player)
        return None, target

    def _execute_hierarchical_action(
        self,
        action_type: int,
        card_idx: int,
        target_idx: Optional[int],
        player: "Player",
        engine: "GameEngine",
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Execute hierarchical action (needs card selection)"""
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
                logger.warning(f"Invalid skill index: {card_idx}")
                return None, None

            target = None
            if target_idx is not None:
                target = self._get_target_by_idx(target_idx, engine, player)

            return card_idx, target

        elif action_type == ActionType.DISCARD:
            if card_idx < 0 or card_idx >= len(player.hand_cards):
                logger.warning(f"Invalid discard index: {card_idx}")
                return None, None

            card = player.hand_cards[card_idx]
            return card, None

        else:
            logger.warning(f"Unsupported action type: {action_type}")
            return None, None

    def _get_target_by_idx(
        self,
        target_idx: int,
        engine: "GameEngine",
        player: "Player",
    ) -> Optional["Player"]:
        """Get target player by index"""
        if target_idx < 0 or target_idx >= len(engine.players):
            return None

        target = engine.players[target_idx]
        if target == player or not target.is_alive:
            return None

        return target


def create_mappo_ai(
    model_path: str,
    player_num: int = 5,
    deterministic: bool = True,
) -> MAPPOAI:
    """
    Factory function to create MAPPOAI instance

    Args:
        model_path: Path to .pt checkpoint file
        player_num: Number of players in game
        deterministic: If True, use argmax; else sample

    Returns:
        MAPPOAI instance
    """
    config = MAPPOAIConfig(
        model_path=model_path,
        player_num=player_num,
        deterministic=deterministic,
    )
    return MAPPOAI(config)
