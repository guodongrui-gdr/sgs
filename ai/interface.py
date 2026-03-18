from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player
    from card.base import Card


@dataclass
class Action:
    action_type: str
    card: Optional["Card"] = None
    target: Optional["Player"] = None
    targets: List["Player"] = None

    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type,
            "card": self.card.to_dict() if self.card else None,
            "target": self.target.idx if self.target else None,
            "targets": [t.idx for t in self.targets] if self.targets else None,
        }


class AIInterface:
    def __init__(self, engine: "GameEngine", player: "Player"):
        self.engine = engine
        self.player = player

    def get_state(self) -> Dict[str, Any]:
        state = self.engine.get_state()
        state["my_hand_cards"] = [c.to_dict() for c in self.player.hand_cards]
        state["my_idx"] = self.player.idx
        return state.to_dict()

    def get_legal_actions(self) -> List[Action]:
        actions = []

        if self.engine.phase.value == "play_phase":
            for i, card in enumerate(self.player.hand_cards):
                if self._can_use_card(card):
                    targets = self._get_legal_targets(card)
                    for target in targets:
                        actions.append(
                            Action(action_type="use_card", card=card, target=target)
                        )

            actions.append(Action(action_type="end_turn"))

        elif self.engine.phase.value == "discard_phase":
            hand_count = len(self.player.hand_cards)
            limit = self.player.hand_limit
            if hand_count > limit:
                for i, card in enumerate(self.player.hand_cards):
                    actions.append(Action(action_type="discard", card=card))

        return actions

    def _can_use_card(self, card: "Card") -> bool:
        if card.name == "杀":
            return self.player.can_use_sha()
        if card.name == "酒":
            return self.player.jiu_count < 1
        if card.name == "桃":
            return self.player.current_hp < self.player.max_hp
        return True

    def _get_legal_targets(self, card: "Card") -> List["Player"]:
        targets = []

        if "another_player" in card.target_types:
            for p in self.engine.players:
                if p != self.player and p.is_alive:
                    if card.name == "杀":
                        if self._is_in_range(p):
                            targets.append(p)
                    else:
                        targets.append(p)

        elif "self" in card.target_types:
            targets.append(self.player)

        elif "all_other_players" in card.target_types:
            for p in self.engine.players:
                if p != self.player and p.is_alive:
                    targets.append(p)

        elif "all_players" in card.target_types:
            for p in self.engine.players:
                if p.is_alive:
                    targets.append(p)

        return targets

    def _is_in_range(self, target: "Player") -> bool:
        distance = self._calculate_distance(self.player, target)
        return distance <= self.player.attack_range

    def _calculate_distance(self, source: "Player", target: "Player") -> int:
        if source == target:
            return 0

        distance = 1
        current = source.next_player
        while current != target:
            distance += 1
            current = current.next_player

        reverse_distance = 1
        current = source.prev_player
        while current != target:
            reverse_distance += 1
            current = current.prev_player

        min_distance = min(distance, reverse_distance)

        if source.equipment.get("进攻坐骑"):
            min_distance = max(1, min_distance - 1)
        if target.equipment.get("防御坐骑"):
            min_distance += 1

        return min_distance

    def step(self, action: Action) -> Tuple[Dict, float, bool]:
        if action.action_type == "use_card":
            success = self.engine.use_card(self.player, action.card, action.target)
            reward = 1.0 if success else 0.0

        elif action.action_type == "discard":
            if action.card in self.player.hand_cards:
                self.player.hand_cards.remove(action.card)
                self.engine.discard_pile.append(action.card)
                reward = 0.0
            else:
                reward = -1.0

        elif action.action_type == "end_turn":
            self.engine.next_turn()
            reward = 0.0

        else:
            reward = -1.0

        done = self.engine.phase.value == "game_over"
        state = self.get_state()

        return state, reward, done
