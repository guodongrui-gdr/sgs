from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class GamePhase(Enum):
    WAITING = "waiting"
    TURN_START = "turn_start"
    PREPARE_PHASE = "prepare_phase"
    JUDGE_PHASE = "judge_phase"
    DRAW_PHASE = "draw_phase"
    PLAY_PHASE = "play_phase"
    DISCARD_PHASE = "discard_phase"
    END_PHASE = "end_phase"
    TURN_END = "turn_end"
    GAME_OVER = "game_over"


@dataclass
class PlayerState:
    player_id: int
    commander_id: str
    commander_name: str
    nation: str
    identity: str
    max_hp: int
    current_hp: int
    hand_cards: List[Dict]
    equipment: Dict[str, Optional[Dict]]
    judge_area: List[Dict]
    skills: List[str]
    is_chained: bool
    is_alive: bool
    sha_count: int
    jiu_count: int
    jiu_effect: int

    def to_dict(self) -> Dict:
        return {
            "player_id": self.player_id,
            "commander_id": self.commander_id,
            "commander_name": self.commander_name,
            "nation": self.nation,
            "identity": self.identity,
            "max_hp": self.max_hp,
            "current_hp": self.current_hp,
            "hand_cards": self.hand_cards,
            "equipment": self.equipment,
            "judge_area": self.judge_area,
            "skills": self.skills,
            "is_chained": self.is_chained,
            "is_alive": self.is_alive,
            "sha_count": self.sha_count,
            "jiu_count": self.jiu_count,
            "jiu_effect": self.jiu_effect,
        }


@dataclass
class GameState:
    phase: GamePhase
    current_player_idx: int
    round_num: int
    players: List[PlayerState]
    deck_count: int
    discard_pile_count: int

    winner: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "phase": self.phase.value,
            "current_player_idx": self.current_player_idx,
            "round_num": self.round_num,
            "players": [p.to_dict() for p in self.players],
            "deck_count": self.deck_count,
            "discard_pile_count": self.discard_pile_count,
            "winner": self.winner,
        }

    def get_alive_players(self) -> List[PlayerState]:
        return [p for p in self.players if p.is_alive]

    def get_player_by_id(self, player_id: int) -> Optional[PlayerState]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None
