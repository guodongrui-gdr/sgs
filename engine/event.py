from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Dict


class EventType(Enum):
	GAME_START = auto()
	GAME_END = auto()

	TURN_START = auto()
	TURN_END = auto()

	PREPARE_PHASE = auto()
	JUDGE_PHASE = auto()
	DRAW_PHASE = auto()
	PLAY_PHASE = auto()
	DISCARD_PHASE = auto()
	END_PHASE = auto()

	PHASE_START = auto()
	PHASE_END = auto()

	DRAW_CARD = auto()

	CARD_USED = auto()
	CARD_TARGETED = auto()
	CARD_EFFECT = auto()
	CARD_DISCARDED = auto()

	DAMAGE_DEALT = auto()
	DAMAGE_TAKEN = auto()
	DAMAGE_PREVENTED = auto()

	HP_CHANGED = auto()

	PLAYER_DYING = auto()
	PLAYER_SAVED = auto()
	PLAYER_DEAD = auto()

	EQUIPMENT_EQUIPPED = auto()
	EQUIPMENT_UNEQUIPPED = auto()

	JUDGE_START = auto()
	JUDGE_RESULT = auto()
	JUDGE_BEFORE = auto()

	SKILL_TRIGGERED = auto()

	CHAIN_LINKED = auto()
	CHAIN_UNLINKED = auto()

	CARD_DRAWN = auto()
	DISCARD_START = auto()
	CARD_LOST = auto()

	BEFORE_USE_CARD = auto()
	AFTER_USE_CARD = auto()

	BEFORE_DAMAGE = auto()
	AFTER_DAMAGE = auto()

	ASK_FOR_SHAN = auto()
	ASK_FOR_SHA = auto()
	ASK_FOR_TAO = auto()


@dataclass
class Event:
	type: EventType
	source: Optional[Any] = None
	target: Optional[Any] = None
	card: Optional[Any] = None
	value: int = 0
	data: Dict = field(default_factory=dict)
	cancelled: bool = False
	engine: Optional[Any] = None

	def cancel(self):
		self.cancelled = True

	def is_cancelled(self) -> bool:
		return self.cancelled
