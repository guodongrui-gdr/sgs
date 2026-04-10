import pygame
from typing import List, Tuple, Optional, Callable, TYPE_CHECKING
from gui.assets import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    CARD_WIDTH,
    CARD_HEIGHT,
    CARD_SPACING,
    HAND_AREA_HEIGHT,
    PLAYER_AREA_WIDTH,
    PLAYER_AREA_HEIGHT,
    COLORS,
    fonts,
)

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player
    from card.base import Card


class InputHandler:
    def __init__(self):
        self.selected_card_indices: List[int] = []
        self.selected_players: List["Player"] = []
        self.hovered_card_index: int = -1
        self.hovered_button: Optional[str] = None

        self._card_rects: List[Tuple[pygame.Rect, int]] = []
        self._player_rects: List[Tuple[pygame.Rect, "Player"]] = []
        self._buttons: dict = {}

    def update_card_rects(
        self, card_data: List[Tuple[pygame.Surface, pygame.Rect, int]]
    ):
        self._card_rects = [(rect, idx) for _, rect, idx in card_data]

    def update_player_rects(self, player_data: List[Tuple[pygame.Rect, "Player"]]):
        self._player_rects = player_data

    def update_buttons(self, buttons: dict):
        self._buttons = buttons

    def handle_event(
        self, event: pygame.event.Event, engine: "GameEngine", human_player: "Player"
    ) -> Optional[dict]:
        if event.type == pygame.MOUSEMOTION:
            return self._handle_mouse_motion(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                return self._handle_left_click(event.pos, engine, human_player)
            elif event.button == 3:
                return self._handle_right_click()

        elif event.type == pygame.KEYDOWN:
            return self._handle_key_press(event.key, engine, human_player)

        return None

    def _handle_mouse_motion(self, pos: Tuple[int, int]) -> dict:
        self.hovered_card_index = -1
        self.hovered_button = None

        for rect, idx in self._card_rects:
            if rect.collidepoint(pos):
                self.hovered_card_index = idx
                break

        for button_name, rect in self._buttons.items():
            if rect.collidepoint(pos):
                self.hovered_button = button_name
                break

        return {"type": "hover", "card_index": self.hovered_card_index}

    def _handle_left_click(
        self, pos: Tuple[int, int], engine: "GameEngine", human_player: "Player"
    ) -> Optional[dict]:
        for rect, idx in self._card_rects:
            if rect.collidepoint(pos):
                if idx in self.selected_card_indices:
                    self.selected_card_indices.remove(idx)
                else:
                    self.selected_card_indices.append(idx)
                return {
                    "type": "card_selected",
                    "card_index": idx,
                    "selected_indices": self.selected_card_indices.copy(),
                }

        for rect, player in self._player_rects:
            if rect.collidepoint(pos):
                if player in self.selected_players:
                    self.selected_players.remove(player)
                else:
                    self.selected_players.append(player)
                return {
                    "type": "player_selected",
                    "player": player,
                    "selected_players": self.selected_players.copy(),
                }

        for button_name, rect in self._buttons.items():
            if rect.collidepoint(pos):
                return {"type": "button_clicked", "button": button_name}

        return {"type": "click", "pos": pos}

    def _handle_right_click(self) -> dict:
        self.clear_selection()
        return {"type": "selection_cleared"}

    def _handle_key_press(
        self, key: int, engine: "GameEngine", human_player: "Player"
    ) -> Optional[dict]:
        if key == pygame.K_ESCAPE:
            self.clear_selection()
            return {"type": "selection_cleared"}

        elif key == pygame.K_RETURN or key == pygame.K_SPACE:
            if self.selected_card_indices:
                return {
                    "type": "confirm",
                    "card_indices": self.selected_card_indices.copy(),
                    "target_players": self.selected_players.copy(),
                }

        elif key == pygame.K_0:
            return {"type": "end_turn"}

        elif pygame.K_1 <= key <= pygame.K_9:
            card_idx = key - pygame.K_1
            if card_idx < len(human_player.hand_cards):
                if card_idx in self.selected_card_indices:
                    self.selected_card_indices.remove(card_idx)
                else:
                    self.selected_card_indices = [card_idx]
                return {
                    "type": "card_selected",
                    "card_index": card_idx,
                    "selected_indices": self.selected_card_indices.copy(),
                }

        return None

    def clear_selection(self):
        self.selected_card_indices.clear()
        self.selected_players.clear()

    def get_selected_card(self, human_player: "Player") -> Optional["Card"]:
        if not self.selected_card_indices:
            return None
        idx = self.selected_card_indices[0]
        if 0 <= idx < len(human_player.hand_cards):
            return human_player.hand_cards[idx]
        return None

    def get_selected_cards(self, human_player: "Player") -> List["Card"]:
        cards = []
        for idx in self.selected_card_indices:
            if 0 <= idx < len(human_player.hand_cards):
                cards.append(human_player.hand_cards[idx])
        return cards
