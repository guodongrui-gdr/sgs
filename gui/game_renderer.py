import pygame
from typing import List, Tuple, Optional, TYPE_CHECKING
from gui.assets import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    COLORS,
    fonts,
    assets,
    CARD_WIDTH,
    CARD_HEIGHT,
    CARD_SPACING,
    HAND_AREA_HEIGHT,
)
from gui.card_renderer import CardRenderer
from gui.player_renderer import PlayerRenderer

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player
    from card.base import Card


class GameRenderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.background: Optional[pygame.Surface] = None
        self._init_background()

    def _init_background(self):
        self.background = assets.get_background()
        if not self.background:
            self.background = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.background.fill(COLORS["background"])

    def render(self, engine: "GameEngine", human_player: "Player"):
        self.screen.blit(self.background, (0, 0))

        self._render_game_info(engine)

        player_positions = PlayerRenderer.calculate_player_positions(
            len(engine.players), WINDOW_WIDTH, WINDOW_HEIGHT
        )

        for i, player in enumerate(engine.players):
            x, y = player_positions[i]
            is_current = i == engine.current_player_idx
            show_hand = player == human_player
            show_identity = player == human_player or player.identity == "主公"

            surface, rect = PlayerRenderer.render_player_area(
                player, x, y, is_current, show_hand, show_identity
            )
            self.screen.blit(surface, rect)

        self._render_hand_area(human_player, engine)

        self._render_deck_info(engine)

    def _render_game_info(self, engine: "GameEngine"):
        font = fonts.get("medium")

        round_text = f"第 {engine.round_num} 轮"
        round_surface = font.render(round_text, True, COLORS["text_normal"])
        self.screen.blit(round_surface, (20, 20))

        if engine.phase:
            from engine.state import GamePhase

            phase_names = {
                GamePhase.TURN_START: "回合开始",
                GamePhase.JUDGE_PHASE: "判定阶段",
                GamePhase.DRAW_PHASE: "摸牌阶段",
                GamePhase.PLAY_PHASE: "出牌阶段",
                GamePhase.DISCARD_PHASE: "弃牌阶段",
                GamePhase.TURN_END: "回合结束",
                GamePhase.GAME_OVER: "游戏结束",
            }
            phase_text = phase_names.get(engine.phase, engine.phase.value)
            phase_surface = font.render(phase_text, True, COLORS["text_highlight"])
            self.screen.blit(phase_surface, (20, 50))

        current_player = engine.players[engine.current_player_idx]
        turn_text = f"当前: {current_player.commander_name}"
        turn_surface = font.render(turn_text, True, COLORS["text_normal"])
        self.screen.blit(turn_surface, (20, 80))

    def _render_hand_area(self, player: "Player", engine: "GameEngine"):
        if not player.is_alive:
            return

        hand_y = WINDOW_HEIGHT - HAND_AREA_HEIGHT

        pygame.draw.rect(
            self.screen,
            (*COLORS["panel_bg"], 150),
            (0, hand_y, WINDOW_WIDTH, HAND_AREA_HEIGHT),
        )
        pygame.draw.line(
            self.screen, COLORS["text_normal"], (0, hand_y), (WINDOW_WIDTH, hand_y), 2
        )

        font = fonts.get("small")
        label = f"你的手牌 ({len(player.hand_cards)}/{player.hand_limit})"
        label_surface = font.render(label, True, COLORS["text_normal"])
        self.screen.blit(label_surface, (20, hand_y + 5))

        if player.hand_cards:
            total_width = (
                len(player.hand_cards) * CARD_WIDTH
                + (len(player.hand_cards) - 1) * CARD_SPACING
            )
            start_x = (WINDOW_WIDTH - total_width) // 2

            card_data = CardRenderer.render_hand_cards(
                player.hand_cards, start_x, hand_y + 30
            )

            for surface, rect, idx in card_data:
                self.screen.blit(surface, rect)

            return card_data
        return []

    def _render_deck_info(self, engine: "GameEngine"):
        font = fonts.get("small")

        deck_x = WINDOW_WIDTH - 150
        deck_y = 20

        deck_text = f"牌堆: {len(engine.deck)}"
        deck_surface = font.render(deck_text, True, COLORS["text_normal"])
        self.screen.blit(deck_surface, (deck_x, deck_y))

        discard_text = f"弃牌: {len(engine.discard_pile)}"
        discard_surface = font.render(discard_text, True, COLORS["text_normal"])
        self.screen.blit(discard_surface, (deck_x, deck_y + 25))

    def render_target_selection(
        self,
        engine: "GameEngine",
        valid_targets: List["Player"],
        selected_targets: List["Player"],
        human_player: "Player",
    ) -> List[Tuple[pygame.Rect, "Player"]]:
        player_positions = PlayerRenderer.calculate_player_positions(
            len(engine.players), WINDOW_WIDTH, WINDOW_HEIGHT
        )

        clickable_areas = []

        for i, player in enumerate(engine.players):
            if player not in valid_targets:
                continue

            x, y = player_positions[i]
            is_selected = player in selected_targets

            surface, rect = PlayerRenderer.render_player_area(
                player, x, y, False, False, False, True, is_selected
            )
            self.screen.blit(surface, rect)
            clickable_areas.append((rect, player))

        return clickable_areas

    def render_message(self, message: str, duration: int = 0):
        font = fonts.get("large")
        text_surface = font.render(message, True, COLORS["text_highlight"])
        text_rect = text_surface.get_rect(
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        )

        bg_rect = text_rect.inflate(40, 20)
        pygame.draw.rect(
            self.screen, (*COLORS["panel_bg"], 230), bg_rect, border_radius=10
        )
        pygame.draw.rect(
            self.screen, COLORS["text_highlight"], bg_rect, 2, border_radius=10
        )

        self.screen.blit(text_surface, text_rect)

    def render_action_log(self, logs: List[str]):
        if not logs:
            return

        font = fonts.get("small")
        log_y = 120
        max_logs = min(8, len(logs))

        for i, log in enumerate(logs[-max_logs:]):
            log_surface = font.render(log, True, COLORS["text_normal"])
            self.screen.blit(log_surface, (20, log_y + i * 22))
