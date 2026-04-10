import pygame
import math
from typing import Optional, List, Dict, TYPE_CHECKING
from gui.assets import COLORS, WINDOW_WIDTH, WINDOW_HEIGHT, fonts

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player
    from card.base import Card


class ResponseRequest:
    TYPE_SHAN = "shan"
    TYPE_TAO = "tao"
    TYPE_WUXIE = "wuxie"
    TYPE_JUDGE = "judge"

    def __init__(
        self,
        request_type: str,
        prompt: str,
        player: "Player",
        source: "Player" = None,
        card: "Card" = None,
        timeout: int = 30000,
        required: bool = True,
    ):
        self.request_type = request_type
        self.prompt = prompt
        self.player = player
        self.source = source
        self.card = card
        self.timeout = timeout
        self.required = required
        self.response = None
        self.start_time = pygame.time.get_ticks()
        self.handled = False


class ResponseManager:
    def __init__(self):
        self.current_request: Optional[ResponseRequest] = None
        self.available_cards: List["Card"] = []
        self.selected_card: Optional["Card"] = None
        self.response_buttons: Dict[str, pygame.Rect] = {}

        self._confirm_callback = None
        self._skip_callback = None

    def has_request(self) -> bool:
        return self.current_request is not None and not self.current_request.handled

    def create_shan_request(
        self, player: "Player", source: "Player", card: "Card"
    ) -> ResponseRequest:
        shan_cards = [c for c in player.hand_cards if c.name == "闪"]

        request = ResponseRequest(
            request_type=ResponseRequest.TYPE_SHAN,
            prompt=f"{source.commander_name} 对你使用了 {card.name}，是否使用闪？",
            player=player,
            source=source,
            card=card,
            required=False,
        )

        self.available_cards = shan_cards
        self.current_request = request
        self.selected_card = None

        return request

    def create_tao_request(
        self, player: "Player", dying_player: "Player"
    ) -> ResponseRequest:
        tao_cards = [c for c in player.hand_cards if c.name == "桃"]
        jiu_cards = []
        if player == dying_player:
            jiu_cards = [c for c in player.hand_cards if c.name == "酒"]

        available = tao_cards + jiu_cards

        request = ResponseRequest(
            request_type=ResponseRequest.TYPE_TAO,
            prompt=f"{dying_player.commander_name} 濒死，是否使用桃/酒救援？",
            player=player,
            source=dying_player,
            required=False,
        )

        self.available_cards = available
        self.current_request = request
        self.selected_card = None

        return request

    def create_wuxie_request(
        self, player: "Player", target_card: "Card", source: "Player"
    ) -> ResponseRequest:
        wuxie_cards = [c for c in player.hand_cards if c.name == "无懈可击"]

        request = ResponseRequest(
            request_type=ResponseRequest.TYPE_WUXIE,
            prompt=f"{source.commander_name} 使用 {target_card.name}，是否无懈可击？",
            player=player,
            source=source,
            card=target_card,
            required=False,
        )

        self.available_cards = wuxie_cards
        self.current_request = request
        self.selected_card = None

        return request

    def select_card(self, card: "Card"):
        if card in self.available_cards:
            self.selected_card = card

    def confirm(self):
        if self.current_request:
            self.current_request.response = self.selected_card
            self.current_request.handled = True

    def skip(self):
        if self.current_request:
            self.current_request.response = None
            self.current_request.handled = True

    def get_response(self) -> Optional["Card"]:
        if self.current_request and self.current_request.handled:
            return self.current_request.response
        return None

    def clear(self):
        self.current_request = None
        self.available_cards.clear()
        self.selected_card = None

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.has_request():
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, card in enumerate(self.available_cards):
                card_rect = self._get_card_rect(i)
                if card_rect.collidepoint(event.pos):
                    if self.selected_card == card:
                        self.selected_card = None
                    else:
                        self.selected_card = card
                    return True

            if "confirm" in self.response_buttons:
                if self.response_buttons["confirm"].collidepoint(event.pos):
                    self.confirm()
                    return True

            if "skip" in self.response_buttons:
                if self.response_buttons["skip"].collidepoint(event.pos):
                    self.skip()
                    return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and self.selected_card:
                self.confirm()
                return True
            elif event.key == pygame.K_ESCAPE:
                self.skip()
                return True
            elif pygame.K_1 <= event.key <= pygame.K_9:
                idx = event.key - pygame.K_1
                if idx < len(self.available_cards):
                    self.selected_card = self.available_cards[idx]
                    return True

        return False

    def _get_card_rect(self, index: int) -> pygame.Rect:
        from gui.assets import CARD_WIDTH, CARD_HEIGHT, CARD_SPACING

        total_width = (
            len(self.available_cards) * CARD_WIDTH
            + (len(self.available_cards) - 1) * CARD_SPACING
        )
        start_x = WINDOW_WIDTH // 2 - total_width // 2
        y = WINDOW_HEIGHT // 2 + 50

        x = start_x + index * (CARD_WIDTH + CARD_SPACING)
        return pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)

    def render(self, surface: pygame.Surface):
        if not self.has_request():
            return

        from gui.card_renderer import CardRenderer
        from gui.ui_elements import Button

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        surface.blit(overlay, (0, 0))

        font_large = fonts.get("large")
        font_medium = fonts.get("medium")

        prompt_surface = font_large.render(
            self.current_request.prompt, True, COLORS["text_highlight"]
        )
        prompt_rect = prompt_surface.get_rect(
            centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT // 2 - 100
        )
        surface.blit(prompt_surface, prompt_rect)

        if self.available_cards:
            hint = "点击选择卡牌，回车确认，ESC跳过"
            hint_surface = font_medium.render(hint, True, COLORS["text_normal"])
            hint_rect = hint_surface.get_rect(
                centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT // 2 - 50
            )
            surface.blit(hint_surface, hint_rect)

            from gui.assets import CARD_WIDTH, CARD_HEIGHT, CARD_SPACING

            total_width = (
                len(self.available_cards) * CARD_WIDTH
                + (len(self.available_cards) - 1) * CARD_SPACING
            )
            start_x = WINDOW_WIDTH // 2 - total_width // 2
            y = WINDOW_HEIGHT // 2 + 20

            self.response_buttons.clear()

            for i, card in enumerate(self.available_cards):
                x = start_x + i * (CARD_WIDTH + CARD_SPACING)

                card_surface = CardRenderer.create_card_surface(card)

                if self.selected_card == card:
                    pygame.draw.rect(
                        card_surface,
                        COLORS["card_selected"],
                        card_surface.get_rect(),
                        3,
                        border_radius=5,
                    )

                surface.blit(card_surface, (x, y))
                self.response_buttons[f"card_{i}"] = pygame.Rect(
                    x, y, CARD_WIDTH, CARD_HEIGHT
                )

            btn_y = y + CARD_HEIGHT + 30
            confirm_btn = Button("确认", WINDOW_WIDTH // 2 - 130, btn_y, 120, 40)
            skip_btn = Button("跳过", WINDOW_WIDTH // 2 + 10, btn_y, 120, 40)

            confirm_btn.enabled = self.selected_card is not None

            confirm_btn.render(surface)
            skip_btn.render(surface)

            self.response_buttons["confirm"] = confirm_btn.rect
            self.response_buttons["skip"] = skip_btn.rect
        else:
            hint = "没有可用的卡牌，按ESC或点击跳过"
            hint_surface = font_medium.render(hint, True, COLORS["text_normal"])
            hint_rect = hint_surface.get_rect(
                centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT // 2
            )
            surface.blit(hint_surface, hint_rect)

            skip_btn = Button(
                "跳过", WINDOW_WIDTH // 2 - 60, WINDOW_HEIGHT // 2 + 50, 120, 40
            )
            skip_btn.render(surface)
            self.response_buttons["skip"] = skip_btn.rect
