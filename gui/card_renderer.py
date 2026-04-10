import pygame
from typing import Optional, Tuple, List
from gui.assets import (
    CARD_WIDTH,
    CARD_HEIGHT,
    COLORS,
    SUITS,
    SUIT_COLORS,
    CARD_TYPE_COLORS,
    POINT_NAMES,
    fonts,
)
from card.base import Card


class CardRenderer:
    @staticmethod
    def get_card_background_color(card: Card) -> Tuple[int, int, int]:
        return CARD_TYPE_COLORS.get(card.card_type, COLORS["card_bg"])

    @staticmethod
    def get_point_display(point: int) -> str:
        if point in POINT_NAMES:
            return POINT_NAMES[point]
        return str(point)

    @staticmethod
    def render_card(
        card: Card,
        x: int,
        y: int,
        selected: bool = False,
        hovered: bool = False,
        scale: float = 1.0,
    ) -> pygame.Rect:
        surface = CardRenderer.create_card_surface(card, scale)

        if selected:
            y += CARD_SELECTED_OFFSET

        rect = surface.get_rect(topleft=(x, y))

        if hovered and not selected:
            highlight = pygame.Surface(
                (rect.width + 4, rect.height + 4), pygame.SRCALPHA
            )
            highlight.fill((*COLORS["card_hover"], 100))
            pygame.draw.rect(
                highlight,
                COLORS["text_highlight"],
                highlight.get_rect(),
                2,
                border_radius=5,
            )

        return surface, rect

    @staticmethod
    def create_card_surface(card: Card, scale: float = 1.0) -> pygame.Surface:
        width = int(CARD_WIDTH * scale)
        height = int(CARD_HEIGHT * scale)
        surface = pygame.Surface((width, height), pygame.SRCALPHA)

        bg_color = CardRenderer.get_card_background_color(card)
        pygame.draw.rect(surface, bg_color, (0, 0, width, height), border_radius=5)
        pygame.draw.rect(
            surface, COLORS["black"], (0, 0, width, height), 2, border_radius=5
        )

        suit = card.color
        point = card.point
        name = card.name

        suit_symbol = SUITS.get(suit, "?")
        suit_color = SUIT_COLORS.get(suit, COLORS["black"])
        point_str = CardRenderer.get_point_display(point)

        font_small = fonts.get("card_point")
        font_title = fonts.get("card_title")

        suit_point_text = f"{suit_symbol}{point_str}"
        suit_point_surface = font_small.render(suit_point_text, True, suit_color)
        surface.blit(suit_point_surface, (5, 5))

        name_surface = font_title.render(name, True, COLORS["black"])
        name_rect = name_surface.get_rect(center=(width // 2, height // 2))
        surface.blit(name_surface, name_rect)

        if card.card_type == "WeaponCard":
            range_text = f"范围:{getattr(card, 'distance', 1)}"
            range_surface = font_small.render(range_text, True, COLORS["black"])
            surface.blit(range_surface, (5, height - 20))
        elif "杀" in name:
            if getattr(card, "is_fire", False):
                pygame.draw.circle(surface, COLORS["fire"], (width - 15, 15), 8)
            elif getattr(card, "is_thunder", False):
                pygame.draw.circle(surface, COLORS["thunder"], (width - 15, 15), 8)

        return surface

    @staticmethod
    def render_card_back(
        x: int, y: int, scale: float = 1.0
    ) -> Tuple[pygame.Surface, pygame.Rect]:
        width = int(CARD_WIDTH * scale)
        height = int(CARD_HEIGHT * scale)
        surface = pygame.Surface((width, height), pygame.SRCALPHA)

        pygame.draw.rect(surface, (60, 80, 60), (0, 0, width, height), border_radius=5)
        pygame.draw.rect(
            surface, (40, 60, 40), (3, 3, width - 6, height - 6), border_radius=4
        )
        pygame.draw.rect(
            surface, COLORS["black"], (0, 0, width, height), 2, border_radius=5
        )

        font = fonts.get("large")
        text = font.render("三", True, (100, 130, 100))
        text_rect = text.get_rect(center=(width // 2, height // 2 - 15))
        surface.blit(text, text_rect)

        text2 = font.render("杀", True, (100, 130, 100))
        text2_rect = text2.get_rect(center=(width // 2, height // 2 + 15))
        surface.blit(text2, text2_rect)

        rect = surface.get_rect(topleft=(x, y))
        return surface, rect

    @staticmethod
    def render_hand_cards(
        cards: List[Card],
        start_x: int,
        y: int,
        selected_indices: List[int] = None,
        hovered_index: int = -1,
    ) -> List[Tuple[pygame.Surface, pygame.Rect, int]]:
        if selected_indices is None:
            selected_indices = []

        result = []
        current_x = start_x

        for i, card in enumerate(cards):
            selected = i in selected_indices
            hovered = i == hovered_index

            surface = CardRenderer.create_card_surface(card)

            card_y = y
            if selected:
                card_y += CARD_SELECTED_OFFSET

            rect = surface.get_rect(topleft=(current_x, card_y))

            result.append((surface, rect, i))
            current_x += CARD_WIDTH + CARD_SPACING

        return result
