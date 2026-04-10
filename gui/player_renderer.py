import pygame
from typing import Optional, List, Tuple, TYPE_CHECKING
from gui.assets import (
    COLORS,
    NATION_COLORS,
    IDENTITY_COLORS,
    fonts,
    PLAYER_AREA_WIDTH,
    PLAYER_AREA_HEIGHT,
)
from gui.card_renderer import CardRenderer

if TYPE_CHECKING:
    from player.player import Player
    from engine.game_engine import GameEngine


class PlayerRenderer:
    @staticmethod
    def render_player_area(
        player: "Player",
        x: int,
        y: int,
        is_current: bool = False,
        show_hand: bool = False,
        show_identity: bool = False,
        selectable: bool = False,
        selected: bool = False,
    ) -> Tuple[pygame.Surface, pygame.Rect]:
        surface = pygame.Surface(
            (PLAYER_AREA_WIDTH, PLAYER_AREA_HEIGHT), pygame.SRCALPHA
        )

        nation_color = NATION_COLORS.get(player.nation, COLORS["qun"])

        border_width = 3 if is_current else 2
        border_color = COLORS["text_highlight"] if is_current else nation_color

        if selected:
            border_color = COLORS["card_selected"]
            border_width = 4

        pygame.draw.rect(
            surface,
            (*COLORS["panel_bg"], 200),
            (0, 0, PLAYER_AREA_WIDTH, PLAYER_AREA_HEIGHT),
            border_radius=8,
        )
        pygame.draw.rect(
            surface,
            border_color,
            (0, 0, PLAYER_AREA_WIDTH, PLAYER_AREA_HEIGHT),
            border_width,
            border_radius=8,
        )

        font_name = fonts.get("player_name")
        font_small = fonts.get("small")
        font_hp = fonts.get("hp")

        name_color = COLORS["text_highlight"] if is_current else COLORS["text_normal"]
        name_surface = font_name.render(player.commander_name, True, name_color)
        surface.blit(name_surface, (10, 8))

        nation_surface = font_small.render(f"[{player.nation}]", True, nation_color)
        surface.blit(nation_surface, (PLAYER_AREA_WIDTH - 45, 8))

        hp_y = 32
        hp_x = 10
        for i in range(player.max_hp):
            if i < player.current_hp:
                hp_color = COLORS["hp_full"]
                hp_text = "♥"
            else:
                hp_color = COLORS["hp_lost"]
                hp_text = "♡"
            hp_surface = font_hp.render(hp_text, True, hp_color)
            surface.blit(hp_surface, (hp_x + i * 18, hp_y))

        identity_y = 52
        if show_identity or player.identity == "主公":
            identity_color = IDENTITY_COLORS.get(player.identity, COLORS["text_normal"])
            identity_surface = font_small.render(
                f"[{player.identity}]", True, identity_color
            )
            surface.blit(identity_surface, (10, identity_y))
        else:
            identity_surface = font_small.render("[???]", True, COLORS["hp_empty"])
            surface.blit(identity_surface, (10, identity_y))

        status_y = 72
        status_parts = []

        if not player.is_alive:
            status_parts.append("已阵亡")
        if player.is_chained:
            status_parts.append("连环")
        if player.judge_area:
            status_parts.append(f"判定×{len(player.judge_area)}")

        if status_parts:
            status_text = " ".join(status_parts)
            status_surface = font_small.render(status_text, True, COLORS["red"])
            surface.blit(status_surface, (10, status_y))

        equip_y = 90
        equip_parts = []
        if player.equipment.get("武器"):
            equip_parts.append(f"武:{player.equipment['武器'].name[:2]}")
        if player.equipment.get("防具"):
            equip_parts.append(f"防:{player.equipment['防具'].name[:2]}")
        if player.equipment.get("进攻坐骑"):
            equip_parts.append("攻马")
        if player.equipment.get("防御坐骑"):
            equip_parts.append("防马")

        if equip_parts:
            equip_text = " ".join(equip_parts)
            equip_surface = font_small.render(equip_text, True, COLORS["text_normal"])
            surface.blit(equip_surface, (10, equip_y))

        if not player.is_human:
            hand_text = f"手牌:{len(player.hand_cards)}"
            hand_surface = font_small.render(hand_text, True, COLORS["text_normal"])
            surface.blit(hand_surface, (PLAYER_AREA_WIDTH - 60, status_y))

        rect = surface.get_rect(topleft=(x, y))
        return surface, rect

    @staticmethod
    def calculate_player_positions(
        player_count: int, screen_width: int, screen_height: int, margin: int = 100
    ) -> List[Tuple[int, int]]:
        positions = []

        center_x = screen_width // 2
        center_y = (screen_height - 150) // 2

        if player_count <= 2:
            radius_x = 300
            radius_y = 200
        else:
            radius_x = min(600, (screen_width - 2 * margin - PLAYER_AREA_WIDTH) // 2)
            radius_y = min(300, (screen_height - 200 - PLAYER_AREA_HEIGHT) // 2)

        for i in range(player_count):
            angle = (2 * 3.14159 * i / player_count) - 3.14159 / 2

            x = (
                center_x + int(radius_x * 0.8 * (i / (player_count - 1) * 2 - 1))
                if player_count > 1
                else center_x
            )
            y = (
                center_y
                + int(radius_y * 0.6 * (1 - abs(i / (player_count - 1) * 2 - 1)))
                if player_count > 1
                else center_y - 100
            )

            if player_count > 2:
                angle = (2 * 3.14159 * i / player_count) - 3.14159 / 2
                x = center_x + int(radius_x * 0.85 * -1 * (3.14159 / 2 - abs(angle)))
                y = center_y + int(
                    radius_y
                    * 0.7
                    * (i % 2 * 2 - 1)
                    * (0.5 + 0.5 * abs(angle) / (3.14159 / 2))
                )

            x = max(
                margin,
                min(
                    screen_width - PLAYER_AREA_WIDTH - margin,
                    x - PLAYER_AREA_WIDTH // 2,
                ),
            )
            y = max(
                margin,
                min(
                    screen_height - PLAYER_AREA_HEIGHT - margin,
                    y - PLAYER_AREA_HEIGHT // 2,
                ),
            )

            positions.append((x, y))

        return positions
