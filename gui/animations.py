import pygame
import math
from typing import Optional, List, Tuple, Callable
from gui.assets import COLORS, WINDOW_WIDTH, WINDOW_HEIGHT, CARD_WIDTH, CARD_HEIGHT


class Animation:
    def __init__(self, duration: int = 500):
        self.duration = duration
        self.start_time = 0
        self.running = False
        self.finished = False

    def start(self):
        self.start_time = pygame.time.get_ticks()
        self.running = True
        self.finished = False

    def update(self) -> bool:
        if not self.running:
            return False

        elapsed = pygame.time.get_ticks() - self.start_time
        if elapsed >= self.duration:
            self.running = False
            self.finished = True
            return True

        return False

    def get_progress(self) -> float:
        if not self.running:
            return 1.0 if self.finished else 0.0
        elapsed = pygame.time.get_ticks() - self.start_time
        return min(1.0, elapsed / self.duration)

    def render(self, surface: pygame.Surface):
        pass


class CardMoveAnimation(Animation):
    def __init__(
        self,
        card_surface: pygame.Surface,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        duration: int = 300,
    ):
        super().__init__(duration)
        self.card_surface = card_surface
        self.start_pos = start_pos
        self.end_pos = end_pos

    def render(self, surface: pygame.Surface):
        if not self.running:
            return

        progress = self.get_progress()
        eased = self._ease_out_quad(progress)

        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * eased
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * eased

        surface.blit(self.card_surface, (int(x), int(y)))

    def _ease_out_quad(self, t: float) -> float:
        return 1 - (1 - t) * (1 - t)


class DealCardAnimation(Animation):
    def __init__(
        self, num_cards: int, target_pos: Tuple[int, int], on_complete: Callable = None
    ):
        super().__init__(duration=500)
        self.num_cards = num_cards
        self.target_pos = target_pos
        self.on_complete = on_complete
        self.card_rects = []

        deck_x = WINDOW_WIDTH - 100
        deck_y = 80

        for i in range(num_cards):
            offset = i * (CARD_WIDTH + 5)
            self.card_rects.append(
                {
                    "start": (deck_x, deck_y),
                    "end": (target_pos[0] + offset, target_pos[1]),
                }
            )

    def render(self, surface: pygame.Surface):
        if not self.running:
            return

        progress = self.get_progress()

        for i, rect_data in enumerate(self.card_rects):
            delay = i * 0.1
            card_progress = max(0, min(1, (progress - delay) / 0.5))

            if card_progress > 0:
                eased = self._ease_out_quad(card_progress)
                x = (
                    rect_data["start"][0]
                    + (rect_data["end"][0] - rect_data["start"][0]) * eased
                )
                y = (
                    rect_data["start"][1]
                    + (rect_data["end"][1] - rect_data["start"][1]) * eased
                )

                card_surf = pygame.Surface((CARD_WIDTH, CARD_HEIGHT), pygame.SRCALPHA)
                pygame.draw.rect(
                    card_surf,
                    (60, 80, 60),
                    (0, 0, CARD_WIDTH, CARD_HEIGHT),
                    border_radius=5,
                )
                pygame.draw.rect(
                    card_surf,
                    COLORS["black"],
                    (0, 0, CARD_WIDTH, CARD_HEIGHT),
                    2,
                    border_radius=5,
                )
                surface.blit(card_surf, (int(x), int(y)))

    def _ease_out_quad(self, t: float) -> float:
        return 1 - (1 - t) * (1 - t)


class DamageAnimation(Animation):
    def __init__(
        self,
        target_pos: Tuple[int, int],
        damage: int,
        is_fire: bool = False,
        is_thunder: bool = False,
    ):
        super().__init__(duration=800)
        self.target_pos = target_pos
        self.damage = damage
        self.is_fire = is_fire
        self.is_thunder = is_thunder

        if is_fire:
            self.color = COLORS["fire"]
        elif is_thunder:
            self.color = COLORS["thunder"]
        else:
            self.color = COLORS["red"]

    def render(self, surface: pygame.Surface):
        if not self.running:
            return

        progress = self.get_progress()

        if progress < 0.5:
            scale = 1 + 0.5 * math.sin(progress * math.pi)
            alpha = 255
        else:
            scale = 1
            alpha = int(255 * (1 - (progress - 0.5) * 2))

        font = pygame.font.Font(None, 72)
        text = f"-{self.damage}"
        text_surface = font.render(text, True, self.color)

        scaled_width = int(text_surface.get_width() * scale)
        scaled_height = int(text_surface.get_height() * scale)
        scaled_surface = pygame.transform.scale(
            text_surface, (scaled_width, scaled_height)
        )

        scaled_surface.set_alpha(alpha)

        x = self.target_pos[0] - scaled_width // 2
        y = self.target_pos[1] - scaled_height // 2 - int(50 * progress)

        surface.blit(scaled_surface, (x, y))


class HealAnimation(Animation):
    def __init__(self, target_pos: Tuple[int, int], amount: int = 1):
        super().__init__(duration=600)
        self.target_pos = target_pos
        self.amount = amount

    def render(self, surface: pygame.Surface):
        if not self.running:
            return

        progress = self.get_progress()

        alpha = int(255 * (1 - progress))
        y_offset = int(-80 * progress)

        font = pygame.font.Font(None, 48)
        text = f"+{self.amount}"
        text_surface = font.render(text, True, COLORS["hp_full"])
        text_surface.set_alpha(alpha)

        x = self.target_pos[0] - text_surface.get_width() // 2
        y = self.target_pos[1] - text_surface.get_height() // 2 + y_offset

        surface.blit(text_surface, (x, y))


class TextFloatAnimation(Animation):
    def __init__(
        self,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int] = None,
        duration: int = 1000,
    ):
        super().__init__(duration)
        self.text = text
        self.pos = pos
        self.color = color or COLORS["text_highlight"]

    def render(self, surface: pygame.Surface):
        if not self.running:
            return

        progress = self.get_progress()
        alpha = int(255 * (1 - progress))
        y_offset = int(-100 * progress)

        font = pygame.font.Font(None, 36)
        text_surface = font.render(self.text, True, self.color)
        text_surface.set_alpha(alpha)

        x = self.pos[0] - text_surface.get_width() // 2
        y = self.pos[1] - text_surface.get_height() // 2 + y_offset

        surface.blit(text_surface, (x, y))


class AnimationManager:
    def __init__(self):
        self.animations: List[Animation] = []

    def add(self, animation: Animation):
        animation.start()
        self.animations.append(animation)

    def update(self):
        finished = []
        for anim in self.animations:
            if anim.update():
                finished.append(anim)

        for anim in finished:
            self.animations.remove(anim)

    def render(self, surface: pygame.Surface):
        for anim in self.animations:
            anim.render(surface)

    def clear(self):
        self.animations.clear()

    def is_empty(self) -> bool:
        return len(self.animations) == 0
