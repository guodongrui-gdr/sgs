import pygame
from typing import Optional, Tuple, List, Callable
from gui.assets import COLORS, fonts, WINDOW_WIDTH, WINDOW_HEIGHT


class Button:
    def __init__(
        self,
        text: str,
        x: int,
        y: int,
        width: int = 120,
        height: int = 40,
        callback: Optional[Callable] = None,
    ):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.callback = callback
        self.hovered = False
        self.enabled = True

    def render(self, surface: pygame.Surface) -> pygame.Rect:
        color = COLORS["button_hover"] if self.hovered else COLORS["button_normal"]
        if not self.enabled:
            color = COLORS["hp_empty"]

        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, COLORS["text_normal"], self.rect, 2, border_radius=8)

        font = fonts.get("button")
        text_color = COLORS["button_text"] if self.enabled else COLORS["hp_lost"]
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

        return self.rect

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.enabled and self.callback:
                self.callback()
                return True
        return False


class Dialog:
    def __init__(
        self, title: str, message: str, buttons: List[Tuple[str, Callable]] = None
    ):
        self.title = title
        self.message = message
        self.visible = False

        width = 400
        height = 200
        x = (WINDOW_WIDTH - width) // 2
        y = (WINDOW_HEIGHT - height) // 2
        self.rect = pygame.Rect(x, y, width, height)

        self.buttons: List[Button] = []
        if buttons:
            btn_width = 100
            btn_height = 35
            btn_spacing = 20
            total_width = len(buttons) * btn_width + (len(buttons) - 1) * btn_spacing
            start_x = x + (width - total_width) // 2

            for i, (btn_text, callback) in enumerate(buttons):
                btn_x = start_x + i * (btn_width + btn_spacing)
                btn_y = y + height - 55
                self.buttons.append(
                    Button(btn_text, btn_x, btn_y, btn_width, btn_height, callback)
                )

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def render(self, surface: pygame.Surface):
        if not self.visible:
            return

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        surface.blit(overlay, (0, 0))

        pygame.draw.rect(surface, COLORS["panel_bg"], self.rect, border_radius=10)
        pygame.draw.rect(surface, COLORS["text_normal"], self.rect, 2, border_radius=10)

        font_title = fonts.get("large")
        title_surface = font_title.render(self.title, True, COLORS["text_highlight"])
        title_rect = title_surface.get_rect(
            centerx=self.rect.centerx, y=self.rect.y + 20
        )
        surface.blit(title_surface, title_rect)

        font_msg = fonts.get("medium")
        lines = self._wrap_text(self.message, font_msg, self.rect.width - 40)
        for i, line in enumerate(lines):
            msg_surface = font_msg.render(line, True, COLORS["text_normal"])
            msg_rect = msg_surface.get_rect(
                centerx=self.rect.centerx, y=self.rect.y + 60 + i * 25
            )
            surface.blit(msg_surface, msg_rect)

        for button in self.buttons:
            button.render(surface)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        for button in self.buttons:
            if button.handle_event(event):
                return True

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.hide()
            return True

        return False

    def _wrap_text(
        self, text: str, font: pygame.font.Font, max_width: int
    ) -> List[str]:
        lines = []
        words = text
        current_line = ""

        for char in words:
            test_line = current_line + char
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)

        return lines


class TargetSelectPanel:
    def __init__(self, title: str = "选择目标"):
        self.title = title
        self.visible = False
        self.targets: List = []
        self.selected: List = []
        self.min_select: int = 1
        self.max_select: int = 1
        self.on_confirm: Optional[Callable] = None
        self.on_cancel: Optional[Callable] = None

        self._confirm_btn = Button("确认", 0, 0, 100, 35)
        self._cancel_btn = Button("取消", 0, 0, 100, 35)

    def show(
        self,
        targets: List,
        min_select: int = 1,
        max_select: int = 1,
        on_confirm: Callable = None,
        on_cancel: Callable = None,
    ):
        self.targets = targets
        self.selected = []
        self.min_select = min_select
        self.max_select = max_select
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.visible = True

    def hide(self):
        self.visible = False

    def toggle_target(self, target):
        if target in self.selected:
            self.selected.remove(target)
        elif len(self.selected) < self.max_select:
            self.selected.append(target)

    def can_confirm(self) -> bool:
        return self.min_select <= len(self.selected) <= self.max_select

    def confirm(self):
        if self.can_confirm() and self.on_confirm:
            self.on_confirm(self.selected)
        self.hide()

    def cancel(self):
        if self.on_cancel:
            self.on_cancel()
        self.hide()

    def render(
        self, surface: pygame.Surface, target_rects: List[Tuple[pygame.Rect, object]]
    ):
        if not self.visible:
            return

        font = fonts.get("medium")
        title_surface = font.render(self.title, True, COLORS["text_highlight"])
        title_rect = title_surface.get_rect(
            centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT // 2 - 50
        )
        surface.blit(title_surface, title_rect)

        hint = f"已选择 {len(self.selected)}/{self.max_select} 个目标"
        hint_surface = font.render(hint, True, COLORS["text_normal"])
        hint_rect = hint_surface.get_rect(
            centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT // 2 - 20
        )
        surface.blit(hint_surface, hint_rect)

        btn_y = WINDOW_HEIGHT // 2 + 20
        self._confirm_btn.rect.centerx = WINDOW_WIDTH // 2 - 70
        self._confirm_btn.rect.y = btn_y
        self._confirm_btn.enabled = self.can_confirm()

        self._cancel_btn.rect.centerx = WINDOW_WIDTH // 2 + 70
        self._cancel_btn.rect.y = btn_y

        self._confirm_btn.render(surface)
        self._cancel_btn.render(surface)

    def handle_event(
        self, event: pygame.event.Event, target_rects: List[Tuple[pygame.Rect, object]]
    ) -> bool:
        if not self.visible:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, target in target_rects:
                if rect.collidepoint(event.pos) and target in self.targets:
                    self.toggle_target(target)
                    return True

            if self._confirm_btn.rect.collidepoint(event.pos):
                self.confirm()
                return True

            if self._cancel_btn.rect.collidepoint(event.pos):
                self.cancel()
                return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.cancel()
                return True
            elif event.key == pygame.K_RETURN and self.can_confirm():
                self.confirm()
                return True

        return False


class MessageBox:
    def __init__(self):
        self.message = ""
        self.visible = False
        self.duration = 0
        self.start_time = 0

    def show(self, message: str, duration: int = 2000):
        self.message = message
        self.duration = duration
        self.visible = True
        self.start_time = pygame.time.get_ticks()

    def update(self) -> bool:
        if not self.visible:
            return False

        elapsed = pygame.time.get_ticks() - self.start_time
        if elapsed >= self.duration:
            self.visible = False
            return True
        return False

    def render(self, surface: pygame.Surface):
        if not self.visible:
            return

        font = fonts.get("large")
        text_surface = font.render(self.message, True, COLORS["text_highlight"])
        text_rect = text_surface.get_rect(
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100)
        )

        bg_rect = text_rect.inflate(40, 20)
        pygame.draw.rect(surface, (*COLORS["panel_bg"], 230), bg_rect, border_radius=10)
        pygame.draw.rect(
            surface, COLORS["text_highlight"], bg_rect, 2, border_radius=10
        )
        surface.blit(text_surface, text_rect)


class CardInfoPanel:
    def __init__(self):
        self.card = None
        self.visible = False

    def show(self, card):
        self.card = card
        self.visible = True

    def hide(self):
        self.visible = False

    def render(self, surface: pygame.Surface, x: int, y: int):
        if not self.visible or not self.card:
            return

        width = 200
        height = 120

        panel_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(
            surface, (*COLORS["panel_bg"], 240), panel_rect, border_radius=8
        )
        pygame.draw.rect(surface, COLORS["text_normal"], panel_rect, 2, border_radius=8)

        font = fonts.get("medium")
        name_surface = font.render(self.card.name, True, COLORS["text_highlight"])
        surface.blit(name_surface, (x + 10, y + 10))

        type_text = f"类型: {self.card.card_type}"
        type_surface = font.render(type_text, True, COLORS["text_normal"])
        surface.blit(type_surface, (x + 10, y + 40))

        color_text = f"花色: {self.card.color} {self.card.point}"
        color_surface = font.render(color_text, True, COLORS["text_normal"])
        surface.blit(color_surface, (x + 10, y + 65))

        if self.card.card_type == "WeaponCard":
            range_text = f"攻击范围: {getattr(self.card, 'distance', 1)}"
            range_surface = font.render(range_text, True, COLORS["text_normal"])
            surface.blit(range_surface, (x + 10, y + 90))
