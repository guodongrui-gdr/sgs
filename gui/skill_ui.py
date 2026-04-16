import pygame
from typing import Optional, List, Callable, TYPE_CHECKING
from gui.assets import WINDOW_WIDTH, WINDOW_HEIGHT, COLORS, fonts

if TYPE_CHECKING:
    from skills.base import Skill
    from player.player import Player
    from engine.game_engine import GameEngine


class SkillTriggerUI:
    def __init__(self):
        self.visible = False
        self.skill: Optional["Skill"] = None
        self.player: Optional["Player"] = None
        self.message: str = ""
        self.options: List[str] = []
        self.on_confirm: Optional[Callable] = None
        self.on_cancel: Optional[Callable] = None

        self.selected_option: int = -1
        self.rect = pygame.Rect(0, 0, 400, 200)
        self.rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

    def show(
        self,
        skill: "Skill",
        player: "Player",
        message: str,
        options: List[str] = None,
        on_confirm: Callable = None,
        on_cancel: Callable = None,
    ):
        self.skill = skill
        self.player = player
        self.message = message
        self.options = options or ["确认", "取消"]
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.selected_option = -1
        self.visible = True

        height = 150 + len(self.options) * 40
        self.rect = pygame.Rect(0, 0, 400, height)
        self.rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

    def hide(self):
        self.visible = False
        self.skill = None
        self.player = None

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            option_y = self.rect.y + 100
            for i, option in enumerate(self.options):
                btn_rect = pygame.Rect(
                    self.rect.centerx - 80, option_y + i * 45, 160, 35
                )
                if btn_rect.collidepoint(mouse_pos):
                    self.selected_option = i
                    if i == 0 and self.on_confirm:
                        self.on_confirm()
                    elif i == 1 and self.on_cancel:
                        self.on_cancel()
                    self.hide()
                    return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_y:
                if self.on_confirm:
                    self.on_confirm()
                self.hide()
                return True
            elif event.key == pygame.K_ESCAPE or event.key == pygame.K_n:
                if self.on_cancel:
                    self.on_cancel()
                self.hide()
                return True

        return False

    def render(self, surface: pygame.Surface):
        if not self.visible:
            return

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        surface.blit(overlay, (0, 0))

        pygame.draw.rect(surface, COLORS["panel_bg"], self.rect, border_radius=10)
        pygame.draw.rect(
            surface, COLORS["text_highlight"], self.rect, 3, border_radius=10
        )

        font_large = fonts.get("large")
        font_medium = fonts.get("medium")

        if self.skill:
            skill_text = f"【{self.skill.name}】"
            skill_surface = font_large.render(
                skill_text, True, COLORS["text_highlight"]
            )
            skill_rect = skill_surface.get_rect(
                centerx=self.rect.centerx, y=self.rect.y + 15
            )
            surface.blit(skill_surface, skill_rect)

        lines = self._wrap_text(self.message, font_medium, self.rect.width - 40)
        for i, line in enumerate(lines):
            msg_surface = font_medium.render(line, True, COLORS["text_normal"])
            msg_rect = msg_surface.get_rect(
                centerx=self.rect.centerx, y=self.rect.y + 55 + i * 25
            )
            surface.blit(msg_surface, msg_rect)

        option_y = self.rect.y + 100 + len(lines) * 10
        for i, option in enumerate(self.options):
            btn_rect = pygame.Rect(self.rect.centerx - 80, option_y + i * 45, 160, 35)

            mouse_pos = pygame.mouse.get_pos()
            hovered = btn_rect.collidepoint(mouse_pos)

            color = COLORS["button_hover"] if hovered else COLORS["button_normal"]
            pygame.draw.rect(surface, color, btn_rect, border_radius=5)
            pygame.draw.rect(
                surface, COLORS["text_normal"], btn_rect, 2, border_radius=5
            )

            opt_surface = font_medium.render(option, True, COLORS["button_text"])
            opt_rect = opt_surface.get_rect(center=btn_rect.center)
            surface.blit(opt_surface, opt_rect)

    def _wrap_text(
        self, text: str, font: pygame.font.Font, max_width: int
    ) -> List[str]:
        lines = []
        current_line = ""

        for char in text:
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


class GameLogPanel:
    def __init__(self, max_lines: int = 10):
        self.logs: List[str] = []
        self.max_lines = max_lines
        self.visible = True
        self.scroll_offset = 0
        self.rect = pygame.Rect(10, 100, 350, 250)

    def add_log(self, message: str):
        self.logs.append(message)
        if len(self.logs) > 100:
            self.logs.pop(0)
        self.scroll_offset = max(0, len(self.logs) - self.max_lines)

    def clear(self):
        self.logs.clear()
        self.scroll_offset = 0

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        if event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                self.scroll_offset = max(
                    0,
                    min(len(self.logs) - self.max_lines, self.scroll_offset - event.y),
                )
                return True

        return False

    def render(self, surface: pygame.Surface):
        if not self.visible or not self.logs:
            return

        panel_surface = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        pygame.draw.rect(
            panel_surface,
            (*COLORS["panel_bg"], 180),
            (0, 0, self.rect.width, self.rect.height),
            border_radius=5,
        )

        font = fonts.get("small")
        line_height = 22

        visible_logs = self.logs[
            self.scroll_offset : self.scroll_offset + self.max_lines
        ]

        for i, log in enumerate(visible_logs):
            log_surface = font.render(log, True, COLORS["text_normal"])
            panel_surface.blit(log_surface, (10, 5 + i * line_height))

        if len(self.logs) > self.max_lines:
            scroll_height = self.rect.height - 10
            scroll_ratio = self.scroll_offset / max(1, len(self.logs) - self.max_lines)
            scrollbar_y = int(scroll_ratio * (scroll_height - 30))

            pygame.draw.rect(
                panel_surface,
                (*COLORS["text_normal"], 100),
                (self.rect.width - 8, scrollbar_y + 5, 4, 30),
                border_radius=2,
            )

        surface.blit(panel_surface, self.rect.topleft)


class SettingsPanel:
    def __init__(self):
        self.visible = False
        self.settings = {
            "sound": True,
            "volume": 0.7,
            "show_log": True,
            "animation_speed": 1.0,
            "ai_type": "heuristic",  # "heuristic" or "mappo"
            "mappo_model_path": "",  # Path to MAPPO model
        }
        self.rect = pygame.Rect(0, 0, 500, 480)
        self.rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

        self._sliders = {}

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def toggle(self):
        self.visible = not self.visible

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            close_btn = pygame.Rect(self.rect.right - 40, self.rect.y + 10, 30, 30)
            if close_btn.collidepoint(mouse_pos):
                self.hide()
                return True

            sound_btn = pygame.Rect(self.rect.x + 30, self.rect.y + 100, 200, 40)
            if sound_btn.collidepoint(mouse_pos):
                self.settings["sound"] = not self.settings["sound"]
                return True

            log_btn = pygame.Rect(self.rect.x + 30, self.rect.y + 160, 200, 40)
            if log_btn.collidepoint(mouse_pos):
                self.settings["show_log"] = not self.settings["show_log"]
                return True

            ai_btn = pygame.Rect(self.rect.x + 30, self.rect.y + 220, 200, 40)
            if ai_btn.collidepoint(mouse_pos):
                current = self.settings["ai_type"]
                self.settings["ai_type"] = (
                    "mappo" if current == "heuristic" else "heuristic"
                )
                return True

        elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
            mouse_pos = event.pos

            volume_slider = pygame.Rect(self.rect.x + 30, self.rect.y + 230, 440, 20)
            if volume_slider.collidepoint(mouse_pos):
                self.settings["volume"] = (
                    mouse_pos[0] - volume_slider.x
                ) / volume_slider.width
                return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.hide()
                return True

        return False

    def render(self, surface: pygame.Surface):
        if not self.visible:
            return

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))

        pygame.draw.rect(surface, COLORS["panel_bg"], self.rect, border_radius=15)
        pygame.draw.rect(
            surface, COLORS["text_highlight"], self.rect, 3, border_radius=15
        )

        font_large = fonts.get("large")
        font_medium = fonts.get("medium")

        title = font_large.render("设置", True, COLORS["text_highlight"])
        title_rect = title.get_rect(centerx=self.rect.centerx, y=self.rect.y + 20)
        surface.blit(title, title_rect)

        close_btn = pygame.Rect(self.rect.right - 40, self.rect.y + 10, 30, 30)
        pygame.draw.rect(surface, COLORS["red"], close_btn, border_radius=5)
        x_surface = font_medium.render("X", True, COLORS["white"])
        x_rect = x_surface.get_rect(center=close_btn.center)
        surface.blit(x_surface, x_rect)

        sound_label = font_medium.render("音效", True, COLORS["text_normal"])
        surface.blit(sound_label, (self.rect.x + 30, self.rect.y + 75))

        sound_btn = pygame.Rect(self.rect.x + 30, self.rect.y + 100, 200, 40)
        sound_color = (
            COLORS["button_hover"] if self.settings["sound"] else COLORS["hp_empty"]
        )
        pygame.draw.rect(surface, sound_color, sound_btn, border_radius=5)
        sound_text = "开启" if self.settings["sound"] else "关闭"
        sound_surface = font_medium.render(sound_text, True, COLORS["button_text"])
        surface.blit(sound_surface, sound_surface.get_rect(center=sound_btn.center))

        log_label = font_medium.render("游戏日志", True, COLORS["text_normal"])
        surface.blit(log_label, (self.rect.x + 30, self.rect.y + 145))

        log_btn = pygame.Rect(self.rect.x + 30, self.rect.y + 170, 200, 40)
        log_color = (
            COLORS["button_hover"] if self.settings["show_log"] else COLORS["hp_empty"]
        )
        pygame.draw.rect(surface, log_color, log_btn, border_radius=5)
        log_text = "显示" if self.settings["show_log"] else "隐藏"
        log_surface = font_medium.render(log_text, True, COLORS["button_text"])
        surface.blit(log_surface, log_surface.get_rect(center=log_btn.center))

        ai_label = font_medium.render("AI 类型", True, COLORS["text_normal"])
        surface.blit(ai_label, (self.rect.x + 30, self.rect.y + 215))

        ai_btn = pygame.Rect(self.rect.x + 30, self.rect.y + 240, 200, 40)
        ai_color = (
            COLORS["button_hover"]
            if self.settings["ai_type"] == "mappo"
            else COLORS["hp_empty"]
        )
        pygame.draw.rect(surface, ai_color, ai_btn, border_radius=5)
        ai_text = "MAPPO" if self.settings["ai_type"] == "mappo" else "规则AI"
        ai_surface = font_medium.render(ai_text, True, COLORS["button_text"])
        surface.blit(ai_surface, ai_surface.get_rect(center=ai_btn.center))

        volume_label = font_medium.render(
            f"音量: {int(self.settings['volume'] * 100)}%", True, COLORS["text_normal"]
        )
        surface.blit(volume_label, (self.rect.x + 30, self.rect.y + 300))

        volume_slider = pygame.Rect(self.rect.x + 30, self.rect.y + 340, 440, 20)
        pygame.draw.rect(surface, COLORS["hp_empty"], volume_slider, border_radius=5)

        filled_width = int(volume_slider.width * self.settings["volume"])
        filled_rect = pygame.Rect(
            volume_slider.x, volume_slider.y, filled_width, volume_slider.height
        )
        pygame.draw.rect(surface, COLORS["button_hover"], filled_rect, border_radius=5)

        pygame.draw.rect(
            surface, COLORS["text_normal"], volume_slider, 2, border_radius=5
        )

    def get(self, key: str, default=None):
        return self.settings.get(key, default)
