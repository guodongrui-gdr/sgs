import pygame
import sys
import random
import json
from typing import Optional, List, Callable
from pathlib import Path

from gui.assets import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    FPS,
    fonts,
    assets,
    PLAYER_AREA_WIDTH,
    PLAYER_AREA_HEIGHT,
    COLORS,
)
from gui.game_renderer import GameRenderer
from gui.input_handler import InputHandler
from gui.ui_elements import Button, Dialog, TargetSelectPanel, MessageBox
from gui.card_renderer import CardRenderer
from gui.player_renderer import PlayerRenderer
from gui.animations import (
    AnimationManager,
    DamageAnimation,
    HealAnimation,
    TextFloatAnimation,
    DealCardAnimation,
)
from gui.response_manager import ResponseManager, ResponseRequest
from gui.audio import audio
from gui.skill_ui import SkillTriggerUI, GameLogPanel, SettingsPanel

from engine.game_engine import GameEngine
from engine.state import GamePhase
from player.player import Player
from card.base import is_sha_card, Card
from skills.registry import SkillRegistry


class GameState:
    MENU = "menu"
    PLAYING = "playing"
    TARGET_SELECT = "target_select"
    RESPONSE = "response"
    DISCARD = "discard"
    SKILL_TRIGGER = "skill_trigger"
    SETTINGS = "settings"
    GAME_OVER = "game_over"


class GameWindow:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_caption("三国杀 - 人机对战版")

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        fonts.init()
        assets.init()
        audio.init()

        self.renderer = GameRenderer(self.screen)
        self.input_handler = InputHandler()
        self.animation_manager = AnimationManager()
        self.response_manager = ResponseManager()
        self.skill_ui = SkillTriggerUI()
        self.log_panel = GameLogPanel(max_lines=12)
        self.settings_panel = SettingsPanel()

        self.state = GameState.MENU
        self.engine: Optional[GameEngine] = None
        self.human_player: Optional[Player] = None

        self.selected_card_index: int = -1
        self.selected_targets: List[Player] = []
        self.valid_targets: List[Player] = []

        self.message_box = MessageBox()
        self.target_panel = TargetSelectPanel()

        self.player_num = 5
        self.ai_delay = 400

        self._setup_menu_buttons()
        self._setup_key_bindings()

    def _setup_menu_buttons(self):
        btn_width = 200
        btn_height = 50
        center_x = WINDOW_WIDTH // 2 - btn_width // 2
        start_y = 350

        self.menu_buttons = {
            "start": Button("开始游戏", center_x, start_y, btn_width, btn_height),
            "player_5": Button("5人", center_x - 160, start_y + 65, 70, 40),
            "player_6": Button("6人", center_x - 80, start_y + 65, 70, 40),
            "player_7": Button("7人", center_x, start_y + 65, 70, 40),
            "player_8": Button("8人", center_x + 80, start_y + 65, 70, 40),
            "quit": Button("退出游戏", center_x, start_y + 130, btn_width, btn_height),
        }

        self.game_buttons = {
            "end_turn": Button(
                "结束回合", WINDOW_WIDTH - 140, WINDOW_HEIGHT - 55, 120, 40
            ),
            "use_card": Button(
                "使用卡牌", WINDOW_WIDTH - 280, WINDOW_HEIGHT - 55, 120, 40
            ),
            "settings": Button("设置", WINDOW_WIDTH - 140, 20, 100, 35),
        }

    def _setup_key_bindings(self):
        self.key_actions = {
            pygame.K_ESCAPE: self._on_escape,
            pygame.K_F1: self._toggle_settings,
            pygame.K_SPACE: self._on_space,
        }

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key in self.key_actions:
                    self.key_actions[event.key]()
                    continue

            if self.settings_panel.visible:
                if self.settings_panel.handle_event(event):
                    continue

            if self.state == GameState.MENU:
                self._handle_menu_event(event)
            elif self.state == GameState.PLAYING:
                self._handle_game_event(event)
            elif self.state == GameState.TARGET_SELECT:
                self._handle_target_select_event(event)
            elif self.state == GameState.DISCARD:
                self._handle_discard_event(event)
            elif self.state == GameState.RESPONSE:
                self._handle_response_event(event)
            elif self.state == GameState.SKILL_TRIGGER:
                self._handle_skill_event(event)
            elif self.state == GameState.GAME_OVER:
                self._handle_game_over_event(event)

            if self.log_panel.handle_event(event):
                continue

    def _handle_menu_event(self, event: pygame.event.Event):
        for name, button in self.menu_buttons.items():
            if button.handle_event(event):
                audio.play("click")
                if name == "start":
                    self._start_game()
                elif name.startswith("player_"):
                    self.player_num = int(name.split("_")[1])
                elif name == "quit":
                    self.running = False

    def _handle_game_event(self, event: pygame.event.Event):
        current = self.engine.players[self.engine.current_player_idx]

        if current != self.human_player:
            return

        result = self.input_handler.handle_event(event, self.engine, self.human_player)

        if not result:
            return

        if result["type"] == "card_selected":
            self.selected_card_index = result["card_index"]

        elif result["type"] == "button_clicked":
            if result["button"] == "end_turn":
                self._end_turn()
            elif result["button"] == "use_card":
                self._try_use_card()
            elif result["button"] == "settings":
                self.settings_panel.show()

        elif result["type"] == "selection_cleared":
            self.selected_card_index = -1
            self.selected_targets.clear()

    def _handle_target_select_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, player in self._get_clickable_player_rects():
                if rect.collidepoint(event.pos) and player in self.valid_targets:
                    if player in self.selected_targets:
                        self.selected_targets.remove(player)
                    else:
                        if len(self.selected_targets) < self.target_panel.max_select:
                            self.selected_targets.append(player)
                    audio.play("click")
                    break

            confirm_rect = pygame.Rect(
                WINDOW_WIDTH // 2 - 130, WINDOW_HEIGHT // 2 + 50, 120, 40
            )
            cancel_rect = pygame.Rect(
                WINDOW_WIDTH // 2 + 10, WINDOW_HEIGHT // 2 + 50, 120, 40
            )

            if confirm_rect.collidepoint(event.pos) and self.target_panel.can_confirm():
                self.target_panel.selected = self.selected_targets.copy()
                self.target_panel.confirm()
            elif cancel_rect.collidepoint(event.pos):
                self.target_panel.cancel()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.target_panel.cancel()
            elif event.key == pygame.K_RETURN and self.target_panel.can_confirm():
                self.target_panel.selected = self.selected_targets.copy()
                self.target_panel.confirm()

    def _handle_discard_event(self, event: pygame.event.Event):
        result = self.input_handler.handle_event(event, self.engine, self.human_player)

        if result and result["type"] == "card_selected":
            need_discard = (
                len(self.human_player.hand_cards) - self.human_player.hand_limit
            )
            selected_count = len(result["selected_indices"])

            if selected_count == need_discard:
                self._execute_discard(result["selected_indices"])

    def _handle_response_event(self, event: pygame.event.Event):
        if self.response_manager.handle_event(event):
            response = self.response_manager.get_response()
            if response:
                audio.play("card_use")

    def _handle_skill_event(self, event: pygame.event.Event):
        if self.skill_ui.handle_event(event):
            audio.play("card_use")

    def _handle_game_over_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            self.state = GameState.MENU
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.state = GameState.MENU

    def _on_escape(self):
        if self.settings_panel.visible:
            self.settings_panel.hide()
        elif self.state == GameState.TARGET_SELECT:
            self.target_panel.cancel()
        elif self.state == GameState.PLAYING:
            self.input_handler.clear_selection()
            self.selected_card_index = -1

    def _toggle_settings(self):
        self.settings_panel.toggle()

    def _on_space(self):
        if self.state == GameState.PLAYING:
            self._try_use_card()

    def _update(self):
        self.animation_manager.update()
        self.message_box.update()

        if self.state == GameState.PLAYING:
            self._update_game()
        elif self.state == GameState.RESPONSE:
            if not self.response_manager.has_request():
                self.state = GameState.PLAYING

    def _update_game(self):
        if not self.engine:
            return

        if self.engine.phase == GamePhase.GAME_OVER:
            self.state = GameState.GAME_OVER
            if audio.is_enabled():
                audio.play("victory")
            return

        current = self.engine.players[self.engine.current_player_idx]

        if current != self.human_player:
            self._ai_turn(current)

    def _render(self):
        if self.state == GameState.MENU:
            self._render_menu()
        elif self.state in [
            GameState.PLAYING,
            GameState.TARGET_SELECT,
            GameState.DISCARD,
            GameState.RESPONSE,
            GameState.SKILL_TRIGGER,
        ]:
            self._render_game()
        elif self.state == GameState.GAME_OVER:
            self._render_game_over()

        if self.settings_panel.visible:
            self.settings_panel.render(self.screen)

        pygame.display.flip()

    def _render_menu(self):
        bg = assets.get_background()
        if bg:
            self.screen.blit(bg, (0, 0))
        else:
            self.screen.fill((45, 80, 45))

        font_title = fonts.get("large")
        font_medium = fonts.get("medium")

        title = font_title.render("三国杀 - 人机对战版", True, (255, 220, 100))
        title_rect = title.get_rect(centerx=WINDOW_WIDTH // 2, y=180)
        self.screen.blit(title, title_rect)

        subtitle = font_medium.render("PyGame GUI版", True, (200, 200, 200))
        subtitle_rect = subtitle.get_rect(centerx=WINDOW_WIDTH // 2, y=240)
        self.screen.blit(subtitle, subtitle_rect)

        player_label = font_medium.render("选择人数:", True, (240, 240, 240))
        player_label_rect = player_label.get_rect(centerx=WINDOW_WIDTH // 2, y=320)
        self.screen.blit(player_label, player_label_rect)

        for name, button in self.menu_buttons.items():
            if name.startswith("player_"):
                num = int(name.split("_")[1])
                if num == self.player_num:
                    button.hovered = True
            button.render(self.screen)

        self.animation_manager.render(self.screen)

    def _render_game(self):
        self.renderer.render(self.engine, self.human_player)

        card_data = self.renderer._render_hand_area(self.human_player, self.engine)
        self.input_handler.update_card_rects(card_data)

        current = self.engine.players[self.engine.current_player_idx]
        if current == self.human_player and self.state == GameState.PLAYING:
            for button in self.game_buttons.values():
                button.render(self.screen)

            self.input_handler.update_buttons(
                {name: btn.rect for name, btn in self.game_buttons.items()}
            )

        if self.settings_panel.get("show_log", True):
            self.log_panel.render(self.screen)

        self.animation_manager.render(self.screen)

        if self.state == GameState.TARGET_SELECT:
            self._render_target_overlay()

        if self.state == GameState.DISCARD:
            self._render_discard_overlay()

        if self.state == GameState.RESPONSE:
            self.response_manager.render(self.screen)

        if self.state == GameState.SKILL_TRIGGER:
            self.skill_ui.render(self.screen)

        self.message_box.render(self.screen)

    def _render_target_overlay(self):
        for player, rect in zip(self.engine.players, self._get_player_rects()):
            is_valid = player in self.valid_targets
            is_selected = player in self.selected_targets

            if is_valid:
                border_color = (100, 255, 100) if is_selected else (255, 220, 100)
                pygame.draw.rect(self.screen, border_color, rect, 4, border_radius=8)

                if is_selected:
                    font = fonts.get("small")
                    select_text = font.render("[已选]", True, (100, 255, 100))
                    self.screen.blit(
                        select_text, (rect.x + rect.width - 50, rect.y + 5)
                    )

        font = fonts.get("medium")
        title = font.render(
            f"选择目标 ({len(self.selected_targets)}/{self.target_panel.max_select})",
            True,
            COLORS["text_highlight"],
        )
        title_rect = title.get_rect(
            centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT // 2 - 80
        )

        bg_rect = title_rect.inflate(40, 20)
        pygame.draw.rect(
            self.screen, (*COLORS["panel_bg"], 230), bg_rect, border_radius=8
        )
        self.screen.blit(title, title_rect)

        confirm_btn = Button(
            "确认", WINDOW_WIDTH // 2 - 130, WINDOW_HEIGHT // 2 - 30, 120, 40
        )
        cancel_btn = Button(
            "取消", WINDOW_WIDTH // 2 + 10, WINDOW_HEIGHT // 2 - 30, 120, 40
        )

        confirm_btn.enabled = self.target_panel.can_confirm()
        confirm_btn.render(self.screen)
        cancel_btn.render(self.screen)

    def _render_discard_overlay(self):
        need_discard = len(self.human_player.hand_cards) - self.human_player.hand_limit
        if need_discard > 0:
            font = fonts.get("large")
            text = font.render(
                f"请弃置 {need_discard} 张牌 (点击选择)", True, COLORS["text_highlight"]
            )
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, 130))

            bg_rect = text_rect.inflate(40, 20)
            pygame.draw.rect(
                self.screen, (*COLORS["panel_bg"], 230), bg_rect, border_radius=10
            )
            pygame.draw.rect(
                self.screen, COLORS["text_highlight"], bg_rect, 2, border_radius=10
            )
            self.screen.blit(text, text_rect)

    def _render_game_over(self):
        self.renderer.render(self.engine, self.human_player)

        state = self.engine.get_state()
        winner = state.winner

        winner_map = {
            "主公": "主公阵营胜利!",
            "反贼": "反贼阵营胜利!",
            "内奸": "内奸胜利!",
        }
        winner_text = winner_map.get(winner, "游戏结束")

        font_large = fonts.get("large")
        font_medium = fonts.get("medium")

        text = font_large.render(winner_text, True, (255, 220, 100))
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80))

        bg_rect = text_rect.inflate(120, 80)
        pygame.draw.rect(self.screen, (40, 60, 40), bg_rect, border_radius=15)
        pygame.draw.rect(self.screen, (255, 220, 100), bg_rect, 4, border_radius=15)
        self.screen.blit(text, text_rect)

        stats_y = WINDOW_HEIGHT // 2

        stats = [
            f"回合数: {self.engine.round_num}",
            f"存活人数: {len([p for p in self.engine.players if p.is_alive])}",
        ]

        for i, stat in enumerate(stats):
            stat_surface = font_medium.render(stat, True, (200, 200, 200))
            stat_rect = stat_surface.get_rect(
                centerx=WINDOW_WIDTH // 2, y=stats_y + i * 30
            )
            self.screen.blit(stat_surface, stat_rect)

        hint = font_medium.render("按回车键返回主菜单", True, (150, 150, 150))
        hint_rect = hint.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT // 2 + 100)
        self.screen.blit(hint, hint_rect)

        self.animation_manager.render(self.screen)

    def _get_player_rects(self) -> List[pygame.Rect]:
        positions = PlayerRenderer.calculate_player_positions(
            len(self.engine.players), WINDOW_WIDTH, WINDOW_HEIGHT
        )
        return [
            pygame.Rect(x, y, PLAYER_AREA_WIDTH, PLAYER_AREA_HEIGHT)
            for x, y in positions
        ]

    def _get_clickable_player_rects(self) -> List[tuple]:
        rects = self._get_player_rects()
        return list(zip(rects, self.engine.players))

    def _start_game(self):
        self.engine = self._setup_game(self.player_num)
        self.human_player = self.engine.players[0]
        self.state = GameState.PLAYING

        self.log_panel.clear()
        self.selected_card_index = -1
        self.selected_targets.clear()
        self.animation_manager.clear()

        audio.play("card_use")

        self._log("=" * 30)
        self._log("游戏开始!")
        self._log(f"你的身份: {self.human_player.identity}")
        self._log(
            f"武将: {self.human_player.commander_name} ({self.human_player.nation})"
        )

        skills = [s.name for s in self.human_player.skills]
        if skills:
            self._log(f"技能: {', '.join(skills)}")
        self._log("=" * 30)

    def _setup_game(self, player_num: int) -> GameEngine:
        config_path = Path(__file__).parent.parent / "data" / "commanders.json"
        with open(config_path, encoding="utf-8") as f:
            commander_configs = json.load(f)

        available_ids = list(commander_configs.keys())
        selected_ids = random.sample(available_ids, player_num)

        engine = GameEngine(player_num, selected_ids, human_player_idx=0)

        players = []
        for i, cid in enumerate(selected_ids):
            config = commander_configs[cid]
            skill_names = config.get("skills", [])
            skills = []
            for skill_name in skill_names:
                skill = SkillRegistry.get(skill_name)
                if skill:
                    skills.append(skill)

            player = Player(
                commander_id=cid,
                commander_name=config["name"],
                nation=config["nation"],
                max_hp=config["max_hp"],
                current_hp=config["max_hp"],
                skills=skills,
                is_human=(i == 0),
            )
            players.append(player)

        engine.setup_game(players)
        return engine

    def _end_turn(self):
        current = self.engine.players[self.engine.current_player_idx]
        if current != self.human_player:
            return

        self._human_discard_phase()
        self.engine.next_turn()
        self.selected_card_index = -1
        self.selected_targets.clear()
        self.input_handler.clear_selection()
        self._log("你结束了回合")

    def _human_discard_phase(self):
        need_discard = len(self.human_player.hand_cards) - self.human_player.hand_limit

        if need_discard <= 0:
            return

        self.state = GameState.DISCARD
        self._log(f"需要弃置 {need_discard} 张牌")

        discarded = 0
        while discarded < need_discard:
            pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pass

                result = self.input_handler.handle_event(
                    event, self.engine, self.human_player
                )

                if result and result["type"] == "card_selected":
                    selected = result["selected_indices"]
                    if len(selected) == need_discard:
                        self._execute_discard(selected)
                        discarded = need_discard
                        break

            self._render()
            self.clock.tick(FPS)

        self.state = GameState.PLAYING
        self.input_handler.clear_selection()

    def _execute_discard(self, indices: List[int]):
        indices_sorted = sorted(indices, reverse=True)
        for idx in indices_sorted:
            if 0 <= idx < len(self.human_player.hand_cards):
                card = self.human_player.hand_cards.pop(idx)
                self.engine.discard_pile.append(card)
                self._log(f"弃置了 {card.name}")

        audio.play("card_use")
        self.input_handler.clear_selection()

    def _try_use_card(self):
        if self.selected_card_index < 0:
            self.message_box.show("请先选择一张卡牌", 1500)
            return

        if self.selected_card_index >= len(self.human_player.hand_cards):
            return

        card = self.human_player.hand_cards[self.selected_card_index]
        self._use_card_with_logic(card)

    def _use_card_with_logic(self, card: Card):
        if card.name == "桃":
            if self.human_player.current_hp >= self.human_player.max_hp:
                self.message_box.show("体力已满", 1500)
                return
            self.engine.use_card(self.human_player, card, self.human_player)
            self._log(f"使用了 {card.name}")
            self._play_heal_animation(self.human_player)
            audio.play("heal")
            self._finish_card_use()
            return

        if card.name == "酒":
            if self.human_player.jiu_count >= 1:
                self.message_box.show("本回合已使用过酒", 1500)
                return
            self.engine.use_card(self.human_player, card, self.human_player)
            self._log("使用酒，下一张杀伤害+1")
            audio.play("card_use")
            self._finish_card_use()
            return

        if card.name == "无中生有":
            self.engine.use_card(self.human_player, card, None)
            self._log("使用无中生有，摸2张牌")
            audio.play("draw")
            self._finish_card_use()
            return

        if card.card_type in [
            "WeaponCard",
            "ArmourCard",
            "AttackHorseCard",
            "DefenseHorseCard",
            "TreasureCard",
        ]:
            self.engine.use_card(self.human_player, card, None)
            self._log(f"装备了 {card.name}")
            audio.play("equip")
            self._finish_card_use()
            return

        if is_sha_card(card):
            if not self.human_player.can_use_sha():
                self.message_box.show("本回合已使用过杀", 1500)
                return
            targets = self._get_sha_targets(self.human_player)
            if not targets:
                self.message_box.show("没有可攻击的目标", 1500)
                return
            self._request_target_selection(card, targets, 1, 1)
            return

        if card.name == "决斗":
            targets = [
                p for p in self.engine.players if p != self.human_player and p.is_alive
            ]
            self._request_target_selection(card, targets, 1, 1)
            return

        if card.name in ["南蛮入侵", "万箭齐发"]:
            self.engine.use_card(self.human_player, card, None)
            self._log(f"使用 {card.name}")
            audio.play("card_use")
            self._finish_card_use()
            return

        if card.name == "桃园结义":
            self.engine.use_card(self.human_player, card, None)
            self._log("使用桃园结义")
            audio.play("heal")
            self._finish_card_use()
            return

        if card.name == "五谷丰登":
            self.engine.use_card(self.human_player, card, None)
            self._log("使用五谷丰登")
            audio.play("draw")
            self._finish_card_use()
            return

        if card.name == "火攻":
            targets = [
                p
                for p in self.engine.players
                if p != self.human_player and p.is_alive and p.hand_cards
            ]
            if not targets:
                self.message_box.show("没有手牌的目标", 1500)
                return
            self._request_target_selection(card, targets, 1, 1)
            return

        if card.name == "过河拆桥":
            targets = [
                p for p in self.engine.players if p != self.human_player and p.is_alive
            ]
            targets = [
                t
                for t in targets
                if t.hand_cards or any(t.equipment.values()) or t.judge_area
            ]
            if not targets:
                self.message_box.show("没有可用目标", 1500)
                return
            self._request_target_selection(card, targets, 1, 1)
            return

        if card.name == "顺手牵羊":
            targets = [
                p for p in self.engine.players if p != self.human_player and p.is_alive
            ]
            targets = [
                t
                for t in targets
                if (t.hand_cards or any(t.equipment.values()))
                and self._calculate_distance(self.human_player, t) <= 1
            ]
            if not targets:
                self.message_box.show("没有距离为1的目标", 1500)
                return
            self._request_target_selection(card, targets, 1, 1)
            return

        if card.name == "铁索连环":
            targets = [p for p in self.engine.players if p.is_alive]
            self._request_target_selection(card, targets, 1, 2)
            return

        if card.name == "乐不思蜀":
            targets = [
                p for p in self.engine.players if p != self.human_player and p.is_alive
            ]
            targets = [
                t
                for t in targets
                if not any(c.name == "乐不思蜀" for c in t.judge_area)
            ]
            if not targets:
                self.message_box.show("没有可用目标", 1500)
                return
            self._request_target_selection(card, targets, 1, 1)
            return

        if card.name == "兵粮寸断":
            targets = [
                p for p in self.engine.players if p != self.human_player and p.is_alive
            ]
            targets = [
                t
                for t in targets
                if not any(c.name == "兵粮寸断" for c in t.judge_area)
            ]
            targets = [
                t
                for t in targets
                if self._calculate_distance(self.human_player, t) <= 1
            ]
            if not targets:
                self.message_box.show("没有距离为1的目标", 1500)
                return
            self._request_target_selection(card, targets, 1, 1)
            return

        if card.name == "闪电":
            self.engine.use_card(self.human_player, card, None)
            self._log("放置闪电")
            audio.play("card_use")
            self._finish_card_use()
            return

        if card.name == "借刀杀人":
            targets = [
                p for p in self.engine.players if p != self.human_player and p.is_alive
            ]
            targets = [t for t in targets if t.equipment.get("武器")]
            if not targets:
                self.message_box.show("没有装备武器的玩家", 1500)
                return
            self._request_target_selection(card, targets, 1, 1)
            return

        self.message_box.show(f"暂不支持: {card.name}", 1500)

    def _finish_card_use(self):
        self.selected_card_index = -1
        self.input_handler.clear_selection()

    def _request_target_selection(
        self, card: Card, targets: List[Player], min_select: int, max_select: int
    ):
        self.valid_targets = targets
        self.selected_targets.clear()
        self.state = GameState.TARGET_SELECT

        self.target_panel.show(
            targets=targets,
            min_select=min_select,
            max_select=max_select,
            on_confirm=lambda sel: self._on_target_selected(card, sel),
            on_cancel=self._on_target_cancel,
        )

    def _on_target_selected(self, card: Card, targets: List[Player]):
        if not targets:
            self._on_target_cancel()
            return

        if card.name == "铁索连环":
            for target in targets[:2]:
                target.is_chained = not target.is_chained
                status = "连环" if target.is_chained else "重置"
                self._log(f"{target.commander_name} {status}")
            self.human_player.hand_cards.remove(card)
            self.engine.discard_pile.append(card)
        else:
            target = targets[0]
            success = self.engine.use_card(self.human_player, card, target)

            if success:
                self._log(f"对 {target.commander_name} 使用 {card.name}")
                audio.play("card_use")

                if is_sha_card(card):
                    self._play_damage_animation(target)

        self._finish_card_use()
        self.state = GameState.PLAYING

    def _on_target_cancel(self):
        self.selected_targets.clear()
        self.state = GameState.PLAYING
        self.input_handler.clear_selection()

    def _ai_turn(self, player: Player):
        if not player.is_alive:
            self.engine.next_turn()
            return

        pygame.time.wait(self.ai_delay)

        self._ai_judge_phase(player)
        self._ai_draw_phase(player)
        self._ai_play_phase(player)
        self._ai_discard_phase(player)

        self.engine.next_turn()

    def _ai_judge_phase(self, player: Player):
        if player.judge_area:
            result = self.engine.judge_phase(player)
            self._log(f"{player.commander_name} 进行判定")

    def _ai_draw_phase(self, player: Player):
        drawn = self.engine.draw_cards(player, 2)
        player.hand_cards.extend(drawn)
        self._log(f"{player.commander_name} 摸 {len(drawn)} 张牌")

    def _ai_play_phase(self, player: Player):
        used_sha = False
        max_actions = 12
        action_count = 0

        while action_count < max_actions and player.is_alive:
            action_count += 1
            card_to_use = None
            target = None

            for card in player.hand_cards:
                if is_sha_card(card):
                    if not used_sha and player.can_use_sha():
                        targets = self._get_sha_targets(player)
                        if targets:
                            card_to_use = card
                            target = min(targets, key=lambda t: t.current_hp)
                            used_sha = True
                            break
                elif card.name == "桃" and player.current_hp < player.max_hp:
                    card_to_use = card
                    target = player
                    break
                elif card.name == "酒" and player.jiu_count < 1:
                    card_to_use = card
                    target = player
                    break
                elif card.name == "无中生有":
                    card_to_use = card
                    break
                elif card.name in ["南蛮入侵", "万箭齐发"]:
                    card_to_use = card
                    break
                elif card.name in ["过河拆桥", "顺手牵羊"]:
                    targets = [
                        p for p in self.engine.players if p != player and p.is_alive
                    ]
                    if targets:
                        card_to_use = card
                        target = random.choice(targets)
                    break
                elif card.card_type in [
                    "WeaponCard",
                    "ArmourCard",
                    "AttackHorseCard",
                    "DefenseHorseCard",
                ]:
                    card_to_use = card
                    break

            if card_to_use:
                self.engine.use_card(player, card_to_use, target)
                target_name = target.commander_name if target else ""
                self._log(
                    f"{player.commander_name} 使用 {card_to_use.name}"
                    + (f" → {target_name}" if target_name else "")
                )

                if target and target != player and is_sha_card(card_to_use):
                    self._play_damage_animation(target)

                pygame.time.wait(200)
            else:
                break

        player.reset_turn_state()

    def _ai_discard_phase(self, player: Player):
        while len(player.hand_cards) > player.hand_limit:
            card = random.choice(player.hand_cards)
            player.hand_cards.remove(card)
            self.engine.discard_pile.append(card)
            self._log(f"{player.commander_name} 弃置 {card.name}")

    def _get_sha_targets(self, player: Player) -> List[Player]:
        targets = []
        for p in self.engine.players:
            if p != player and p.is_alive:
                dist = self._calculate_distance(player, p)
                if dist <= player.attack_range:
                    targets.append(p)
        return targets

    def _calculate_distance(self, source: Player, target: Player) -> int:
        if source == target:
            return 0

        dist = 1
        current = source.next_player
        while current != target:
            dist += 1
            current = current.next_player

        reverse_dist = 1
        current = source.prev_player
        while current != target:
            reverse_dist += 1
            current = current.prev_player

        dist = min(dist, reverse_dist)

        if source.equipment.get("进攻坐骑"):
            dist = max(1, dist - 1)
        if target.equipment.get("防御坐骑"):
            dist += 1

        return dist

    def _play_damage_animation(self, target: Player):
        rects = self._get_player_rects()
        target_idx = self.engine.players.index(target)
        if target_idx < len(rects):
            rect = rects[target_idx]
            center = (rect.centerx, rect.centery)
            self.animation_manager.add(DamageAnimation(center, 1))
            audio.play("damage")

    def _play_heal_animation(self, player: Player):
        rects = self._get_player_rects()
        player_idx = self.engine.players.index(player)
        if player_idx < len(rects):
            rect = rects[player_idx]
            center = (rect.centerx, rect.centery)
            self.animation_manager.add(HealAnimation(center, 1))

    def _log(self, message: str):
        self.log_panel.add_log(message)


def main():
    window = GameWindow()
    window.run()


if __name__ == "__main__":
    main()
