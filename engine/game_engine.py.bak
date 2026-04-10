import json
import logging
import random
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from .event import Event, EventType
from .event_bus import EventBus
from .judge import JudgeSystem, DelayedTrickHandler
from .response import ResponseSystem, CardResolver
from .state import GameState, PlayerState, GamePhase

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from player.player import Player
    from card.base import Card, is_sha_card
else:
    from card.base import is_sha_card


class GameEngine:
    def __init__(
        self, player_num: int, commander_ids: List[str], human_player_idx: int = 0
    ):
        self.player_num = player_num
        self.commander_ids = commander_ids
        self.human_player_idx = human_player_idx

        self.event_bus = EventBus()
        self.players: List["Player"] = []
        self.deck: List["Card"] = []
        self.discard_pile: List["Card"] = []
        self.tmp_cards: List["Card"] = []

        self.current_player_idx = 0
        self.round_num = 1
        self.phase = GamePhase.WAITING
        self._winner: Optional[str] = None

        self.action_log: List[str] = []

        self._load_commanders()
        self._load_cards()

        self.response_system = ResponseSystem(self)
        self.card_resolver = CardResolver(self)
        self.judge_system = JudgeSystem(self)
        self.delayed_trick_handler = DelayedTrickHandler(self)

    def log(self, message: str):
        self.action_log.append(message)
        logger.info(message)

    def _load_commanders(self):
        config_path = Path(__file__).parent.parent / "data" / "commanders.json"
        with open(config_path, encoding="utf-8") as f:
            self.commander_configs = json.load(f)

    def _load_cards(self):
        from card.factory import CardFactory

        config_path = Path(__file__).parent.parent / "data" / "cards.json"
        self.deck = CardFactory.load_from_config(config_path)
        random.shuffle(self.deck)

    def setup_game(self, player_classes: List["Player"]):
        self.players = player_classes

        random.shuffle(self.players)

        identities = self._distribute_identities()
        for i, player in enumerate(self.players):
            player.identity = identities[i]
            player.idx = i + 1

            if player.identity == "主公":
                player.max_hp += 1
                player.current_hp += 1

            for skill in player.skills:
                skill.bind_player(player)
                for event_type in skill.trigger_events:
                    self.event_bus.subscribe(event_type, skill.on_event)

        self._setup_seating()

        for player in self.players:
            player.hand_cards = self.draw_cards(player, 4)

        self._emit_event(EventType.GAME_START)

    def _distribute_identities(self) -> List[str]:
        from config import IDENTITY_CONFIG

        identities = IDENTITY_CONFIG[self.player_num].copy()
        random.shuffle(identities)
        return identities

    def _setup_seating(self):
        n = len(self.players)
        for i, player in enumerate(self.players):
            player.next_player = self.players[(i + 1) % n]
            player.prev_player = self.players[(i - 1) % n]

    def draw_cards(self, player: "Player", num: int) -> List["Card"]:
        drawn = []
        for _ in range(num):
            if not self.deck:
                if not self.discard_pile:
                    break
                self.deck = self.discard_pile.copy()
                self.discard_pile.clear()
                random.shuffle(self.deck)

            if self.deck:
                card = self.deck.pop()
                drawn.append(card)

        self._emit_event(EventType.DRAW_CARD, source=player, value=num)
        return drawn

    def judge_phase(self, player: "Player") -> dict:
        self.phase = GamePhase.JUDGE_PHASE
        return self.judge_system.process_judge_phase(player)

    def use_card(
        self, player: "Player", card: "Card", target: Optional["Player"] = None
    ) -> bool:
        if card not in player.hand_cards:
            return False

        if is_sha_card(card):
            if player.sha_count >= self._get_max_sha(player):
                return False
            player.sha_count += 1

        if card.name == "酒":
            if player.jiu_count >= 1:
                return False
            player.jiu_count += 1
            player.jiu_effect += 1

        player.hand_cards.remove(card)
        self.tmp_cards.append(card)

        before_event = self._emit_event(
            EventType.BEFORE_USE_CARD, source=player, target=target, card=card
        )

        if before_event.is_cancelled():
            player.hand_cards.append(card)
            self.tmp_cards.remove(card)
            return False

        event = self._emit_event(
            EventType.CARD_USED, source=player, target=target, card=card
        )

        if target:
            target_event = self._emit_event(
                EventType.CARD_TARGETED, source=player, target=target, card=card
            )
            event.data.update(target_event.data)

        if not event.is_cancelled():
            self._resolve_card(player, card, target, event)

        return True

    def _resolve_card(
        self,
        player: "Player",
        card: "Card",
        target: Optional["Player"],
        event: Optional["Event"] = None,
    ):
        if card.name == "桃":
            if player.current_hp < player.max_hp:
                player.current_hp += 1
                self.log(
                    f"{player.commander_name} 使用 {card}，恢复1点体力，当前体力: {player.current_hp}/{player.max_hp}"
                )
            self._move_to_discard(card)

        elif card.name == "无中生有":
            drawn = self.draw_cards(player, 2)
            player.hand_cards.extend(drawn)
            self.log(f"{player.commander_name} 使用 {card}，摸了2张牌")
            self._move_to_discard(card)

        elif is_sha_card(card) and target:
            damage = 1 + player.jiu_effect
            is_elemental = getattr(card, "is_elemental", False)
            self.log(f"{player.commander_name} 对 {target.commander_name} 使用 {card}")
            self.card_resolver.resolve_sha(
                player, target, card, event.data if event else None
            )
            player.jiu_effect = 0
            self._move_to_discard(card)

        elif card.name == "决斗" and target:
            self.log(f"{player.commander_name} 对 {target.commander_name} 使用 {card}")
            self.card_resolver.resolve_juedou(player, target)
            self._move_to_discard(card)

        elif card.name == "南蛮入侵":
            self.log(f"{player.commander_name} 使用 {card}")
            self.card_resolver.resolve_namaninru(player)
            self._move_to_discard(card)

        elif card.name == "万箭齐发":
            self.log(f"{player.commander_name} 使用 {card}")
            self.card_resolver.resolve_wanjianqifa(player)
            self._move_to_discard(card)

        elif card.name == "火攻" and target:
            self.log(f"{player.commander_name} 对 {target.commander_name} 使用 {card}")
            self.card_resolver.resolve_huogong(player, target)
            self._move_to_discard(card)

        elif card.name == "过河拆桥" and target:
            self.log(f"{player.commander_name} 对 {target.commander_name} 使用过河拆桥")
            self._resolve_chaiqiao(player, target)
            self._move_to_discard(card)

        elif card.name == "顺手牵羊" and target:
            self._resolve_shunshou(player, target)
            self._move_to_discard(card)

        elif card.name == "铁索连环":
            self._resolve_tiesuo(player, target)
            self._move_to_discard(card)

        elif card.name == "五谷丰登":
            self._resolve_wugu(player)
            self._move_to_discard(card)

        elif card.name == "桃园结义":
            self._resolve_taoyuan(player)
            self._move_to_discard(card)

        elif card.name == "乐不思蜀" and target:
            self.delayed_trick_handler.use_lebusishu(player, target, card)

        elif card.name == "兵粮寸断" and target:
            self.delayed_trick_handler.use_bingliangcunduan(player, target, card)

        elif card.name == "闪电":
            self.delayed_trick_handler.use_shandian(player, card)

        elif card.name == "借刀杀人" and target:
            self._resolve_jiedaosharen(player, target, card)
            self._move_to_discard(card)

        elif card.card_type == "WeaponCard":
            self._equip_weapon(player, card)
            self.log(f"{player.commander_name} 装备了 {card.name}")
            if card.name == "诸葛连弩":
                player.unlimited_sha = True

        elif card.card_type == "ArmourCard":
            self._equip_armour(player, card)
            self.log(f"{player.commander_name} 装备了 {card.name}")

        elif card.card_type == "AttackHorseCard":
            self._equip_attack_horse(player, card)
            self.log(f"{player.commander_name} 装备了 {card.name}")

        elif card.card_type == "DefenseHorseCard":
            self._equip_defense_horse(player, card)
            self.log(f"{player.commander_name} 装备了 {card.name}")

        elif card.card_type == "TreasureCard":
            self._equip_treasure(player, card)
            self.log(f"{player.commander_name} 装备了 {card.name}")

    def _resolve_chaiqiao(self, source: "Player", target: "Player"):
        if source.is_human:
            print(f"\n{target.commander_name} 的区域:")
            print(f"  手牌数: {len(target.hand_cards)}")
            print(
                f"  装备: {[(k, v.name if v else '无') for k, v in target.equipment.items()]}"
            )
            print(f"  判定区: {[c.name for c in target.judge_area]}")
            choice = input(
                "选择弃置区域 (h=手牌, w=武器, a=防具, j=进攻马, d=防御马, t=宝物, p=判定区): "
            )
        else:
            if target.hand_cards:
                choice = "h"
            elif target.equipment.get("武器"):
                choice = "w"
            elif target.equipment.get("防具"):
                choice = "a"
            elif target.judge_area:
                choice = "p"
            else:
                return

        if choice == "h" and target.hand_cards:
            import random

            card = random.choice(target.hand_cards)
            target.hand_cards.remove(card)
            self.discard_pile.append(card)
            self.log(f"弃置了 {target.commander_name} 的 {card}")
        elif choice == "w" and target.equipment.get("武器"):
            self.discard_pile.append(target.equipment["武器"])
            self.log(
                f"弃置了 {target.commander_name} 的 {target.equipment['武器'].name}"
            )
            target.equipment["武器"] = None
        elif choice == "a" and target.equipment.get("防具"):
            self.discard_pile.append(target.equipment["防具"])
            self.log(
                f"弃置了 {target.commander_name} 的 {target.equipment['防具'].name}"
            )
            target.equipment["防具"] = None
        elif choice == "j" and target.equipment.get("进攻坐骑"):
            self.discard_pile.append(target.equipment["进攻坐骑"])
            self.log(
                f"弃置了 {target.commander_name} 的 {target.equipment['进攻坐骑'].name}"
            )
            target.equipment["进攻坐骑"] = None
        elif choice == "d" and target.equipment.get("防御坐骑"):
            self.discard_pile.append(target.equipment["防御坐骑"])
            self.log(
                f"弃置了 {target.commander_name} 的 {target.equipment['防御坐骑'].name}"
            )
            target.equipment["防御坐骑"] = None
        elif choice == "t" and target.equipment.get("宝物"):
            self.discard_pile.append(target.equipment["宝物"])
            self.log(
                f"弃置了 {target.commander_name} 的 {target.equipment['宝物'].name}"
            )
            target.equipment["宝物"] = None
        elif choice == "p" and target.judge_area:
            card = target.judge_area.pop()
            self.discard_pile.append(card)
            self.log(f"弃置了 {target.commander_name} 判定区的 {card.name}")

    def _resolve_shunshou(self, source: "Player", target: "Player"):
        if source.is_human:
            print(f"\n{target.commander_name} 的区域:")
            print(f"  手牌数: {len(target.hand_cards)}")
            print(
                f"  装备: {[(k, v.name if v else '无') for k, v in target.equipment.items()]}"
            )
            choice = input(
                "选择获得区域 (h=手牌, w=武器, a=防具, j=进攻马, d=防御马, t=宝物): "
            )
        else:
            if target.hand_cards:
                choice = "h"
            elif target.equipment.get("武器"):
                choice = "w"
            elif target.equipment.get("防具"):
                choice = "a"
            elif target.equipment.get("进攻坐骑"):
                choice = "j"
            elif target.equipment.get("防御坐骑"):
                choice = "d"
            else:
                return

        if choice == "h" and target.hand_cards:
            import random

            card = random.choice(target.hand_cards)
            target.hand_cards.remove(card)
            source.hand_cards.append(card)
            self.log(f"获得了 {target.commander_name} 的 {card}")
        elif choice == "w" and target.equipment.get("武器"):
            source.hand_cards.append(target.equipment["武器"])
            self.log(
                f"获得了 {target.commander_name} 的 {target.equipment['武器'].name}"
            )
            target.equipment["武器"] = None
        elif choice == "a" and target.equipment.get("防具"):
            source.hand_cards.append(target.equipment["防具"])
            self.log(
                f"获得了 {target.commander_name} 的 {target.equipment['防具'].name}"
            )
            target.equipment["防具"] = None
        elif choice == "j" and target.equipment.get("进攻坐骑"):
            source.hand_cards.append(target.equipment["进攻坐骑"])
            self.log(
                f"获得了 {target.commander_name} 的 {target.equipment['进攻坐骑'].name}"
            )
            target.equipment["进攻坐骑"] = None
        elif choice == "d" and target.equipment.get("防御坐骑"):
            source.hand_cards.append(target.equipment["防御坐骑"])
            self.log(
                f"获得了 {target.commander_name} 的 {target.equipment['防御坐骑'].name}"
            )
            target.equipment["防御坐骑"] = None
        elif choice == "t" and target.equipment.get("宝物"):
            source.hand_cards.append(target.equipment["宝物"])
            self.log(
                f"获得了 {target.commander_name} 的 {target.equipment['宝物'].name}"
            )
            target.equipment["宝物"] = None

    def _resolve_tiesuo(self, source: "Player", target: Optional["Player"]):
        if target:
            target.is_chained = not target.is_chained
            status = "连环" if target.is_chained else "重置"
            self.log(f"{target.commander_name} {status}")
        else:
            for p in self.players:
                if p.is_alive:
                    p.is_chained = not p.is_chained

    def _resolve_wugu(self, source: "Player"):
        alive_count = len([p for p in self.players if p.is_alive])
        cards = self.draw_cards(source, min(alive_count, 5))
        if not cards:
            return

        self.log(f"五谷丰登，翻开 {len(cards)} 张牌:")
        for i, c in enumerate(cards):
            self.log(f"  {i + 1}. {c}")

        self.tmp_cards.extend(cards)

        current = source
        for i, card in enumerate(cards):
            if len(self.tmp_cards) > 0 and current.is_alive:
                if current.is_human:
                    print(
                        f"\n可选牌: {list(enumerate([str(c) for c in self.tmp_cards], 1))}"
                    )
                    idx = int(input(f"{current.commander_name} 选择获得: ")) - 1
                    if 0 <= idx < len(self.tmp_cards):
                        chosen = self.tmp_cards.pop(idx)
                        current.hand_cards.append(chosen)
                        self.log(f"{current.commander_name} 获得了 {chosen}")
                else:
                    if self.tmp_cards:
                        chosen = self.tmp_cards.pop(0)
                        current.hand_cards.append(chosen)
                        self.log(f"{current.commander_name} 获得了 {chosen}")
            current = current.next_player

    def _resolve_taoyuan(self, source: "Player"):
        for player in self.players:
            if player.is_alive and player.current_hp < player.max_hp:
                player.current_hp += 1
                self.log(f"{player.commander_name} 恢复1点体力")

    def _resolve_jiedaosharen(self, source: "Player", target: "Player", card: "Card"):
        if not target.equipment.get("武器"):
            self.log(f"{target.commander_name} 没有武器，借刀杀人无效")
            source.hand_cards.append(card)
            return

        kill_targets = []
        for p in self.players:
            if p != target and p.is_alive:
                dist = self._calculate_distance(target, p)
                if dist <= target.attack_range:
                    kill_targets.append(p)

        if not kill_targets:
            self.log(f"{target.commander_name} 没有可攻击的目标")
            source.hand_cards.append(card)
            return

        kill_target = None
        if source.is_human:
            print(f"\n可选的被杀目标:")
            for i, t in enumerate(kill_targets, 1):
                print(f"  {i}. {t.commander_name}")
            idx = int(input("选择被杀目标: ")) - 1
            if 0 <= idx < len(kill_targets):
                kill_target = kill_targets[idx]
        else:
            import random

            kill_target = random.choice(kill_targets)

        if kill_target:
            self.card_resolver.resolve_jiedaosharen(source, target, kill_target, card)

    def _calculate_distance(self, source: "Player", target: "Player") -> int:
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

    def _equip_weapon(self, player: "Player", card: "Card"):
        if player.equipment.get("武器"):
            self.discard_pile.append(player.equipment["武器"])
        player.equipment["武器"] = card
        # print(f"{player.commander_name} 装备了 {card.name}")
        self.tmp_cards.remove(card)

    def _equip_armour(self, player: "Player", card: "Card"):
        old_armour = player.equipment.get("防具")
        if old_armour:
            if old_armour.name == "白银狮子":
                if player.current_hp < player.max_hp:
                    player.current_hp += 1
                    self.log(f"{player.commander_name} 失去白银狮子，回复1点体力")
            self.discard_pile.append(old_armour)
        player.equipment["防具"] = card
        self.tmp_cards.remove(card)

    def _equip_attack_horse(self, player: "Player", card: "Card"):
        if player.equipment.get("进攻坐骑"):
            self.discard_pile.append(player.equipment["进攻坐骑"])
        player.equipment["进攻坐骑"] = card
        # print(f"{player.commander_name} 装备了 {card.name}")
        self.tmp_cards.remove(card)

    def _equip_defense_horse(self, player: "Player", card: "Card"):
        if player.equipment.get("防御坐骑"):
            self.discard_pile.append(player.equipment["防御坐骑"])
        player.equipment["防御坐骑"] = card
        # print(f"{player.commander_name} 装备了 {card.name}")
        self.tmp_cards.remove(card)

    def _equip_treasure(self, player: "Player", card: "Card"):
        if player.equipment.get("宝物"):
            self.discard_pile.append(player.equipment["宝物"])
        player.equipment["宝物"] = card
        # print(f"{player.commander_name} 装备了 {card.name}")
        self.tmp_cards.remove(card)

    def deal_damage(
        self,
        source: "Player",
        target: "Player",
        card: Optional["Card"],
        damage: int,
        is_elemental: bool = False,
        is_fire: bool = False,
        is_thunder: bool = False,
    ):
        logger.debug(
            f"deal_damage: target={target.idx}, damage={damage}, is_elemental={is_elemental}, is_fire={is_fire}, is_thunder={is_thunder}"
        )
        event = self._emit_event(
            EventType.DAMAGE_DEALT,
            source=source,
            target=target,
            card=card,
            value=damage,
        )

        if event.is_cancelled():
            return

        actual_damage = event.value

        if target.is_chained and is_elemental:
            target.is_chained = False

        self._emit_event(
            EventType.DAMAGE_TAKEN,
            source=source,
            target=target,
            card=card,
            value=actual_damage,
        )

        target.current_hp -= actual_damage

        if source:
            target.last_damage_source = source.idx
            self.log(
                f"{target.commander_name} 受到 {actual_damage} 点伤害（来自{source.commander_name}），当前体力: {target.current_hp}/{target.max_hp}"
            )
        else:
            self.log(
                f"{target.commander_name} 受到 {actual_damage} 点伤害，当前体力: {target.current_hp}/{target.max_hp}"
            )

        self._emit_event(EventType.HP_CHANGED, target=target, value=-actual_damage)

        if target.current_hp <= 0:
            killer_idx = source.idx if source else None
            self._handle_dying(target, killer_idx)

        if is_elemental:
            self._propagate_chain_damage(
                source, target, card, actual_damage, is_fire, is_thunder
            )

    def _propagate_chain_damage(
        self,
        source: "Player",
        original_target: "Player",
        card: Optional["Card"],
        damage: int,
        is_fire: bool = False,
        is_thunder: bool = False,
    ):
        logger.debug(
            f"_propagate_chain_damage start: original_target={original_target.idx}"
        )
        original_idx = original_target.idx
        alive_players = [p for p in self.players if p.is_alive]
        n = len(alive_players)
        if n <= 1:
            logger.debug("_propagate_chain_damage end: only one alive player")
            return

        start_idx = None
        for i, p in enumerate(alive_players):
            if p.idx == original_idx:
                start_idx = i
                break

        if start_idx is None:
            logger.debug(
                "_propagate_chain_damage end: original_target not in alive players"
            )
            return

        for offset in range(1, n):
            current = alive_players[(start_idx + offset) % n]
            logger.debug(
                f"  checking player {current.idx}, is_chained={current.is_chained}"
            )
            if current.is_chained:
                current.is_chained = False
                logger.debug(f"  propagating damage to {current.idx}")
                self.deal_damage(
                    source, current, card, damage, True, is_fire, is_thunder
                )
        logger.debug(f"_propagate_chain_damage end")

    def _handle_dying(self, player: "Player", killer_idx: int = None):
        logger.debug(f"_handle_dying: player={player.idx}, hp={player.current_hp}")
        self._emit_event(EventType.PLAYER_DYING, target=player)

        current = player
        saved = False

        for i in range(len(self.players)):
            logger.debug(f"  _handle_dying loop {i}, checking player {current.idx}")
            if self._ask_for_peach(current, player):
                saved = True
                break
            current = current.next_player

        if not saved and player.current_hp <= 0:
            logger.debug(f"  player {player.idx} died")
            if killer_idx is not None:
                player.last_kill = killer_idx
            self._handle_death(player)

    def _ask_for_peach(self, responder: "Player", dying_player: "Player") -> bool:
        logger.debug(
            f"_ask_for_peach: responder={responder.idx}, dying={dying_player.idx}"
        )
        peaches = [c for c in responder.hand_cards if c.name == "桃"]
        if responder == dying_player:
            wines = [c for c in responder.hand_cards if c.name == "酒"]
        else:
            wines = []

        if not peaches and not wines:
            return False

        if responder.is_human:
            if peaches:
                pass
            if wines:
                pass
            choice = input("使用桃/酒救人? (y/n): ")
            if choice.lower() == "y":
                if peaches:
                    peach = peaches[0]
                    responder.hand_cards.remove(peach)
                    self.discard_pile.append(peach)
                    hp_gain = 1
                    if (
                        dying_player.identity == "主公"
                        and hasattr(dying_player, "skills")
                        and any(s.name == "救援" for s in dying_player.skills)
                    ):
                        if responder != dying_player and responder.nation == "吴":
                            hp_gain = 2
                    dying_player.current_hp += hp_gain
                    return True
                elif wines:
                    wine = wines[0]
                    responder.hand_cards.remove(wine)
                    self.discard_pile.append(wine)
                    dying_player.current_hp += 1
                    return True
        else:
            if dying_player.current_hp <= 0:
                if peaches:
                    peach = peaches[0]
                    responder.hand_cards.remove(peach)
                    self.discard_pile.append(peach)
                    hp_gain = 1
                    if (
                        dying_player.identity == "主公"
                        and hasattr(dying_player, "skills")
                        and any(s.name == "救援" for s in dying_player.skills)
                    ):
                        if responder != dying_player and responder.nation == "吴":
                            hp_gain = 2
                    dying_player.current_hp += hp_gain
                    return True
                elif wines:
                    wine = wines[0]
                    responder.hand_cards.remove(wine)
                    self.discard_pile.append(wine)
                    dying_player.current_hp += 1
                    return True

        return False

    def _handle_death(self, player: "Player"):
        player.is_alive = False
        self._emit_event(EventType.PLAYER_DEAD, target=player)

        for card in player.hand_cards:
            self.discard_pile.append(card)
        player.hand_cards.clear()

        for slot, card in player.equipment.items():
            if card:
                self.discard_pile.append(card)
        player.equipment = {
            "武器": None,
            "防具": None,
            "进攻坐骑": None,
            "防御坐骑": None,
            "宝物": None,
        }

        for card in player.judge_area:
            self.discard_pile.append(card)
        player.judge_area.clear()

        player.prev_player.next_player = player.next_player
        player.next_player.prev_player = player.prev_player

        winner = self._check_victory()
        if winner:
            self._winner = winner
            self.phase = GamePhase.GAME_OVER

    def _check_victory(self) -> Optional[str]:
        alive_identities = [p.identity for p in self.players if p.is_alive]

        lord_alive = "主公" in alive_identities
        rebel_alive = "反贼" in alive_identities
        spy_alive = "内奸" in alive_identities

        if not lord_alive:
            if len(alive_identities) == 1 and alive_identities[0] == "内奸":
                return "内奸"
            return "反贼"

        if not rebel_alive and not spy_alive:
            return "主公"

        return None

    def _get_max_sha(self, player: "Player") -> int:
        if player.unlimited_sha:
            return 999
        if player.equipment.get("武器"):
            weapon = player.equipment["武器"]
            if weapon and weapon.name == "诸葛连弩":
                return 999
        return 1

    def _move_to_discard(self, card: "Card"):
        if card in self.tmp_cards:
            self.tmp_cards.remove(card)
        self.discard_pile.append(card)

    def _emit_event(
        self,
        event_type: EventType,
        source: "Player" = None,
        target: "Player" = None,
        card: "Card" = None,
        value: int = 0,
    ) -> Event:
        event = Event(
            type=event_type,
            source=source,
            target=target,
            card=card,
            value=value,
            engine=self,
        )
        return self.event_bus.emit(event)

    def get_state(self) -> GameState:
        player_states = []
        for p in self.players:
            ps = PlayerState(
                player_id=p.idx,
                commander_id=p.commander_id,
                commander_name=p.commander_name,
                nation=p.nation,
                identity=p.identity,
                max_hp=p.max_hp,
                current_hp=p.current_hp,
                hand_cards=[c.to_dict() for c in p.hand_cards],
                equipment={
                    k: v.to_dict() if v else None for k, v in p.equipment.items()
                },
                judge_area=[c.to_dict() for c in p.judge_area],
                skills=[s.name for s in p.skills],
                is_chained=p.is_chained,
                is_alive=p.is_alive,
                sha_count=p.sha_count,
                jiu_count=p.jiu_count,
                jiu_effect=p.jiu_effect,
            )
            player_states.append(ps)

        return GameState(
            phase=self.phase,
            current_player_idx=self.current_player_idx,
            round_num=self.round_num,
            players=player_states,
            deck_count=len(self.deck),
            discard_pile_count=len(self.discard_pile),
            winner=self._winner,
        )

    def next_turn(self):
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        loop_count = 0
        while not self.players[self.current_player_idx].is_alive:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            loop_count += 1
            if loop_count > len(self.players):
                self.phase = GamePhase.GAME_OVER
                return

        if self.current_player_idx == 0:
            self.round_num += 1

        player = self.players[self.current_player_idx]
        player.sha_count = 0
        player.jiu_count = 0
        player.jiu_effect = 0

        self.phase = GamePhase.TURN_START
        self._emit_event(EventType.TURN_START, source=player)

        self.phase = GamePhase.PREPARE_PHASE
        self._emit_event(EventType.PREPARE_PHASE, source=player)

    def end_turn(self, player: "Player"):
        self.phase = GamePhase.END_PHASE
        self._emit_event(EventType.END_PHASE, source=player)

        self.phase = GamePhase.TURN_END
        self._emit_event(EventType.TURN_END, source=player)
