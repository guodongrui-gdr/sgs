from typing import Optional, List, Callable, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from player.player import Player
    from card.base import Card
    from engine.game_engine import GameEngine
    from engine.event import EventType


class ResponseType(Enum):
    SHAN = "闪"
    SHA = "杀"
    TAO = "桃"
    WUXIE = "无懈可击"
    ANY = "任意牌"


@dataclass
class ResponseRequest:
    response_type: ResponseType
    prompt: str
    source: Optional["Player"] = None
    target: Optional["Player"] = None
    card: Optional["Card"] = None
    can_skip: bool = True
    min_count: int = 1
    max_count: int = 1


class ResponseSystem:
    def __init__(self, engine: "GameEngine"):
        self.engine = engine
        self._response_callback: Optional[Callable] = None
        self.wuxie_chain: List["Player"] = []

    def ask_for_response(
        self,
        player: "Player",
        request: ResponseRequest,
        human_handler: Callable = None,
        ai_handler: Callable = None,
    ) -> Optional["Card"]:
        available_cards = self._get_available_cards(player, request.response_type)

        if not available_cards:
            return None

        if player.is_human:
            if human_handler:
                return human_handler(player, available_cards, request)
            return self._default_human_response(player, available_cards, request)
        else:
            if ai_handler:
                return ai_handler(player, available_cards, request)
            return self._default_ai_response(player, available_cards, request)

    def _get_available_cards(
        self, player: "Player", response_type: ResponseType
    ) -> List["Card"]:
        if response_type == ResponseType.SHAN:
            return [c for c in player.hand_cards if c.name == "闪"]
        elif response_type == ResponseType.SHA:
            return [c for c in player.hand_cards if "杀" in c.name]
        elif response_type == ResponseType.TAO:
            return [c for c in player.hand_cards if c.name == "桃"]
        elif response_type == ResponseType.WUXIE:
            return [c for c in player.hand_cards if c.name == "无懈可击"]
        elif response_type == ResponseType.ANY:
            return player.hand_cards.copy()
        return []

    def _default_human_response(
        self, player: "Player", available_cards: List["Card"], request: ResponseRequest
    ) -> Optional["Card"]:
        print(f"\n{request.prompt}")
        print(f"可用卡牌: {list(enumerate([str(c) for c in available_cards], 1))}")

        if request.can_skip:
            print("输入 0 跳过")

        try:
            choice = int(input("选择: "))
            if choice == 0 and request.can_skip:
                return None
            if 1 <= choice <= len(available_cards):
                return available_cards[choice - 1]
        except ValueError:
            pass

        return None

    def _default_ai_response(
        self, player: "Player", available_cards: List["Card"], request: ResponseRequest
    ) -> Optional["Card"]:
        import random

        if request.response_type == ResponseType.SHAN:
            if player.current_hp <= 1:
                return available_cards[0] if available_cards else None
            if random.random() < 0.8:
                return available_cards[0] if available_cards else None
            return None

        elif request.response_type == ResponseType.SHA:
            if request.source and request.target:
                if request.source == player:
                    return available_cards[0] if available_cards else None
            return available_cards[0] if available_cards else None

        elif request.response_type == ResponseType.TAO:
            if player.current_hp <= 1:
                return available_cards[0] if available_cards else None
            return None

        elif request.response_type == ResponseType.WUXIE:
            if request.target and request.target == player:
                return available_cards[0] if available_cards else None
            if random.random() < 0.3:
                return available_cards[0] if available_cards else None
            return None

        return available_cards[0] if available_cards else None

    def ask_for_wuxie(
        self, source: "Player", target_card: "Card", original_target: "Player" = None
    ) -> bool:
        wuxie_count = 0
        return self._wuxie_chain(source, target_card, original_target, wuxie_count)

    def _wuxie_chain(
        self,
        source: "Player",
        target_card: "Card",
        original_target: "Player",
        wuxie_count: int,
    ) -> bool:
        max_wuxie_depth = 10
        if wuxie_count >= max_wuxie_depth:
            import logging

            logging.warning(f"无懈可击链达到最大深度 {max_wuxie_depth}")
            return False

        for player in self.engine.players:
            if not player.is_alive:
                continue

            wuxie_cards = [c for c in player.hand_cards if c.name == "无懈可击"]
            if not wuxie_cards:
                continue

            prompt = f"{player.commander_name} 是否使用无懈可击? (目标: {target_card.name if hasattr(target_card, 'name') else target_card})"
            if original_target:
                prompt += f" -> {original_target.commander_name}"

            request = ResponseRequest(
                response_type=ResponseType.WUXIE,
                prompt=prompt,
                source=source,
                target=original_target,
                card=target_card,
                can_skip=True,
            )

            response = self.ask_for_response(player, request)

            if response:
                player.hand_cards.remove(response)
                self.engine.discard_pile.append(response)

                new_wuxie_count = wuxie_count + 1
                countered = self._wuxie_chain(
                    player, response, original_target, new_wuxie_count
                )

                if not countered:
                    if wuxie_count == 0:
                        #     f">>> 无懈可击生效，{target_card.name if hasattr(target_card, 'name') else target_card} 被抵消!"
                        # )
                        pass
                    return True
                else:
                    return False

        return False


class CardResolver:
    def __init__(self, engine: "GameEngine"):
        self.engine = engine
        self.response_system = ResponseSystem(engine)

    def resolve_sha(
        self, source: "Player", target: "Player", card: "Card", event_data: dict = None
    ) -> bool:
        from engine.event import EventType
        import random

        damage = 1 + source.jiu_effect
        is_elemental = getattr(card, "is_elemental", False)
        is_fire = getattr(card, "is_fire", False) if card else False
        is_thunder = getattr(card, "is_thunder", False) if card else False

        weapon = source.equipment.get("武器")
        weapon_name = weapon.name if weapon else None

        ask_event = self.engine._emit_event(
            EventType.ASK_FOR_SHA,
            source=source,
            target=target,
            card=card,
        )
        if ask_event.is_cancelled():
            return False

        ignore_armour = weapon_name == "青釭剑"

        if target.equipment.get("防具") and not ignore_armour:
            armour = target.equipment["防具"]
            if armour.name == "仁王盾" and card.color in ["黑桃", "梅花"]:
                self.engine.log(f"仁王盾生效，黑杀无效!")
                return False
            if armour.name == "藤甲" and not is_elemental:
                self.engine.log(f"藤甲生效，普通杀无效!")
                return False

        if weapon_name == "雌雄双股剑":
            if source.gender != target.gender and target.gender:
                if source.is_human:
                    choice = input(f"发动雌雄双股剑？对方弃牌(1)/你摸牌(2): ")
                    if choice == "1":
                        if target.hand_cards:
                            discard = random.choice(target.hand_cards)
                            target.hand_cards.remove(discard)
                            self.engine.discard_pile.append(discard)
                            self.engine.log(
                                f"雌雄双股剑：{target.commander_name} 弃置了一张牌"
                            )
                    elif choice == "2":
                        drawn = self.engine.draw_cards(source, 1)
                        source.hand_cards.extend(drawn)
                        self.engine.log(
                            f"雌雄双股剑：{source.commander_name} 摸了一张牌"
                        )
                else:
                    if random.random() < 0.5 and target.hand_cards:
                        discard = random.choice(target.hand_cards)
                        target.hand_cards.remove(discard)
                        self.engine.discard_pile.append(discard)
                        self.engine.log(
                            f"雌雄双股剑：{target.commander_name} 弃置了一张牌"
                        )
                    else:
                        drawn = self.engine.draw_cards(source, 1)
                        source.hand_cards.extend(drawn)
                        self.engine.log(
                            f"雌雄双股剑：{source.commander_name} 摸了一张牌"
                        )

        wushuang_active = event_data and event_data.get("wushuang_sha", False)
        shan_needed = 2 if wushuang_active else 1

        if (
            target.equipment.get("防具")
            and target.equipment["防具"].name == "八卦阵"
            and not ignore_armour
        ):
            judge_card = self.engine.deck.pop() if self.engine.deck else None
            if judge_card:
                self.engine.discard_pile.append(judge_card)
                self.engine.log(f"八卦阵判定: {judge_card}")
                if judge_card.color in ["红桃", "方块"]:
                    self.engine.log(f"{target.commander_name} 的八卦阵生效，视为出闪!")
                    shan_needed -= 1
                    if shan_needed <= 0:
                        return False

        shan_played = 0
        while shan_played < shan_needed:
            request = ResponseRequest(
                response_type=ResponseType.SHAN,
                prompt=f"{target.commander_name} 需要出闪 ({shan_played + 1}/{shan_needed})",
                source=source,
                target=target,
                card=card,
                can_skip=True,
            )

            shan = self.response_system.ask_for_response(target, request)

            if not shan:
                break

            target.hand_cards.remove(shan)
            self.engine.discard_pile.append(shan)
            shan_played += 1
            self.engine.log(
                f"{target.commander_name} 打出了闪 ({shan_played}/{shan_needed})"
            )

        if shan_played >= shan_needed:
            if weapon_name == "贯石斧":
                if len(source.hand_cards) >= 2:
                    if source.is_human:
                        choice = input("发动贯石斧，弃置两张牌强命？(y/n): ")
                        if choice.lower() == "y":
                            for _ in range(2):
                                if source.hand_cards:
                                    discard = source.hand_cards.pop()
                                    self.engine.discard_pile.append(discard)
                            self.engine.log(
                                f"{source.commander_name} 发动贯石斧，杀依然生效!"
                            )
                            shan_played = 0
                    else:
                        if random.random() < 0.7:
                            for _ in range(2):
                                if source.hand_cards:
                                    discard = source.hand_cards.pop()
                                    self.engine.discard_pile.append(discard)
                            self.engine.log(
                                f"{source.commander_name} 发动贯石斧，杀依然生效!"
                            )
                            shan_played = 0

            if weapon_name == "青龙偃月刀" and shan_played >= shan_needed:
                sha_cards = [c for c in source.hand_cards if "杀" in c.name]
                if sha_cards:
                    if source.is_human:
                        choice = input("发动青龙偃月刀，追杀？(y/n): ")
                        if choice.lower() == "y":
                            new_sha = sha_cards[0]
                            source.hand_cards.remove(new_sha)
                            self.engine.log(
                                f"{source.commander_name} 发动青龙偃月刀追杀!"
                            )
                            return self.resolve_sha(source, target, new_sha, event_data)
                    else:
                        if random.random() < 0.5:
                            new_sha = sha_cards[0]
                            source.hand_cards.remove(new_sha)
                            self.engine.log(
                                f"{source.commander_name} 发动青龙偃月刀追杀!"
                            )
                            return self.resolve_sha(source, target, new_sha, event_data)

            if shan_played >= shan_needed:
                if wushuang_active:
                    self.engine.log(
                        f"{target.commander_name} 打出了{shan_played}张闪，躲避了杀（无双）"
                    )
                return False

        actual_damage = damage

        if weapon_name == "古锭刀" and len(target.hand_cards) == 0:
            actual_damage += 1
            self.engine.log(f"{source.commander_name} 发动古锭刀，伤害+1!")

        if (
            target.equipment.get("防具")
            and target.equipment["防具"].name == "藤甲"
            and is_fire
            and not ignore_armour
        ):
            actual_damage += 1
            self.engine.log(f"{target.commander_name} 穿着藤甲，火焰伤害+1!")
        if (
            target.equipment.get("防具")
            and target.equipment["防具"].name == "白银狮子"
            and actual_damage > 1
            and not ignore_armour
        ):
            actual_damage = 1
            self.engine.log(f"{target.commander_name} 穿着白银狮子，伤害改为1!")

        ice_sword_used = False
        if weapon_name == "寒冰剑" and (
            target.hand_cards
            or target.equipment.get("武器")
            or target.equipment.get("防具")
            or target.equipment.get("进攻坐骑")
            or target.equipment.get("防御坐骑")
        ):
            if source.is_human:
                choice = input("发动寒冰剑，防止伤害并弃置对方两张牌？(y/n): ")
                if choice.lower() == "y":
                    ice_sword_used = True
                    self._ice_sword_discard(source, target, 2)
            else:
                if random.random() < 0.7:
                    ice_sword_used = True
                    self._ice_sword_discard(source, target, 2)

        if not ice_sword_used:
            if weapon_name == "麒麟弓":
                mount = target.equipment.get("进攻坐骑") or target.equipment.get(
                    "防御坐骑"
                )
                if mount:
                    if source.is_human:
                        choice = input(f"发动麒麟弓，弃置{mount.name}？(y/n): ")
                        if choice.lower() == "y":
                            self._qilin_discard(target)
                    else:
                        self._qilin_discard(target)

            self.engine.deal_damage(
                source, target, card, actual_damage, is_elemental, is_fire, is_thunder
            )
        return True

    def _ice_sword_discard(self, source: "Player", target: "Player", count: int):
        import random

        discarded = 0
        while discarded < count:
            options = []
            if target.hand_cards:
                options.append(("hand", None))
            for slot, card in target.equipment.items():
                if card:
                    options.append(("equip", slot))
            if not options:
                break
            choice_type, slot = random.choice(options)
            if choice_type == "hand":
                card = random.choice(target.hand_cards)
                target.hand_cards.remove(card)
                self.engine.discard_pile.append(card)
                self.engine.log(f"寒冰剑弃置了 {target.commander_name} 的 {card}")
            else:
                card = target.equipment[slot]
                target.equipment[slot] = None
                self.engine.discard_pile.append(card)
                self.engine.log(f"寒冰剑弃置了 {target.commander_name} 的 {card.name}")
            discarded += 1

    def _qilin_discard(self, target: "Player"):
        if target.equipment.get("进攻坐骑"):
            card = target.equipment["进攻坐骑"]
            target.equipment["进攻坐骑"] = None
            self.engine.discard_pile.append(card)
            self.engine.log(f"麒麟弓弃置了 {target.commander_name} 的 {card.name}")
        elif target.equipment.get("防御坐骑"):
            card = target.equipment["防御坐骑"]
            target.equipment["防御坐骑"] = None
            self.engine.discard_pile.append(card)
            self.engine.log(f"麒麟弓弃置了 {target.commander_name} 的 {card.name}")

    def resolve_juedou(self, source: "Player", target: "Player") -> bool:

        if self.response_system.ask_for_wuxie(
            source, type("MockCard", (), {"name": "决斗"})(), target
        ):
            return False

        current = target
        loser = None
        max_juedou_rounds = 50
        round_count = 0

        while round_count < max_juedou_rounds:
            round_count += 1
            request = ResponseRequest(
                response_type=ResponseType.SHA,
                prompt=f"{current.commander_name} 请出杀 (决斗)",
                can_skip=True,
            )

            sha = self.response_system.ask_for_response(current, request)

            if not sha:
                loser = current
                break

            current.hand_cards.remove(sha)
            self.engine.discard_pile.append(sha)

            current = source if current == target else target

        if round_count >= max_juedou_rounds:
            import logging

            logging.warning(f"决斗达到最大轮数 {max_juedou_rounds}，强制结束")
            loser = target

        if loser:
            self.engine.deal_damage(source, loser, None, 1, False, False, False)
            return True

        return False

    def resolve_namaninru(self, source: "Player") -> List["Player"]:

        if self.response_system.ask_for_wuxie(
            source, type("MockCard", (), {"name": "南蛮入侵"})()
        ):
            return []

        damaged_players = []

        for player in self.engine.players:
            if player == source or not player.is_alive:
                continue

            if self.response_system.ask_for_wuxie(
                source, type("MockCard", (), {"name": "南蛮入侵"})(), player
            ):
                continue

            request = ResponseRequest(
                response_type=ResponseType.SHA,
                prompt=f"{player.commander_name} 受到南蛮入侵! 请出杀",
                can_skip=True,
            )

            sha = self.response_system.ask_for_response(player, request)

            if sha:
                player.hand_cards.remove(sha)
                self.engine.discard_pile.append(sha)
            else:
                self.engine.deal_damage(source, player, None, 1, False)
                damaged_players.append(player)

        return damaged_players

    def resolve_wanjianqifa(self, source: "Player") -> List["Player"]:
        from engine.event import EventType

        if self.response_system.ask_for_wuxie(
            source, type("MockCard", (), {"name": "万箭齐发"})()
        ):
            return []

        damaged_players = []

        for player in self.engine.players:
            if player == source or not player.is_alive:
                continue

            if self.response_system.ask_for_wuxie(
                source, type("MockCard", (), {"name": "万箭齐发"})(), player
            ):
                continue

            ask_event = self.engine._emit_event(
                EventType.ASK_FOR_SHAN,
                source=source,
                target=player,
            )
            if ask_event.is_cancelled():
                continue

            request = ResponseRequest(
                response_type=ResponseType.SHAN,
                prompt=f"{player.commander_name} 受到万箭齐发! 请出闪",
                can_skip=True,
            )

            shan = self.response_system.ask_for_response(player, request)

            if shan:
                player.hand_cards.remove(shan)
                self.engine.discard_pile.append(shan)
            else:
                self.engine.deal_damage(source, player, None, 1, False, False, False)
                damaged_players.append(player)

        return damaged_players

    def resolve_huogong(self, source: "Player", target: "Player") -> bool:

        if self.response_system.ask_for_wuxie(
            source, type("MockCard", (), {"name": "火攻"})(), target
        ):
            return False

        if not target.hand_cards:
            return False

        if target.is_human:
            #     f"\n你的手牌: {list(enumerate([str(c) for c in target.hand_cards], 1))}"
            # )
            idx = int(input("选择一张牌展示: ")) - 1
            show_card = (
                target.hand_cards[idx]
                if 0 <= idx < len(target.hand_cards)
                else target.hand_cards[0]
            )
        else:
            import random

            show_card = random.choice(target.hand_cards)


        source_cards = [c for c in source.hand_cards if c.color == show_card.color]
        if not source_cards:
            return False

        if source.is_human:
            available = list(enumerate([str(c) for c in source_cards], 1))
            idx = int(input("选择: ")) - 1
            if idx < 0:
                return False
            discard = source_cards[idx] if idx < len(source_cards) else None
        else:
            discard = source_cards[0]

        if discard:
            source.hand_cards.remove(discard)
            self.engine.discard_pile.append(discard)
            self.engine.deal_damage(source, target, None, 1, True, True, False)
            return True

        return False

    def resolve_jiedaosharen(
        self,
        source: "Player",
        target: "Player",
        kill_target: "Player",
        card: "Card",
    ) -> bool:
        if not target.equipment.get("武器"):
            return False

        if self.response_system.ask_for_wuxie(
            source, type("MockCard", (), {"name": "借刀杀人"})(), target
        ):
            return False

        request = ResponseRequest(
            response_type=ResponseType.SHA,
            prompt=f"{target.commander_name} 请出杀，否则交出武器",
            can_skip=True,
        )

        sha = self.response_system.ask_for_response(target, request)

        if sha:
            target.hand_cards.remove(sha)

            dist = self._calculate_distance(target, kill_target)
            if dist <= target.attack_range:
                is_fire = getattr(sha, "is_fire", False)
                is_thunder = getattr(sha, "is_thunder", False)
                is_elemental = is_fire or is_thunder
                self.engine.deal_damage(
                    target, kill_target, sha, 1, is_elemental, is_fire, is_thunder
                )
                self.engine.discard_pile.append(sha)
                return True
            else:
                target.hand_cards.append(sha)
                return False
        else:
            weapon = target.equipment["武器"]
            source.hand_cards.append(weapon)
            target.equipment["武器"] = None
            #     f"{target.commander_name} 将 {weapon.name} 交给了 {source.commander_name}"
            # )
            return True

    def _calculate_distance(self, source: "Player", target: "Player") -> int:
        if source == target:
            return 0

        dist = 1
        current = source.next_player
        while current and current != target:
            dist += 1
            current = current.next_player

        reverse_dist = 1
        current = source.prev_player
        while current and current != target:
            reverse_dist += 1
            current = current.prev_player

        dist = min(dist, reverse_dist)

        if source.equipment.get("进攻坐骑"):
            dist = max(1, dist - 1)
        if target.equipment.get("防御坐骑"):
            dist += 1

        return dist
