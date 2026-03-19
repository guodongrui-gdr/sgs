from typing import Optional, List, Callable, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from player.player import Player
    from card.base import Card
    from engine.game_engine import GameEngine


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
        # print(f"\n{request.prompt}")
        # print(f"可用卡牌: {list(enumerate([str(c) for c in available_cards], 1))}")

        if request.can_skip:
            # print("输入 0 跳过")
            pass

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
                # print(f">>> {player.commander_name} 使用无懈可击!")
                player.hand_cards.remove(response)
                self.engine.discard_pile.append(response)

                new_wuxie_count = wuxie_count + 1
                countered = self._wuxie_chain(
                    player, response, original_target, new_wuxie_count
                )

                if not countered:
                    if wuxie_count == 0:
                        # print(
                        #     f">>> 无懈可击生效，{target_card.name if hasattr(target_card, 'name') else target_card} 被抵消!"
                        # )
                        pass
                    return True
                else:
                    # print(f">>> {player.commander_name} 的无懈可击被抵消!")
                    return False

        return False


class CardResolver:
    def __init__(self, engine: "GameEngine"):
        self.engine = engine
        self.response_system = ResponseSystem(engine)

    def resolve_sha(self, source: "Player", target: "Player", card: "Card") -> bool:
        damage = 1 + source.jiu_effect
        is_elemental = "火" in card.name or "雷" in card.name

        if target.equipment.get("防具"):
            armour = target.equipment["防具"]
            if armour.name == "仁王盾" and card.color in ["黑桃", "梅花"]:
                # print(f"仁王盾生效，黑杀无效!")
                return False
            if armour.name == "藤甲" and not is_elemental:
                # print(f"藤甲生效，普通杀无效!")
                return False

        request = ResponseRequest(
            response_type=ResponseType.SHAN,
            prompt=f"{target.commander_name} 被杀! 是否出闪?",
            source=source,
            target=target,
            card=card,
            can_skip=True,
        )

        if target.equipment.get("防具") and target.equipment["防具"].name == "八卦阵":
            import random

            judge_card = self.engine.deck.pop() if self.engine.deck else None
            if judge_card:
                self.engine.discard_pile.append(judge_card)
                # print(f"八卦阵判定: {judge_card}")
                if judge_card.color in ["红桃", "方块"]:
                    # print("八卦阵生效，视为出闪!")
                    return False

        shan = self.response_system.ask_for_response(target, request)

        if shan:
            target.hand_cards.remove(shan)
            self.engine.discard_pile.append(shan)
            # print(f"{target.commander_name} 打出了闪")
            return False

        actual_damage = damage
        if (
            target.equipment.get("防具")
            and target.equipment["防具"].name == "藤甲"
            and is_elemental
        ):
            actual_damage += 1
            # print("藤甲使火焰伤害+1!")
        if target.equipment.get("防具") and target.equipment["防具"].name == "白银狮子":
            actual_damage = max(1, actual_damage - 1)
            # print("白银狮子使伤害-1!")

        self.engine.deal_damage(source, target, card, actual_damage, is_elemental)
        return True

    def resolve_juedou(self, source: "Player", target: "Player") -> bool:
        # print(f"\n{source.commander_name} 对 {target.commander_name} 发起决斗!")

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
            # print(f"{current.commander_name} 打出了杀")

            current = source if current == target else target

        if round_count >= max_juedou_rounds:
            import logging

            logging.warning(f"决斗达到最大轮数 {max_juedou_rounds}，强制结束")
            loser = target

        if loser:
            self.engine.deal_damage(source, loser, None, 1, False)
            return True

        return False

    def resolve_namaninru(self, source: "Player") -> List["Player"]:
        # print(f"\n{source.commander_name} 使用南蛮入侵!")

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
                # print(f"{player.commander_name} 打出了杀")
            else:
                self.engine.deal_damage(source, player, None, 1, False)
                damaged_players.append(player)

        return damaged_players

    def resolve_wanjianqifa(self, source: "Player") -> List["Player"]:
        # print(f"\n{source.commander_name} 使用万箭齐发!")

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

            request = ResponseRequest(
                response_type=ResponseType.SHAN,
                prompt=f"{player.commander_name} 受到万箭齐发! 请出闪",
                can_skip=True,
            )

            shan = self.response_system.ask_for_response(player, request)

            if shan:
                player.hand_cards.remove(shan)
                self.engine.discard_pile.append(shan)
                # print(f"{player.commander_name} 打出了闪")
            else:
                self.engine.deal_damage(source, player, None, 1, False)
                damaged_players.append(player)

        return damaged_players

    def resolve_huogong(self, source: "Player", target: "Player") -> bool:
        # print(f"\n{source.commander_name} 对 {target.commander_name} 使用火攻!")

        if self.response_system.ask_for_wuxie(
            source, type("MockCard", (), {"name": "火攻"})(), target
        ):
            return False

        if not target.hand_cards:
            # print(f"{target.commander_name} 没有手牌")
            return False

        if target.is_human:
            # print(
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

        # print(f"{target.commander_name} 展示了 {show_card}")

        source_cards = [c for c in source.hand_cards if c.color == show_card.color]
        if not source_cards:
            # print(f"{source.commander_name} 没有相同花色的牌")
            return False

        if source.is_human:
            available = list(enumerate([str(c) for c in source_cards], 1))
            # print(f"\n可弃置的牌: {available}")
            # print("输入 0 跳过")
            idx = int(input("选择: ")) - 1
            if idx < 0:
                return False
            discard = source_cards[idx] if idx < len(source_cards) else None
        else:
            discard = source_cards[0]

        if discard:
            source.hand_cards.remove(discard)
            self.engine.discard_pile.append(discard)
            # print(f"{source.commander_name} 弃置了 {discard}")
            self.engine.deal_damage(source, target, None, 1, True)
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
            # print(f"{target.commander_name} 没有武器")
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
            # print(f"{target.commander_name} 打出了杀")

            dist = self._calculate_distance(target, kill_target)
            if dist <= target.attack_range:
                self.engine.deal_damage(target, kill_target, sha, 1, False)
                self.engine.discard_pile.append(sha)
                return True
            else:
                # print(f"距离不够，无法对 {kill_target.commander_name} 出杀")
                target.hand_cards.append(sha)
                return False
        else:
            weapon = target.equipment["武器"]
            source.hand_cards.append(weapon)
            target.equipment["武器"] = None
            # print(
            #     f"{target.commander_name} 将 {weapon.name} 交给了 {source.commander_name}"
            # )
            return True

    def _calculate_distance(self, source: "Player", target: "Player") -> int:
        if source == target:
            return 0

        dist = 1
        current = source.next_player
        while current != target:
            dist += 1
            current = current.next_player

        if source.equipment.get("进攻坐骑"):
            dist = max(1, dist - 1)
        if target.equipment.get("防御坐骑"):
            dist += 1

        return dist
