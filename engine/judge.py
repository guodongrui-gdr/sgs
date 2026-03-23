import random
from typing import Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from player.player import Player
    from card.base import Card
    from engine.game_engine import GameEngine


class JudgeResult(Enum):
    SUCCESS = "success"
    FAIL = "fail"
    PASS = "pass"


@dataclass
class JudgeCard:
    name: str
    success_condition: callable

    def check(self, card: "Card") -> JudgeResult:
        result = self.success_condition(card)
        if result:
            return JudgeResult.SUCCESS
        return JudgeResult.FAIL


JUDGE_CARDS = {
    "乐不思蜀": JudgeCard(
        name="乐不思蜀", success_condition=lambda c: c.color == "红桃"
    ),
    "兵粮寸断": JudgeCard(
        name="兵粮寸断", success_condition=lambda c: c.color == "梅花"
    ),
    "闪电": JudgeCard(
        name="闪电", success_condition=lambda c: c.color == "黑桃" and 2 <= c.point <= 9
    ),
}


class JudgeSystem:
    def __init__(self, engine: "GameEngine"):
        self.engine = engine

    def judge(
        self, player: "Player", judge_card_name: str, show_result: bool = True
    ) -> tuple:
        if not self.engine.deck:
            if self.engine.discard_pile:
                self.engine.deck = self.engine.discard_pile.copy()
                self.engine.discard_pile.clear()
                random.shuffle(self.engine.deck)
            else:
                return JudgeResult.FAIL, None

        judge_card = self.engine.deck.pop()
        self.engine.discard_pile.append(judge_card)

        if show_result:
            # print(f"判定牌: {judge_card}")
            pass

        judge_config = JUDGE_CARDS.get(judge_card_name)
        if judge_config:
            result = judge_config.check(judge_card)
            if result == JudgeResult.SUCCESS:
                if show_result:
                    # print(f"判定成功! {judge_card_name} 无效")
                    pass
                return JudgeResult.SUCCESS, judge_card
            else:
                if show_result:
                    # print(f"判定失败! {judge_card_name} 生效")
                    pass
                return JudgeResult.FAIL, judge_card

        return JudgeResult.PASS, judge_card

    def process_judge_phase(self, player: "Player") -> dict:
        results = {
            "skip_draw": False,
            "skip_play": False,
            "lightning_damage": 0,
            "processed_cards": [],
        }

        cards_to_remove = []

        for i, card in enumerate(player.judge_area):
            # print(f"\n判定: {card.name}")

            result, judge_card = self.judge(player, card.name)

            if card.name == "乐不思蜀":
                if result == JudgeResult.FAIL:
                    results["skip_play"] = True
                    # print("本回合不能出牌!")

            elif card.name == "兵粮寸断":
                if result == JudgeResult.FAIL:
                    results["skip_draw"] = True
                    # print("本回合不能摸牌!")

            elif card.name == "闪电":
                if result == JudgeResult.FAIL:
                    results["lightning_damage"] = 3
                    # print("受到3点雷电伤害!")
                else:
                    next_player = player.next_player
                    while not next_player.is_alive and next_player != player:
                        next_player = next_player.next_player

                    if next_player != player:
                        next_player.judge_area.append(card)
                        cards_to_remove.append(i)
                        # print(f"闪电传递给 {next_player.commander_name}")

            if result != JudgeResult.PASS:
                results["processed_cards"].append({"card": card, "result": result})

        for i in sorted(cards_to_remove, reverse=True):
            if i < len(player.judge_area):
                player.judge_area.pop(i)

        for card_info in results["processed_cards"]:
            card = card_info["card"]
            if card_info["result"] == JudgeResult.SUCCESS or card.name != "闪电":
                if card in player.judge_area:
                    player.judge_area.remove(card)
                self.engine.discard_pile.append(card)

        return results

    def add_judge_card(self, target: "Player", card: "Card") -> bool:
        for existing in target.judge_area:
            if existing.name == card.name:
                # print(f"{target.commander_name} 判定区已有同名卡牌")
                return False

        target.judge_area.append(card)
        # print(f"{target.commander_name} 判定区被放置了 {card.name}")
        return True


class DelayedTrickHandler:
    def __init__(self, engine: "GameEngine"):
        self.engine = engine
        self.judge_system = JudgeSystem(engine)

    def use_lebusishu(self, source: "Player", target: "Player", card: "Card") -> bool:
        if not self.judge_system.add_judge_card(target, card):
            return False

        if card in source.hand_cards:
            source.hand_cards.remove(card)
        return True

    def use_bingliangcunduan(
        self, source: "Player", target: "Player", card: "Card"
    ) -> bool:
        distance = self._calculate_distance(source, target)
        if distance > 1:
            # print("目标距离超过1")
            return False

        if not self.judge_system.add_judge_card(target, card):
            return False

        if card in source.hand_cards:
            source.hand_cards.remove(card)
        return True
        return True

    def use_shandian(self, source: "Player", card: "Card") -> bool:
        if not self.judge_system.add_judge_card(source, card):
            return False

        if card in source.hand_cards:
            source.hand_cards.remove(card)
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
