from typing import Optional, List, TYPE_CHECKING
from skills.base import TriggerSkill, ActiveSkill
from skills.registry import SkillRegistry
from engine.event import Event, EventType

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player
    from card.base import Card


@SkillRegistry.register
class Rende(ActiveSkill):
    def __init__(self):
        super().__init__(
            name="仁德",
            trigger_events=[],
            description="出牌阶段，你可以将任意数量的手牌交给其他角色，每以此法给出两张及以上的牌，你回复1点体力",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        return self.player is not None and len(self.player.hand_cards) > 0

    def is_available(self, engine: "GameEngine") -> bool:
        """检查仁德是否可用：需要有手牌且有其他存活角色"""
        if self.player is None:
            return False
        if len(self.player.hand_cards) == 0:
            return False
        # 检查是否有其他存活角色
        others = [p for p in engine.players if p != self.player and p.is_alive]
        return len(others) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not self.player.hand_cards:
            return event

        print(f">>> {self.player.commander_name} 发动【仁德】")

        given_count = 0
        while self.player.hand_cards:
            others = [p for p in engine.players if p != self.player and p.is_alive]
            if not others:
                break

            if self.player.is_human:
                print(
                    f"手牌: {list(enumerate([str(c) for c in self.player.hand_cards], 1))}"
                )
                print("可选目标:")
                for i, p in enumerate(others, 1):
                    print(f"  {i}. {p.commander_name}")
                    pass
                print("输入 0 结束")

                try:
                    card_idx = int(input("选择卡牌: ")) - 1
                    if card_idx < 0:
                        break
                    target_idx = int(input("选择目标: ")) - 1
                    if 0 <= card_idx < len(
                        self.player.hand_cards
                    ) and 0 <= target_idx < len(others):
                        card = self.player.hand_cards.pop(card_idx)
                        others[target_idx].hand_cards.append(card)
                        given_count += 1
                        print(f"将 {card} 交给 {others[target_idx].commander_name}")
                except ValueError:
                    break
            else:
                import random

                if random.random() < 0.3:
                    break
                card = random.choice(self.player.hand_cards)
                target = random.choice(others)
                self.player.hand_cards.remove(card)
                target.hand_cards.append(card)
                given_count += 1
                print(f"将 {card} 交给 {target.commander_name}")

        if given_count >= 2 and self.player.current_hp < self.player.max_hp:
            self.player.current_hp += 1
            print(
                f">>> {self.player.commander_name} 给出了 {given_count} 张牌，回复1点体力"
            )

        return event


@SkillRegistry.register
class Wusheng(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="武圣",
            trigger_events=[EventType.BEFORE_USE_CARD, EventType.ASK_FOR_SHA],
            description="你可以将一张红色牌当杀使用或打出",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if self.player is None:
            return False

        red_cards = [c for c in self.player.hand_cards if c.is_red()]
        if not red_cards:
            return False

        if event.type == EventType.BEFORE_USE_CARD:
            return event.source == self.player
        elif event.type == EventType.ASK_FOR_SHA:
            return event.target == self.player

        return False

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        red_cards = [c for c in self.player.hand_cards if c.is_red()]
        if not red_cards:
            return event

        if not self.ask_player("是否发动【武圣】将红色牌当杀使用?"):
            return event

        if self.player.is_human:
            print(f"可选牌: {list(enumerate([str(c) for c in red_cards], 1))}")
            try:
                idx = int(input("选择卡牌: ")) - 1
                if 0 <= idx < len(red_cards):
                    event.data["use_as_sha"] = red_cards[idx]
            except ValueError:
                pass
        else:
            from ai.skill_decision import SkillDecisionType

            selected = self.ask_decision(
                SkillDecisionType.SELECT_CARDS,
                red_cards,
                min_selections=1,
                max_selections=1,
                default=[0],
            )

            if selected:
                idx = selected[0] if isinstance(selected, list) else selected
                if 0 <= idx < len(red_cards):
                    event.data["use_as_sha"] = red_cards[idx]

        return event


@SkillRegistry.register
class Paoxiao(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="咆哮",
            trigger_events=[EventType.TURN_START],
            description="你可以出任意数量的杀",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        return event.source == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if self.player:
            self.player.unlimited_sha = True
            print(f">>> {self.player.commander_name} 的【咆哮】生效，可出任意数量的杀")
        return event


@SkillRegistry.register
class Guanxing(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="观星",
            trigger_events=[EventType.TURN_START],
            description="准备阶段，你可以观看牌堆顶的X张牌（X为存活角色数且至多为5），然后以任意顺序放回牌堆顶或牌堆底",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        return event.source == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if not self.ask_player("是否发动【观星】?"):
            return event

        alive_count = min(5, len([p for p in engine.players if p.is_alive]))
        cards = []
        for _ in range(alive_count):
            if engine.deck:
                cards.append(engine.deck.pop())

        if not cards:
            return event

        if self.player.is_human:
            print(f"牌堆顶牌: {list(enumerate([str(c) for c in cards], 1))}")
            print("选择放回顺序 (从牌堆顶到底，用逗号分隔):")
            order = input("顺序: ")
            indices = [int(x.strip()) - 1 for x in order.split(",") if x.strip()]

            ordered_cards = []
            for idx in indices:
                if 0 <= idx < len(cards):
                    ordered_cards.append(cards[idx])

            remaining = [c for i, c in enumerate(cards) if i not in indices]
            ordered_cards.extend(remaining)

            for c in reversed(ordered_cards):
                engine.deck.append(c)
        else:
            order = self.ask_select_order(cards)

            if order is None:
                import random

                random.shuffle(cards)
                for c in cards:
                    engine.deck.append(c)
            else:
                ordered_cards = [cards[i] for i in order]
                for c in reversed(ordered_cards):
                    engine.deck.append(c)

        return event


@SkillRegistry.register
class Kongcheng(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="空城",
            trigger_events=[EventType.BEFORE_DAMAGE],
            description="锁定技，当你没有手牌时，你不能成为杀或决斗的目标",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.BEFORE_DAMAGE:
            return False
        if event.target != self.player:
            return False
        if not self.player:
            return False
        if self.player.hand_cards:
            return False
        if not event.card:
            return False
        return event.card.name in ["杀", "决斗"]

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if self.player and not self.player.hand_cards:
            print(
                f">>> {self.player.commander_name} 的【空城】生效，不能成为杀或决斗的目标"
            )
            event.cancel()
        return event


@SkillRegistry.register
class Longdan(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="龙胆",
            trigger_events=[
                EventType.BEFORE_USE_CARD,
                EventType.ASK_FOR_SHAN,
                EventType.ASK_FOR_SHA,
            ],
            description="你可以将杀当闪使用或打出，或将闪当杀使用或打出",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False

        if event.type == EventType.ASK_FOR_SHAN:
            sha_cards = [c for c in self.player.hand_cards if "杀" in c.name]
            return len(sha_cards) > 0
        elif event.type in [EventType.BEFORE_USE_CARD, EventType.ASK_FOR_SHA]:
            shan_cards = [c for c in self.player.hand_cards if c.name == "闪"]
            return len(shan_cards) > 0

        return False

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if event.type == EventType.ASK_FOR_SHAN:
            sha_cards = [c for c in self.player.hand_cards if "杀" in c.name]
            if not sha_cards:
                return event
            if not self.ask_player("是否发动【龙胆】将杀当闪使用?"):
                return event

            if self.player.is_human:
                print(f"可选牌: {list(enumerate([str(c) for c in sha_cards], 1))}")
                try:
                    idx = int(input("选择卡牌: ")) - 1
                    if 0 <= idx < len(sha_cards):
                        card = sha_cards[idx]
                        event.data["shan_used"] = True
                        self.player.hand_cards.remove(card)
                        engine.discard_pile.append(card)
                except ValueError:
                    pass
            else:
                from ai.skill_decision import SkillDecisionType

                selected = self.ask_decision(
                    SkillDecisionType.SELECT_CARDS,
                    sha_cards,
                    min_selections=1,
                    max_selections=1,
                    default=[0],
                )

                if selected:
                    idx = selected[0] if isinstance(selected, list) else selected
                    if 0 <= idx < len(sha_cards):
                        card = sha_cards[idx]
                        event.data["shan_used"] = True
                        self.player.hand_cards.remove(card)
                        engine.discard_pile.append(card)

        elif event.type in [EventType.BEFORE_USE_CARD, EventType.ASK_FOR_SHA]:
            shan_cards = [c for c in self.player.hand_cards if c.name == "闪"]
            if not shan_cards:
                return event
            if not self.ask_player("是否发动【龙胆】将闪当杀使用?"):
                return event

            if self.player.is_human:
                print(f"可选牌: {list(enumerate([str(c) for c in shan_cards], 1))}")
                try:
                    idx = int(input("选择卡牌: ")) - 1
                    if 0 <= idx < len(shan_cards):
                        event.data["use_as_sha"] = shan_cards[idx]
                except ValueError:
                    pass
            else:
                from ai.skill_decision import SkillDecisionType

                selected = self.ask_decision(
                    SkillDecisionType.SELECT_CARDS,
                    shan_cards,
                    min_selections=1,
                    max_selections=1,
                    default=[0],
                )

                if selected:
                    idx = selected[0] if isinstance(selected, list) else selected
                    if 0 <= idx < len(shan_cards):
                        event.data["use_as_sha"] = shan_cards[idx]

        return event

        return event


@SkillRegistry.register
class Mashu(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="马术",
            trigger_events=[EventType.TURN_START],
            description="锁定技，你计算与其他角色的距离-1",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        return event.source == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if self.player:
            self.player.has_mashu = True
        return event


@SkillRegistry.register
class Tieqi(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="铁骑",
            trigger_events=[EventType.CARD_TARGETED],
            description="当你使用杀指定目标后，你可以进行判定：若结果为红色，此杀不可被闪响应",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.CARD_TARGETED:
            return False
        if event.source != self.player:
            return False
        if not event.card or "杀" not in event.card.name:
            return False
        return True

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not event.card:
            return event

        if self.ask_player("是否发动【铁骑】?"):
            judge_card = engine.deck.pop() if engine.deck else None
            if judge_card:
                engine.discard_pile.append(judge_card)
                print(
                    f">>> {self.player.commander_name} 发动【铁骑】，判定结果: {judge_card}"
                )

                if judge_card.color in ["红桃", "方块"]:
                    event.data["tieqi_success"] = True
                    print(f">>> 判定为红色，此杀不可被闪响应！")
                else:
                    print(f">>> 判定为黑色，铁骑失败")
                    pass

        return event


@SkillRegistry.register
class Jizhi(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="集智",
            trigger_events=[EventType.CARD_USED],
            description="当你使用一张锦囊牌时，你可以摸一张牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.CARD_USED:
            return False
        if event.source != self.player:
            return False
        if not event.card:
            return False
        return event.card.card_type in ["CommonJinnangCard", "YanshiJinnangCard"]

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【集智】摸一张牌?"):
            drawn = engine.draw_cards(self.player, 1)
            self.player.hand_cards.extend(drawn)

        return event


@SkillRegistry.register
class Qicai(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="奇才",
            trigger_events=[EventType.BEFORE_USE_CARD],
            description="锁定技，你使用锦囊牌无距离限制",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.BEFORE_USE_CARD:
            return False
        if event.source != self.player:
            return False
        if not event.card:
            return False
        return event.card.card_type in ["CommonJinnangCard", "YanshiJinnangCard"]

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if self.player and event.card:
            event.data["no_distance_limit"] = True
        return event


@SkillRegistry.register
class Jijiang(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="激将",
            trigger_events=[EventType.ASK_FOR_SHA],
            description="主公技，当你需要使用或打出一张杀时，你可以发动激将。所有蜀势力角色按行动顺序决定是否打出一张杀"
            "：若有角色（或你是蜀势力角色）如此做，视为你使用或打出了一张杀；若没有，你结束发动激将",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.ASK_FOR_SHA:
            return False
        if event.target != self.player:
            return False
        if not self.player:
            return False
        shu_players = [
            p
            for p in engine.players
            if p.is_alive and p.nation == "蜀" and p.hand_cards
        ]
        return len(shu_players) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【激将】?"):
            shu_players = [
                p
                for p in engine.players
                if p.is_alive and p.nation == "蜀" and p != self.player and p.hand_cards
            ]

            for responder in shu_players:
                sha_cards = [c for c in responder.hand_cards if "杀" in c.name]
                if sha_cards:
                    if responder.is_human:
                        if self.ask_player(
                            f"{responder.commander_name} 是否响应【激将】打出杀?"
                        ):
                            card = sha_cards[0]
                            responder.hand_cards.remove(card)
                            engine.discard_pile.append(card)
                            event.data["sha_used"] = True
                            return event
                    else:
                        import random

                        if random.random() < 0.7:
                            card = sha_cards[0]
                            responder.hand_cards.remove(card)
                            engine.discard_pile.append(card)
                            event.data["sha_used"] = True
                            return event

            if self.player.nation == "蜀":
                sha_cards = [c for c in self.player.hand_cards if "杀" in c.name]
                if sha_cards:
                    event.data["sha_used"] = True

        return event
