from typing import Optional, List, TYPE_CHECKING
from skills.base import TriggerSkill, ActiveSkill
from skills.registry import SkillRegistry
from engine.event import Event, EventType

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player
    from card.base import Card


@SkillRegistry.register
class Zhiheng(ActiveSkill):
    def __init__(self):
        super().__init__(
            name="制衡",
            trigger_events=[],
            description="出牌阶段限一次，你可以弃置任意数量的牌，然后摸等量的牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False
        if self.player.zhiheng_used:
            return False
        return len(self.player.hand_cards) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        # print(f">>> {self.player.commander_name} 发动【制衡】")

        if self.player.is_human:
            # print(
            #     f"手牌: {list(enumerate([str(c) for c in self.player.hand_cards], 1))}"
            # )
            # print("选择弃置的牌 (用逗号分隔，0结束):")
            indices = [
                int(x.strip()) - 1
                for x in input().split(",")
                if x.strip() and int(x.strip()) > 0
            ]
            cards_to_discard = [
                self.player.hand_cards[i]
                for i in indices
                if 0 <= i < len(self.player.hand_cards)
            ]
        else:
            import random

            count = random.randint(0, len(self.player.hand_cards))
            cards_to_discard = (
                random.sample(self.player.hand_cards, count) if count > 0 else []
            )

        if cards_to_discard:
            for card in cards_to_discard:
                self.player.hand_cards.remove(card)
                engine.discard_pile.append(card)

            drawn = engine.draw_cards(self.player, len(cards_to_discard))
            self.player.hand_cards.extend(drawn)
            self.player.zhiheng_used = True
            # print(f">>> 弃置了 {len(cards_to_discard)} 张牌，摸了 {len(drawn)} 张牌")

        return event


@SkillRegistry.register
class Qixi(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="奇袭",
            trigger_events=[EventType.BEFORE_USE_CARD],
            description="你可以将一张黑色牌当过河拆桥使用",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.BEFORE_USE_CARD:
            return False
        if event.source != self.player:
            return False
        if not self.player:
            return False
        black_cards = [c for c in self.player.hand_cards if c.color in ["黑桃", "梅花"]]
        return len(black_cards) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        black_cards = [c for c in self.player.hand_cards if c.color in ["黑桃", "梅花"]]
        if black_cards and self.ask_player("是否发动【奇袭】将黑色牌当过河拆桥使用?"):
            if self.player.is_human:
                # print(f"可选牌: {list(enumerate([str(c) for c in black_cards], 1))}")
                idx = int(input("选择卡牌: ")) - 1
                if 0 <= idx < len(black_cards):
                    card = black_cards[idx]
                    event.data["use_as_chaiqiao"] = card
                    # print(
                    #     f">>> {self.player.commander_name} 发动【奇袭】，将 {card} 当过河拆桥使用"
                    # )
            else:
                card = black_cards[0]
                event.data["use_as_chaiqiao"] = card
                # print(
                #     f">>> {self.player.commander_name} 发动【奇袭】，将 {card} 当过河拆桥使用"
                # )

        return event


@SkillRegistry.register
class Keji(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="克己",
            trigger_events=[EventType.DISCARD_START],
            description="锁定技，若你未于出牌阶段内使用或打出过杀，则你的手牌上限+X（X为你当前的体力值）",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.DISCARD_START:
            return False
        if event.source != self.player:
            return False
        if not self.player:
            return False
        return self.player.sha_count == 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if self.player and self.player.sha_count == 0:
            self.player.keji_active = True
            # print(f">>> {self.player.commander_name} 的【克己】生效，本回合不用弃牌")
            event.cancel()
        return event


@SkillRegistry.register
class Kurou(ActiveSkill):
    def __init__(self):
        super().__init__(
            name="苦肉",
            trigger_events=[],
            description="出牌阶段，你可以失去1点体力，然后摸两张牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False
        return self.player.current_hp > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【苦肉】?"):
            self.player.current_hp -= 1
            # print(f">>> {self.player.commander_name} 发动【苦肉】，失去1点体力")

            drawn = engine.draw_cards(self.player, 2)
            self.player.hand_cards.extend(drawn)
            # print(f">>> 摸了 {len(drawn)} 张牌")

        return event


@SkillRegistry.register
class Yingzi(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="英姿",
            trigger_events=[EventType.DRAW_PHASE],
            description="锁定技，摸牌阶段，你多摸一张牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        return event.source == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if self.player:
            self.player.cards_to_draw = 3
            # print(f">>> {self.player.commander_name} 的【英姿】生效，摸牌阶段多摸一张")
        return event


@SkillRegistry.register
class Fanjian(ActiveSkill):
    def __init__(self):
        super().__init__(
            name="反间",
            trigger_events=[],
            description="出牌阶段限一次，你可以展示一张手牌并交给一名其他角色，然后该角色选择一项：展示所有手牌，弃置所有与你展示的牌花色相同的牌；或失去1点体力",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False
        if self.player.fanjian_used:
            return False
        return len(self.player.hand_cards) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not self.player.hand_cards:
            return event

        others = [p for p in engine.players if p != self.player and p.is_alive]
        if not others:
            return event

        if self.player.is_human:
            # print(
            #     f"手牌: {list(enumerate([str(c) for c in self.player.hand_cards], 1))}"
            # )
            card_idx = int(input("选择要展示的牌: ")) - 1
            if not (0 <= card_idx < len(self.player.hand_cards)):
                return event

            # print("可选目标:")
            for i, p in enumerate(others, 1):
                # print(f"  {i}. {p.commander_name}")
                pass
            target_idx = int(input("选择目标: ")) - 1
            if not (0 <= target_idx < len(others)):
                return event

            card = self.player.hand_cards.pop(card_idx)
            target = others[target_idx]
        else:
            import random

            card = random.choice(self.player.hand_cards)
            self.player.hand_cards.remove(card)
            target = random.choice(others)

        # print(
        #     f">>> {self.player.commander_name} 发动【反间】，展示 {card} 并交给 {target.commander_name}"
        # )
        target.hand_cards.append(card)

        target_color = card.color

        if target.is_human:
            # print(f"你获得了 {card}，花色为 {target_color}")
            # print("1. 展示手牌并弃置相同花色的牌")
            # print("2. 失去1点体力")
            choice = input("选择: ")

            if choice == "1":
                same_color = [c for c in target.hand_cards if c.color == target_color]
                if same_color:
                    # print(f"弃置: {same_color}")
                    for c in same_color:
                        target.hand_cards.remove(c)
                        engine.discard_pile.append(c)
            else:
                target.current_hp -= 1
                # print(f"{target.commander_name} 失去1点体力")
        else:
            import random

            same_color = [c for c in target.hand_cards if c.color == target_color]
            if len(same_color) <= 1 and target.current_hp > 1:
                target.current_hp -= 1
                # print(f"{target.commander_name} 选择失去1点体力")
            elif same_color:
                for c in same_color:
                    target.hand_cards.remove(c)
                    engine.discard_pile.append(c)
                # print(f"{target.commander_name} 弃置了 {len(same_color)} 张牌")

        self.player.fanjian_used = True
        return event


@SkillRegistry.register
class Guose(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="国色",
            trigger_events=[EventType.BEFORE_USE_CARD],
            description="你可以将一张方块手牌当乐不思蜀使用",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False
        diamond_cards = [c for c in self.player.hand_cards if c.color == "方块"]
        return len(diamond_cards) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        diamond_cards = [c for c in self.player.hand_cards if c.color == "方块"]
        if diamond_cards and self.ask_player("是否发动【国色】将方块牌当乐不思蜀使用?"):
            if self.player.is_human:
                # print(f"可选牌: {list(enumerate([str(c) for c in diamond_cards], 1))}")
                idx = int(input("选择卡牌: ")) - 1
                if 0 <= idx < len(diamond_cards):
                    card = diamond_cards[idx]
                    event.data["use_as_lebusishu"] = card
                    # print(
                    #     f">>> {self.player.commander_name} 发动【国色】，将 {card} 当乐不思蜀使用"
                    # )
            else:
                card = diamond_cards[0]
                event.data["use_as_lebusishu"] = card
                # print(
                #     f">>> {self.player.commander_name} 发动【国色】，将 {card} 当乐不思蜀使用"
                # )

        return event


@SkillRegistry.register
class Liuli(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="流离",
            trigger_events=[EventType.CARD_TARGETED],
            description="当你成为杀的目标时，你可以弃置一张牌将此杀转移给你攻击范围内的一名其他角色",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.CARD_TARGETED:
            return False
        if event.target != self.player:
            return False
        if not event.card or "杀" not in event.card.name:
            return False
        if not self.player:
            return False
        if not self.player.hand_cards:
            return False
        return True

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【流离】转移杀的目标?"):
            if self.player.is_human:
                # print(
                #     f"手牌: {list(enumerate([str(c) for c in self.player.hand_cards], 1))}"
                # )
                idx = int(input("选择弃置的牌: ")) - 1
                if not (0 <= idx < len(self.player.hand_cards)):
                    return event
                discard_card = self.player.hand_cards.pop(idx)
            else:
                import random

                discard_card = random.choice(self.player.hand_cards)
                self.player.hand_cards.remove(discard_card)

            engine.discard_pile.append(discard_card)

            others = [p for p in engine.players if p != self.player and p.is_alive]
            in_range = []
            for p in others:
                dist = (
                    engine._calculate_distance(self.player, p)
                    if hasattr(engine, "_calculate_distance")
                    else 1
                )
                if dist <= self.player.attack_range:
                    in_range.append(p)

            if in_range:
                if self.player.is_human:
                    # print("可选目标:")
                    for i, p in enumerate(in_range, 1):
                        # print(f"  {i}. {p.commander_name}")
                        pass
                    idx = int(input("选择目标: ")) - 1
                    if 0 <= idx < len(in_range):
                        new_target = in_range[idx]
                else:
                    import random

                    new_target = random.choice(in_range)

                event.target = new_target
                # print(
                #     f">>> {self.player.commander_name} 发动【流离】，将杀转移给 {new_target.commander_name}"
                # )

        return event


@SkillRegistry.register
class Qianxun(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="谦逊",
            trigger_events=[EventType.CARD_TARGETED],
            description="锁定技，你不能成为顺手牵羊和乐不思蜀的目标",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.CARD_TARGETED:
            return False
        if event.target != self.player:
            return False
        if not event.card:
            return False
        return event.card.name in ["顺手牵羊", "乐不思蜀"]

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if self.player:
            # print(
            #     f">>> {self.player.commander_name} 的【谦逊】生效，不能成为顺手牵羊和乐不思蜀的目标"
            # )
            event.cancel()
        return event


@SkillRegistry.register
class Lianying(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="连营",
            trigger_events=[EventType.CARD_LOST],
            description="当你失去最后的手牌时，你可以令至多X名角色各摸一张牌（X为你失去的手牌数）",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.target != self.player:
            return False
        if not self.player:
            return False
        return len(self.player.hand_cards) == 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【连营】?"):
            alive = [p for p in engine.players if p.is_alive]
            x = min(len(alive), event.value if event.value else 1)

            if self.player.is_human:
                # print(f"可选 {x} 名角色")
                for i, p in enumerate(alive, 1):
                    # print(f"  {i}. {p.commander_name}")
                    pass
                indices = [
                    int(x.strip()) - 1
                    for x in input("选择目标(逗号分隔): ").split(",")
                    if x.strip()
                ]
                targets = [alive[i] for i in indices if 0 <= i < len(alive)][:x]
            else:
                import random

                targets = random.sample(alive, x)

            for target in targets:
                drawn = engine.draw_cards(target, 1)
                target.hand_cards.extend(drawn)
                # print(f">>> {target.commander_name} 摸了 {drawn[0] if drawn else '牌'}")

        return event


@SkillRegistry.register
class Jieyin(ActiveSkill):
    def __init__(self):
        super().__init__(
            name="结姻",
            trigger_events=[],
            description="出牌阶段限一次，你可以弃置两张手牌并选择一名男性角色，你与其各回复1点体力",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False
        if self.player.jieyin_used:
            return False
        if len(self.player.hand_cards) < 2:
            return False

        males = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and p.gender == "male"
        ]
        return len(males) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or len(self.player.hand_cards) < 2:
            return event

        males = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and p.gender == "male"
        ]
        if not males:
            return event

        if self.ask_player("是否发动【结姻】?"):
            if self.player.is_human:
                # print(
                #     f"手牌: {list(enumerate([str(c) for c in self.player.hand_cards], 1))}"
                # )
                indices = [
                    int(x.strip()) - 1
                    for x in input("选择两张牌(逗号分隔): ").split(",")
                    if x.strip()
                ]
                if len(indices) < 2:
                    return event

                cards = [
                    self.player.hand_cards[i]
                    for i in indices[:2]
                    if 0 <= i < len(self.player.hand_cards)
                ]
                if len(cards) < 2:
                    return event

                # print("可选目标:")
                for i, p in enumerate(males, 1):
                    # print(f"  {i}. {p.commander_name}")
                    pass
                target_idx = int(input("选择目标: ")) - 1
                if not (0 <= target_idx < len(males)):
                    return event
                target = males[target_idx]
            else:
                import random

                cards = random.sample(self.player.hand_cards, 2)
                target = random.choice(males)

            for card in cards:
                self.player.hand_cards.remove(card)
                engine.discard_pile.append(card)

            if self.player.current_hp < self.player.max_hp:
                self.player.current_hp += 1
            if target.current_hp < target.max_hp:
                target.current_hp += 1

            self.player.jieyin_used = True
            # print(
            #     f">>> {self.player.commander_name} 发动【结姻】，与 {target.commander_name} 各回复1点体力"
            # )

        return event


@SkillRegistry.register
class Xiaoji(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="枭姬",
            trigger_events=[EventType.EQUIPMENT_UNEQUIPPED],
            description="当你失去装备区里的牌后，你可以摸两张牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        return event.target == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【枭姬】摸两张牌?"):
            drawn = engine.draw_cards(self.player, 2)
            self.player.hand_cards.extend(drawn)

        return event


@SkillRegistry.register
class JiuYuan(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="救援",
            trigger_events=[EventType.PLAYER_DYING],
            description="主公技，当其他吴势力角色在你濒死状态下对你使用桃时，你回复1点额外体力",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.PLAYER_DYING:
            return False
        if event.target != self.player:
            return False
        return True

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event
        event.data["jiuyuan_active"] = True
        return event
