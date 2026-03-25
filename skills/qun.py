from typing import Optional, List, TYPE_CHECKING
from skills.base import TriggerSkill, ActiveSkill
from skills.registry import SkillRegistry
from engine.event import Event, EventType
from card.base import is_sha_card

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player
    from card.base import Card


@SkillRegistry.register
class Jiuji(ActiveSkill):
    def __init__(self):
        super().__init__(
            name="急救",
            trigger_events=[],
            description="你的回合外，你可以将一张红色牌当桃使用",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False
        red_cards = [c for c in self.player.hand_cards if c.is_red()]
        return len(red_cards) > 0

    def is_available(self, engine: "GameEngine") -> bool:
        """检查急救是否可用：需要有红色牌"""
        if not self.player:
            return False
        red_cards = [
            c for c in self.player.hand_cards if hasattr(c, "is_red") and c.is_red()
        ]
        return len(red_cards) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        red_cards = [c for c in self.player.hand_cards if c.is_red()]

        if self.player.is_human:
            print(f"可选牌: {list(enumerate([str(c) for c in red_cards], 1))}")
            idx = int(input("选择卡牌当桃使用: ")) - 1
            if 0 <= idx < len(red_cards):
                card = red_cards[idx]
                event.data["use_as_tao"] = card
                print(
                    f">>> {self.player.commander_name} 发动【急救】，将 {card} 当桃使用"
                )
        else:
            card = red_cards[0]
            event.data["use_as_tao"] = card
            print(f">>> {self.player.commander_name} 发动【急救】，将 {card} 当桃使用")

        return event


@SkillRegistry.register
class Qingnang(ActiveSkill):
    def __init__(self):
        super().__init__(
            name="青囊",
            trigger_events=[],
            description="出牌阶段限一次，你可以弃置一张手牌并选择一名其他角色，令其回复1点体力",
        )
        self.max_uses_per_turn = 1

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False
        if not self.can_use():
            return False
        if not self.player.hand_cards:
            return False
        others_need_heal = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and p.current_hp < p.max_hp
        ]
        return len(others_need_heal) > 0

    def is_available(self, engine: "GameEngine") -> bool:
        """检查青囊是否可用：需要有手牌、本回合未使用、有需要治疗的角色"""
        if not self.player:
            return False
        if not self.can_use():
            return False
        if not self.player.hand_cards:
            return False
        others_need_heal = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and p.current_hp < p.max_hp
        ]
        return len(others_need_heal) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not self.player.hand_cards:
            return event

        others = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and p.current_hp < p.max_hp
        ]
        if not others:
            return event

        if not self.ask_player("是否发动【青囊】?"):
            return event

        if self.player.is_human:
            print(
                f"手牌: {list(enumerate([str(c) for c in self.player.hand_cards], 1))}"
            )
            try:
                card_idx = int(input("选择弃置的牌: ")) - 1
                if not (0 <= card_idx < len(self.player.hand_cards)):
                    return event

                print("可选目标:")
                for i, p in enumerate(others, 1):
                    print(f"  {i}. {p.commander_name} ({p.current_hp}/{p.max_hp})")
                target_idx = int(input("选择目标: ")) - 1
                if not (0 <= target_idx < len(others)):
                    return event

                card = self.player.hand_cards.pop(card_idx)
                target = others[target_idx]
            except ValueError:
                return event
        else:
            from ai.skill_decision import SkillDecisionType

            card_selected = self.ask_decision(
                SkillDecisionType.SELECT_CARDS,
                self.player.hand_cards,
                min_selections=1,
                max_selections=1,
                default=[0],
            )

            target_selected = self.ask_decision(
                SkillDecisionType.SELECT_TARGETS,
                others,
                min_selections=1,
                max_selections=1,
                default=[0],
            )

            if card_selected and target_selected:
                card_idx = (
                    card_selected[0]
                    if isinstance(card_selected, list)
                    else card_selected
                )
                target_idx = (
                    target_selected[0]
                    if isinstance(target_selected, list)
                    else target_selected
                )

                if 0 <= card_idx < len(
                    self.player.hand_cards
                ) and 0 <= target_idx < len(others):
                    card = self.player.hand_cards.pop(card_idx)
                    target = others[target_idx]
                else:
                    return event
            else:
                return event

        engine.discard_pile.append(card)
        target.current_hp = min(target.max_hp, target.current_hp + 1)
        self.use()

        return event


@SkillRegistry.register
class Wushuang(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="无双",
            trigger_events=[EventType.CARD_TARGETED],
            description="锁定技，当你使用杀指定目标后，该角色需使用两张闪才能抵消；当你使用决斗时，该角色需打出两张杀才能抵消",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.source != self.player:
            return False
        if not event.card:
            return False

        card = event.card
        is_sha = is_sha_card(card)
        is_juedou = card.name == "决斗"

        return is_sha or is_juedou

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not event.card:
            return event

        card = event.card

        if is_sha_card(card):
            event.data["wushuang_sha"] = True
            engine.log(f"{self.player.commander_name} 的【无双】生效，目标需要出两张闪")
        elif card.name == "决斗":
            event.data["wushuang_juedou"] = True
            engine.log(f"{self.player.commander_name} 的【无双】生效，目标需要出两张杀")

        return event


@SkillRegistry.register
class Lijian(ActiveSkill):
    def __init__(self):
        super().__init__(
            name="离间",
            trigger_events=[],
            description="出牌阶段限一次，你可以弃置一张手牌并选择两名男性角色，令其中一名角色对另一名角色使用杀",
        )
        self.max_uses_per_turn = 1

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if not self.player:
            return False
        if not self.can_use():
            return False
        if not self.player.hand_cards:
            return False
        males = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and p.gender == "male"
        ]
        return len(males) >= 2

    def is_available(self, engine: "GameEngine") -> bool:
        """检查离间是否可用：需要有手牌、本回合未使用、至少2名男性角色"""
        if not self.player:
            return False
        if not self.can_use():
            return False
        if not self.player.hand_cards:
            return False
        males = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and getattr(p, "gender", "") == "male"
        ]
        return len(males) >= 2

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not self.player.hand_cards:
            return event

        males = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and p.gender == "male"
        ]
        if len(males) < 2:
            return event

        if not self.ask_player("是否发动【离间】?"):
            return event

        if self.player.is_human:
            print(
                f"手牌: {list(enumerate([str(c) for c in self.player.hand_cards], 1))}"
            )
            try:
                card_idx = int(input("选择弃置的牌: ")) - 1
                if not (0 <= card_idx < len(self.player.hand_cards)):
                    return event

                print("选择出杀的角色:")
                for i, p in enumerate(males, 1):
                    print(f"  {i}. {p.commander_name}")
                source_idx = int(input("选择: ")) - 1
                if not (0 <= source_idx < len(males)):
                    return event

                source = males.pop(source_idx)

                print("选择被杀的目标:")
                for i, p in enumerate(males, 1):
                    print(f"  {i}. {p.commander_name}")
                target_idx = int(input("选择: ")) - 1
                if not (0 <= target_idx < len(males)):
                    return event
                target = males[target_idx]

                card = self.player.hand_cards.pop(card_idx)
            except (ValueError, IndexError):
                return event
        else:
            from ai.skill_decision import SkillDecisionType

            card_idx = 0
            if len(self.player.hand_cards) > 1:
                card_request = self.ask_decision(
                    SkillDecisionType.SELECT_CARDS,
                    self.player.hand_cards,
                    min_selections=1,
                    max_selections=1,
                )
                if card_request:
                    card_idx = card_request[0]

            pair = self.ask_select_pair(males)

            if pair is None:
                import random

                card = random.choice(self.player.hand_cards)
                self.player.hand_cards.remove(card)
                source, target = random.sample(males, 2)
            else:
                card = self.player.hand_cards.pop(card_idx)
                source = males[pair[0]]
                target = males[pair[1]] if pair[1] < len(males) else males[0]

        engine.discard_pile.append(card)
        self.use()

        sha_cards = [c for c in source.hand_cards if "杀" in c.name]
        if sha_cards:
            sha = sha_cards[0]
            source.hand_cards.remove(sha)
            engine.card_resolver.resolve_sha(source, target, sha)
            engine.discard_pile.append(sha)
        else:
            engine.deal_damage(self.player, source, None, 1, False, False, False)

        return event


@SkillRegistry.register
class Biyue(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="闭月",
            trigger_events=[EventType.TURN_END],
            description="结束阶段，你可以摸一张牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        return event.source == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【闭月】摸一张牌?"):
            drawn = engine.draw_cards(self.player, 1)
            self.player.hand_cards.extend(drawn)
            print(
                f">>> {self.player.commander_name} 发动【闭月】，摸了 {drawn[0] if drawn else '牌'}"
            )

        return event
