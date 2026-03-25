from typing import Optional, List, TYPE_CHECKING
from skills.base import TriggerSkill, ActiveSkill
from skills.registry import SkillRegistry
from engine.event import Event, EventType

if TYPE_CHECKING:
    from engine.game_engine import GameEngine
    from player.player import Player
    from card.base import Card


@SkillRegistry.register
class JianXiong(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="奸雄",
            trigger_events=[EventType.DAMAGE_TAKEN],
            description="当你受到伤害后，你可以获得造成伤害的牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.DAMAGE_TAKEN:
            return False
        if event.target != self.player:
            return False
        if not event.card:
            return False
        return True

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if self.player and event.card:
            if self.ask_player("是否发动【奸雄】获得造成伤害的牌?"):
                self.player.hand_cards.append(event.card)
                if event.card in engine.tmp_cards:
                    engine.tmp_cards.remove(event.card)
                print(f">>> {self.player.commander_name} 发动【奸雄】获得 {event.card}")
        return event


@SkillRegistry.register
class Guicai(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="鬼才",
            trigger_events=[EventType.JUDGE_BEFORE],
            description="在判定生效前，你可以打出一张手牌代替判定牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.JUDGE_BEFORE:
            return False
        if not self.player or not self.player.hand_cards:
            return False
        return True

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not self.player.hand_cards:
            return event

        judge_target = event.data.get("judge_target")
        if judge_target and judge_target != self.player:
            if not self.ask_player("是否发动【鬼才】改判?"):
                return event

        if self.player.is_human:
            print(
                f"手牌: {list(enumerate([str(c) for c in self.player.hand_cards], 1))}"
            )
            print("输入 -1 跳过")
            try:
                idx = int(input("选择一张牌改判: ")) - 1
                if 0 <= idx < len(self.player.hand_cards):
                    card = self.player.hand_cards.pop(idx)
                    engine.discard_pile.append(event.card)
                    event.card = card
            except ValueError:
                pass
        else:
            from ai.skill_decision import SkillDecisionType

            selected = self.ask_decision(
                SkillDecisionType.SELECT_CARDS,
                self.player.hand_cards,
                min_selections=1,
                max_selections=1,
                default=[0],
            )

            if selected:
                idx = selected[0] if isinstance(selected, list) else selected
                if 0 <= idx < len(self.player.hand_cards):
                    card = self.player.hand_cards.pop(idx)
                    engine.discard_pile.append(event.card)
                    event.card = card

        return event


@SkillRegistry.register
class Fankui(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="反馈",
            trigger_events=[EventType.DAMAGE_TAKEN],
            description="当你受到伤害后，你可以获得伤害来源的一张牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.DAMAGE_TAKEN:
            return False
        if event.target != self.player:
            return False
        if not event.source:
            return False
        source = event.source
        if not source.hand_cards and not any(source.equipment.values()):
            return False
        return True

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not event.source:
            return event

        source = event.source
        if self.ask_player(f"是否发动【反馈】获得 {source.commander_name} 的一张牌?"):
            available = source.hand_cards.copy()
            for slot, equip in source.equipment.items():
                if equip:
                    available.append(equip)

            if available:
                import random

                card = random.choice(available)
                if card in source.hand_cards:
                    source.hand_cards.remove(card)
                else:
                    for slot, equip in source.equipment.items():
                        if equip == card:
                            source.equipment[slot] = None
                            break
                self.player.hand_cards.append(card)
                print(
                    f">>> {self.player.commander_name} 发动【反馈】获得 {source.commander_name} 的 {card}"
                )

        return event


@SkillRegistry.register
class Ganglie(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="刚烈",
            trigger_events=[EventType.DAMAGE_TAKEN],
            description="当你受到伤害后，你可以进行判定：若结果不为红桃，则伤害来源选择弃置一张手牌或受到1点伤害",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.DAMAGE_TAKEN:
            return False
        if event.target != self.player:
            return False
        if not event.source:
            return False
        return True

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not event.source:
            return event

        if self.ask_player("是否发动【刚烈】?"):
            judge_card = engine.deck.pop() if engine.deck else None
            if judge_card:
                engine.discard_pile.append(judge_card)
                print(
                    f">>> {self.player.commander_name} 发动【刚烈】，判定结果: {judge_card}"
                )

                if judge_card.color != "红桃":
                    source = event.source
                    print(f">>> {source.commander_name} 需要弃置一张手牌或受到1点伤害")

                    if source.is_human:
                        print("1. 弃置一张手牌")
                        print("2. 受到1点伤害")
                        choice = input("选择: ")
                        if choice == "1" and source.hand_cards:
                            import random

                            card = random.choice(source.hand_cards)
                            source.hand_cards.remove(card)
                            engine.discard_pile.append(card)
                            print(f"{source.commander_name} 弃置了 {card}")
                        else:
                            engine.deal_damage(
                                self.player, source, None, 1, False, False, False
                            )
                    else:
                        if source.hand_cards:
                            import random

                            card = random.choice(source.hand_cards)
                            source.hand_cards.remove(card)
                            engine.discard_pile.append(card)
                            print(f"{source.commander_name} 弃置了 {card}")
                        else:
                            engine.deal_damage(
                                self.player, source, None, 1, False, False, False
                            )

        return event


@SkillRegistry.register
class Tuxi(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="突袭",
            trigger_events=[EventType.DRAW_PHASE],
            description="摸牌阶段，你可以改为获得至多两名角色的各一张手牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.DRAW_PHASE:
            return False
        if event.source != self.player:
            return False
        others_with_cards = [
            p
            for p in engine.players
            if p != self.player and p.is_alive and p.hand_cards
        ]
        return len(others_with_cards) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【突袭】?"):
            others = [
                p
                for p in engine.players
                if p != self.player and p.is_alive and p.hand_cards
            ]

            if self.player.is_human:
                print("可选目标:")
                for i, p in enumerate(others, 1):
                    print(f"  {i}. {p.commander_name} ({len(p.hand_cards)}张手牌)")
                    pass

                targets = []
                for _ in range(min(2, len(others))):
                    try:
                        idx = int(input("选择目标 (0结束): ")) - 1
                        if idx < 0:
                            break
                        if 0 <= idx < len(others):
                            targets.append(others.pop(idx))
                    except ValueError:
                        break
            else:
                import random

                targets = random.sample(others, min(2, len(others)))

            for target in targets:
                if target.hand_cards:
                    import random

                    card = random.choice(target.hand_cards)
                    target.hand_cards.remove(card)
                    self.player.hand_cards.append(card)
                    print(
                        f">>> {self.player.commander_name} 发动【突袭】获得 {target.commander_name} 的 {card}"
                    )

            event.cancel()
            return None

        return event


@SkillRegistry.register
class Luoyi(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="裸衣",
            trigger_events=[EventType.DRAW_PHASE],
            description="摸牌阶段，你可以少摸一张牌，然后本回合你使用杀或决斗造成的伤害+1",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.DRAW_PHASE:
            return False
        return event.source == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【裸衣】?"):
            self.player.luoyi_active = True
            self.player.cards_to_draw = 1
            print(f">>> {self.player.commander_name} 发动【裸衣】，少摸一张牌，伤害+1")

        return event


@SkillRegistry.register
class Tiandu(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="天妒",
            trigger_events=[EventType.JUDGE_RESULT],
            description="当判定牌生效后，你可以获得此牌",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.JUDGE_RESULT:
            return False
        if not event.card:
            return False
        judge_target = event.data.get("judge_target")
        return judge_target == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player or not event.card:
            return event

        if self.ask_player(f"是否发动【天妒】获得判定牌 {event.card}?"):
            self.player.hand_cards.append(event.card)
            if event.card in engine.discard_pile:
                engine.discard_pile.remove(event.card)
            print(f">>> {self.player.commander_name} 发动【天妒】获得 {event.card}")

        return event


@SkillRegistry.register
class Yiji(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="遗计",
            trigger_events=[EventType.DAMAGE_TAKEN],
            description="当你受到1点伤害后，你可以观看牌堆顶的两张牌，然后将这些牌交给任意角色",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.DAMAGE_TAKEN:
            return False
        return event.target == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        damage = event.value if event.value else 1
        for _ in range(damage):
            if not self.ask_player("是否发动【遗计】?"):
                continue

            cards = []
            for _ in range(2):
                if engine.deck:
                    cards.append(engine.deck.pop())

            if not cards:
                continue

            if self.player.is_human:
                while cards:
                    alive = [p for p in engine.players if p.is_alive]

                    print(f"剩余牌: {list(enumerate([str(c) for c in cards], 1))}")
                    print("可选目标:")
                    for i, p in enumerate(alive, 1):
                        print(f"  {i}. {p.commander_name}")

                    try:
                        card_idx = int(input("选择卡牌: ")) - 1
                        target_idx = int(input("选择目标: ")) - 1

                        if 0 <= card_idx < len(cards) and 0 <= target_idx < len(alive):
                            card = cards.pop(card_idx)
                            alive[target_idx].hand_cards.append(card)
                    except (ValueError, IndexError):
                        pass
            else:
                alive = [p for p in engine.players if p.is_alive]

                distribution = self.ask_distribute(cards, alive)

                if distribution is None:
                    import random

                    for card in cards:
                        target = random.choice(alive)
                        target.hand_cards.append(card)
                else:
                    for card_idx, target_idx in distribution.items():
                        if card_idx < len(cards) and target_idx < len(alive):
                            target = alive[target_idx]
                            target.hand_cards.append(cards[card_idx])

        return event


@SkillRegistry.register
class Luoshen(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="洛神",
            trigger_events=[EventType.TURN_START],
            description="准备阶段，你可以进行判定：若结果为黑色，你获得判定牌，并可重复此流程",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.TURN_START:
            return False
        return event.source == self.player

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【洛神】?"):
            print(f">>> {self.player.commander_name} 发动【洛神】")
            while True:
                if not engine.deck:
                    break

                judge_card = engine.deck.pop()
                engine.discard_pile.append(judge_card)
                print(f"判定结果: {judge_card}")

                if judge_card.color in ["黑桃", "梅花"]:
                    self.player.hand_cards.append(judge_card)
                    engine.discard_pile.remove(judge_card)
                    print(f"{self.player.commander_name} 获得 {judge_card}")

                    if not self.ask_player("是否继续判定?"):
                        break
                else:
                    break

        return event


@SkillRegistry.register
class Qingguo(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="倾国",
            trigger_events=[EventType.ASK_FOR_SHAN],
            description="你可以将一张黑色手牌当闪使用或打出",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.ASK_FOR_SHAN:
            return False
        if event.target != self.player:
            return False
        black_cards = [c for c in self.player.hand_cards if c.color in ["黑桃", "梅花"]]
        return len(black_cards) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        black_cards = [c for c in self.player.hand_cards if c.color in ["黑桃", "梅花"]]
        if not black_cards:
            return event

        if not self.ask_player("是否发动【倾国】将黑色牌当闪使用?"):
            return event

        if self.player.is_human:
            print(f"可选牌: {list(enumerate([str(c) for c in black_cards], 1))}")
            try:
                idx = int(input("选择卡牌: ")) - 1
                if 0 <= idx < len(black_cards):
                    card = black_cards[idx]
                    self.player.hand_cards.remove(card)
                    engine.discard_pile.append(card)
                    event.data["shan_used"] = True
            except ValueError:
                pass
        else:
            from ai.skill_decision import SkillDecisionType

            selected = self.ask_decision(
                SkillDecisionType.SELECT_CARDS,
                black_cards,
                min_selections=1,
                max_selections=1,
                default=[0],
            )

            if selected:
                idx = selected[0] if isinstance(selected, list) else selected
                if 0 <= idx < len(black_cards):
                    card = black_cards[idx]
                    self.player.hand_cards.remove(card)
                    engine.discard_pile.append(card)
                    event.data["shan_used"] = True

        return event


@SkillRegistry.register
class Hujia(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="护驾",
            trigger_events=[EventType.ASK_FOR_SHAN],
            description="主公技，当你需要使用或打出一张闪时，你可以发动护驾。所有魏势力角色按行动顺序决定是否打出一张闪："
            "若有角色（或你是魏势力角色）如此做，视为你使用或打出了一张闪；若没有，你结束发动护驾",
        )

    def can_activate(self, event: Event, engine: "GameEngine") -> bool:
        if event.type != EventType.ASK_FOR_SHAN:
            return False
        if event.target != self.player:
            return False
        if not self.player:
            return False
        wei_players = [
            p
            for p in engine.players
            if p.is_alive and p.nation == "魏" and p.hand_cards
        ]
        return len(wei_players) > 0

    def execute(self, event: Event, engine: "GameEngine") -> Optional[Event]:
        if not self.player:
            return event

        if self.ask_player("是否发动【护驾】?"):
            wei_players = [
                p
                for p in engine.players
                if p.is_alive and p.nation == "魏" and p != self.player and p.hand_cards
            ]

            for responder in wei_players:
                shan_cards = [c for c in responder.hand_cards if c.name == "闪"]
                if shan_cards:
                    if responder.is_human:
                        if self.ask_player(
                            f"{responder.commander_name} 是否响应【护驾】打出闪?"
                        ):
                            card = shan_cards[0]
                            responder.hand_cards.remove(card)
                            engine.discard_pile.append(card)
                            event.data["shan_used"] = True
                            return event
                    else:
                        import random

                        if random.random() < 0.7:
                            card = shan_cards[0]
                            responder.hand_cards.remove(card)
                            engine.discard_pile.append(card)
                            event.data["shan_used"] = True
                            return event

            if self.player.nation == "魏":
                shan_cards = [c for c in self.player.hand_cards if c.name == "闪"]
                if shan_cards:
                    event.data["shan_used"] = True

        return event
