import random
import json
from pathlib import Path

from config import MIN_PLAYERS, MAX_PLAYERS, IDENTITY_CONFIG
from engine import GameEngine, EventType
from player import Player
from card import CardFactory
from skills.registry import SkillRegistry
import skills.wei
import skills.shu
import skills.wu
import skills.qun


def load_commanders():
    config_path = Path(__file__).parent / "data" / "commanders.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def create_player(
    commander_id: str, commander_config: dict, is_human: bool = False
) -> Player:
    skill_names = commander_config.get("skills", [])
    skills = []
    for skill_name in skill_names:
        skill = SkillRegistry.get(skill_name)
        if skill:
            skills.append(skill)

    return Player(
        commander_id=commander_id,
        commander_name=commander_config["name"],
        nation=commander_config["nation"],
        max_hp=commander_config["max_hp"],
        current_hp=commander_config["max_hp"],
        skills=skills,
        is_human=is_human,
    )


def select_commanders(commander_configs: dict, player_num: int) -> list:
    available_ids = list(commander_configs.keys())
    selected = random.sample(available_ids, player_num)
    return selected


def setup_game(player_num: int, human_player_idx: int = 0):
    if player_num < MIN_PLAYERS or player_num > MAX_PLAYERS:
        raise ValueError(f"玩家数量必须在 {MIN_PLAYERS} 到 {MAX_PLAYERS} 之间")

    commander_configs = load_commanders()
    commander_ids = select_commanders(commander_configs, player_num)

    engine = GameEngine(player_num, commander_ids, human_player_idx)

    players = []
    for i, cid in enumerate(commander_ids):
        is_human = i == human_player_idx
        player = create_player(cid, commander_configs[cid], is_human)
        players.append(player)

    engine.setup_game(players)

    return engine


def print_game_state(engine: GameEngine):
    print("\n" + "=" * 50)
    print(f"第 {engine.round_num} 轮")
    current = engine.players[engine.current_player_idx]
    print(f"当前回合: {current.idx}号位 {current.commander_name} ({current.identity})")
    print(f"体力: {current.current_hp}/{current.max_hp}")
    print(f"手牌数: {len(current.hand_cards)}")
    if current.equipment.get("武器"):
        print(f"武器: {current.equipment['武器'].name}")
    if current.judge_area:
        print(f"判定区: {[c.name for c in current.judge_area]}")
    print("=" * 50)


def print_player_info(player: Player, show_hand: bool = False):
    print(f"\n{player.idx}号位 {player.commander_name}")
    print(f"  身份: {player.identity}")
    print(f"  体力: {current_hp_display(player)}")
    if show_hand:
        print(f"  手牌: {[str(c) for c in player.hand_cards]}")
    else:
        print(f"  手牌数: {len(player.hand_cards)}")
    print(f"  装备: {equipment_display(player)}")
    if player.judge_area:
        print(f"  判定区: {[c.name for c in player.judge_area]}")
    print(f"  技能: {[s.name for s in player.skills]}")


def current_hp_display(player: Player) -> str:
    hearts = "♥" * player.current_hp
    empty = "♡" * (player.max_hp - player.current_hp)
    return hearts + empty


def equipment_display(player: Player) -> str:
    parts = []
    for slot, card in player.equipment.items():
        if card:
            parts.append(f"{slot}:{card.name}")
    return ", ".join(parts) if parts else "无"


def human_turn(engine: GameEngine, player: Player):
    while True:
        print(f"\n{'=' * 40}")
        print(f"你的回合 - {player.commander_name}")
        print(f"体力: {current_hp_display(player)}")
        print(f"手牌: {list(enumerate([str(c) for c in player.hand_cards], 1))}")
        print(f"装备: {equipment_display(player)}")
        if player.judge_area:
            print(f"判定区: {[c.name for c in player.judge_area]}")

        print("\n操作:")
        print("  1. 出牌 (输入卡牌序号)")
        print("  2. 结束回合 (输入 0)")
        print("  3. 查看所有玩家 (输入 -1)")

        try:
            choice = int(input("\n请选择: "))
        except ValueError:
            print("请输入有效数字")
            continue

        if choice == 0:
            print("结束回合")
            return "end_turn"
        elif choice == -1:
            print("\n所有玩家信息:")
            for p in engine.players:
                if p.is_alive:
                    print_player_info(p, show_hand=(p == player))
        elif 1 <= choice <= len(player.hand_cards):
            card = player.hand_cards[choice - 1]
            result = handle_card_play(engine, player, card)
            if result:
                return result
        else:
            print("无效选择")


def handle_card_play(engine: GameEngine, player: Player, card):
    print(f"\n选择使用: {card}")

    if card.name == "桃":
        if player.current_hp >= player.max_hp:
            print("体力已满，无法使用桃")
            return None
        engine.use_card(player, card, player)
        print(f"恢复1点体力，当前体力: {current_hp_display(player)}")
        return None

    if card.name == "酒":
        if player.jiu_count >= 1:
            print("本回合已使用过酒")
            return None
        engine.use_card(player, card, player)
        print("酒效果生效，下一张杀伤害+1")
        return None

    if card.name == "无中生有":
        engine.use_card(player, card, None)
        print("摸了2张牌")
        return None

    if "杀" in card.name:
        if not player.can_use_sha():
            print("本回合已使用过杀")
            return None

        targets = get_sha_targets(engine, player)
        if not targets:
            print("没有可攻击的目标")
            return None

        print("\n可选目标:")
        for i, t in enumerate(targets, 1):
            print(
                f"  {i}. {t.idx}号位 {t.commander_name} (距离:{calculate_distance(player, t)})"
            )

        try:
            target_idx = int(input("选择目标: ")) - 1
            if 0 <= target_idx < len(targets):
                target = targets[target_idx]
                success = engine.use_card(player, card, target)
                if success:
                    print(f"对 {target.commander_name} 使用{card.name}")
                return None
        except ValueError:
            print("无效选择")
        return None

    if card.name == "决斗":
        targets = [p for p in engine.players if p != player and p.is_alive]
        if not targets:
            print("没有可用目标")
            return None
        print("\n可选目标:")
        for i, t in enumerate(targets, 1):
            print(f"  {i}. {t.idx}号位 {t.commander_name}")
        try:
            target_idx = int(input("选择目标: ")) - 1
            if 0 <= target_idx < len(targets):
                target = targets[target_idx]
                engine.use_card(player, card, target)
                return None
        except ValueError:
            pass
        return None

    if card.name == "南蛮入侵":
        engine.use_card(player, card, None)
        print("使用南蛮入侵!")
        return None

    if card.name == "万箭齐发":
        engine.use_card(player, card, None)
        print("使用万箭齐发!")
        return None

    if card.name == "火攻":
        targets = [
            p for p in engine.players if p != player and p.is_alive and p.hand_cards
        ]
        if not targets:
            print("没有可用目标")
            return None
        print("\n可选目标:")
        for i, t in enumerate(targets, 1):
            print(f"  {i}. {t.idx}号位 {t.commander_name}")
        try:
            target_idx = int(input("选择目标: ")) - 1
            if 0 <= target_idx < len(targets):
                target = targets[target_idx]
                engine.use_card(player, card, target)
                return None
        except ValueError:
            pass
        return None

    if card.name in ["过河拆桥", "顺手牵羊"]:
        targets = [
            p for p in engine.players if p != player and p.is_alive and has_cards(p)
        ]
        if not targets:
            print("没有可用的目标")
            return None

        if card.name == "顺手牵羊":
            targets = [t for t in targets if calculate_distance(player, t) <= 1]
            if not targets:
                print("没有距离为1的目标")
                return None

        print("\n可选目标:")
        for i, t in enumerate(targets, 1):
            dist = calculate_distance(player, t)
            print(f"  {i}. {t.idx}号位 {t.commander_name} (距离:{dist})")

        try:
            target_idx = int(input("选择目标: ")) - 1
            if 0 <= target_idx < len(targets):
                target = targets[target_idx]
                engine.use_card(player, card, target)
                return None
        except ValueError:
            pass
        return None

    if card.name == "铁索连环":
        print("\n选择:")
        print("  1. 使用 (连环1-2名角色)")
        print("  2. 重铸 (摸一张牌)")
        try:
            choice = int(input("选择: "))
            if choice == 2:
                player.hand_cards.remove(card)
                engine.discard_pile.append(card)
                drawn = engine.draw_cards(player, 1)
                player.hand_cards.extend(drawn)
                print("重铸成功，摸了1张牌")
                return None
        except ValueError:
            pass

        targets = [p for p in engine.players if p.is_alive]
        print("\n可选目标 (可多选，用逗号分隔):")
        for i, t in enumerate(targets, 1):
            status = " [已连环]" if t.is_chained else ""
            print(f"  {i}. {t.idx}号位 {t.commander_name}{status}")

        try:
            input_str = input("选择目标: ")
            indices = [int(x.strip()) - 1 for x in input_str.split(",") if x.strip()]
            selected_targets = [targets[i] for i in indices if 0 <= i < len(targets)]
            if selected_targets:
                for t in selected_targets[:2]:
                    t.is_chained = not t.is_chained
                    status = "连环" if t.is_chained else "重置"
                    print(f"{t.commander_name} {status}")
                player.hand_cards.remove(card)
                engine.discard_pile.append(card)
            return None
        except ValueError:
            pass
        return None

    if card.name == "五谷丰登":
        engine.use_card(player, card, None)
        return None

    if card.name == "桃园结义":
        engine.use_card(player, card, None)
        return None

    if card.name == "乐不思蜀":
        targets = [p for p in engine.players if p != player and p.is_alive]
        targets = [
            t for t in targets if not any(c.name == "乐不思蜀" for c in t.judge_area)
        ]
        if not targets:
            print("没有可用目标")
            return None
        print("\n可选目标:")
        for i, t in enumerate(targets, 1):
            print(f"  {i}. {t.idx}号位 {t.commander_name}")
        try:
            target_idx = int(input("选择目标: ")) - 1
            if 0 <= target_idx < len(targets):
                target = targets[target_idx]
                engine.use_card(player, card, target)
                return None
        except ValueError:
            pass
        return None

    if card.name == "兵粮寸断":
        targets = [p for p in engine.players if p != player and p.is_alive]
        targets = [
            t for t in targets if not any(c.name == "兵粮寸断" for c in t.judge_area)
        ]
        targets = [t for t in targets if calculate_distance(player, t) <= 1]
        if not targets:
            print("没有距离为1的可用目标")
            return None
        print("\n可选目标:")
        for i, t in enumerate(targets, 1):
            print(f"  {i}. {t.idx}号位 {t.commander_name}")
        try:
            target_idx = int(input("选择目标: ")) - 1
            if 0 <= target_idx < len(targets):
                target = targets[target_idx]
                engine.use_card(player, card, target)
                return None
        except ValueError:
            pass
        return None

    if card.name == "闪电":
        engine.use_card(player, card, None)
        print("在自己判定区放置闪电")
        return None

    if card.card_type == "WeaponCard":
        engine.use_card(player, card, None)
        return None

    if card.card_type == "ArmourCard":
        engine.use_card(player, card, None)
        return None

    if card.card_type == "AttackHorseCard":
        engine.use_card(player, card, None)
        return None

    if card.card_type == "DefenseHorseCard":
        engine.use_card(player, card, None)
        return None

    if card.card_type == "TreasureCard":
        engine.use_card(player, card, None)
        return None

    print(f"暂不支持此卡牌: {card.name} ({card.card_type})")
    return None


def get_sha_targets(engine: GameEngine, player: Player):
    targets = []
    for p in engine.players:
        if p != player and p.is_alive:
            dist = calculate_distance(player, p)
            if dist <= player.attack_range:
                targets.append(p)
    return targets


def calculate_distance(source: Player, target: Player) -> int:
    if source == target:
        return 0

    dist_forward = 0
    current = source
    while current != target:
        dist_forward += 1
        current = current.next_player

    dist_backward = 0
    current = source
    while current != target:
        dist_backward += 1
        current = current.prev_player

    dist = min(dist_forward, dist_backward)

    if source.equipment.get("进攻坐骑"):
        dist = max(1, dist - 1)
    if target.equipment.get("防御坐骑"):
        dist += 1

    return dist


def has_cards(player: Player) -> bool:
    if player.hand_cards:
        return True
    for card in player.equipment.values():
        if card:
            return True
    if player.judge_area:
        return True
    return False


def ai_turn(engine: GameEngine, player: Player):
    print(f"\nAI回合: {player.commander_name}")

    if player.judge_area:
        print("  处理判定区...")

    used_sha = False
    cards_to_use = player.hand_cards.copy()

    for card in cards_to_use:
        if not player.is_alive:
            break

        if card.name == "桃" and player.current_hp < player.max_hp:
            engine.use_card(player, card, player)
            print(f"  使用桃，恢复体力")

        elif card.name == "酒" and player.jiu_count < 1:
            engine.use_card(player, card, player)
            print(f"  使用酒")

        elif card.name == "无中生有":
            engine.use_card(player, card, None)
            print(f"  使用无中生有")

        elif "杀" in card.name and not used_sha and player.can_use_sha():
            targets = get_sha_targets(engine, player)
            if targets:
                min_hp_target = min(targets, key=lambda t: t.current_hp)
                engine.use_card(player, card, min_hp_target)
                used_sha = True
                print(f"  对 {min_hp_target.commander_name} 使用杀")

        elif card.name == "南蛮入侵":
            engine.use_card(player, card, None)
            print(f"  使用南蛮入侵")

        elif card.name == "万箭齐发":
            engine.use_card(player, card, None)
            print(f"  使用万箭齐发")

        elif card.name in ["过河拆桥", "顺手牵羊"]:
            targets = [
                p for p in engine.players if p != player and p.is_alive and has_cards(p)
            ]
            if card.name == "顺手牵羊":
                targets = [t for t in targets if calculate_distance(player, t) <= 1]
            if targets:
                target = random.choice(targets)
                engine.use_card(player, card, target)
                print(f"  对 {target.commander_name} 使用{card.name}")

        elif card.card_type in [
            "WeaponCard",
            "ArmourCard",
            "AttackHorseCard",
            "DefenseHorseCard",
        ]:
            engine.use_card(player, card, None)
            print(f"  装备 {card.name}")

    while len(player.hand_cards) > player.hand_limit:
        card = random.choice(player.hand_cards)
        player.hand_cards.remove(card)
        engine.discard_pile.append(card)
        print(f"  弃置 {card}")

    player.reset_turn_state()


def game_loop(engine: GameEngine):
    print("\n游戏开始!")

    while engine.phase.value != "game_over":
        current = engine.players[engine.current_player_idx]

        if not current.is_alive:
            engine.next_turn()
            continue

        print_game_state(engine)

        current.reset_turn_state()

        judge_result = engine.judge_phase(current)

        if judge_result["lightning_damage"] > 0:
            engine.deal_damage(
                None, current, None, judge_result["lightning_damage"], True
            )

        if not judge_result["skip_draw"]:
            drawn = engine.draw_cards(current, 2)
            current.hand_cards.extend(drawn)
            print(f"摸了{len(drawn)}张牌")

        if not judge_result["skip_play"]:
            if current.is_human:
                result = human_turn(engine, current)
            else:
                ai_turn(engine, current)

        while len(current.hand_cards) > current.hand_limit:
            if current.is_human:
                print(f"\n需要弃置 {len(current.hand_cards) - current.hand_limit} 张牌")
                print(
                    f"手牌: {list(enumerate([str(c) for c in current.hand_cards], 1))}"
                )
                try:
                    idx = int(input("选择弃置的牌: ")) - 1
                    if 0 <= idx < len(current.hand_cards):
                        card = current.hand_cards.pop(idx)
                        engine.discard_pile.append(card)
                except ValueError:
                    pass
            else:
                card = random.choice(current.hand_cards)
                current.hand_cards.remove(card)
                engine.discard_pile.append(card)
                print(f"  弃置 {card}")

        engine.next_turn()

    print("\n游戏结束!")
    state = engine.get_state()
    if state.winner:
        winner_map = {"主公": "主公阵营", "反贼": "反贼阵营", "内奸": "内奸"}
        print(f"胜利方: {winner_map.get(state.winner, state.winner)}")


def main():
    print("=" * 50)
    print("三国杀 - 人机对战版")
    print("=" * 50)

    print(f"\n支持 {MIN_PLAYERS}-{MAX_PLAYERS} 人游戏")

    try:
        player_num = int(input("请输入玩家数量 (默认2): ") or "2")
    except ValueError:
        player_num = 2

    player_num = max(MIN_PLAYERS, min(MAX_PLAYERS, player_num))

    print(f"\n游戏人数: {player_num}")
    print("你是1号位玩家\n")

    engine = setup_game(player_num, human_player_idx=0)

    print("\n武将分配:")
    for p in engine.players:
        human_mark = " [玩家]" if p.is_human else " [AI]"
        print(f"  {p.idx}号位: {p.commander_name} ({p.nation}){human_mark}")

    print("\n身份分配:")
    for p in engine.players:
        if p.identity == "主公":
            print(f"  {p.idx}号位 {p.commander_name} 是主公")

    input("\n按回车开始游戏...")

    game_loop(engine)


if __name__ == "__main__":
    main()
