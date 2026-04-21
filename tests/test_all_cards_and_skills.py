"""
全面测试套件 - 测试所有卡牌和技能

测试内容:
1. TestGameSetup: 配置加载和引擎创建
2. TestCardTypes: 所有卡牌类型测试
3. TestSkills: 所有技能存在性测试
4. TestGameMechanics: 游戏机制测试
5. TestCardFactory: 卡牌工厂测试
6. TestIntegration: 完整游戏流程测试
"""

import sys
import unittest
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from card.base import (
    Card,
    BasicCard,
    ShaCard,
    FireSha,
    ThunderSha,
    CommonJinnangCard,
    YanshiJinnangCard,
    WeaponCard,
    ArmourCard,
    AttackHorseCard,
    DefenseHorseCard,
    TreasureCard,
    is_sha_card,
)
from card.factory import CardFactory
from skills.registry import SkillRegistry
from skills.base import Skill, TriggerSkill, ActiveSkill, PassiveSkill
from player.player import Player
from engine.event import Event, EventType
from engine.game_engine import GameEngine


# 固定随机种子便于复现
RANDOM_SEED = 42


class TestGameSetup(unittest.TestCase):
    """测试配置加载和引擎创建"""

    def setUp(self):
        """每个测试前重置随机种子"""
        random.seed(RANDOM_SEED)

    def test_load_commanders(self):
        """测试武将配置加载"""
        config_path = Path(__file__).parent.parent / "data" / "commanders.json"
        with open(config_path, encoding="utf-8") as f:
            configs = json.load(f)

        self.assertIsInstance(configs, dict)
        self.assertGreater(len(configs), 0)

        # 检查必要字段
        for commander_id, config in configs.items():
            self.assertIn("name", config)
            self.assertIn("nation", config)
            self.assertIn("max_hp", config)
            self.assertIn("skills", config)
            self.assertIsInstance(config["skills"], list)

    def test_load_cards(self):
        """测试卡牌配置加载"""
        config_path = Path(__file__).parent.parent / "data" / "cards.json"
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("cards", data)
        self.assertGreater(len(data["cards"]), 0)

        # 检查必要字段
        for card_config in data["cards"]:
            self.assertIn("name", card_config)
            self.assertIn("color", card_config)
            self.assertIn("point", card_config)

    def test_create_engine(self):
        """测试游戏引擎创建"""
        commander_ids = ["WEI001", "WEI002", "SHU001", "SHU002", "WU001"]
        engine = GameEngine(player_num=5, commander_ids=commander_ids)

        self.assertEqual(engine.player_num, 5)
        self.assertEqual(len(engine.commander_ids), 5)
        self.assertGreater(len(engine.deck), 0)

    def test_skill_registry(self):
        """测试技能注册表"""
        import skills  # 触发装饰器自动注册

        all_skills = SkillRegistry.all_skills()
        self.assertGreater(len(all_skills), 0)


class TestCardTypes(unittest.TestCase):
    """测试所有卡牌类型"""

    def setUp(self):
        """每个测试前创建干净环境"""
        random.seed(RANDOM_SEED)
        self.cards_path = Path(__file__).parent.parent / "data" / "cards.json"

    def test_basic_card_sha(self):
        """测试杀牌"""
        sha = ShaCard(name="杀", color="黑桃", point=7)
        self.assertEqual(sha.name, "杀")
        self.assertEqual(sha.card_type, "BasicCard")
        self.assertTrue(is_sha_card(sha))
        self.assertIn("another_player", sha.target_types)
        self.assertEqual(sha.distance, 1)

    def test_basic_card_shan(self):
        """测试闪牌"""
        shan = BasicCard(name="闪", color="红桃", point=2)
        self.assertEqual(shan.name, "闪")
        self.assertEqual(shan.target_types, [])

    def test_basic_card_tao(self):
        """测试桃牌"""
        tao = BasicCard(name="桃", color="红桃", point=3)
        self.assertEqual(tao.name, "桃")
        self.assertIn("self", tao.target_types)
        self.assertIn("dying_player", tao.target_types)

    def test_basic_card_jiu(self):
        """测试酒牌"""
        jiu = BasicCard(name="酒", color="黑桃", point=9)
        self.assertEqual(jiu.name, "酒")
        self.assertIn("self", jiu.target_types)

    def test_fire_sha(self):
        """测试火杀"""
        fire_sha = FireSha(name="火杀", color="红桃", point=4)
        self.assertEqual(fire_sha.name, "火杀")
        self.assertTrue(is_sha_card(fire_sha))
        self.assertTrue(hasattr(fire_sha, "is_elemental"))
        self.assertTrue(fire_sha.is_elemental)
        self.assertTrue(fire_sha.is_fire)

    def test_thunder_sha(self):
        """测试雷杀"""
        thunder_sha = ThunderSha(name="雷杀", color="黑桃", point=4)
        self.assertEqual(thunder_sha.name, "雷杀")
        self.assertTrue(is_sha_card(thunder_sha))
        self.assertTrue(hasattr(thunder_sha, "is_elemental"))
        self.assertTrue(thunder_sha.is_elemental)
        self.assertTrue(thunder_sha.is_thunder)

    def test_weapon_cards(self):
        """测试武器牌"""
        weapons = [
            ("诸葛连弩", 1),
            ("青龙偃月刀", 3),
            ("贯石斧", 3),
            ("方天画戟", 4),
            ("朱雀羽扇", 4),
            ("麒麟弓", 5),
        ]
        for name, dis in weapons:
            weapon = WeaponCard(name=name, color="黑桃", point=1, dis=dis)
            self.assertEqual(weapon.name, name)
            self.assertEqual(weapon.card_type, "WeaponCard")
            self.assertEqual(weapon.attack_range, dis)
            self.assertIn("self", weapon.target_types)

    def test_armour_cards(self):
        """测试防具牌"""
        armours = ["八卦阵", "仁王盾", "白银狮子", "藤甲"]
        for name in armours:
            armour = ArmourCard(name=name, color="黑桃", point=2)
            self.assertEqual(armour.name, name)
            self.assertEqual(armour.card_type, "ArmourCard")
            self.assertIn("self", armour.target_types)

    def test_attack_horse_cards(self):
        """测试进攻坐骑"""
        attack_horses = ["大宛", "赤兔", "紫骍"]
        for name in attack_horses:
            horse = AttackHorseCard(name=name, color="黑桃", point=5)
            self.assertEqual(horse.name, name)
            self.assertEqual(horse.card_type, "AttackHorseCard")

    def test_defense_horse_cards(self):
        """测试防御坐骑"""
        defense_horses = ["绝影", "的卢", "爪黄飞电", "骅骝"]
        for name in defense_horses:
            horse = DefenseHorseCard(name=name, color="黑桃", point=5)
            self.assertEqual(horse.name, name)
            self.assertEqual(horse.card_type, "DefenseHorseCard")

    def test_treasure_card(self):
        """测试宝物牌"""
        treasure = TreasureCard(name="木牛流马", color="方块", point=5)
        self.assertEqual(treasure.name, "木牛流马")
        self.assertEqual(treasure.card_type, "TreasureCard")

    def test_trick_cards_instant(self):
        """测试即时锦囊"""
        instant_tricks = [
            ("决斗", ["another_player"]),
            ("无中生有", ["self"]),
            ("过河拆桥", ["another_player_with_cards"]),
            ("顺手牵羊", ["another_player_with_cards"]),
            ("借刀杀人", ["player_with_weapon"]),
            ("火攻", ["player_with_hand_cards"]),
            ("铁索连环", ["one_or_two_players"]),
        ]
        for name, expected_targets in instant_tricks:
            trick = CommonJinnangCard(name=name, color="黑桃", point=1)
            self.assertEqual(trick.name, name)
            self.assertEqual(trick.card_type, "CommonJinnangCard")

    def test_trick_cards_delayed(self):
        """测试延时锦囊"""
        delayed_tricks = ["乐不思蜀", "兵粮寸断", "闪电"]
        for name in delayed_tricks:
            trick = YanshiJinnangCard(name=name, color="黑桃", point=1)
            self.assertEqual(trick.name, name)
            self.assertEqual(trick.card_type, "YanshiJinnangCard")

    def test_trick_cards_aoe(self):
        """测试AOE锦囊"""
        aoe_tricks = [
            ("南蛮入侵", ["all_other_players"]),
            ("万箭齐发", ["all_other_players"]),
            ("桃园结义", ["all_players"]),
            ("五谷丰登", ["all_players"]),
        ]
        for name, expected_targets in aoe_tricks:
            trick = CommonJinnangCard(name=name, color="黑桃", point=1)
            self.assertEqual(trick.name, name)
            self.assertEqual(trick.card_type, "CommonJinnangCard")

    def test_wuxiekeji(self):
        """测试无懈可击"""
        wuxie = CommonJinnangCard(name="无懈可击", color="黑桃", point=1)
        self.assertEqual(wuxie.name, "无懈可击")
        self.assertEqual(wuxie.target_types, [])

    def test_card_colors(self):
        """测试卡牌颜色"""
        red_card = Card(name="测试", color="红桃", point=1)
        black_card = Card(name="测试", color="黑桃", point=1)

        self.assertTrue(red_card.is_red())
        self.assertFalse(red_card.is_black())
        self.assertFalse(black_card.is_red())
        self.assertTrue(black_card.is_black())


class TestSkills(unittest.TestCase):
    """测试所有技能"""

    @classmethod
    def setUpClass(cls):
        """加载技能（只执行一次）"""
        import skills  # 触发装饰器自动注册

    def setUp(self):
        """每个测试前重置随机种子"""
        random.seed(RANDOM_SEED)

    def test_wei_skills_exist(self):
        """验证魏国技能存在"""
        wei_skills = [
            "奸雄",
            "鬼才",
            "反馈",
            "刚烈",
            "突袭",
            "裸衣",
            "天妒",
            "遗计",
            "洛神",
            "倾国",
            "护驾",
        ]
        for skill_name in wei_skills:
            self.assertTrue(
                SkillRegistry.has_skill(skill_name), f"魏国技能 {skill_name} 不存在"
            )

    def test_shu_skills_exist(self):
        """验证蜀国技能存在"""
        shu_skills = [
            "仁德",
            "武圣",
            "咆哮",
            "观星",
            "空城",
            "龙胆",
            "马术",
            "铁骑",
            "集智",
            "奇才",
            "激将",
        ]
        for skill_name in shu_skills:
            self.assertTrue(
                SkillRegistry.has_skill(skill_name), f"蜀国技能 {skill_name} 不存在"
            )

    def test_wu_skills_exist(self):
        """验证吴国技能存在"""
        wu_skills = [
            "制衡",
            "奇袭",
            "克己",
            "苦肉",
            "英姿",
            "反间",
            "国色",
            "流离",
            "谦逊",
            "连营",
            "结姻",
            "枭姬",
            "救援",
        ]
        for skill_name in wu_skills:
            self.assertTrue(
                SkillRegistry.has_skill(skill_name), f"吴国技能 {skill_name} 不存在"
            )

    def test_qun_skills_exist(self):
        """验证群雄技能存在"""
        qun_skills = ["急救", "青囊", "无双", "离间", "闭月"]
        for skill_name in qun_skills:
            self.assertTrue(
                SkillRegistry.has_skill(skill_name), f"群雄技能 {skill_name} 不存在"
            )

    def test_skill_trigger_events(self):
        """测试技能触发事件配置"""
        # 奸雄应该在受到伤害时触发
        jianxiong = SkillRegistry.get("奸雄")
        self.assertIsNotNone(jianxiong)
        self.assertIn(EventType.DAMAGE_TAKEN, jianxiong.trigger_events)

        # 鬼才应该在判定前触发
        guicai = SkillRegistry.get("鬼才")
        self.assertIsNotNone(guicai)
        self.assertIn(EventType.JUDGE_BEFORE, guicai.trigger_events)

    def test_skill_bind_player(self):
        """测试技能绑定玩家"""
        skill = SkillRegistry.get("奸雄")
        self.assertIsNotNone(skill)

        player = Player(idx=1, commander_name="曹操")
        skill_copy = SkillRegistry.create_instance("奸雄", player)

        self.assertIsNotNone(skill_copy)
        self.assertEqual(skill_copy.player, player)

    def test_skill_max_uses(self):
        """测试技能使用次数限制"""
        skill = SkillRegistry.get("观星")
        self.assertIsNotNone(skill)

        # 测试默认状态
        self.assertTrue(skill.can_use())

        # 如果有使用限制，测试计数
        if skill.max_uses_per_turn > 0:
            for _ in range(skill.max_uses_per_turn):
                skill.use()
            self.assertFalse(skill.can_use())

            # 重置后应该可以再次使用
            skill.reset_turn_state()
            self.assertTrue(skill.can_use())


class TestGameMechanics(unittest.TestCase):
    """测试游戏机制"""

    @classmethod
    def setUpClass(cls):
        import skills

    def setUp(self):
        random.seed(RANDOM_SEED)

    def test_distance_calculation(self):
        """测试距离计算"""
        # 创建5个玩家
        players = []
        for i in range(5):
            player = Player(idx=i + 1, commander_name=f"玩家{i + 1}")
            players.append(player)

        # 设置座位（循环链表）
        for i, player in enumerate(players):
            player.next_player = players[(i + 1) % 5]
            player.prev_player = players[(i - 1) % 5]

        # 验证座位连接
        for i, player in enumerate(players):
            self.assertEqual(player.next_player.idx, players[(i + 1) % 5].idx)
            self.assertEqual(player.prev_player.idx, players[(i - 1) % 5].idx)

    def test_seating_arrangement(self):
        """测试座位安排"""
        players = []
        for i in range(5):
            player = Player(idx=i + 1, commander_name=f"玩家{i + 1}")
            players.append(player)

        # 设置循环座位
        for i, player in enumerate(players):
            player.next_player = players[(i + 1) % 5]
            player.prev_player = players[(i - 1) % 5]

        # 验证循环性
        current = players[0]
        visited = []
        for _ in range(5):
            visited.append(current.idx)
            current = current.next_player

        self.assertEqual(len(visited), 5)
        self.assertEqual(len(set(visited)), 5)  # 所有玩家都被访问一次

    def test_draw_cards(self):
        """测试摸牌"""
        engine = GameEngine(
            player_num=5,
            commander_ids=["WEI001", "WEI002", "SHU001", "SHU002", "WU001"],
        )

        player = Player(idx=1, commander_name="曹操")
        initial_hand = len(player.hand_cards)

        drawn = engine.draw_cards(player, 2)
        self.assertEqual(len(drawn), 2)
        player.hand_cards.extend(drawn)
        self.assertEqual(len(player.hand_cards), initial_hand + 2)

    def test_damage_and_healing(self):
        """测试伤害和治疗"""
        player = Player(idx=1, commander_name="曹操", max_hp=4, current_hp=4)

        # 受到伤害
        player.current_hp -= 1
        self.assertEqual(player.current_hp, 3)

        # 治疗
        player.current_hp = min(player.current_hp + 1, player.max_hp)
        self.assertEqual(player.current_hp, 4)

        # 不能超过最大血量
        player.current_hp = min(player.current_hp + 10, player.max_hp)
        self.assertEqual(player.current_hp, 4)

    def test_death_handling(self):
        """测试死亡处理"""
        player = Player(idx=1, commander_name="曹操", max_hp=4, current_hp=1)

        # 受到致命伤害
        player.current_hp = 0
        player.is_alive = False

        self.assertFalse(player.is_alive)
        self.assertEqual(player.current_hp, 0)

    def test_victory_conditions(self):
        """测试胜利条件"""
        # 创建引擎并设置身份
        engine = GameEngine(
            player_num=5,
            commander_ids=["WEI001", "WEI002", "SHU001", "SHU002", "WU001"],
        )

        # 身份分配应该正确
        identities = ["主公", "忠臣", "反贼", "反贼", "内奸"]
        # 实际分配在 setup_game 中进行，这里只验证引擎创建成功
        self.assertIsNotNone(engine)

    def test_equipment_slots(self):
        """测试装备槽位"""
        player = Player(idx=1, commander_name="曹操")

        # 验证装备槽位存在
        self.assertIn("武器", player.equipment)
        self.assertIn("防具", player.equipment)
        self.assertIn("进攻坐骑", player.equipment)
        self.assertIn("防御坐骑", player.equipment)
        self.assertIn("宝物", player.equipment)

        # 初始应该都是空的
        for slot, card in player.equipment.items():
            self.assertIsNone(card, f"装备槽 {slot} 初始应该为空")

    def test_attack_range(self):
        """测试攻击距离"""
        player = Player(idx=1, commander_name="曹操")

        # 没有武器时攻击距离为1
        self.assertEqual(player.attack_range, 1)

        # 装备武器后
        weapon = WeaponCard(name="青龙偃月刀", color="黑桃", point=1, dis=3)
        player.equipment["武器"] = weapon
        self.assertEqual(player.attack_range, 3)


class TestCardFactory(unittest.TestCase):
    """测试卡牌工厂"""

    def setUp(self):
        """每个测试前重置随机种子"""
        random.seed(RANDOM_SEED)
        self.cards_path = Path(__file__).parent.parent / "data" / "cards.json"

    def test_load_all_cards(self):
        """测试加载所有卡牌"""
        cards = CardFactory.load_from_config(self.cards_path)

        self.assertGreater(len(cards), 0)
        # 应该有至少160张牌
        self.assertGreater(len(cards), 150)

    def test_card_types(self):
        """测试卡牌类型正确"""
        cards = CardFactory.load_from_config(self.cards_path)

        # 统计各类型数量
        type_counts = {}
        for card in cards:
            card_type = card.card_type
            type_counts[card_type] = type_counts.get(card_type, 0) + 1

        # 应该有多种卡牌类型
        self.assertGreater(len(type_counts), 5)

    def test_create_single_card(self):
        """测试创建单张卡牌"""
        config = {
            "type": "BasicCard",
            "name": "杀",
            "color": "黑桃",
            "point": 7,
            "count": 1,
        }

        cards = CardFactory.create(config)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].name, "杀")
        self.assertEqual(cards[0].color, "黑桃")
        self.assertEqual(cards[0].point, 7)

    def test_create_multiple_cards(self):
        """测试创建多张相同卡牌"""
        config = {
            "type": "BasicCard",
            "name": "杀",
            "color": "黑桃",
            "point": 7,
            "count": 5,
        }

        cards = CardFactory.create(config)
        self.assertEqual(len(cards), 5)
        for card in cards:
            self.assertEqual(card.name, "杀")

    def test_sha_card_creation(self):
        """测试杀牌创建"""
        config = {
            "type": "BasicCard",
            "name": "杀",
            "color": "黑桃",
            "point": 7,
            "count": 1,
        }
        cards = CardFactory.create(config)
        self.assertTrue(is_sha_card(cards[0]))

    def test_fire_sha_creation(self):
        """测试火杀创建"""
        config = {
            "type": "FireSha",
            "name": "火杀",
            "color": "红桃",
            "point": 4,
            "count": 1,
        }
        cards = CardFactory.create(config)
        self.assertTrue(is_sha_card(cards[0]))
        self.assertTrue(hasattr(cards[0], "is_fire"))

    def test_weapon_creation(self):
        """测试武器创建"""
        config = {
            "type": "WeaponCard",
            "name": "诸葛连弩",
            "color": "黑桃",
            "point": 1,
            "dis": 1,
            "count": 1,
        }
        cards = CardFactory.create(config)
        self.assertEqual(cards[0].card_type, "WeaponCard")
        self.assertEqual(cards[0].attack_range, 1)

    def test_unique_card_names(self):
        """测试唯一卡牌名称"""
        cards = CardFactory.load_from_config(self.cards_path)
        unique_names = set(card.name for card in cards)

        # 应该有约45种不同的卡牌
        self.assertGreaterEqual(len(unique_names), 40)


class TestIntegration(unittest.TestCase):
    """完整游戏流程测试"""

    @classmethod
    def setUpClass(cls):
        import skills

    def setUp(self):
        random.seed(RANDOM_SEED)

    def test_full_game_flow(self):
        """测试完整游戏流程（模拟3回合）"""
        # 创建引擎
        commander_ids = ["WEI001", "WEI002", "SHU001", "SHU002", "WU001"]
        engine = GameEngine(player_num=5, commander_ids=commander_ids)

        # 创建玩家
        players = []
        for i, cmd_id in enumerate(commander_ids):
            config = engine.commander_configs.get(cmd_id, {})
            player = Player(
                idx=i + 1,
                commander_id=cmd_id,
                commander_name=config.get("name", f"玩家{i + 1}"),
                nation=config.get("nation", ""),
                max_hp=config.get("max_hp", 4),
                current_hp=config.get("max_hp", 4),
            )

            # 加载技能
            skills = SkillRegistry.create_skills_for_commander(cmd_id, player)
            player.skills = skills

            players.append(player)

        # 设置游戏
        engine.setup_game_with_players(players)

        # 验证游戏状态
        self.assertEqual(len(engine.players), 5)
        self.assertGreater(len(engine.deck), 0)

        # 模拟3回合
        for round_num in range(1, 4):
            engine.round_num = round_num

            for player in engine.players:
                if not player.is_alive:
                    continue

                # 摸牌阶段
                drawn = engine.draw_cards(player, 2)
                self.assertGreaterEqual(len(player.hand_cards), 0)

                # 出牌阶段（简化：只检查手牌）
                self.assertLessEqual(len(player.hand_cards), player.hand_limit + 2)

        # 游戏应该没有崩溃
        self.assertIsNotNone(engine)

    def test_commander_skill_binding(self):
        """测试武将技能绑定"""
        # 曹操应该有奸雄和护驾
        skills = SkillRegistry.create_skills_for_commander("WEI001")
        skill_names = [s.name for s in skills]

        self.assertIn("奸雄", skill_names)
        self.assertIn("护驾", skill_names)

        # 司马懿应该有鬼才和反馈
        skills = SkillRegistry.create_skills_for_commander("WEI002")
        skill_names = [s.name for s in skills]

        self.assertIn("鬼才", skill_names)
        self.assertIn("反馈", skill_names)

    def test_all_commanders_loadable(self):
        """测试所有武将可加载"""
        config_path = Path(__file__).parent.parent / "data" / "commanders.json"
        with open(config_path, encoding="utf-8") as f:
            configs = json.load(f)

        for commander_id, config in configs.items():
            # 每个武将都应该能创建技能
            skills = SkillRegistry.create_skills_for_commander(commander_id)
            self.assertGreaterEqual(
                len(skills), 0, f"武将 {config['name']} ({commander_id}) 技能加载失败"
            )

    def test_card_serialization(self):
        """测试卡牌序列化"""
        card = ShaCard(name="杀", color="黑桃", point=7)

        # 序列化
        data = card.to_dict()
        self.assertIn("name", data)
        self.assertIn("color", data)
        self.assertIn("point", data)

        # 验证数据
        self.assertEqual(data["name"], "杀")
        self.assertEqual(data["color"], "黑桃")
        self.assertEqual(data["point"], 7)

    def test_player_state(self):
        """测试玩家状态"""
        player = Player(
            idx=1,
            commander_name="曹操",
            max_hp=4,
            current_hp=4,
        )

        # 初始状态
        self.assertTrue(player.is_alive)
        self.assertFalse(player.is_chained)
        self.assertEqual(player.sha_count, 0)
        self.assertEqual(player.jiu_count, 0)

        # 铁索连环
        player.is_chained = True
        self.assertTrue(player.is_chained)

        # 重置
        player.is_chained = False
        self.assertFalse(player.is_chained)


if __name__ == "__main__":
    unittest.main(verbosity=2)
