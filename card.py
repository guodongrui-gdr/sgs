# card.py

import random


# 游戏牌
class Card:
    def __init__(self, name, color, point, target=None):
        self.name = name  # 牌名
        self.color = color  # 花色
        self.point = point  # 点数
        self.target = target  # 目标
        self.is_huoyan = False
        self.is_leidian = False
        self.is_shuxing = False


# 基本牌
class BasicCard(Card):
    def __init__(self, name, color, point):
        super(BasicCard, self).__init__(name, color, point, target=None)
        if '杀' in self.name:
            self.dis = 1
            self.target = ['another player']
            self.need_shan = 1  # 抵消杀所需闪的数量
            self.is_huoyan = False
            self.is_leidian = False
            self.is_shuxing = self.is_huoyan or self.is_leidian
            if '火' in self.name:
                self.is_huoyan = True
                self.is_shuxing = True
            elif '雷' in self.name:
                self.leidian = True
                self.is_shuxing = True
        if self.name == '酒':
            self.target = ['player']
        elif self.name == '桃':
            self.target = ['player', 'binsi_player']
        elif self.name == '闪':
            self.target = ['杀']


# 锦囊牌
class JinnangCard(Card):
    def __init__(self, name, color, point):
        super(JinnangCard, self).__init__(name, color, point, target=None)


# 普通锦囊牌
class CommonJinnangCard(JinnangCard):
    def __init__(self, name, color, point):
        super(CommonJinnangCard, self).__init__(name, color, point)
        if self.name == '顺手牵羊':
            self.target = ['another player']
            self.dis = 1
        elif self.name == '万箭齐发' or self.name == '南蛮入侵':  # 万箭齐发和南蛮入侵的目标为所有其他角色
            self.target = ['all other players']
        elif self.name == '五谷丰登' or self.name == '桃园结义':  # 五谷丰登和桃园结义的目标为所有角色
            self.target = ['all players']
        elif self.name == '无中生有':
            self.target = ['player']
        elif self.name == '火攻':
            self.is_huoyan = True
            self.is_shuxing = True
            self.target = ['one player']
        elif self.name == '铁索连环':
            self.target = ['one player', 'two players']
        elif self.name == '无懈可击':
            self.target = []
        else:
            self.target = ['another player']


# 延时锦囊牌
class YanshiJinnangCard(JinnangCard):
    def __init__(self, name, color, point):
        super(YanshiJinnangCard, self).__init__(name, color, point)
        if name == '兵粮寸断':
            self.dis = 1
            self.target = ['another player']
        elif name == '闪电':
            self.target = ['player']
            self.is_leidian = True
            self.is_shuxing = True
        else:
            self.target = ['another player']


# 装备牌
class EquipmentCard(Card):
    def __init__(self, name, color, point):
        super(EquipmentCard, self).__init__(name, color, point, target='player')
        self.dis = 0


# 武器牌
class WeaponCard(EquipmentCard):
    def __init__(self, name, color, point, dis):
        super(WeaponCard, self).__init__(name, color, point)
        self.dis = dis


# 防具牌
class ArmourCard(EquipmentCard):
    def __init__(self, name, color, point):
        super(ArmourCard, self).__init__(name, color, point)


# 进攻坐骑
class AttackHorseCard(EquipmentCard):
    def __init__(self, name, color, point):
        super(AttackHorseCard, self).__init__(name, color, point)


# 防守坐骑
class DefenseHorseCard(EquipmentCard):
    def __init__(self, name, color, point):
        super(DefenseHorseCard, self).__init__(name, color, point)


# 宝物牌
class TreasureCard(EquipmentCard):
    def __init__(self, name, color, point):
        super(TreasureCard, self).__init__(name, color, point)


class GetCardHeap:  # 摸牌堆
    def __init__(self):
        self.card_list = card_list

    def init_card_heap(self):  # 初始化摸牌堆
        card_heap_cache = []  # 创建一个缓存区用于存放牌
        for i in range(len(card_list)):
            idx = random.randint(0, len(self.card_list) - 1)
            card_heap_cache.append(self.card_list[idx])
            del self.card_list[idx]
        self.card_list = card_heap_cache.copy()
        del card_heap_cache

    def shuffle(self, left_card_heap):  # 洗牌
        card_heap_cache = []  # 创建一个缓存区用于存放牌
        for i in range(len(left_card_heap.card_list)):
            idx = random.randint(0, len(left_card_heap.card_list) - 1)
            card_heap_cache.append(left_card_heap.card_list[idx])
            del left_card_heap.card_list[idx]
        self.card_list = card_heap_cache.copy()
        del card_heap_cache

    def get_card(self, num, left_card_heap):  # 摸n张牌
        """

        num: 摸牌数量

        """
        return_card = []
        if len(self.card_list) >= num:
            return_card = self.card_list[:num]
            del self.card_list[:num]
        elif len(left_card_heap.card_list) + len(self.card_list) >= num > len(self.card_list) >= 0:
            return_card = self.card_list
            self.shuffle(left_card_heap)
            return_card = return_card + self.get_card(num - len(return_card), left_card_heap)
        elif len(left_card_heap.card_list) + len(self.card_list) < num:
            print('游戏结束\n平局')
            exit(0)
        return return_card


class IdnetityCardHeap:  # 身份牌堆
    def __init__(self):
        self.card_dic = {2: ['主公', '反贼'],
                         4: ['主公', '忠臣', '内奸', '反贼'],
                         5: ['主公', '忠臣', '内奸', '反贼', '反贼'],
                         }
        self.card_list = []

    def init_card_heap(self, player_num):  # 初始化身份牌堆
        card_heap_cache = []  # 创建一个缓存区用于存放牌
        for i in range(len(self.card_dic[player_num])):
            idx = random.randint(0, len(self.card_dic[player_num]) - 1)
            card_heap_cache.append(self.card_dic[player_num][idx])
            del self.card_dic[player_num][idx]
        self.card_list = card_heap_cache.copy()
        del card_heap_cache

    def get_identity(self):
        return_card = self.card_list[0]
        del self.card_list[0]
        return return_card


class LeftCardHeap:  # 弃牌堆
    def __init__(self):
        self.card_list = []


card_list = [BasicCard('普通杀', '黑桃', 7),
             BasicCard('普通杀', '黑桃', 8),
             BasicCard('普通杀', '黑桃', 8),
             BasicCard('普通杀', '黑桃', 9),
             BasicCard('普通杀', '黑桃', 9),
             BasicCard('普通杀', '黑桃', 10),
             BasicCard('普通杀', '黑桃', 10),
             BasicCard('普通杀', '红桃', 10),
             BasicCard('普通杀', '红桃', 10),
             BasicCard('普通杀', '红桃', 11),
             BasicCard('普通杀', '梅花', 2),
             BasicCard('普通杀', '梅花', 3),
             BasicCard('普通杀', '梅花', 4),
             BasicCard('普通杀', '梅花', 5),
             BasicCard('普通杀', '梅花', 6),
             BasicCard('普通杀', '梅花', 7),
             BasicCard('普通杀', '梅花', 8),
             BasicCard('普通杀', '梅花', 8),
             BasicCard('普通杀', '梅花', 9),
             BasicCard('普通杀', '梅花', 9),
             BasicCard('普通杀', '梅花', 10),
             BasicCard('普通杀', '梅花', 10),
             BasicCard('普通杀', '梅花', 11),
             BasicCard('普通杀', '梅花', 11),
             BasicCard('普通杀', '方块', 6),
             BasicCard('普通杀', '方块', 7),
             BasicCard('普通杀', '方块', 8),
             BasicCard('普通杀', '方块', 9),
             BasicCard('普通杀', '方块', 10),
             BasicCard('普通杀', '方块', 13),

             BasicCard('火杀', '红桃', 4),
             BasicCard('火杀', '红桃', 7),
             BasicCard('火杀', '红桃', 10),
             BasicCard('火杀', '方块', 4),
             BasicCard('火杀', '方块', 5),

             BasicCard('雷杀', '黑桃', 4),
             BasicCard('雷杀', '黑桃', 5),
             BasicCard('雷杀', '黑桃', 6),
             BasicCard('雷杀', '黑桃', 7),
             BasicCard('雷杀', '黑桃', 8),
             BasicCard('雷杀', '梅花', 5),
             BasicCard('雷杀', '梅花', 6),
             BasicCard('雷杀', '梅花', 7),
             BasicCard('雷杀', '梅花', 8),

             BasicCard('闪', '红桃', 2),
             BasicCard('闪', '红桃', 2),
             BasicCard('闪', '红桃', 8),
             BasicCard('闪', '红桃', 9),
             BasicCard('闪', '红桃', 11),
             BasicCard('闪', '红桃', 12),
             BasicCard('闪', '红桃', 13),
             BasicCard('闪', '方块', 2),
             BasicCard('闪', '方块', 2),
             BasicCard('闪', '方块', 3),
             BasicCard('闪', '方块', 4),
             BasicCard('闪', '方块', 5),
             BasicCard('闪', '方块', 6),
             BasicCard('闪', '方块', 6),
             BasicCard('闪', '方块', 7),
             BasicCard('闪', '方块', 7),
             BasicCard('闪', '方块', 8),
             BasicCard('闪', '方块', 8),
             BasicCard('闪', '方块', 9),
             BasicCard('闪', '方块', 10),
             BasicCard('闪', '方块', 10),
             BasicCard('闪', '方块', 11),
             BasicCard('闪', '方块', 11),
             BasicCard('闪', '方块', 11),

             BasicCard('桃', '红桃', 3),
             BasicCard('桃', '红桃', 4),
             BasicCard('桃', '红桃', 5),
             BasicCard('桃', '红桃', 6),
             BasicCard('桃', '红桃', 6),
             BasicCard('桃', '红桃', 7),
             BasicCard('桃', '红桃', 8),
             BasicCard('桃', '红桃', 9),
             BasicCard('桃', '红桃', 12),
             BasicCard('桃', '方块', 2),
             BasicCard('桃', '方块', 3),
             BasicCard('桃', '方块', 12),

             BasicCard('酒', '黑桃', 3, ),
             BasicCard('酒', '黑桃', 9, ),
             BasicCard('酒', '梅花', 3, ),
             BasicCard('酒', '梅花', 9, ),
             BasicCard('酒', '方块', 9, ),
             WeaponCard('诸葛连弩', '梅花', 1, 1),
             WeaponCard('诸葛连弩', '方块', 1, 1),
             WeaponCard('雌雄双股剑', '黑桃', 2, 2),
             WeaponCard('寒冰剑', '黑桃', 2, 2),
             WeaponCard('青釭剑', '黑桃', 2, 2),
             WeaponCard('古锭刀', '黑桃', 2, 2),
             WeaponCard('青龙偃月刀', '黑桃', 5, 3),
             WeaponCard('贯石斧', '方块', 5, 3),
             WeaponCard('丈八蛇矛', '黑桃', 12, 3),
             WeaponCard('方天画戟', '方块', 12, 4),
             WeaponCard('朱雀羽扇', '方块', 1, 4),
             WeaponCard('麒麟弓', '红桃', 5, 5),

             ArmourCard('八卦阵', '黑桃', 2),
             ArmourCard('八卦阵', '梅花', 2),
             ArmourCard('白银狮子', '梅花', 1),
             ArmourCard('仁王盾', '梅花', 2),
             ArmourCard('藤甲', '黑桃', 2),
             ArmourCard('藤甲', '梅花', 2),

             AttackHorseCard('大宛', '黑桃', 13),
             AttackHorseCard('赤兔', '红桃', 5),
             AttackHorseCard('紫骍', '黑桃', 13),

             DefenseHorseCard('绝影', '黑桃', 5),
             DefenseHorseCard('爪黄飞电', '红桃', 13),
             DefenseHorseCard('的卢', '梅花', 5),
             DefenseHorseCard('骅骝', '方块', 13),

             TreasureCard('木牛流马', '方块', 5),

             CommonJinnangCard('决斗', '黑桃', 1),
             CommonJinnangCard('决斗', '梅花', 1),
             CommonJinnangCard('决斗', '方块', 1),

             CommonJinnangCard('无中生有', '红桃', 7),
             CommonJinnangCard('无中生有', '红桃', 8),
             CommonJinnangCard('无中生有', '红桃', 9),
             CommonJinnangCard('无中生有', '红桃', 11),

             CommonJinnangCard('过河拆桥', '黑桃', 3),
             CommonJinnangCard('过河拆桥', '黑桃', 4),
             CommonJinnangCard('过河拆桥', '黑桃', 12),
             CommonJinnangCard('过河拆桥', '红桃', 12),
             CommonJinnangCard('过河拆桥', '梅花', 3),
             CommonJinnangCard('过河拆桥', '梅花', 4),

             CommonJinnangCard('顺手牵羊', '黑桃', 3),
             CommonJinnangCard('顺手牵羊', '黑桃', 4),
             CommonJinnangCard('顺手牵羊', '黑桃', 11),
             CommonJinnangCard('顺手牵羊', '方块', 3),
             CommonJinnangCard('顺手牵羊', '方块', 4),

             CommonJinnangCard('借刀杀人', '梅花', 12),
             CommonJinnangCard('借刀杀人', '梅花', 13),

             CommonJinnangCard('南蛮入侵', '黑桃', 7),
             CommonJinnangCard('南蛮入侵', '黑桃', 12),
             CommonJinnangCard('南蛮入侵', '梅花', 7),

             CommonJinnangCard('万箭齐发', '红桃', 1),

             CommonJinnangCard('桃园结义', '红桃', 1),

             CommonJinnangCard('五谷丰登', '红桃', 3),
             CommonJinnangCard('五谷丰登', '红桃', 4),

             CommonJinnangCard('无懈可击', '黑桃', 11),
             CommonJinnangCard('无懈可击', '黑桃', 13),
             CommonJinnangCard('无懈可击', '梅花', 12),
             CommonJinnangCard('无懈可击', '梅花', 13),
             CommonJinnangCard('无懈可击', '方块', 12),
             CommonJinnangCard('无懈可击', '红桃', 1),
             CommonJinnangCard('无懈可击', '红桃', 12),

             CommonJinnangCard('火攻', '红桃', 2),
             CommonJinnangCard('火攻', '红桃', 3),
             CommonJinnangCard('火攻', '方块', 12),

             CommonJinnangCard('铁索连环', '黑桃', 11),
             CommonJinnangCard('铁索连环', '黑桃', 12),
             CommonJinnangCard('铁索连环', '梅花', 10),
             CommonJinnangCard('铁索连环', '梅花', 11),
             CommonJinnangCard('铁索连环', '梅花', 12),
             CommonJinnangCard('铁索连环', '梅花', 13),

             YanshiJinnangCard('乐不思蜀', '黑桃', 6),
             YanshiJinnangCard('乐不思蜀', '红桃', 6),
             YanshiJinnangCard('乐不思蜀', '梅花', 6),

             YanshiJinnangCard('闪电', '黑桃', 1),
             YanshiJinnangCard('闪电', '红桃', 12),

             YanshiJinnangCard('兵粮寸断', '黑桃', 10),
             YanshiJinnangCard('兵粮寸断', '梅花', 4),

             ]
