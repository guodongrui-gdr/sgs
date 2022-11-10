# 玩家类
import Card


class player:
    def __init__(self, commander):
        self.commander = commander
        self.max_HP = commander.max_HP  # 体力上限
        self.current_HP = commander.HP  # 当前体力值
        self.max_HandCards = self.current_HP  # 手牌上限
        self.idx = 0  # 座次
        self.identity = ''  # 身份信息
        self.equipment_area = {  # 装备区
            'weapon': Card.weapon_card(None, None, None, None),
            'armour': Card.armour_card(None, None, None),
            'horse+1': Card.attack_horse_card(None, None, None),
            'horse-1': Card.defense_horse_card(None, None, None)
        }
        self.HandCards_area = []  # 手牌区
        self.pandin_area = []  # 判定区
        self.pre = None  # 上家
        self.next = None  # 下家
        self.use_sha_count = 0  # 使用杀的次数
        self.use_jiu_count = 0  # 使用酒的次数
        self.jiu = 0  # 是否喝酒
        self.hengzhi = False  # 是否处于横置状态
