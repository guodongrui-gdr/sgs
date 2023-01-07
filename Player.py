# Player.py

# 玩家类
import Card
import Commander

class player:
    def __init__(self, commander):
        self.commander = commander  # 武将
        self.max_HP = 0  # 体力上限
        self.current_HP = 0  # 当前体力值
        self.max_HandCards = 0  # 手牌上限
        self.idx = 0  # 座次
        self.identity = ''  # 身份信息
        self.equipment_area = {  # 装备区
            '武器': Card.weapon_card(None, None, None, None),
            '防具': Card.armour_card(None, None, None),
            '进攻坐骑': Card.defense_horse_card(None, None, None),
            '防御坐骑': Card.attack_horse_card(None, None, None),
            '宝物': Card.treasure_card(None, None, None)
        }
        self.HandCards_area = []  # 手牌区
        self.pandin_area = []  # 判定区
        self.pre: player = None  # 上家
        self.next: player = None  # 下家
        self.use_sha_count = 0  # 使用杀的次数
        self.use_jiu_count = 0  # 使用酒的次数
        self.jiu = 0  # 是否喝酒
        self.hengzhi = False  # 是否处于横置状态
