# commander.py
from skill import *


#  武将类
class Commander:
    def __init__(self, id, name, max_hp, hp, nation, skills):
        self.ID = id  # 武将ID
        self.name = name  # 武将姓名
        self.max_HP = max_hp  # 武将最大体力值
        self.HP = hp  # 武将初始体力值
        self.nation = nation  # 武将势力
        self.skills = skills  # 武将技能列表


Caocao = Commander('WEI001', '曹操', 4, 4, '魏', [JianXiong])
