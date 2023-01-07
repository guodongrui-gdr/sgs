# Commander.py
import Skill


#  武将类
class commander:
    def __init__(self, ID, name, max_HP, HP, nation, skills):
        self.ID = ID  # 武将ID
        self.name = name  # 武将姓名
        self.max_HP = max_HP  # 武将最大体力值
        self.HP = HP  # 武将初始体力值
        self.nation = nation  # 武将势力
        self.skills = skills  # 武将技能列表


Caocao = commander('WEI001', '曹操', 4, 4, '魏', [Skill.Jianxiong])
