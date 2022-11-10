# 技能类
class Skill:
    def __init__(self, skill_type, occ_time, seffect):
        self.skill_type = skill_type  # 技能类型
        self.occ_time = occ_time  # 技能发动时机
        # if self.type=='suodinji': # 若技能为锁定技,则技能能发动时必须发动
