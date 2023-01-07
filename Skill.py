# 技能类
import Player


class Skill:
    def __init__(self, name, target=None, label=None, consume=None):
        """

        name: 技能名称
        target: 目标
        consume: 消耗
        label: 技能标签(锁定技、觉醒技、限定技、主公技)

        """
        self.name = name
        self.target = target
        self.consume = consume
        self.label = label


# 状态技
class StateSkill(Skill):
    def __init__(self, name, target=None, label=None, consume=None):
        super(StateSkill, self).__init__(name, target, label, consume)


class JianXiong(Skill):
    def __init__(self):
        self.name = '奸雄'
        super(JianXiong).__init__(self.name)

    def trigger(self, channel):
        self.target.HandCards_area.append(channel)
        del channel


Jianxiong = JianXiong()
