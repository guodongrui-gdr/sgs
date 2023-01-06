# 技能类
import Player


class Skill:
    def __init__(self, name, player: Player.player, target: Player.player, label=None, consume=None):
        """

        name: 技能名称
        player: 发动角色
        target: 目标
        consume: 消耗
        label: 技能标签(锁定技、觉醒技、限定技、主公技)

        """
        self.name = name
        self.player = player
        self.target = target
        self.consume = consume
        self.label = label


# 状态技
class StateSkill(Skill):
    def __init__(self, name, player, target, label=None, consume=None):
        super(StateSkill, self).__init__(name, player, target, label, consume)


# 触发技
class TriggerSkill(Skill):
    def __init__(self, name, player, target, time, label=None, consume=None):
        super(TriggerSkill, self).__init__(name, player, target, label, consume)
        self.time = time

    def JianXiong(self, channel):
        self.player.HandCards_area += channel
