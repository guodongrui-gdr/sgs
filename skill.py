# skill.py
from player import Player
from items import Items


# 技能类

class Skill:
    def __init__(self, name: str, time: str, label=None):
        self.name = name
        self.time = time
        self.label = label
        self.use_player: Player = None
        self.target_player: Player = None

    def check_time(self, time):  # 检查时机
        pass

    def check_con(self, Items: Items):  # 检查条件
        pass

    def state(self):  # 若为触发技,询问是否声明
        pass

    def trigger(self, Items: Items):  # 触发
        pass


class JianXiong(Skill):
    def __init__(self, name='奸雄', time='af_hurt'):
        super(JianXiong, self).__init__(name, time)

    def check_time(self, time):  # 检查时机
        if time == self.time:
            return True

    def check_con(self, Items: Items):  # 检查条件
        if self.use_player == Items.TmpCard[-1].target:
            return True

    def state(self) -> bool:
        return eval(input('是否发动奸雄:'))

    def trigger(self, Items: Items):
        for c in Items.TmpCard:
            self.use_player.HandCards_area.append(c)
            del c


JianXiong = JianXiong()
