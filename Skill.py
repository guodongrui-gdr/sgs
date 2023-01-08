# Skill.py

# 技能类

class JianXiong():
    def __init__(self):
        self.name = '奸雄'
        self.target = None

    def trigger(self, Items):
        if eval(input('是否发动奸雄:')):
            for i in range(len(Items.TmpCard)):
                self.target.HandCards_area.append(Items.TmpCard[-1])
                del Items.TmpCard[-1]


Jianxiong = JianXiong()
