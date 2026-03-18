# 三国杀 - 人机对战版

一个基于 Python 的三国杀卡牌游戏，支持人机对战。

## 特性

- 支持 2-8 人对战
- 可扩展的武将、卡牌、技能系统
- AI 交互接口，支持外部强化学习模型对接
- 配置化设计，易于扩展

## 快速开始

```bash
python new_main.py
```

## 项目结构

```
sgs/
├── config.py                # 全局配置
├── new_main.py              # 新版入口
│
├── data/
│   ├── cards.json           # 卡牌配置 (160+张)
│   └── commanders.json      # 武将配置 (22个)
│
├── engine/
│   ├── event.py             # 事件类型定义
│   ├── event_bus.py         # 事件总线 (技能触发)
│   ├── state.py             # 游戏状态序列化
│   └── game_engine.py       # 游戏引擎核心
│
├── card/
│   ├── base.py              # 卡牌基类
│   └── factory.py           # 卡牌工厂 (从JSON创建)
│
├── skills/
│   ├── base.py              # 技能基类
│   ├── registry.py          # 技能注册中心
│   └── jianxiong.py         # 示例技能 (奸雄、武圣、咆哮)
│
├── player/
│   └── player.py            # 玩家类
│
├── ai/
│   └── interface.py         # AI交互接口
│
└── gui/                     # GUI模块 (待开发)
```

## 扩展指南

### 新增武将

编辑 `data/commanders.json`:

```json
{
  "SHU010": {
    "name": "新武将",
    "nation": "蜀",
    "max_hp": 4,
    "skills": ["skill_name"]
  }
}
```

### 新增卡牌

编辑 `data/cards.json`:

```json
{
  "cards": [
    {"type": "BasicCard", "name": "新牌", "color": "红桃", "point": 1, "count": 2}
  ]
}
```

### 新增技能

在 `skills/` 目录新建模块:

```python
from skills.base import TriggerSkill
from skills.registry import SkillRegistry

@SkillRegistry.register
class NewSkill(TriggerSkill):
    def __init__(self):
        super().__init__(
            name="新技能",
            trigger_events=["damage_taken"],
            description="技能描述"
        )
    
    def can_activate(self, event, engine) -> bool:
        return True
    
    def execute(self, event, engine):
        return event
```

## AI 接口

```python
from ai.interface import AIInterface

interface = AIInterface(engine, player)

# 获取当前状态
state = interface.get_state()

# 获取合法动作
actions = interface.get_legal_actions()

# 执行动作
new_state, reward, done = interface.step(action)
```

## 已实现功能

- [x] 核心游戏引擎
- [x] 卡牌系统 (基本牌、锦囊牌、装备牌)
- [x] 武将系统 (22个武将)
- [x] 技能系统 (事件驱动)
- [x] AI 接口
- [x] 命令行界面
- [ ] GUI 界面
- [ ] 网络对战

## 旧文件

以下文件保留作为参考，新项目使用新架构：

- `main.py` / `process.py` / `card.py` / `skill.py` - 旧代码
- `new_main.py` + `engine/` + `card/` + `skills/` - 新架构

## License

MIT