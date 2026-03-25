# 三国杀 - 人机对战版

一个基于 Python 的三国杀卡牌游戏，支持人机对战，包含命令行和图形界面。

## 特性

- 支持 2-8 人对战
- 可扩展的武将、卡牌、技能系统
- AI 交互接口，支持外部强化学习模型对接
- PyGame 图形界面，支持动画和音效
- 配置化设计，易于扩展

## 快速开始

### 命令行模式

```bash
python main.py --player-num 5
```

### 图形界面模式

```bash
python main.py --gui
```

或使用快捷脚本：
```bash
./run_gui.sh
```

## 项目结构

```
sgs/
├── main.py                  # 程序入口
├── config.py                # 全局配置
│
├── data/
│   ├── cards.json           # 卡牌配置 (160+张)
│   └── commanders.json      # 武将配置 (22个)
│
├── engine/
│   ├── event.py             # 事件类型定义
│   ├── event_bus.py         # 事件总线 (技能触发)
│   ├── state.py             # 游戏状态序列化
│   ├── game_engine.py       # 游戏引擎核心
│   ├── judge.py             # 判定系统
│   └── response.py          # 响应系统
│
├── card/
│   ├── base.py              # 卡牌基类
│   └── factory.py           # 卡牌工厂
│
├── skills/
│   ├── base.py              # 技能基类
│   ├── registry.py          # 技能注册中心
│   ├── wei.py               # 魏国技能
│   ├── shu.py               # 蜀国技能
│   ├── wu.py                # 吴国技能
│   └── qun.py               # 群雄技能
│
├── player/
│   └── player.py            # 玩家类
│
├── ai/
│   ├── interface.py         # AI交互接口
│   ├── rl_ai.py             # 强化学习AI
│   ├── rule_ai.py           # 规则AI
│   └── gym_wrapper.py       # Gym环境封装
│
├── train/
│   └── train_sb3.py         # 训练脚本
│
├── gui/
│   ├── __init__.py
│   ├── main_window.py       # 主窗口
│   ├── game_renderer.py     # 游戏渲染
│   ├── card_renderer.py     # 卡牌绘制
│   ├── player_renderer.py   # 玩家渲染
│   ├── input_handler.py     # 输入处理
│   ├── ui_elements.py       # UI组件
│   ├── animations.py        # 动画系统
│   ├── audio.py             # 音效管理
│   ├── response_manager.py  # 响应管理
│   ├── skill_ui.py          # 技能UI
│   └── assets.py            # 资源管理
│
└── 素材/
    ├── 背景.jpg             # 游戏背景
    └── ...
```

## GUI 操作说明

### 主菜单
- 选择游戏人数（5-8人）
- 点击"开始游戏"进入

### 游戏界面
- **选择卡牌**: 点击手牌区域
- **使用卡牌**: 选中卡牌后点击"使用卡牌"按钮或按空格键
- **选择目标**: 点击玩家区域选择目标
- **结束回合**: 点击"结束回合"按钮
- **设置**: 点击右上角"设置"按钮或按F1
- **游戏日志**: 左侧面板显示游戏记录

### 快捷键
- `1-9`: 选择对应位置的卡牌
- `空格`: 使用选中的卡牌
- `回车`: 确认选择
- `ESC`: 取消选择/关闭面板
- `F1`: 打开设置

## 支持的卡牌

### 基本牌
- 杀、闪、桃、酒、火杀、雷杀

### 锦囊牌
- 决斗、无中生有、南蛮入侵、万箭齐发
- 桃园结义、五谷丰登、火攻
- 过河拆桥、顺手牵羊、铁索连环、借刀杀人
- 无懈可击

### 延时锦囊
- 乐不思蜀、兵粮寸断、闪电

### 装备牌
- 武器、防具、进攻坐骑、防御坐骑、宝物

## 已实现功能

- [x] 核心游戏引擎
- [x] 卡牌系统 (基本牌、锦囊牌、装备牌)
- [x] 武将系统 (22个武将)
- [x] 技能系统 (事件驱动)
- [x] AI 接口
- [x] 命令行界面
- [x] GUI 界面 (PyGame)
  - [x] 主菜单
  - [x] 游戏桌面渲染
  - [x] 卡牌交互
  - [x] 目标选择
  - [x] 动画效果
  - [x] 音效系统
  - [x] 设置面板
  - [x] 游戏日志
- [ ] 网络对战

## 依赖

```
pygame>=2.5.0
```

安装依赖：
```bash
pip install pygame
```

## License

MIT