# 三国杀 RL 训练项目 - 当前状态

最后更新: 2026-04-02

## 项目目标

训练一个强化学习模型 (MaskablePPO) 来玩三国杀卡牌游戏。关键特性：

- 让 RL 参与技能内部决策（如观星的牌序选择、遗计的分配等）
- 分步输出技能决策，而非一次性决定

---

## 重要说明

- **不要提前结束技能决策**：技能决策必须分步进行，不能提前终止
- **技能决策由 RL 驱动**：技能触发后的内部决策由 RL 模型参与，而非纯规则
- **代码推送到 GitHub**：当用户要求时执行
- **游戏逻辑与官方规则一致**：代码实现需符合 RULES.md 官方规则
- **支持 GUI 和 CLI 两种模式**：可通过 `python main.py --gui` 启动图形界面

---

## 当前状态

### 训练状态: ✅ 正常运行

最近一次成功训练 (2026-03-23 15:33):

- 训练步数: 100K timesteps
- ep_rew_mean: 正常波动 (-0.64 → 2.75 → 0.58)
- ep_len_mean: ~220-250 步
- FPS: ~1500
- explained_variance: ~0.04 (正值，说明价值函数学习正常)

### 已完成的修复

#### 游戏规则一致性修复 ✅ (2026-03-23)

根据 RULES.md 官方规则文档，修复了多项与规则不一致的问题：

| 优先级 | 问题                 | 状态    |
|-----|--------------------|-------|
| P0  | 白银狮子伤害计算错误（减1→改为1） | ✅ 已修复 |
| P0  | 白银狮子失去回血效果缺失       | ✅ 已修复 |
| P0  | 判定返回值类型错误          | ✅ 已修复 |
| P1  | 藤甲对雷电伤害错误+1        | ✅ 已修复 |
| P1  | 铁索传递伤害丢失属性         | ✅ 已修复 |
| P1  | 距离计算未取双向最小值        | ✅ 已修复 |
| P2  | 缺少银月枪卡牌            | ✅ 已修复 |
| P2  | 武器技能未实现            | ✅ 已修复 |
| P2  | 铁索连环重铸             | ✅ 已存在 |
| P3  | 回合阶段不完整            | ✅ 已修复 |

**武器技能实现**：

- 青釭剑：无视防具
- 雌雄双股剑：杀异性目标效果
- 寒冰剑：弃置对方牌代替伤害
- 古锭刀：无手牌伤害+1
- 贯石斧：弃两牌强命
- 青龙偃月刀：追杀
- 麒麟弓：弃置坐骑

**修改的文件**：

- `engine/game_engine.py` - 白银狮子、回合阶段
- `engine/response.py` - 伤害计算、距离计算、武器技能
- `engine/judge.py` - 判定返回值、距离计算
- `engine/event.py` - 回合阶段事件
- `engine/state.py` - GamePhase 枚举
- `card.py` - 银月枪
- `data/cards.json` - 银月枪

#### 1. 技能决策阻塞问题 ✅ 已修复

**问题**：当技能（如观星）触发时，环境会等待 RL 提供决策输入，但 RL 不知道如何响应，导致训练卡住。

**解决方案**：在 `step()` 方法中添加自动处理逻辑：

```python
# 自动处理技能决策（训练时使用默认策略）
while self.skill_decision_context.has_pending_decision():
    request = self.skill_decision_context.active_request
    if request is None:
        break
    mask = self._get_skill_decision_mask()
    valid_options = np.where(mask > 0)[0]
    if len(valid_options) == 0:
        self.skill_decision_context.clear()
        break
    auto_action = valid_options[0]
    self._handle_skill_decision(int(auto_action))
```

**注意**：这是临时方案，后续应该让 RL 真正参与决策。

#### 2. Player.__eq__ 方法 ✅ 已修复

**问题**：`event.source == self.player` 当 source 是 int 时会崩溃。

**解决方案**：添加 `__eq__` 方法支持 int 比较：

```python
def __eq__(self, other):
    if isinstance(other, Player):
        return self.idx == other.idx and self.commander_id == other.commander_id
    if isinstance(other, int):
        return self.idx == other
    return NotImplemented
```

#### 3. Invalid Action Truncated 标志 ✅ 已修复

**问题**：无效动作返回 `truncated=True`，导致 Monitor 错误地在第 1 步就结束 episode。

**解决方案**：改为返回 `truncated=False`：

```python
# 修复前
return obs, -0.1, False, True, {"error": "Invalid action"}
# 修复后
return obs, -0.1, False, False, {"error": "Invalid action"}
```

---

## 技能决策系统

### 架构

文件: `ai/skill_decision.py`

决策类型：

```python
class SkillDecisionType(IntEnum):
    YES_NO = 1        # 是/否决策
    SELECT_CARDS = 2  # 选择卡牌
    SELECT_ORDER = 3  # 选择顺序（观星）
    DISTRIBUTE = 4    # 分配资源（遗计）
    SELECT_PAIR = 5   # 选择一对（离间）
```

### 已重构的技能

| 技能 | 武将  | 决策类型         | 文件            |
|----|-----|--------------|---------------|
| 观星 | 诸葛亮 | SELECT_ORDER | skills/shu.py |
| 遗计 | 郭嘉  | DISTRIBUTE   | skills/wei.py |
| 离间 | 貂蝉  | SELECT_PAIR  | skills/qun.py |
| 鬼才 | 司马懿 | SELECT_CARDS | skills/wei.py |
| 青囊 | 华佗  | SELECT_CARDS | skills/qun.py |
| 武圣 | 关羽  | SELECT_CARDS | skills/shu.py |
| 龙胆 | 赵云  | SELECT_CARDS | skills/shu.py |
| 倾国 | 甄姬  | SELECT_CARDS | skills/wei.py |
| 奇袭 | 甘宁  | SELECT_CARDS | skills/wu.py  |
| 国色 | 大乔  | SELECT_CARDS | skills/wu.py  |
| 流离 | 小乔  | SELECT_CARDS | skills/wu.py  |
| 结姻 | 孙尚香 | SELECT_CARDS | skills/wu.py  |

### 观察空间扩展

在 `ai/gym_wrapper.py` 中添加了技能决策相关的观察：

```python
obs["skill_decision_type"] = int(request.decision_type)  # 决策类型
obs["skill_decision_mask"] = self._get_skill_decision_mask()  # 有效选项掩码
```

---

## Autoresearch 自动实验模式（新增）

基于 karpathy/autoresearch 思想实现的自动实验系统，让 AI agent 自主进行 RL 训练实验。

### 核心思想

- **自主迭代**：agent 持续运行实验，不停止询问用户
- **固定时间预算**：每次实验运行固定 10 分钟，便于比较不同架构
- **唯一修改文件**：agent 只修改 train/train.py，不修改其他文件
- **标准化评估**：使用固定的评估函数和指标

### 文件结构

| 文件                  | 用途                     | 可修改性    |
|---------------------|------------------------|---------|
| train/prepare.py    | 固定配置、评估函数、环境创建       | ❌ 不可修改 |
| train/train.py      | 超参数、奖励配置、模型架构、训练循环    | ✅ 可修改   |
| train/program.md    | agent 指令文件，指导 agent 如何实验  | 人类修改    |
| train/results.tsv   | 实验记录（不提交到 git）        | agent 维护 |

### 运行方式

**启动自动实验**：

```bash
# 1. 创建实验分支
git checkout -b autoresearch/apr2

# 2. 运行单个实验（10分钟）
python train/train.py > train/run.log 2>&1

# 3. 提取结果
grep "^win_rate:\|^mean_reward:" train/run.log

# 4. 查看实验记录
cat train/results.tsv
```

**让 agent 自主运行**：

向 agent（如 Claude/Codex）发送：

```
查看 train/program.md 并启动新的自动实验！
```

agent 会按照 program.md 的指令，自主迭代实验。

### 关键设计

**时间预算**：固定 10 分钟（600秒）
- 每次实验约 6 次/小时
- 一夜（8小时）可运行约 48 次实验

**评估指标**：
- `win_rate`：游戏胜率（主要指标）
- `mean_reward`：平均奖励（辅助指标）
- 综合评分 = win_rate * 100 + mean_reward

**agent 可修改范围**：
- HyperparametersConfig：学习率、gamma、clip_range等
- RewardConfig：奖励函数参数
- ModelConfig：模型架构（MLP/Transformer）
- train_main()：训练循环逻辑

**agent 不可修改**：
- prepare.py 中的固定配置
- evaluate_model() 评估函数
- 固定时间预算（10分钟）
- 环境创建逻辑

### 典型实验流程

agent 的实验循环：

1. 查看 git 状态
2. 修改 train/train.py（尝试新配置）
3. git commit
4. 运行实验：`python train/train.py > train/run.log 2>&1`
5. 提取结果：`grep "^win_rate:\|^mean_reward:" train/run.log`
6. 记录到 results.tsv
7. 如果改进，保留 commit；否则 git reset
8. 重复循环（永不停止）

### results.tsv 格式

```
commit	win_rate	mean_reward	memory_mb	status	description
a1b2c3d	0.150000	12.34	4096.0	keep	baseline
b2c3d4e	0.180000	15.67	4100.0	keep	increase LR to 1e-3
c3d4e5f	0.120000	10.23	4096.0	discard	decrease gamma to 0.95
d4e5f6g	0.000000	0.00	0.0	crash	OOM with large model
```

### 可尝试的方向

1. **超参数调整**：LR、gamma、clip_range、ent_coef
2. **奖励函数优化**：调整 win_reward、damage_reward_scale
3. **模型架构变化**：Transformer vs MLP、网络深度/宽度
4. **训练策略改进**：课程学习、早停、多环境训练

### 与原有训练脚本的关系

- `train/train_sb3.py`：保留原有训练脚本，用于完整训练
- `train/train.py`：新增自动实验脚本，用于快速迭代
- 两套系统并存，互不干扰

---

## GUI 图形界面（新增）

### 功能特性

基于 Pygame 实现的图形界面，提供直观的三国杀游戏体验：

- **完整的游戏界面**：玩家区域、卡牌显示、手牌管理
- **动画效果**：伤害动画、治疗动画、发牌动画、文字浮动提示
- **交互系统**：鼠标点击选牌、目标选择、技能触发 UI
- **响应机制**：杀/闪响应、无懈可击、技能决策界面
- **游戏日志**：实时显示游戏事件和技能触发信息

### 运行方式

```bash
# 方式 1：使用启动脚本
bash run_gui.sh

# 方式 2：直接运行
python main.py --gui

# 方式 3：在游戏中指定模式
python main.py --gui --mode 2p  # 双人对战模式
```

### 技术实现

- **动画系统**：`gui/animations.py` 管理所有动画效果
- **渲染分离**：游戏逻辑与渲染完全分离，便于维护
- **响应管理**：`gui/response_manager.py` 处理游戏响应请求
- **音频支持**：`gui/audio.py` 提供音效播放

---

## 训练配置

### 当前参数

文件: `train/train_sb3.py`

```python
TrainingConfig:
    n_steps: 4096
    batch_size: 256
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.98
    ent_coef: 0.05
    vf_coef: 0.5
    clip_range: 0.2
    learning_rate: 5e-4 (cosine schedule)

SGSConfig:
    max_rounds: 15  # 从 100 减少，加快训练
```

### 运行命令

```bash
# 快速测试
.venv/bin/python train/train_sb3.py --n-steps 256 --timesteps 500 --n-envs 1

# 正常训练
.venv/bin/python train/train_sb3.py --n-steps 4096 --timesteps 100000 --n-envs 4

# 使用 GPU
.venv/bin/python train/train_sb3.py --device cuda
```

---

## 文件结构

### 核心文件

| 文件                     | 用途                       |
|------------------------|--------------------------|
| `ai/gym_wrapper.py`    | Gym 环境包装器，动作空间、观察空间、奖励计算 |
| `ai/skill_decision.py` | 技能决策框架，决策类型、请求结构、缓存机制    |
| `ai/reward.py`         | 奖励配置和计算                  |
| `ai/action_encoder.py` | 动作编码/解码                  |
| `ai/action_mask.py`    | 动作掩码生成                   |
| `train/train_sb3.py`   | 训练脚本入口（完整训练）             |
| `train/train.py`       | 自动实验脚本入口（快速迭代）            |
| `train/prepare.py`     | 自动实验固定配置和评估函数            |
| `train/program.md`     | 自动实验 agent 指令文件           |
| `main.py`              | 主入口，支持 GUI 和 CLI 模式      |

### 游戏引擎

| 文件                      | 用途         |
|-------------------------|------------|
| `engine/game_engine.py` | 游戏引擎核心     |
| `engine/event.py`       | 事件类型定义     |
| `engine/event_bus.py`   | 事件总线（技能触发） |
| `engine/state.py`       | 游戏状态序列化    |

### GUI 图形界面（新增）

| 文件                          | 用途              |
|-----------------------------|-----------------|
| `gui/main_window.py`        | GUI 主窗口和游戏循环     |
| `gui/game_renderer.py`      | 游戏画面渲染          |
| `gui/player_renderer.py`    | 玩家区域渲染          |
| `gui/card_renderer.py`      | 卡牌渲染            |
| `gui/ui_elements.py`        | UI 组件（按钮、对话框等）   |
| `gui/skill_ui.py`           | 技能触发 UI 和游戏日志    |
| `gui/animations.py`         | 动画效果            |
| `gui/response_manager.py`   | 响应请求管理          |
| `gui/input_handler.py`       | 输入处理            |
| `gui/assets.py`             | 资源加载和配置         |
| `gui/audio.py`              | 音频播放            |
| `run_gui.sh`                | GUI 启动脚本        |

### 技能文件

| 文件              | 武将                               |
|-----------------|----------------------------------|
| `skills/shu.py` | 蜀国武将：诸葛亮、关羽、张飞、赵云、马超、黄月英         |
| `skills/wei.py` | 魏国武将：曹操、司马懿、郭嘉、甄姬、许褚、夏侯惇         |
| `skills/wu.py`  | 吴国武将：孙权、甘宁、吕蒙、黄盖、周瑜、大乔、陆逊、孙尚香、小乔 |
| `skills/qun.py` | 群雄武将：吕布、貂蝉、华佗                    |

---

## 已知问题

### 1. 偶发性 SystemError

```
SystemError: Objects/dictobject.c:1605: bad argument to internal function
```

- 在 `engine/state.py` 的 `to_dict()` 方法中偶发
- 可能与特定游戏状态有关
- 大多数训练运行正常

### 2. 待完成：RL 参与技能决策

当前技能决策被自动处理（使用默认策略），应该让 RL 真正学习决策。

---

## 最近发现的根本问题

### ep_rew_mean 不变化的原因

1. **Episode 过长**：每个 episode 500-1000+ 步
2. **Rollout 太小**：n_steps=2048 意味着每轮只有 2-4 个 episode
3. **Buffer 更新慢**：Monitor 的 ep_rew_mean buffer 更新不够频繁

**解决方案**：

- 减少 `max_rounds` (100 → 15)
- 增加 `n_steps` (2048 → 4096)
- 确保技能决策不阻塞环境

---

## Git 状态

### 工作区状态

**干净** - 所有之前的更改已提交。

### 新增未跟踪文件

```
gui/              # GUI 图形界面模块（新增）
run_gui.sh        # GUI 启动脚本
```

### 最近提交

- `384c790` - fix tab
- `dade628` - refactor: 清理无用代码和旧文件
- `69ed7dd` - fix: 游戏规则一致性修复
- `bacb2e0` - docs: 更新BUG_FIXES和PROJECT_STATUS，添加RULES官方规则文档
- `d5e9118` - feat: RL训练优化与技能决策系统

---

## 下一步计划

1. **让 RL 参与技能决策**
    - 当前自动处理是临时方案
    - 需要扩展动作空间，让 RL 输出技能决策

2. **解决偶发性 SystemError**
    - 添加更多调试信息
    - 检查 `to_dict()` 中的数据类型

3. **长期训练**
    - 训练更长时间 (1M+ timesteps)
    - 观察奖励曲线是否持续上升

4. **实现银月枪技能**
    - 当前只添加了卡牌，技能尚未实现
    - 规则：你的回合外，每当你使用一张黑色手牌，可立即对攻击范围内一名角色使用一张杀

5. **完善 GUI 功能**
    - 完善技能决策 UI
    - 优化动画效果
    - 添加更多游戏提示

6. **运行 Autoresearch 实验**
    - 创建第一个实验分支（如 autoresearch/apr2）
    - 运行 baseline 实验建立初始指标
    - 让 agent 自主迭代优化模型

---

## 参考命令

```bash
# 检查语法错误
python3 -m py_compile ai/gym_wrapper.py

# 运行训练（完整训练）
.venv/bin/python train/train_sb3.py --n-steps 4096 --timesteps 100000 --n-envs 4

# 运行自动实验（10分钟快速迭代）
python train/train.py > train/run.log 2>&1

# 提取自动实验结果
grep "^win_rate:\|^mean_reward:" train/run.log

# 验证自动实验环境
python train/prepare.py

# 运行 GUI 图形界面
bash run_gui.sh

# 或直接运行
python main.py --gui

# 查看日志
ls -la train/logs/

# Git 提交
git add -A && git commit -m "message" && git push
```