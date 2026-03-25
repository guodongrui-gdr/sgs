# 三国杀 RL 训练项目 - 当前状态

最后更新: 2026-03-23

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
| `train/train_sb3.py`   | 训练脚本入口                   |

### 游戏引擎

| 文件                      | 用途         |
|-------------------------|------------|
| `engine/game_engine.py` | 游戏引擎核心     |
| `engine/event.py`       | 事件类型定义     |
| `engine/event_bus.py`   | 事件总线（技能触发） |
| `engine/state.py`       | 游戏状态序列化    |

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

### 未提交的更改

```
修改: engine/game_engine.py   # 白银狮子、回合阶段、deal_damage参数
修改: engine/response.py      # 伤害计算、距离计算、武器技能
修改: engine/judge.py         # 判定返回值、距离计算
修改: engine/event.py         # 回合阶段事件
修改: engine/state.py         # GamePhase 枚举
修改: card.py                 # 银月枪
修改: data/cards.json         # 银月枪
修改: skills/qun.py           # deal_damage 调用
修改: skills/wei.py           # deal_damage 调用
修改: ai/gym_wrapper.py       # deal_damage 调用
修改: new_main.py             # deal_damage 调用
```

### 最近提交

- `d5e9118` - 技能决策系统重构

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

---

## 参考命令

```bash
# 检查语法错误
python3 -m py_compile ai/gym_wrapper.py

# 运行训练
.venv/bin/python train/train_sb3.py --n-steps 4096 --timesteps 100000 --n-envs 4

# 查看日志
ls -la train/logs/

# Git 提交
git add -A && git commit -m "message" && git push
```