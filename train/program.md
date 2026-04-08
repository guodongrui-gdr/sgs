# SGS RL 自动研究实验

这是一个让 AI agent 自主进行 RL 训练实验的系统。agent 会持续迭代，尝试不同的配置，目标是提高三国杀 AI 的胜率和平均奖励。

## 设置

要与用户一起设置一个新的实验：

1. **确定实验标签**：根据日期提议一个标签（例如 `apr2`）。分支 `autoresearch/<tag>` 必须不存在 — 这是一个全新的实验运行。
2. **创建分支**：从当前 master 创建：`git checkout -b autoresearch/<tag>`
3. **读取相关文件**：仓库很小。读取这些文件了解完整上下文：
   - `README.md` — 仓库背景
   - `PROJECT_STATUS.md` — 项目当前状态
   - `train/prepare.py` — 固定配置、评估函数（不可修改）
   - `train/train.py` — 可修改的训练脚本（你修改的唯一文件）
4. **验证环境**：检查 `train/prepare.py` 可以正常运行：`python train/prepare.py`
5. **初始化 results.tsv**：创建 `train/results.tsv`，只有标题行。baseline 将在第一次运行后记录。
6. **确认并开始**：确认设置看起来正常。

一旦获得确认，就开始实验。

## 实验循环

每个实验在单个 GPU 上运行。训练脚本运行 **固定时间预算：10分钟**（墙钟训练时间，不包括启动时间）。

启动方式：`python train/train.py`

**你可以修改的内容**：
- `train/train.py` — 这是唯一可以编辑的文件。一切都是公平的：
  - `HyperparametersConfig` — 学习率、gamma、clip_range 等
  - `RewardConfig` — 奖励函数参数
  - `ModelConfig` — 模型架构选择（MLP vs Transformer）
  - `create_model()` — 模型创建逻辑
  - `inject_reward_config()` — 奖励注入逻辑
  - `train_main()` — 训练循环逻辑

**你不能修改的内容**：
- `prepare.py` — 它是只读的。包含固定的评估、环境创建和常量。
- 安装新包或添加依赖。只能使用已安装的包。
- 修改评估逻辑。`evaluate_model()` 函数是基准指标。
- 固定时间预算（10分钟）。

**目标很简单：获得最高的综合评分**。综合评分 = `win_rate * 100 + mean_reward`。

由于时间预算固定，你不需要担心训练时间 — 总是10分钟。一切都是公平的：改变架构、优化器、超参数、批大小、模型大小。唯一约束是代码能运行不崩溃并在时间预算内完成。

**VRAM** 是软约束。一些增加对有意义的改进是可接受的，但不应该爆炸性增长。上限是 8GB。

**简洁性标准**：在同等条件下，简单更好。一个增加丑陋复杂性的小改进不值得。相反，删除代码获得相同或更好的结果是一个很好的成果 — 那是一个简化胜利。在评估是否保留改变时，权衡复杂性成本与改进幅度。一个 0.001 win_rate 改进增加了20行代码？可能不值得。一个 0.001 win_rate 改进来自删除代码？绝对保留。改进 ~0 但代码更简单？保留。

**第一次运行**：第一次运行应该始终建立 baseline，所以你要运行当前训练脚本。

## 输出格式

脚本完成后打印摘要，格式如下：

```
---
win_rate:         0.123456
mean_reward:      12.345678
std_reward:       5.123456
training_seconds: 600.1
total_steps:      123456
peak_memory_mb:   4096.2
fps:              205.7
```

注意脚本配置为总是在10分钟后停止，所以根据这台计算机的计算平台，数字可能看起来不同。你可以从日志文件中提取关键指标：

```bash
grep "^win_rate:\|^mean_reward:" train/run.log
```

## 记录结果

实验完成后，记录到 `train/results.tsv`（tab分隔，**不是逗号分隔** —逗号在描述中会破坏）。

TSV 有标题行和7列：

```
commit	win_rate	mean_reward	memory_mb	status	description
```

1. git commit hash（短，7字符）
2. win_rate 达到（例如 0.123456）
3. mean_reward 达到（例如 12.34）
4. 峰值内存（MB，例如 4096.0）
5. status：`keep`、`discard`或`crash`
6. 简短文本描述这个实验尝试了什么

示例：

```
commit	win_rate	mean_reward	memory_mb	status	description
a1b2c3d	0.150000	12.34	4096.0	keep	baseline
b2c3d4e	0.180000	15.67	4100.0	keep	increase LR to 1e-3
c3d4e5f	0.120000	10.23	4096.0	discard	decrease gamma to 0.95
d4e5f6g	0.000000	0.00	0.0	crash	OOM with large model
```

## 实验循环

实验在专用分支上运行（例如 `autoresearch/apr2` 或 `autoresearch/apr2-gpu0`）。

**永久循环**：

1. 查看 git 状态：当前所在的分支/commit
2. 通过直接修改代码调整 `train/train.py`，尝试实验想法
3. git commit
4. 运行实验：`python train/train.py > train/run.log 2>&1`（重定向所有内容 — **不要使用 tee 或让输出淹没你的上下文**）
5. 读出结果：`grep "^win_rate:\|^mean_reward:\|^peak_memory_mb:" train/run.log`
6. 如果 grep 输出为空，运行崩溃。运行 `tail -n 50 train/run.log` 读取 Python 栈跟踪并尝试修复。如果几次尝试后仍无法工作，放弃。
7. 在 tsv 中记录结果（注意：不要 commit results.tsv 文件，保持它不被 git 跟踪）
8. 如果 win_rate + mean_reward 改进（更高），你"推进"分支，保留 git commit
9. 如果 win_rate + mean_reward 相等或更差，你 git reset 回到起点

想法是你是一个完全自主的研究者尝试各种方法。如果它们工作，保留。如果不工作，丢弃。你在推进分支以便迭代。如果你感觉卡住了，你可以回退但你应该非常非常少这样做（如果有的话）。

**超时**：每个实验应该花费 ~10分钟（+几秒启动和评估开销）。如果一个运行超过15分钟，杀掉它并视为失败（丢弃并恢复）。

**崩溃**：如果运行崩溃（OOM、bug等），使用你的判断：如果是愚蠢且容易修复的（例如拼写错误、缺少导入），修复并重新运行。如果想法本身根本上坏了，跳过它，在 tsv 中记录"crash"作为状态，继续。

**永不停止**：一旦实验循环开始（初始设置后），**不要暂停询问用户是否应该继续**。不要问"我应该继续吗？"或"这是一个好的停止点吗？"。用户可能在睡觉或离开计算机并期望你继续工作 *无限期* 直到被手动停止。你是自主的。如果你用完想法，更努力思考 — 读代码中引用的论文、重读相关文件找新角度、尝试组合之前的接近成功的、尝试更激进的架构改变。循环运行直到用户中断你，周期。

作为示例使用案例，用户可能在你运行时睡觉。如果每个实验花费你 ~10分钟，那么你可以运行大约 6/小时，在平均人类睡眠期间总共约 60。用户醒来看到实验结果，所有由你在他们睡觉时完成！

## 可尝试的方向

以下是一些你可以尝试的方向（不限于这些）：

### 1. 超参数调整
- **学习率调度**：尝试不同调度（cosine decay、linear warmup、constant）
- **学习率值**：尝试 1e-4, 3e-4, 1e-3, 3e-3
- **折扣因子**：尝试 gamma = 0.95, 0.99, 0.995
- **GAE lambda**：尝试 0.9, 0.95, 0.98
- **熵系数**：尝试 0.01, 0.05, 0.1（鼓励探索）
- **Clip范围**：尝试 0.1, 0.2, 0.3

### 2. 奖励函数优化
- **奖励缩放**：调整 win_reward、lose_penalty 的比例
- **奖励平衡**：调整伤害奖励、技能奖励的权重
- **奖励整形**：尝试添加中间奖励（如回合存活奖励）
- **奖励稀疏性**：尝试只使用最终奖励 vs 使用密集奖励

### 3. 模型架构变化
- **Transformer vs MLP**：切换 `use_transformer`
- **网络深度**：修改 `transformer_depth` 或 MLP hidden layers
- **网络宽度**：修改 hidden sizes 或 embed_dim
- **注意力机制**：尝试不同的 transformer_heads 数量

### 4. 训练策略改进
- **早停策略**：根据中间评估结果提前停止
- **课程学习**：从简单场景开始逐渐增加难度
- **多环境训练**：调整 n_envs（1, 4, 8）
- **批次大小**：调整 n_steps 和 batch_size

### 5. 其他创新想法
- **探索策略**：调整熵系数鼓励更多探索
- **价值函数**：调整 vf_coef 权重
- **梯度裁剪**：尝试不同的 max_grad_norm

## 成功案例示例

以下是一些可能的改进路径：

1. **学习率优化**：
   - baseline: win_rate=0.15, mean_reward=12.34
   - 实验: LR=1e-3 → win_rate=0.18, mean_reward=15.67
   - 状态: keep

2. **奖励调整**：
   - baseline: win_rate=0.18, mean_reward=15.67
   - 实验: win_reward=200, lose_penalty=-100 → win_rate=0.20, mean_reward=18.45
   - 状态: keep

3. **架构简化**：
   - baseline: win_rate=0.20, mean_reward=18.45
   - 实验: 移除复杂奖励 → win_rate=0.21, mean_reward=19.23
   - 状态: keep（简化胜利）

4. **失败案例**：
   - 实验: 大模型 → crash（OOM）
   - 实验: gamma=0.9 → win_rate=0.12, mean_reward=10.23
   - 状态: discard

## 重要提示

1. **保持结果文件的整洁**：results.tsv 不要被 git 跟踪，保持 untracked
2. **记录足够详细**：description 列应该让用户理解你尝试了什么
3. **诚实记录**：即使实验失败，也要记录 crash 状态
4. **不要过度优化**：简洁性比小改进更重要
5. **自主学习**：如果用完想法，参考 PROJECT_STATUS.md、README.md 找新方向

开始实验吧！祝你好运！ 🎯