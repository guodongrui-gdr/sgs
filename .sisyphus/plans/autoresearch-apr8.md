# Autonomous RL Experiment Loop - April 8 Execution

## TL;DR

> **Quick Summary**: 执行 program.md 中定义的自主 RL 实验循环，目标是持续优化三国杀 AI 的胜率和平均奖励。
>
> **Deliverables**:
> - 完成实验环境设置（环境验证、results.tsv初始化）
> - 运行第一个 baseline 实验（建立初始基准）
> - 启动自主实验循环（agent 自主迭代优化）
> - results.tsv 记录所有实验结果（baseline + 后续迭代）
>
> **Estimated Effort**: Medium（设置 + 10分钟baseline + 持续循环）
> **Parallel Execution**: Phase 1 可并行，Phase 2-3 顺序执行
> **Critical Path**: Setup → Baseline Experiment → Initiate Loop → Autonomous Iteration

---

## Context

### Original Request
查看 train/program.md 并启动新的自动实验

### Interview Summary
**关键发现**：
- program.md 已经是完整的自主循环规范（194行），无需重新设计循环机制
- 正确的意图分类：Mid-sized Task（设置 + 执行），而非 Architecture
- 采用3阶段方法：Setup → First Experiment → Initiate Loop

**Metis Review分析**：
- program.md IS the loop specification - Prometheus 应委托执行而非重新设计
- 计划完成标准：循环已启动（而非"10个实验完成"）
- Phase 3 任务必须读取 program.md 并遵循其规范

**当前状态**：
- ✓ Git 分支已存在：`autoresearch/apr8`
- ✓ 环境已验证：Dependencies OK, Environment OK
- ✓ results.tsv 已创建（含标题行）
- ✓ program.md, PROJECT_STATUS.md, train.py 已读取
- ⚠️ 待完成：提交初始文件到 git

### Research Findings
**从 train/program.md 提取的关键信息**：
- 固定时间预算：600秒（10分钟）
- 综合评分公式：`win_rate * 100 + mean_reward`
- VRAM 上限：8192 MB
- 唯一可修改文件：`train/train.py`
- 不可修改：`prepare.py`, 评估函数, 时间预算
- 简洁性标准：删除代码获得相同或更好结果 = 简化胜利

**从 PROJECT_STATUS.md 提取的背景**：
- 项目目标：训练 MaskablePPO 玩三国杀
- 技能决策系统已重构（观星、遗计、离间等）
- 最近训练状态：正常运行（100K timesteps）
- Autoresearch 模式：基于 karpathy/autoresearch 思想实现

**从 train/train.py 提取的可修改内容**：
- `HyperparametersConfig`: LR, gamma, clip_range, ent_coef, n_steps, batch_size
- `RewardConfig`: win_reward, lose_penalty, damage奖励, skill奖励
- `ModelConfig`: use_transformer, MLP hidden sizes, Transformer配置
- `train_main()`: 训练循环逻辑

---

## Work Objectives

### Core Objective
让 agent 自主执行 RL 训练实验循环，持续迭代优化模型参数，目标是提高三国杀 AI 的胜率（win_rate）和平均奖励（mean_reward）。

### Concrete Deliverables
- Git 分支：`autoresearch/apr8`（已存在）
- 文件提交：`train/prepare.py`, `train/train.py`, `train/program.md`（初始提交）
- 实验记录：`train/results.tsv`（baseline + 所有后续实验）
- Baseline 实验：第一个10分钟训练运行（建立初始基准）
- 自主循环：agent 读取 program.md 并执行永久循环

### Definition of Done
- [ ] Git 状态：初始文件已提交（commit hash 记录）
- [ ] Baseline 记录：results.tsv 第一行包含 baseline 数据
- [ ] 循环启动：agent 正在执行 program.md 的永久循环
- [ ] 结果验证：grep 能从 run.log 提取 win_rate 和 mean_reward

### Must Have
- 环境正常运行（prepare.py 验证通过）
- Baseline 实验成功完成（无崩溃）
- results.tsv 正确格式（tab分隔，非逗号）
- Git commit 包含正确的描述信息

### Must NOT Have (Guardrails)
- **绝不修改 prepare.py**：这是只读文件（固定配置和评估函数）
- **绝不修改评估逻辑**：evaluate_model() 函数不可变
- **绝不修改时间预算**：600秒固定（不可调整）
- **绝不暂停询问用户**：循环必须自主运行（"永不停止"要求）
- **绝不提交 results.tsv 到 git**：该文件保持 untracked（便于记录所有实验）
- **绝不在循环中使用 "pause to ask user"**：agent 应自主决策所有后续实验

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - Phase 1-2 的验证是 agent-executed，Phase 3 是自主循环无验证终点。

### Test Decision
- **Infrastructure exists**: YES（.venv/bin/python 可用）
- **Automated tests**: NO（这是实验运行，不是单元测试）
- **Framework**: .venv/bin/python（虚拟环境中的 Python）
- **Baseline Experiment**: 运行 train/train.py，10分钟时长，无测试框架

### QA Policy
Phase 1-2 的每个任务都有 agent-executed QA 验证：
- **Phase 1 (Setup)**: Bash 命令验证（git status, python prepare.py, results.tsv 存在）
- **Phase 2 (Baseline)**: Bash 命令提取结果（grep from run.log），验证 results.tsv 更新
- **Phase 3 (Loop)**: 无验证终点（循环永久运行直到用户手动中断）

---

## Execution Strategy

### Parallel Execution Waves

> Phase 1 大部分已完成，Phase 2-3 顺序执行。
> 计划自然结束于"循环已启动"（循环本身永久运行）。

```
Wave 1 (Setup - 大部分已完成):
├── Task 1: 验证 git 分支 [quick] - ✓ 已存在 autoresearch/apr8
├── Task 2: 验证环境 [quick] - ✓ Dependencies OK, Environment OK
├── Task 3: 初始化 results.tsv [quick] - ✓ 已创建含标题行
├── Task 4: 读取 program.md [quick] - ✓ 已读取理解循环机制
└── Task 5: 提交初始文件 [quick] - ⚠️ 待完成（git add + commit）

Wave 2 (Baseline Experiment - Sequential):
├── Task 6: 运行 baseline 实验 [unspecified-low] - 等待10分钟
└── Task 7: 记录 baseline [quick] - 提取结果 + git commit

Wave 3 (Initiate Loop - Delegation):
└── Task 8: 启动自主循环 [deep] - 读取 program.md 并执行永久循环

Wave FINAL (无验证 - 循环永久运行):
→ 循环运行直到用户手动中断（Ctrl+C 或关闭终端）
→ 计划完成：循环已启动
```

### Dependency Matrix

- **1-5**: - - Wave 2-3, parallel setup tasks
- **6**: 5 - 7, sequential（等待10分钟）
- **7**: 6 - 8, sequential（依赖 baseline 结果）
- **8**: 7 - 无后续任务（循环永久运行）

> Wave 1 任务可并行（大部分已完成），Wave 2 必须顺序（10分钟等待），Wave 3 是终点。

### Agent Dispatch Summary

- **Wave 1**: **5** - T1-T4 已完成，T5 → `quick`（git commit）
- **Wave 2**: **2** - T6 → `unspecified-low`（长时间运行），T7 → `quick`（记录结果）
- **Wave 3**: **1** - T8 → `deep`（自主循环，委托给 program.md）
- **Wave FINAL**: 无（循环永久运行，无验证终点）

---

## TODOs

> Phase 1 大部分已完成，Phase 2-3 是核心任务。
> Phase 3 是终点：启动循环后，计划完成。

- [ ] 1. 验证 Git 分支状态

  **What to do**:
  - 确认当前在 `autoresearch/apr8` 分支上
  - 检查分支是否存在且状态正常
  - 如果分支不存在，从 master 创建：`git checkout -b autoresearch/apr8`

  **Must NOT do**:
  - 不要创建其他日期的分支（必须是 apr8）
  - 不要切换到其他分支（保持在 autoresearch/apr8）

  **Recommended Agent Profile**:
  - **Category**: `quick`（快速验证任务）
    - Reason: 只需运行 git status 命令验证，无需复杂逻辑
  - **Skills**: [`git-master`]
    - `git-master`: Git 操作专业知识，确保分支操作正确

  **Parallelization**:
  - **Can Run In Parallel**: YES（与其他 Phase 1 任务并行）
  - **Parallel Group**: Wave 1 (with Tasks 2-5)
  - **Blocks**: Task 5（git commit 依赖分支状态）
  - **Blocked By**: None（可立即执行）

  **References**:
  - `train/program.md:9-11` - 实验分支创建要求："分支 `autoresearch/<tag>` 必须不存在"
  - Git 命令参考：`git branch`, `git checkout -b`

  **Acceptance Criteria**:
  - [ ] `git branch` 输出包含 `* autoresearch/apr8`
  - [ ] 当前在正确的分支上（不是 master 或其他）

  **QA Scenarios**:
  ```
  Scenario: Git 分支验证
    Tool: Bash
    Preconditions: Git 仓库存在
    Steps:
      1. git branch | grep "autoresearch/apr8"
      2. 验证输出包含 "* autoresearch/apr8"
    Expected Result: 分支名称匹配，有 * 标记表示当前分支
    Failure Indicators: 输出为空，或分支名称不匹配
    Evidence: .sisyphus/evidence/task-1-git-branch.txt
  ```

  **Evidence to Capture**:
  - [ ] git branch 输出保存到 evidence 文件

  **Commit**: NO（这是验证任务，不提交）

- [ ] 2. 验证训练环境

  **What to do**:
  - 运行 `train/prepare.py` 验证环境和依赖
  - 确认输出包含 "✓ Dependencies OK" 和 "✓ Environment OK"
  - 检查固定常量输出正确（Time budget: 600s, Eval episodes: 50, etc.）

  **Must NOT do**:
  - 不要修改 prepare.py（这是只读文件）
  - 不要修改环境配置（PLAYER_NUM, MAX_ROUNDS 等固定）
  - 不要跳过验证（必须确认环境正常）

  **Recommended Agent Profile**:
  - **Category**: `quick`（快速验证任务）
    - Reason: 单一命令验证，输出检查简单
  - **Skills**: []
    - 无需特殊技能（Python 运行和环境检查是基础能力）

  **Parallelization**:
  - **Can Run In Parallel**: YES（与其他 Phase 1 任务并行）
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-5)
  - **Blocks**: Task 6（baseline 实验依赖环境正常）
  - **Blocked By**: None（可立即执行）

  **References**:
  - `train/prepare.py:283-340` - verify_environment() 和 verify_dependencies() 函数
  - `train/program.md:16` - "验证环境：检查 `train/prepare.py` 可以正常运行"
  - PROJECT_STATUS.md:194-209 - Autoresearch 模式说明

  **Acceptance Criteria**:
  - [ ] 输出包含 "✓ Dependencies OK"
  - [ ] 输出包含 "✓ Environment OK"
  - [ ] 输出显示正确的固定常量（Time budget: 600s）

  **QA Scenarios**:
  ```
  Scenario: 环境验证成功
    Tool: Bash
    Preconditions: Python 虚拟环境存在（.venv/bin/python）
    Steps:
      1. .venv/bin/python train/prepare.py
      2. 检查输出包含 "✓ Dependencies OK"
      3. 检查输出包含 "✓ Environment OK"
      4. 检查输出包含 "Time budget: 600s"
    Expected Result: 所有验证标记为 ✓，无错误信息
    Failure Indicators: ImportError, ModuleNotFoundError, 或缺少 ✓ 标记
    Evidence: .sisyphus/evidence/task-2-env-verify.txt

  Scenario: 环境验证失败（假设场景）
    Tool: Bash
    Preconditions: 缺少依赖包（模拟）
    Steps:
      1. 运行 prepare.py
      2. 检查错误信息：ModuleNotFoundError
    Expected Result: 返回 False，报告缺失依赖
    Failure Indicators: Traceback, ImportError
    Evidence: .sisyphus/evidence/task-2-env-error.txt
  ```

  **Evidence to Capture**:
  - [ ] prepare.py 完整输出保存到 evidence 文件

  **Commit**: NO（这是验证任务，不提交）

- [ ] 3. 初始化 results.tsv

  **What to do**:
  - 创建 `train/results.tsv` 文件，包含标题行
  - 标题格式：`commit\twin_rate\tmean_reward\tmemory_mb\tstatus\tdescription`（tab分隔，**非逗号**）
  - 确认文件存在且格式正确
  - 注意：此文件**不提交到 git**（保持 untracked，便于记录所有实验）

  **Must NOT do**:
  - 不要使用逗号分隔（会导致 description 列损坏）
  - 不要提交 results.tsv 到 git（根据 program.md，它应保持 untracked）
  - 不要添加初始数据行（只有标题，baseline 在 Task 7 添加）

  **Recommended Agent Profile**:
  - **Category**: `quick`（文件创建任务）
    - Reason: 单一文件创建，格式简单
  - **Skills**: []
    - 无需特殊技能（文件创建和格式检查是基础能力）

  **Parallelization**:
  - **Can Run In Parallel**: YES（与其他 Phase 1 任务并行）
  - **Parallel Group**: Wave 1 (with Tasks 1-2, 4-5)
  - **Blocks**: Task 7（记录 baseline 需要此文件存在）
  - **Blocked By**: None（可立即执行）

  **References**:
  - `train/program.md:76-99` - results.tsv 格式说明（7列，tab分隔）
  - `train/program.md:188-189` - "保持结果文件的整洁：results.tsv 不要被 git 跟踪"
  - .gitignore - 应包含 `train/results.tsv` 排除规则

  **Acceptance Criteria**:
  - [ ] 文件存在：`ls train/results.tsv` 返回文件路径
  - [ ] 标题行正确：包含7个列名，tab分隔
  - [ ] 文件未被 git 跟踪：`git status` 不显示 results.tsv（或显示在 .gitignore 提示中）

  **QA Scenarios**:
  ```
  Scenario: results.tsv 创建成功
    Tool: Bash
    Preconditions: train/ 目录存在
    Steps:
      1. printf "commit\twin_rate\tmean_reward\tmemory_mb\tstatus\tdescription\n" > train/results.tsv
      2. cat train/results.tsv（验证内容）
      3. git status --short（验证未被跟踪）
    Expected Result: 文件内容为标题行，git status 显示 ?? train/results.tsv（untracked）
    Failure Indicators: 文件不存在，或内容格式错误
    Evidence: .sisyphus/evidence/task-3-tsv-create.txt

  Scenario: 格式验证（tab分隔检查）
    Tool: Bash
    Preconditions: results.tsv 已创建
    Steps:
      1. head -1 train/results.tsv | od -c（检查字符）
      2. 确认包含 \t（tab字符），而非逗号
    Expected Result: od 输出显示 \t 字符
    Failure Indicators: 显示逗号而非 \t
    Evidence: .sisyphus/evidence/task-3-tsv-format.txt
  ```

  **Evidence to Capture**:
  - [ ] results.tsv 内容保存到 evidence 文件
  - [ ] git status 输出（显示 untracked）

  **Commit**: NO（results.tsv 保持 untracked）

- [ ] 4. 读取 program.md 理解循环机制

  **What to do**:
  - 读取 `train/program.md` 全文，理解自主实验循环的完整流程
  - 关键要点提取：
    - 固定时间预算（600秒）
    - 唯一可修改文件（train/train.py）
    - 实验循环步骤（git → modify → run → record → keep/discard）
    - "永不停止"要求（循环永久运行）
  - 确认理解简洁性标准和综合评分公式

  **Must NOT do**:
  - 不要修改 program.md（这是人类维护的指令文件）
  - 不要跳过关键章节（必须理解完整循环）
  - 不要误解"永不停止"要求（agent 必须自主运行，不询问用户）

  **Recommended Agent Profile**:
  - **Category**: `quick`（文档阅读任务）
    - Reason: 读取和理解文档，为后续任务做准备
  - **Skills**: []
    - 无需特殊技能（文档阅读是基础能力）

  **Parallelization**:
  - **Can Run In Parallel**: YES（与其他 Phase 1 任务并行）
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5)
  - **Blocks**: Task 8（启动循环依赖对此文档的理解）
  - **Blocked By**: None（可立即执行）

  **References**:
  - `train/program.md:全文` - 完整的自主实验指令（194行）
  - `train/program.md:101-123` - 永久循环流程（核心）
  - `train/program.md:28-51` - 可修改和不可修改约束
  - `train/program.md:123-126` - "永不停止"和"自主学习"要求

  **Acceptance Criteria**:
  - [ ] 文件已读取（确认 program.md 内容理解）
  - [ ] 提取关键信息：时间预算、约束、循环流程、评分公式
  - [ ] 理解"永不停止"要求：循环必须自主运行

  **QA Scenarios**:
  ```
  Scenario: program.md 内容验证
    Tool: Read
    Preconditions: train/program.md 存在
    Steps:
      1. 读取文件全文
      2. 检查包含 "固定时间预算：10分钟"
      3. 检查包含 "永不停止"
      4. 检查包含 "综合评分 = win_rate * 100 + mean_reward"
    Expected Result: 所有关键章节存在
    Failure Indicators: 文件缺失或关键内容缺失
    Evidence: .sisyphus/evidence/task-4-program-content.txt

  Scenario: 循环流程理解验证
    Tool: Bash（人工验证）
    Preconditions: program.md 已读取
    Steps:
      1. 确认理解 8 步循环：git → modify → commit → run → extract → record → keep/reset → repeat
      2. 确认理解约束：prepare.py不可变，train.py可变
    Expected Result: 能清晰描述循环步骤
    Failure Indicators: 无法描述循环流程
    Evidence: .sisyphus/evidence/task-4-loop-understanding.txt
  ```

  **Evidence to Capture**:
  - [ ] program.md 关键章节内容保存

  **Commit**: NO（这是理解任务，不提交）

- [ ] 5. 提交初始设置文件到 Git

  **What to do**:
  - 添加实验设置相关文件到 git：`train/prepare.py`, `train/train.py`, `train/program.md`
  - 注意：**不要添加 results.tsv**（应保持 untracked，便于记录所有实验）
  - 创建初始提交，描述为："feat(train): add autoresearch experiment setup files"
  - 确认提交成功（commit hash 记录，用于后续 baseline）

  **Must NOT do**:
  - 不要添加 results.tsv（应保持 untracked）
  - 不要添加 .sisyphus/ 目录（这是 Prometheus 的计划工作区）
  - 不要使用 -f 强制添加被 .gitignore 排除的文件
  - 不要提交 train/run.log（这是临时实验输出）

  **Recommended Agent Profile**:
  - **Category**: `quick`（Git 提交任务）
    - Reason: 标准的 git add + commit 操作
  - **Skills**: [`git-master`]
    - `git-master`: Git 操作专业知识，确保提交正确

  **Parallelization**:
  - **Can Run In Parallel**: YES（与其他 Phase 1 任务并行）
  - **Parallel Group**: Wave 1 (with Tasks 1-4)
  - **Blocks**: Task 6-7（后续实验依赖此 commit baseline）
  - **Blocked By**: Task 1（需要确认在正确分支上）

  **References**:
  - `train/program.md:17` - "初始化 results.tsv：创建只有标题行"（不提交）
  - `.gitignore` - 应排除 results.tsv, run.log, logs/
  - Git 命令参考：`git add`, `git commit`, `git rev-parse --short HEAD`

  **Acceptance Criteria**:
  - [ ] git status 显示 prepare.py, train.py, program.md 为已添加（A 状态）
  - [ ] git commit 成功（commit hash 7字符短码）
  - [ ] results.tsv 显示为 untracked（?? 状态）
  - [ ] .sisyphus/ 不在提交中

  **QA Scenarios**:
  ```
  Scenario: Git 提交成功
    Tool: Bash
    Preconditions: 在 autoresearch/apr8 分支，文件未提交
    Steps:
      1. git add train/prepare.py train/train.py train/program.md
      2. git status --short（验证 A 状态）
      3. git commit -m "feat(train): add autoresearch experiment setup files"
      4. git rev-parse --short HEAD（获取 commit hash）
      5. git log --oneline -1（验证提交）
    Expected Result: commit hash 显示，提交信息正确
    Failure Indicators: git add 失败，或 commit 失败
    Evidence: .sisyphus/evidence/task-5-git-commit.txt

  Scenario: results.tsv 未被跟踪验证
    Tool: Bash
    Preconditions: results.tsv 已创建，.gitignore 正确配置
    Steps:
      1. git status --short
      2. 检查 train/results.tsv 显示为 ??（untracked）
    Expected Result: ?? train/results.tsv（不被 git 管理）
    Failure Indicators: results.tsv 显示为 A（被添加）或修改状态
    Evidence: .sisyphus/evidence/task-5-results-untracked.txt
  ```

  **Evidence to Capture**:
  - [ ] git status 输出（显示 A 状态文件）
  - [ ] git log 输出（显示新 commit）
  - [ ] commit hash 记录（用于后续 baseline）

  **Commit**: YES
  - Message: `feat(train): add autoresearch experiment setup files`
  - Files: `train/prepare.py`, `train/train.py`, `train/program.md`, `.gitignore`（如果修改）
  - Pre-commit: 无（这是初始设置文件）

- [ ] 6. 运行第一个 Baseline 实验

  **What to do**:
  - 运行当前 train/train.py 配置作为 baseline（建立初始基准）
  - 命令：`.venv/bin/python train/train.py > train/run.log 2>&1`（重定向所有输出）
  - **等待完成**：实验运行约10分钟（600秒时间预算）+ 启动开销
  - 提取关键指标：`grep "^win_rate:\|^mean_reward:\|^peak_memory_mb:" train/run.log`
  - **重要**：这是第一个实验，必须记录 baseline，不要修改 train.py（保持原始配置）

  **Must NOT do**:
  - 不要修改 train/train.py（baseline 必须使用原始配置）
  - 不要使用 tee 或让输出淹没上下文（必须重定向到 run.log）
  - 不要提前结束（必须等待完整的600秒）
  - 不要跳过结果提取（必须 grep 输出指标）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`（长时间运行任务）
    - Reason: 需要等待10分钟，不需要复杂决策（只是运行 baseline）
  - **Skills**: []
    - 无需特殊技能（Python 运行是基础能力）

  **Parallelization**:
  - **Can Run In Parallel**: NO（必须等待完成）
  - **Parallel Group**: Wave 2（与 Task 7 顺序执行）
  - **Blocks**: Task 7（记录 baseline 依赖结果）
  - **Blocked By**: Task 2, 5（需要环境验证和 git 提交）

  **References**:
  - `train/program.md:52` - "第一次运行：第一次运行应该始终建立 baseline"
  - `train/program.md:110` - "运行实验：python train/train.py > train/run.log 2>&1"
  - `train/program.md:111` - "提取结果：grep "^win_rate:\|^mean_reward:" train/run.log"
  - `train/prepare.py:47-50` - FIXED_TIME_BUDGET = 600（固定时间预算）
  - `train/train.py:当前配置` - HyperparametersConfig, RewardConfig, ModelConfig

  **Acceptance Criteria**:
  - [ ] train/run.log 文件存在且非空
  - [ ] grep 提取到 win_rate, mean_reward, peak_memory_mb（数值格式正确）
  - [ ] 实验运行约10分钟（training_seconds ≈ 600）
  - [ ] 无崩溃（没有 Python traceback）

  **QA Scenarios**:
  ```
  Scenario: Baseline 实验成功完成
    Tool: Bash
    Preconditions: 环境验证通过，git 提交完成
    Steps:
      1. .venv/bin/python train/train.py > train/run.log 2>&1（启动训练）
      2. 等待完成（约10分钟）
      3. grep "^win_rate:\|^mean_reward:\|^peak_memory_mb:" train/run.log
      4. 检查数值格式（win_rate: 0.XXX, mean_reward: XX.XX）
    Expected Result: 提取到3个数值，training_seconds ≈ 600
    Failure Indicators: grep 输出为空，或文件不存在
    Evidence: .sisyphus/evidence/task-6-baseline-success.txt

  Scenario: Baseline 实验崩溃（错误场景）
    Tool: Bash
    Preconditions: 环境异常（模拟）
    Steps:
      1. 运行 train/train.py
      2. tail -n 50 train/run.log（查看错误）
      3. 检查 Python traceback
    Expected Result: 发现崩溃原因（如 ImportError, OOM）
    Failure Indicators: Traceback, MemoryError, 或其他异常
    Evidence: .sisyphus/evidence/task-6-baseline-crash.txt
  ```

  **Evidence to Capture**:
  - [ ] train/run.log 关键部分（grep 输出）
  - [ ] training_seconds 数值（验证时间预算）
  - [ ] 如果崩溃：tail -n 50 train/run.log

**Commit**: NO（baseline 运行，不提交代码修改）
  - 注意：Task 7 会提交 baseline 记录（git commit "experiment: baseline"）

- [ ] 7. 记录 Baseline 结果到 results.tsv

  **What to do**:
  - 获取当前 commit hash：`git rev-parse --short HEAD`（7字符短码）
  - 从 run.log 提取数值：win_rate, mean_reward, peak_memory_mb
  - 添加到 results.tsv：`<hash>\t<win_rate>\t<mean_reward>\t<memory>\tkeep\tbaseline`（tab分隔）
  - Git commit baseline 记录：`git commit --allow-empty -m "experiment: baseline"`
    - 使用 --allow-empty（因为 results.tsv 不提交，需要空 commit 记录 milestone）
  - 验证 results.tsv 更新：`tail -1 train/results.tsv`

  **Must NOT do**:
  - 不要提交 results.tsv（应保持 untracked）
  - 不要使用逗号分隔（会导致 description 列损坏）
  - 不要修改数值精度（保持 grep 提取的原始精度）
  - 不要跳过 git commit（需要记录 baseline milestone）

  **Recommended Agent Profile**:
  - **Category**: `quick`（数据记录和 git 操作）
    - Reason: 提取数值、添加行、git commit - 简单操作
  - **Skills**: [`git-master`]
    - `git-master`: Git 操作专业知识，确保 commit 正确

  **Parallelization**:
  - **Can Run In Parallel**: NO（依赖 Task 6 的结果）
  - **Parallel Group**: Wave 2（与 Task 6 顺序执行）
  - **Blocks**: Task 8（启动循环依赖 baseline 已记录）
  - **Blocked By**: Task 6（需要 baseline 实验结果）

  **References**:
  - `train/program.md:74-99` - results.tsv 格式和记录说明
  - `train/program.md:94` - 示例："a1b2c3d  0.150000  12.34  4096.0  keep  baseline"
  - `train/program.md:114-116` - "如果 win_rate + mean_reward 改进（更高），你'推进'分支，保留 git commit"

  **Acceptance Criteria**:
  - [ ] results.tsv 第二行存在（第一行是标题，第二行是 baseline）
  - [ ] commit hash 正确（7字符，匹配 git rev-parse 输出）
  - [ ] 数值格式正确（win_rate: 0.XXXXXX, mean_reward: XX.XXXXXX）
  - [ ] status = "keep"（baseline 保留）
  - [ ] description = "baseline"（标识第一个实验）
  - [ ] git log 显示 "experiment: baseline" commit

  **QA Scenarios**:
  ```
  Scenario: Baseline 记录成功
    Tool: Bash
    Preconditions: Task 6 完成，run.log 有结果
    Steps:
      1. git rev-parse --short HEAD（获取 commit hash）
      2. grep "^win_rate:\|^mean_reward:\|^peak_memory_mb:" train/run.log（提取数值）
      3. printf "<hash>\t<win_rate>\t<mean_reward>\t<memory>\tkeep\tbaseline\n" >> train/results.tsv
      4. tail -1 train/results.tsv（验证添加）
      5. git commit --allow-empty -m "experiment: baseline"
    Expected Result: results.tsv 有 baseline 行，git commit 成功
    Failure Indicators: results.tsv 无新行，或 git commit 失败
    Evidence: .sisyphus/evidence/task-7-baseline-record.txt

  Scenario: TSV 格式验证
    Tool: Bash
    Preconditions: results.tsv 有 baseline 行
    Steps:
      1. tail -1 train/results.tsv | od -c（检查 tab字符）
      2. 验证包含 \t（tab），而非逗号
      3. wc -w（验证7个字段）
    Expected Result: od 显示 \t 字符，wc 输出 7（7个字段）
    Failure Indicators: 显示逗号，或字段数不为7
    Evidence: .sisyphus/evidence/task-7-tsv-format.txt
  ```

  **Evidence to Capture**:
  - [ ] git rev-parse 输出（commit hash）
  - [ ] grep 输出（提取的数值）
  - [ ] tail -1 train/results.tsv（baseline 行）
  - [ ] git log --oneline -2（显示 baseline commit）

  **Commit**: YES
  - Message: `experiment: baseline`
  - Files: 无文件修改（使用 --allow-empty 创建 milestone commit）
  - Pre-commit: 无

- [ ] 8. 启动自主实验循环

  **What to do**:
  - **CRITICAL**: 读取并执行 `train/program.md` 的自主循环规范（lines 101-123）
  - 遵循"永久循环"指令：
    1. 查看 git 状态（确认当前 commit）
    2. 修改 train/train.py（尝试新配置）
    3. Git commit: "experiment: <description>"
    4. 运行实验：`.venv/bin/python train/train.py > train/run.log 2>&1`
    5. 等待10-15分钟（固定时间预算600秒）
    6. 提取结果：`grep "^win_rate:\|^mean_reward:\|^peak_memory_mb:" train/run.log`
    7. 记录到 results.tsv：添加新行（commit, win_rate, mean_reward, memory, status, description）
    8. 决策：
       - 如果 win_rate + mean_reward 改进：保留 commit（status = "keep"）
       - 如果退步：git reset --hard HEAD~1（status = "discard"）
    9. **重复循环 INFINITELY**（永不停止）
  - **自主决策**：每次迭代尝试不同的实验方向（参考 program.md lines 127-162）
  - **永不停止**：循环运行直到用户手动中断（Ctrl+C 或关闭终端）

  **Must NOT do**:
  - **绝不修改 prepare.py**：这是只读文件（固定配置和评估函数）
  - **绝不修改时间预算**：FIXED_TIME_BUDGET = 600（不可调整）
  - **绝不暂停询问用户**：循环必须自主运行（"永不停止"要求，program.md line 123）
  - **绝不提交 results.tsv 到 git**：保持 untracked（便于记录所有实验）
  - **绝不提前结束循环**：必须运行无限期（直到用户手动中断）

  **Recommended Agent Profile**:
  - **Category**: `deep`（自主循环，需要理解和决策）
    - Reason: 这是复杂的自主实验循环，需要理解 program.md、决策实验方向、持续迭代
  - **Skills**: [`git-master`]
    - `git-master`: Git 操作专业知识（commit, reset, status）

  **Parallelization**:
  - **Can Run In Parallel**: NO（这是计划终点，单线程循环）
  - **Parallel Group**: Wave 3（计划终点）
  - **Blocks**: 无后续任务（计划完成）
  - **Blocked By**: Task 7（需要 baseline 已记录）

  **References**:
  - `train/program.md:101-123` - **永久循环流程**（核心指令）
  - `train/program.md:123-126` - **永不停止和自主学习**（自主决策要求）
  - `train/program.md:127-162` - **可尝试的方向**（超参数、奖励、架构等）
  - `train/program.md:28-51` - **约束和目标**（prepare.py不可变，时间预算固定，简洁性标准）
  - `train/train.py:可修改部分` - HyperparametersConfig, RewardConfig, ModelConfig, train_main()
  - `train/prepare.py:28-340` - **不可修改部分**（固定常量、环境创建、评估函数）

  **Acceptance Criteria**:
  - [ ] agent 正在执行 program.md 的循环流程（验证：git status, run.log 存在）
  - [ ] 循环自主运行（无"暂停询问用户"步骤）
  - [ ] 遵守所有约束（prepare.py不可变，时间预算600秒，results.tsv不提交）
  - [ ] 计划完成：循环已启动（无验证终点）

  **QA Scenarios**:
  ```
  Scenario: 循环启动验证
    Tool: Bash
    Preconditions: Baseline 已记录，环境正常
    Steps:
      1. 检查 agent 正在读取 program.md（理解循环流程）
      2. 检查 git log 显示 baseline commit
      3. 检查 results.tsv 有 baseline 行
      4. 确认循环开始运行（run.log 正在生成）
    Expected Result: 循环启动，agent 自主运行
    Failure Indicators: agent 暂停询问，或未开始循环
    Evidence: .sisyphus/evidence/task-8-loop-start.txt

  Scenario: 循环自主性验证（关键）
    Tool: Read（检查 agent 行为）
    Preconditions: 循环已启动
    Steps:
      1. 验证 agent 没有"pause to ask user"步骤
      2. 验证 agent 尝试自主实验方向（如修改 LR、gamma、奖励等）
      3. 验证 agent 在迭代中持续运行（不停止）
    Expected Result: agent 自主决策，持续迭代
    Failure Indicators: agent 询问"Should I continue?", 或停止运行
    Evidence: .sisyphus/evidence/task-8-loop-autonomy.txt

  Scenario: 约束遵守验证
    Tool: Bash（检查文件修改）
    Preconditions: 循环运行多次迭代
    Steps:
      1. git diff（检查只修改 train/train.py）
      2. git log --oneline | grep prepare.py（验证无 prepare.py 提交）
      3. grep "FIXED_TIME_BUDGET" train/prepare.py（验证时间预算未改）
      4. git status --short | grep results.tsv（验证 ?? 状态）
    Expected Result: 只修改 train.py，其他文件未变
    Failure Indicators: prepare.py 被修改，或时间预算改变
    Evidence: .sisyphus/evidence/task-8-constraints.txt
  ```

  **Evidence to Capture**:
  - [ ] git log 显示多次 experiment commit（循环迭代证据）
  - [ ] results.tsv 有多行数据（baseline + 后续实验）
  - [ ] git diff 显示只有 train/train.py 修改
  - [ ] 无 agent 询问步骤的证据（循环自主运行）

  **Commit**: YES（循环中的每次迭代）
  - Message 格式：`experiment: <description>`（如 "experiment: increase LR to 1e-3"）
  - Files: `train/train.py`（唯一可修改文件）
  - Pre-commit: 无（直接运行实验）
  - 如果改进：保留 commit（status = "keep"）
  - 如果退步：git reset --hard HEAD~1（status = "discard"）

  **计划终点说明**:
  - 此任务完成后，计划自然结束
  - 循环本身永久运行（无验证终点）
  - 用户需要手动中断（Ctrl+C 或关闭终端）
  - 计划完成条件："循环已启动"

---

## Final Verification Wave

> **无最终验证** - Phase 3 启动循环后，计划完成。循环本身永久运行无验证终点。
> 循环运行直到用户手动中断（Ctrl+C 或关闭终端）。

**计划完成条件**：
- [ ] Phase 1: Git 提交完成（commit hash 记录）
- [ ] Phase 2: Baseline 实验完成（results.tsv 第一行有数据）
- [ ] Phase 3: 循环已启动（agent 正在执行 program.md）
- [ ] 无需用户干预：agent 自主运行后续迭代

---

## Commit Strategy

- **Task 5**: `feat(train): add autoresearch experiment setup files` - train/prepare.py, train/train.py, train/program.md, .gitignore
  - Pre-commit: 无（这些是初始设置文件）

- **Task 7**: `experiment: baseline` - train/train.py（如果有改动），results.tsv 更新但不提交
  - Pre-commit: 无（baseline 运行）

- **Task 8 及后续**: 循环中的每次实验遵循 program.md 的 commit 规范
  - 格式：`experiment: <description>`
  - 如果改进：保留 commit（git keep）
  - 如果退步：reset 到起点（git reset --hard HEAD~1）

---

## Success Criteria

### Verification Commands
```bash
# Phase 1 验证
git branch | grep "autoresearch/apr8"  # Expected: * autoresearch/apr8
.venv/bin/python train/prepare.py | grep "✓"  # Expected: ✓ Dependencies OK, ✓ Environment OK
head -1 train/results.tsv  # Expected: commit\twin_rate\tmean_reward\tmemory_mb\tstatus\tdescription

# Phase 2 验证
grep "^win_rate:\|^mean_reward:" train/run.log  # Expected: win_rate: 0.XX, mean_reward: XX.XX
tail -1 train/results.tsv | grep "baseline"  # Expected: baseline 行存在

# Phase 3 验证（启动检查）
git log --oneline -1  # Expected: 最新的 experiment commit
ls -la train/run.log  # Expected: run.log 文件存在（循环正在运行）
```

### Final Checklist
- [ ] 所有 Phase 1 设置完成（git, env, tsv）
- [ ] Baseline 实验成功运行（10分钟，无崩溃）
- [ ] results.tsv 包含 baseline 数据
- [ ] 循环已启动（agent 正在自主运行）
- [ ] 遵守所有约束（prepare.py不可变，时间预算固定）