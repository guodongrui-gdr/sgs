# 自我对弈训练系统构建

## TL;DR

> **Quick Summary**: 从头构建完整的自我对弈训练系统，支持10个agent池随机配对对抗，联合训练通用策略+技能决策（观星、遗计等高影响技能），大规模并行（8-16环境），GPU加速，2M+训练步数。
> 
> **Deliverables**: 
> - train/self_play_wrapper.py - 自我对弈环境包装器
> - train/agent_pool_manager.py - Agent池管理系统（ELO跟踪）
> - train/policy_extension.py - TransformerPolicy扩展（技能决策支持）
> - train/train_self_play.py - 自我对弈训练循环
> - train/evaluate.py - 评估脚本（胜率、ELO、身份统计）
> - train/visualize.py - TensorBoard可视化工具
> - train/config.py - 训练参数配置
> - train/recovery.py - 训练状态恢复机制
> - tests/ - TDD测试套件
> 
> **Estimated Effort**: XL
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: T1→T2→T3→T4→T5→T6→T10→T11→T12→T13→F1-F4

---

## Context

### Original Request
用户已清空train文件夹，需要从头构建一套全新的自我对弈训练系统。

### Interview Summary
**Key Discussions**:
- 训练算法: 自我对弈（AlphaGo风格）+ 随机配对池（10 agents）
- 训练目标: 多目标组合 - 联合训练通用策略+技能决策
- 训练流程: 基础训练循环 + 评估系统 + 可视化工具
- 训练输出: TensorBoard日志 + checkpoint（每10万步） + 训练报告 + 可视化回放
- 训练规模: 大规模（200万步以上），8-16并行环境，GPU可用
- 测试策略: TDD测试驱动开发
- 部署方式: 单机训练

**Research Findings**:
- 现有策略网络: ai/models/transformer_policy.py（TransformerPolicy，d_model=256，4层encoder）
- 现有环境: ai/gym_wrapper.py（SGSEnv，Dict观察空间~3000维，Discrete动作空间，分层解码）
- 技能决策系统: ai/skill_decision.py（7种决策类型，异步Step 3模式）
- 奖励系统: ai/reward.py（终端奖励~10x中间奖励，身份感知）
- RLAI推理: ai/rl_ai.py（MaskablePPO/PPO加载，action_masks支持）
- Agent池基础设施: ai/policy_pool.py（PolicyPool，ELO系统，IdentityStats）
- 自我对弈模式: ai/self_play.py（SelfPlayTrainer，TrainingMode.MIXED）

### Metis Review
**Identified Gaps** (to be addressed):

**CRITICAL (需用户决策)**:
- 技能决策训练范围: 哪些具体技能？43个全部还是subset？[DECISION NEEDED]

**Defaults Applied** (基于Metis推荐):
- Policy架构: 单策略（current_step路由）而非多头架构
- Agent池配比: 复用现有PolicyPool，包含历史checkpoint+RuleAI baseline（MIXED模式）
- GPU并行策略: 先4-8环境验证内存，后扩展到8-16
- Checkpoint粒度: 策略快照（pool使用）+ 完整训练状态（恢复使用）
- 身份训练策略: 随机身份每episode而非固定身份
- 评估指标: 先胜率+身份胜率+ELO，后续扩展技能质量指标
- TDD测试范围: 全面覆盖（agent池、技能决策掩码、checkpoint兼容性、多头策略验证）

**Guardrails Applied**:
- MUST复用现有模块: ai/gym_wrapper.py, ai/reward.py, ai/skill_decision.py, ai/policy_pool.py
- MUST保持checkpoint兼容: MaskablePPO/PPO .zip格式，RLAI可加载
- MUST保持技能决策Step 3协议: current_step=3时特殊observation
- MUST保持分层动作解码: Discrete(20)动作空间不变
- MUST NOT修改现有ai/模块编码器（StateEncoder, ActionEncoder）
- MUST NOT训练所有43技能（先5个高影响技能）
- MUST NOT重建agent池系统（复用PolicyPool）

---

## Work Objectives

### Core Objective
构建完整的自我对弈训练系统，使AI能够通过与历史版本对抗持续提升策略质量，同时优化复杂技能决策能力。

### Concrete Deliverables
- train/self_play_wrapper.py - 自我对弈环境包装器（继承SGSEnv）
- train/agent_pool_manager.py - Agent池管理（基于PolicyPool扩展）
- train/policy_extension.py - TransformerPolicy技能决策扩展
- train/train_self_play.py - 主训练循环脚本
- train/evaluate.py - 定期评估脚本
- train/visualize.py - TensorBoard可视化配置
- train/config.py - 训练参数配置类
- train/recovery.py - 训练状态恢复工具
- tests/test_self_play_wrapper.py
- tests/test_agent_pool_manager.py
- tests/test_policy_extension.py
- tests/test_train_self_play.py
- tests/test_checkpoint_compatibility.py
- tests/test_skill_decision_masking.py

### Definition of Done
- [ ] 训练脚本可运行并完成100K步训练测试
- [ ] Checkpoint可通过RLAI加载并推理
- [ ] Agent池ELO系统工作（10 agents，采样分布正确）
- [ ] 技能决策掩码正确（Step 3 observation）
- [ ] 评估脚本生成JSON报告（胜率、ELO、身份统计）
- [ ] TensorBoard日志正常（训练曲线、胜率曲线）
- [ ] 所有pytest测试通过

### Must Have
- 自我对弈环境包装器（opponent sampling）
- Agent池管理（10 agents，ELO，采样分布30/30/40）
- 技能决策训练支持（Step 3 routing）
- Checkpoint兼容RLAI（MaskablePPO/PPO .zip）
- TDD测试套件（所有核心模块）
- 训练恢复机制（从checkpoint继续）
- 评估系统（胜率、ELO、身份胜率）
- TensorBoard可视化

### Must NOT Have (Guardrails)
- 不修改ai/gym_wrapper.py（SGSEnv保持不变）
- 不修改ai/reward.py（RewardSystem保持不变）
- 不修改ai/skill_decision.py（SkillDecisionRequest保持不变）
- 不修改ai/state_encoder.py、ai/action_encoder.py
- 不重建agent池系统（复用ai/policy_pool.py）
- 不训练所有43技能（先subset高影响技能）
- 不改变checkpoint格式（必须RLAI兼容）
- 不改变动作空间编码（Discrete(20)不变）

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES（pytest在tests/目录）
- **Automated tests**: TDD（测试驱动开发）
- **Framework**: pytest
- **If TDD**: 每个任务遵循RED（先写测试） → GREEN（实现功能） → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python模块**: pytest运行测试，验证测试通过
- **训练脚本**: 运行训练测试模式（1000步），验证checkpoint生成
- **Checkpoint兼容**: 加载checkpoint并通过RLAI推理
- **评估脚本**: 运行评估，验证JSON报告生成

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - 环境与池管理):
├── Task 1: train/config.py + tests (配置系统)
├── Task 2: train/agent_pool_manager.py + tests (Agent池管理，复用PolicyPool)
├── Task 3: train/self_play_wrapper.py + tests (自我对弈环境包装器)
└── Task 4: train/recovery.py + tests (训练状态恢复)

Wave 2 (Policy Extension - 技能决策支持):
├── Task 5: train/policy_extension.py + tests (TransformerPolicy扩展)
├── Task 6: Skill decision masking validation (验证Step 3协议)
└── Task 7: train/evaluate.py + tests (评估脚本)

Wave 3 (Training Loop - 核心训练):
├── Task 8: train/train_self_play.py (主训练循环)
├── Task 9: train/visualize.py + tests (TensorBoard配置)
└── Task 10: Training integration test (100K步测试)

Wave 4 (Scaling & Optimization):
├── Task 11: GPU memory profiling (验证8-16环境可行性)
├── Task 12: Hyperparameter tuning (训练参数优化)
└── Task 13: 2M步大规模训练验证

Wave FINAL (Verification - 并行review):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix

- **1-4**: - (无依赖，可并行)
- **5**: 1 - (依赖配置)
- **6**: 5 - (依赖policy扩展)
- **7**: 1, 2 - (依赖配置和池)
- **8**: 1, 2, 3, 5 - (依赖所有基础模块)
- **9**: 1 - (依赖配置)
- **10**: 8 - (依赖训练脚本)
- **11**: 10 - (依赖集成测试)
- **12**: 11 - (依赖GPU profiling）
- **13**: 12 - (依赖参数优化）

### Agent Dispatch Summary

- **Wave1**: **4** - T1-T4 → `quick`
- **Wave2**: **3** - T5 → `deep`, T6 → `quick`, T7 → `unspecified-high`
- **Wave3**: **3** - T8 → `deep`, T9 → `quick`, T10 → `unspecified-high`
- **Wave4**: **3** - T11 → `quick`, T12 → `unspecified-high`, T13 → `deep`
- **FINAL**: **4** - F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. **训练配置系统** - train/config.py + tests/test_config.py

  **What to do**:
  - 创建TrainingConfig dataclass（训练参数：timesteps, n_envs, checkpoint_freq, eval_freq, pool_size等）
  - 创建SelfPlayConfig（agent池参数：sampling_distribution, elo_init等）
  - 创建SkillDecisionConfig（技能训练范围：哪些技能、奖励系数等）
  - 编写tests/test_config.py验证配置类（TDD先写测试）
  
  **Must NOT do**:
  - 不修改ai/config.py（仅新增train/专用配置）
  - 不添加与训练无关的配置项

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 配置dataclass简单定义，测试 straightforward
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Tasks 5, 7, 8, 9
  - **Blocked By**: None (可立即开始)

  **References**:
  - ai/config.py:SGSConfig - 环境配置模式参考（dataclass风格）
  - ai/policy_pool.py:PolicyPoolConfig - Agent池配置参考
  - ai/reward.py:RewardConfig - 奖励配置参考
  
  **WHY Each Reference Matters**:
  - ai/config.py:SGSConfig - 学习现有dataclass配置风格（player_num, max_rounds等）
  - ai/policy_pool.py:PolicyPoolConfig - 了解pool配置字段（elo_init, version等）
  - ai/reward.py:RewardConfig - 理解奖励系数配置结构（terminal奖励比例）

  **Acceptance Criteria** (TDD):
  - [ ] Test file created: tests/test_config.py
  - [ ] pytest tests/test_config.py → PASS (配置实例化、字段验证)

  **QA Scenarios**:
  ```
  Scenario: 配置实例化成功
    Tool: Bash (pytest)
    Preconditions: train/config.py已创建
    Steps:
      1. pytest tests/test_config.py -v
    Expected Result: All tests pass, exit code 0
    Evidence: .sisyphus/evidence/task-01-config-test-pass.txt

  Scenario: 配置字段验证失败
    Tool: Bash (pytest)
    Preconditions: 配置类已定义但字段缺失
    Steps:
      1. pytest tests/test_config.py::TestConfigFields -v
      2. Assert missing fields raise AttributeError
    Expected Result: Test catches missing fields
    Evidence: .sisyphus/evidence/task-01-config-field-error.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing all tests pass
  - [ ] Config instance with correct default values

  **Commit**: YES
  - Message: `feat(train): add training config system`
  - Files: train/config.py, tests/test_config.py
  - Pre-commit: pytest tests/test_config.py

- [ ] 2. **Agent池管理系统** - train/agent_pool_manager.py + tests/test_agent_pool_manager.py

  **What to do**:
  - 创建AgentPoolManager类（基于ai/policy_pool.py:PolicyPool扩展）
  - 实现10 agents池管理（add_agent, remove_agent, sample_opponent）
  - 实现ELO系统（update_elo, get_ranking）
  - 实现采样分布（30% latest, 30% best ELO, 40% random）
  - 实现IdentityStats跟踪（每个身份胜率统计）
  - 编写tests/test_agent_pool_manager.py（TDD先写测试）
  
  **Must NOT do**:
  - 不重建全新的池系统（必须复用ai/policy_pool.py:PolicyPool）
  - 不修改ai/policy_pool.py（仅扩展包装）
  - 不实现所有PolicyPool功能（仅扩展训练专用功能）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 包装现有PolicyPool，添加采样逻辑，测试 straightforward
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Tasks 7, 8
  - **Blocked By**: None

  **References**:
  - ai/policy_pool.py:PolicyPool - MUST复用的基础池类
  - ai/policy_pool.py:PolicyPoolConfig - 配置参考
  - ai/policy_pool.py:MatchHistory - ELO计算参考
  - ai/policy_pool.py:IdentityStats - 身份统计参考
  
  **WHY Each Reference Matters**:
  - ai/policy_pool.py:PolicyPool - 核心池类，提供add_agent, get_agent, sample_opponent基础
  - ai/policy_pool.py:MatchHistory - ELO更新算法（已有实现）
  - ai/policy_pool.py:IdentityStats - 身份胜率统计结构

  **Acceptance Criteria** (TDD):
  - [ ] Test file created: tests/test_agent_pool_manager.py
  - [ ] pytest tests/test_agent_pool_manager.py → PASS (池管理、采样分布、ELO更新)

  **QA Scenarios**:
  ```
  Scenario: Agent池初始化并添加agent
    Tool: Bash (pytest)
    Preconditions: AgentPoolManager类已定义
    Steps:
      1. pytest tests/test_agent_pool_manager.py::TestPoolInit -v
      2. Assert pool.size == 10 after init
      3. Assert pool.add_agent('model.zip', elo=1000) succeeds
    Expected Result: All tests pass, pool initialized with 10 slots
    Evidence: .sisyphus/evidence/task-02-pool-init.txt

  Scenario: 采样分布正确（30/30/40）
    Tool: Bash (pytest)
    Preconditions: Agent池有10 agents
    Steps:
      1. pytest tests/test_agent_pool_manager.py::TestSamplingDistribution -v
      2. Run 1000 samples, count distribution
      3. Assert latest ~30%, best_elo ~30%, random ~40%
    Expected Result: Distribution matches 30/30/40 within tolerance
    Evidence: .sisyphus/evidence/task-02-sampling-dist.txt

  Scenario: ELO系统更新正确
    Tool: Bash (pytest)
    Preconditions: Agent池有多个agents
    Steps:
      1. pytest tests/test_agent_pool_manager.py::TestELOUpdate -v
      2. Simulate win/loss, verify elo change
      3. Assert winner elo increases, loser decreases
    Expected Result: ELO updates correctly per MatchHistory algorithm
    Evidence: .sisyphus/evidence/task-02-elo-update.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing pool management tests pass
  - [ ] Sampling distribution histogram (30/30/40)
  - [ ] ELO update before/after values

  **Commit**: YES
  - Message: `feat(train): add agent pool manager`
  - Files: train/agent_pool_manager.py, tests/test_agent_pool_manager.py
  - Pre-commit: pytest tests/test_agent_pool_manager.py

- [ ] 3. **自我对弈环境包装器** - train/self_play_wrapper.py + tests/test_self_play_wrapper.py

  **What to do**:
  - 创建SelfPlayWrapper类（继承ai/gym_wrapper.py:SGSEnv）
  - 实现opponent sampling（从AgentPoolManager采样对手）
  - 实现multi-player控制（controlled_player_idx轮询）
  - 实现opponent observation injection（对手的RLAI推理）
  - 保持checkpoint兼容（observation/action空间不变）
  - 编写tests/test_self_play_wrapper.py（TDD先写测试）
  
  **Must NOT do**:
  - 不修改ai/gym_wrapper.py:SGSEnv（仅继承扩展）
  - 不改变observation_space结构（必须保持兼容）
  - 不改变action_space定义（Discrete(20)不变）
  - 不改变分层动作解码协议（Step 0-3不变）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 继承SGSEnv，添加opponent sampling逻辑
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Tasks 8
  - **Blocked By**: None

  **References**:
  - ai/gym_wrapper.py:SGSEnv - MUST继承的环境基类
  - ai/gym_wrapper.py:SGSEnv.reset() - reset接口参考
  - ai/gym_wrapper.py:SGSEnv.step() - step接口参考
  - ai/rl_ai.py:RLAI - 对手推理接口（select_action）
  - ai/multi_agent_env.py:SelfPlayEnv - 现有自我对弈模式参考
  
  **WHY Each Reference Matters**:
  - ai/gym_wrapper.py:SGSEnv - 核心环境类，提供observation/action空间、reset/step、skill decision
  - ai/rl_ai.py:RLAI - 对手推理接口，用于opponent observation injection
  - ai/multi_agent_env.py:SelfPlayEnv - 现有多玩家控制模式（controlled_player_idx）

  **Acceptance Criteria** (TDD):
  - [ ] Test file created: tests/test_self_play_wrapper.py
  - [ ] pytest tests/test_self_play_wrapper.py → PASS (环境包装、对手采样、多玩家控制)

  **QA Scenarios**:
  ```
  Scenario: SelfPlayWrapper继承SGSEnv
    Tool: Bash (pytest)
    Preconditions: SelfPlayWrapper类已定义
    Steps:
      1. pytest tests/test_self_play_wrapper.py::TestInheritance -v
      2. Assert isinstance(wrapper, SGSEnv)
      3. Assert wrapper.observation_space == SGSEnv.observation_space
    Expected Result: Wrapper correctly inherits SGSEnv
    Evidence: .sisyphus/evidence/task-03-wrapper-inherit.txt

  Scenario: Opponent sampling from pool
    Tool: Bash (pytest)
    Preconditions: Agent池已初始化，wrapper已创建
    Steps:
      1. pytest tests/test_self_play_wrapper.py::TestOpponentSampling -v
      2. Reset wrapper, check info['opponent_version']
      3. Assert opponent sampled from pool
    Expected Result: Opponent version in info, sampled correctly
    Evidence: .sisyphus/evidence/task-03-opponent-sample.txt

  Scenario: Multi-player control轮询
    Tool: Bash (pytest)
    Preconditions: Wrapper已reset
    Steps:
      1. pytest tests/test_self_play_wrapper.py::TestMultiPlayerControl -v
      2. Step through game, track controlled_player_idx
      3. Assert player idx cycles correctly
    Expected Result: Player idx轮询正确
    Evidence: .sisyphus/evidence/task-03-multi-player.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing wrapper tests pass
  - [ ] Opponent version from reset info
  - [ ] Player idx cycle trace

  **Commit**: YES
  - Message: `feat(train): add self-play wrapper`
  - Files: train/self_play_wrapper.py, tests/test_self_play_wrapper.py
  - Pre-commit: pytest tests/test_self_play_wrapper.py

- [ ] 4. **训练状态恢复机制** - train/recovery.py + tests/test_recovery.py

  **What to do**:
  - 创建TrainingState类（model, optimizer, vecnormalize, step_count）
  - 实现save_training_state()（完整训练状态保存）
  - 实现load_training_state()（从checkpoint恢复）
  - 实现resume_training()（继续训练脚本）
  - 编写tests/test_recovery.py（TDD先写测试）
  
  **Must NOT do**:
  - 不改变checkpoint格式（必须MaskablePPO/PPO .zip兼容）
  - 不修改ai/rl_ai.py:RLAI.load()（使用现有加载接口）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 状态保存/加载逻辑简单，测试 straightforward
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Tasks 8
  - **Blocked By**: None

  **References**:
  - ai/rl_ai.py:RLAI.load() - checkpoint加载参考
  - stable_baselines3:PPO.save() - 模型保存接口
  - stable_baselines3:VecNormalize.save/load - normalization stats
  
  **WHY Each Reference Matters**:
  - ai/rl_ai.py:RLAI.load() - 了解现有checkpoint加载方式（MaskablePPO/PPO）
  - stable_baselines3:PPO.save() - SB3保存接口（model.save(path) → .zip）
  - stable_baselines3:VecNormalize - normalization stats需单独保存

  **Acceptance Criteria** (TDD):
  - [ ] Test file created: tests/test_recovery.py
  - [ ] pytest tests/test_recovery.py → PASS (保存、加载、恢复)

  **QA Scenarios**:
  ```
  Scenario: Training state保存并加载
    Tool: Bash (pytest)
    Preconditions: TrainingState类已定义
    Steps:
      1. pytest tests/test_recovery.py::TestSaveLoad -v
      2. Create state, save to .zip, load back
      3. Assert model parameters match
    Expected Result: State saves and loads correctly
    Evidence: .sisyphus/evidence/task-04-save-load.txt

  Scenario: Resume from checkpoint继续训练
    Tool: Bash (pytest)
    Preconditions: checkpoint已保存
    Steps:
      1. pytest tests/test_recovery.py::TestResume -v
      2. Load state, verify step_count continues
      3. Assert not starting from step 0
    Expected Result: Training resumes from correct step
    Evidence: .sisyphus/evidence/task-04-resume.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing recovery tests pass
  - [ ] Step count before/after resume

  **Commit**: YES
  - Message: `feat(train): add training recovery`
  - Files: train/recovery.py, tests/test_recovery.py
  - Pre-commit: pytest tests/test_recovery.py

- [ ] 5. **TransformerPolicy技能决策扩展** - train/policy_extension.py + tests/test_policy_extension.py

  **What to do**:
  - 创建ExtendedTransformerPolicy类（继承ai/models/transformer_policy.py:TransformerPolicy）
  - 扩展forward()方法支持current_step=3 routing（技能决策模式）
  - 添加skill_decision_head（可选：分离头或共享actor头）
  - 保持checkpoint兼容（不改变模型架构，仅添加routing逻辑）
  - 编写tests/test_policy_extension.py（TDD先写测试）
  
  **Must NOT do**:
  - 不修改ai/models/transformer_policy.py:TransformerPolicy（仅继承扩展）
  - 不改变输入维度（~3000维不变）
  - 不改变输出架构（actor/critic头不变，仅routing）
  - 不创建多头架构（Metis推荐单策略routing）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需深入理解TransformerPolicy架构，扩展forward()逻辑，测试复杂
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (依赖Task 1配置)
  - **Blocks**: Tasks 6, 8
  - **Blocked By**: Task 1

  **References**:
  - ai/models/transformer_policy.py:TransformerPolicy - MUST继承的策略网络
  - ai/models/transformer_policy.py:TransformerPolicy.forward() - forward方法参考
  - ai/models/transformer_policy.py:TransformerFeaturesExtractor - 特征提取器
  - ai/gym_wrapper.py:current_step - observation字段（0-3）
  - ai/skill_decision.py:SkillDecisionType - 7种决策类型
  
  **WHY Each Reference Matters**:
  - ai/models/transformer_policy.py:TransformerPolicy - 核心策略网络，d_model=256, nhead=8, num_layers=4
  - ai/models/transformer_policy.py:forward() - 需扩展以支持current_step routing
  - ai/gym_wrapper.py:current_step - Step 3时需特殊处理（skill decision mask）
  - ai/skill_decision.py:SkillDecisionType - 了解决策类型（YES_NO, SELECT_ORDER等）

  **Acceptance Criteria** (TDD):
  - [ ] Test file created: tests/test_policy_extension.py
  - [ ] pytest tests/test_policy_extension.py → PASS (forward routing, skill decision handling)

  **QA Scenarios**:
  ```
  Scenario: ExtendedTransformerPolicy forward routing
    Tool: Bash (pytest)
    Preconditions: ExtendedTransformerPolicy已定义
    Steps:
      1. pytest tests/test_policy_extension.py::TestForwardRouting -v
      2. Test current_step=0,1,2,3的不同routing
      3. Assert skill_decision routing when step=3
    Expected Result: Forward routing correctly handles all steps
    Evidence: .sisyphus/evidence/task-05-forward-routing.txt

  Scenario: Skill decision mask应用
    Tool: Bash (pytest)
    Preconditions: Policy已扩展
    Steps:
      1. pytest tests/test_policy_extension.py::TestSkillDecisionMask -v
      2. Create obs with current_step=3, skill_decision_mask
      3. Assert mask correctly applied to action selection
    Expected Result: Mask restricts skill decision options
    Evidence: .sisyphus/evidence/task-05-skill-mask.txt

  Scenario: Checkpoint兼容性（ExtendedTransformerPolicy加载）
    Tool: Bash (python)
    Preconditions: checkpoint已保存
    Steps:
      1. python -c "from train.policy_extension import ExtendedTransformerPolicy; policy = ExtendedTransformerPolicy.load('checkpoint.zip'); print('OK')"
    Expected Result: Policy loads successfully, "OK" printed
    Evidence: .sisyphus/evidence/task-05-checkpoint-load.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing policy extension tests pass
  - [ ] Forward routing log (step 0-3)
  - [ ] Checkpoint load success message

  **Commit**: YES
  - Message: `feat(train): extend policy for skill decision`
  - Files: train/policy_extension.py, tests/test_policy_extension.py
  - Pre-commit: pytest tests/test_policy_extension.py

- [ ] 6. **技能决策掩码验证** - tests/test_skill_decision_masking.py

  **What to do**:
  - 验证ai/gym_wrapper.py:SGSEnv的skill decision Step 3协议
  - 验证observation包含skill_decision_type和skill_decision_mask
  - 验证mask正确反映valid skill decision options
  - 验证环境返回正常游戏play after skill decision completes
  - TDD：先写测试验证现有协议
  
  **Must NOT do**:
  - 不修改ai/gym_wrapper.py（仅验证现有协议）
  - 不修改ai/skill_decision.py（仅验证接口）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 验证测试，无实现，仅assertion
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (依赖Task 5)
  - **Blocks**: Tasks 8
  - **Blocked By**: Task 5

  **References**:
  - ai/gym_wrapper.py:SGSEnv.step() - step返回observation
  - ai/gym_wrapper.py:current_step - observation字段
  - ai/skill_decision.py:SkillDecisionContext - pending decision检测
  - ai/skill_decision.py:SkillDecisionRequest - request结构
  
  **WHY Each Reference Matters**:
  - ai/gym_wrapper.py:SGSEnv.step() - 需验证返回的observation包含skill_decision字段
  - ai/skill_decision.py:SkillDecisionContext - pending decision检测逻辑（has_pending_decision()）
  - ai/skill_decision.py:SkillDecisionRequest - request结构（decision_type, mask等）

  **Acceptance Criteria** (TDD):
  - [ ] Test file created: tests/test_skill_decision_masking.py
  - [ ] pytest tests/test_skill_decision_masking.py → PASS (Step 3协议、mask正确性)

  **QA Scenarios**:
  ```
  Scenario: Step 3 observation包含skill_decision字段
    Tool: Bash (pytest)
    Preconditions: ai/gym_wrapper.py已存在
    Steps:
      1. pytest tests/test_skill_decision_masking.py::TestStep3Observation -v
      2. Trigger skill decision (e.g., 观星发动)
      3. Assert obs['current_step'] == 3
      4. Assert 'skill_decision_type' in obs
      5. Assert 'skill_decision_mask' in obs
    Expected Result: Step 3 observation has required fields
    Evidence: .sisyphus/evidence/task-06-step3-obs.txt

  Scenario: Skill decision mask反映valid options
    Tool: Bash (pytest)
    Preconditions: Skill decision pending
    Steps:
      1. pytest tests/test_skill_decision_masking.py::TestMaskValidity -v
      2. Check mask values (0/1)
      3. Assert mask matches valid skill decision options
    Expected Result: Mask correctly indicates valid options
    Evidence: .sisyphus/evidence/task-06-mask-valid.txt

  Scenario: Environment returns to normal after skill decision
    Tool: Bash (pytest)
    Preconditions: Skill decision completed
    Steps:
      1. pytest tests/test_skill_decision_masking.py::TestReturnToNormal -v
      2. Complete skill decision (provide RL selection)
      3. Assert current_step != 3
      4. Assert no pending_decision
    Expected Result: Environment exits Step 3 mode
    Evidence: .sisyphus/evidence/task-06-return-normal.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing skill decision masking tests pass
  - [ ] Step 3 observation structure
  - [ ] Mask validity log

  **Commit**: YES
  - Message: `test(train): validate skill decision masking`
  - Files: tests/test_skill_decision_masking.py
  - Pre-commit: pytest tests/test_skill_decision_masking.py

- [ ] 7. **评估脚本** - train/evaluate.py + tests/test_evaluate.py

  **What to do**:
  - 创建evaluate_model()函数（运行100 games vs pool）
  - 实现胜率统计（total_win_rate, identity_win_rates）
  - 实现ELO统计（current elo, elo_change）
  - 实现JSON报告生成（保存到train/logs/{run}/evaluation.json）
  - 编写tests/test_evaluate.py（TDD先写测试）
  
  **Must NOT do**:
  - 不修改ai/reward.py（仅使用现有RewardSystem）
  - 不实现复杂tournament（仅100 games vs pool）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 评估逻辑中等复杂度，需统计计算、JSON生成
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (依赖Tasks 1, 2)
  - **Blocks**: Tasks 10
  - **Blocked By**: Tasks 1, 2

  **References**:
  - ai/policy_pool.py:PolicyPool - opponent采样
  - ai/rl_ai.py:RLAI - model推理
  - ai/reward.py:RewardSystem - 奖励系统（可选）
  - ai/policy_pool.py:IdentityStats - 身份统计
  
  **WHY Each Reference Matters**:
  - ai/policy_pool.py:PolicyPool - 评估时从池采样对手
  - ai/rl_ai.py:RLAI - 加载待评估模型
  - ai/policy_pool.py:IdentityStats - 统计每个身份的胜率

  **Acceptance Criteria** (TDD):
  - [ ] Test file created: tests/test_evaluate.py
  - [ ] pytest tests/test_evaluate.py → PASS (评估、统计、JSON生成)

  **QA Scenarios**:
  ```
  Scenario: Evaluate model vs pool (100 games)
    Tool: Bash (pytest)
    Preconditions: Agent池有agents, model已训练
    Steps:
      1. pytest tests/test_evaluate.py::TestEvaluation -v
      2. Run evaluate_model('model.zip', n_games=100)
      3. Assert win_rate in [0, 1]
      4. Assert identity_win_rates for all identities
    Expected Result: Evaluation completes, stats generated
    Evidence: .sisyphus/evidence/task-07-eval-100.txt

  Scenario: JSON report generated
    Tool: Bash (pytest)
    Preconditions: 评估完成
    Steps:
      1. pytest tests/test_evaluate.py::TestJSONReport -v
      2. Check train/logs/{run}/evaluation.json exists
      3. Assert JSON contains win_rate, identity_win_rates, elo
    Expected Result: JSON report with required fields
    Evidence: .sisyphus/evidence/task-07-json-report.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing evaluation tests pass
  - [ ] Win rate statistics (total + identity)
  - [ ] JSON report file

  **Commit**: YES
  - Message: `feat(train): add evaluation script`
  - Files: train/evaluate.py, tests/test_evaluate.py
  - Pre-commit: pytest tests/test_evaluate.py

- [ ] 8. **自我对弈训练循环** - train/train_self_play.py

  **What to do**:
  - 创建主训练脚本train_self_play.py
  - 实现训练循环（MaskablePPO with action_masks）
  - 集成SelfPlayWrapper、AgentPoolManager、ExtendedTransformerPolicy
  - 实现checkpoint保存（每10万步）
  - 实现定期评估（每5万步，调用evaluate.py）
  - 实现训练恢复（从checkpoint继续，使用recovery.py）
  - 实现命令行参数（timesteps, n_envs, pool_size等）
  - 实现TensorBoard logging（胜率、ELO、loss等）
  
  **Must NOT do**:
  - 不修改ai/gym_wrapper.py（使用SelfPlayWrapper包装）
  - 不改变checkpoint格式（MaskablePPO/PPO .zip）
  - 不实现分布式训练（仅单机）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 集成所有组件，训练循环复杂，需处理状态管理、checkpoint、评估调度
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (依赖Tasks 1-7)
  - **Blocks**: Tasks 10
  - **Blocked By**: Tasks 1, 2, 3, 5

  **References**:
  - train/self_play_wrapper.py:SelfPlayWrapper - 环境包装器
  - train/agent_pool_manager.py:AgentPoolManager - Agent池
  - train/policy_extension.py:ExtendedTransformerPolicy - 策略网络
  - train/evaluate.py:evaluate_model() - 评估函数
  - train/recovery.py:save/load_training_state() - 状态恢复
  - sb3_contrib:MaskablePPO - RL算法（action_masks支持）
  - ai/self_play.py:SelfPlayTrainer - 现有训练循环参考
  
  **WHY Each Reference Matters**:
  - train/self_play_wrapper.py:SelfPlayWrapper - 自我对弈环境，提供opponent sampling
  - train/agent_pool_manager.py:AgentPoolManager - Agent池管理，采样对手
  - train/policy_extension.py:ExtendedTransformerPolicy - 策略网络，支持skill decision
  - sb3_contrib:MaskablePPO - PPO算法，需传入action_masks
  - ai/self_play.py:SelfPlayTrainer - 现有训练循环模式（checkpoint, evaluation, logging）

  **Acceptance Criteria** (无单独测试文件，集成在Task 10):
  - [ ] 训练脚本可运行
  - [ ] 支持--test-mode（1000步测试）
  - [ ] 支持--resume（从checkpoint继续）

  **QA Scenarios**:
  ```
  Scenario: Training script runs with test mode
    Tool: Bash (python)
    Preconditions: 所有依赖组件已实现
    Steps:
      1. .venv/bin/python train/train_self_play.py --test-mode --timesteps 1000 --n-envs 1
      2. Wait for training to complete
      3. Assert checkpoint saved at train/logs/test_run/checkpoints/
    Expected Result: Training completes, checkpoint saved
    Evidence: .sisyphus/evidence/task-08-test-mode.txt

  Scenario: Training uses action_masks
    Tool: Bash (python)
    Preconditions: 训练运行中
    Steps:
      1. Check TensorBoard logs for action_mask usage
      2. Verify MaskablePPO receives masks in predict()
    Expected Result: Action masks correctly applied
    Evidence: .sisyphus/evidence/task-08-action-masks.txt

  Scenario: Checkpoint saved at 10K steps
    Tool: Bash (python)
    Preconditions: 训练运行超过10K步
    Steps:
      1. Run training to 10K+ steps
      2. Check train/logs/{run}/checkpoints/model_step_10000.zip exists
    Expected Result: Checkpoint file exists
    Evidence: .sisyphus/evidence/task-08-checkpoint-10k.txt

  Scenario: Evaluation triggered at 5K steps
    Tool: Bash (python)
    Preconditions: 训练运行超过5K步
    Steps:
      1. Run training to 5K+ steps
      2. Check evaluation.json generated
    Expected Result: Evaluation JSON exists
    Evidence: .sisyphus/evidence/task-08-eval-5k.txt
  ```

  **Evidence to Capture**:
  - [ ] Training completion message
  - [ ] Checkpoint file path
  - [ ] TensorBoard log directory
  - [ ] Evaluation JSON path

  **Commit**: YES
  - Message: `feat(train): add self-play training loop`
  - Files: train/train_self_play.py
  - Pre-commit: .venv/bin/python train/train_self_play.py --test-mode --timesteps 1000 --n-envs 1

- [ ] 9. **TensorBoard可视化配置** - train/visualize.py + tests/test_visualize.py

  **What to do**:
  - 创建TensorBoard配置函数setup_tensorboard()
  - 实现训练曲线logging（loss, entropy, learning_rate）
  - 实现胜率曲线logging（每eval_freq更新）
  - 实现ELO曲线logging（agent池ELO变化）
  - 实现identity胜率logging（每个身份的胜率）
  - 编写tests/test_visualize.py（TDD先写测试）
  
  **Must NOT do**:
  - 不实现复杂可视化（仅基础TensorBoard logging）
  - 不实现游戏回放可视化（后续Phase）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: TensorBoard配置简单，logging straightforward
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (依赖Task 1配置)
  - **Blocks**: Tasks 10
  - **Blocked By**: Task 1

  **References**:
  - stable_baselines3:TensorBoardCallback - SB3内置callback
  - ai/self_play.py:SelfPlayTrainer - 现有TensorBoard logging参考
  
  **WHY Each Reference Matters**:
  - stable_baselines3:TensorBoardCallback - SB3自动logging（loss, entropy等）
  - ai/self_play.py:SelfPlayTrainer - 现有胜率、ELO logging模式

  **Acceptance Criteria** (TDD):
  - [ ] Test file created: tests/test_visualize.py
  - [ ] pytest tests/test_visualize.py → PASS (TensorBoard配置、logging)

  **QA Scenarios**:
  ```
  Scenario: TensorBoard logs generated
    Tool: Bash (pytest)
    Preconditions: 训练运行中
    Steps:
      1. pytest tests/test_visualize.py::TestTensorBoardLogging -v
      2. Check train/logs/{run}/TensorBoard/ directory
      3. Assert loss, entropy, lr events exist
    Expected Result: TensorBoard events file exists
    Evidence: .sisyphus/evidence/task-09-tb-logs.txt

  Scenario: Win rate logged at eval_freq
    Tool: Bash (pytest)
    Preconditions: 评估完成
    Steps:
      1. pytest tests/test_visualize.py::TestWinRateLogging -v
      2. Check TensorBoard for win_rate scalar
      3. Assert logged at correct eval_freq
    Expected Result: Win rate scalar logged
    Evidence: .sisyphus/evidence/task-09-win-rate.txt
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing visualize tests pass
  - [ ] TensorBoard directory structure
  - [ ] Scalar names (loss, entropy, win_rate, elo)

  **Commit**: YES
  - Message: `feat(train): add tensorboard visualization`
  - Files: train/visualize.py, tests/test_visualize.py
  - Pre-commit: pytest tests/test_visualize.py

- [ ] 10. **训练集成测试** - 100K步测试

  **What to do**:
  - 运行完整训练流程（100K步，从T8的test-mode扩展）
  - 验证checkpoint兼容性（加载并通过RLAI推理）
  - 验证agent池ELO系统（采样分布正确）
  - 验证技能决策训练（Step 3 mask正确）
  - 验证评估系统（JSON报告生成）
  - 验证TensorBoard日志（曲线正常）
  
  **Must NOT do**:
  - 不修改已实现的组件（仅验证）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 集成测试复杂，需运行完整训练流程并验证多个方面
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (依赖Task 8)
  - **Blocks**: Tasks 11
  - **Blocked By**: Task 8

  **References**:
  - train/train_self_play.py - 主训练脚本
  - ai/rl_ai.py:RLAI - checkpoint加载验证
  - train/agent_pool_manager.py:AgentPoolManager - ELO验证
  
  **WHY Each Reference Matters**:
  - train/train_self_play.py - 集成所有组件的训练脚本
  - ai/rl_ai.py:RLAI - 验证checkpoint可用性（核心acceptance criteria）
  - train/agent_pool_manager.py - 验证pool系统正确工作

  **Acceptance Criteria**:
  - [ ] 训练完成100K步
  - [ ] Checkpoint可通过RLAI加载并推理
  - [ ] Agent池ELO分布正确
  - [ ] 评估JSON报告生成
  - [ ] TensorBoard日志正常

  **QA Scenarios**:
  ```
  Scenario: Full training integration (100K steps)
    Tool: Bash (python)
    Preconditions: 所有组件已实现
    Steps:
      1. .venv/bin/python train/train_self_play.py --timesteps 100000 --n-envs 4
      2. Wait for training to complete (timeout: 2 hours)
      3. Assert final_model.zip exists
    Expected Result: Training completes 100K steps, final checkpoint saved
    Evidence: .sisyphus/evidence/task-10-full-training.txt

  Scenario: Checkpoint loads via RLAI
    Tool: Bash (python)
    Preconditions: 训练完成，checkpoint saved
    Steps:
      1. python -c "from ai.rl_ai import RLAI; ai = RLAI(RLAIConfig(model_path='train/logs/{run}/final_model.zip')); print('OK')"
    Expected Result: RLAI loads checkpoint successfully, "OK" printed
    Evidence: .sisyphus/evidence/task-10-rlai-load.txt

  Scenario: Agent pool ELO distribution correct
    Tool: Bash (python)
    Preconditions: 训练完成，pool updated
    Steps:
      1. python -c "from train.agent_pool_manager import AgentPoolManager; pool = AgentPoolManager.load('train/logs/{run}/pool.pkl'); stats = pool.get_stats(); assert stats['sampling_dist'] matches 30/30/40"
    Expected Result: Pool sampling distribution matches config
    Evidence: .sisyphus/evidence/task-10-pool-dist.txt

  Scenario: Evaluation JSON report generated
    Tool: Bash (python)
    Preconditions: 评估完成
    Steps:
      1. ls train/logs/{run}/evaluation.json
      2. python -c "import json; r = json.load(open('train/logs/{run}/evaluation.json')); assert 'win_rate' in r"
    Expected Result: JSON report exists with win_rate
    Evidence: .sisyphus/evidence/task-10-eval-json.txt
  ```

  **Evidence to Capture**:
  - [ ] Training completion log (100K steps)
  - [ ] RLAI load success message
  - [ ] Pool sampling distribution stats
  - [ ] Evaluation JSON content

  **Commit**: YES
  - Message: `test(train): integration test 100K steps`
  - Files: N/A (验证已有实现)
  - Pre-commit: .venv/bin/python train/train_self_play.py --timesteps 100000 --n-envs 4

- [ ] 11. **GPU内存性能分析** - 验证8-16环境可行性

  **What to do**:
  - Profiling GPU内存使用（4, 8, 12, 16 envs）
  - 测试MaskablePPO内存占用（TransformerPolicy + VecEnv）
  - 确定最优并行环境数量（基于GPU内存）
  - 验证gradient checkpointing需求（如果内存不足）
  - 生成性能报告（GPU内存曲线）
  
  **Must NOT do**:
  - 不修改训练算法（仅profiling）
  - 不假设GPU内存充足（需实测）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: GPU profiling straightforward，测试不同n_envs配置
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (依赖Task 10)
  - **Blocks**: Tasks 12
  - **Blocked By**: Task 10

  **References**:
  - train/train_self_play.py - 训练脚本（n_envs参数）
  - torch.cuda.memory_allocated() - GPU内存监控
  - ai/models/transformer_policy.py - 策略网络（内存占用）
  
  **WHY Each Reference Matters**:
  - train/train_self_play.py - 需测试不同n_envs的内存占用
  - torch.cuda - GPU内存监控API
  - ai/models/transformer_policy.py - TransformerPolicy内存占用分析

  **Acceptance Criteria**:
  - [ ] Profiling 4, 8, 12, 16 envs
  - [ ] 确定最优n_envs（基于GPU内存）
  - [ ] 性能报告生成

  **QA Scenarios**:
  ```
  Scenario: Profile GPU memory with 4 envs
    Tool: Bash (python)
    Preconditions: 训练脚本可运行
    Steps:
      1. .venv/bin/python train/train_self_play.py --test-mode --timesteps 1000 --n-envs 4
      2. Monitor GPU memory (nvidia-smi or torch.cuda)
      3. Record peak memory usage
    Expected Result: Memory usage recorded, training completes
    Evidence: .sisyphus/evidence/task-11-gpu-4envs.txt

  Scenario: Profile GPU memory with 8 envs
    Tool: Bash (python)
    Preconditions: 4-env测试成功
    Steps:
      1. .venv/bin/python train/train_self_play.py --test-mode --timesteps 1000 --n-envs 8
      2. Monitor GPU memory
    Expected Result: Memory usage recorded
    Evidence: .sisyphus/evidence/task-11-gpu-8envs.txt

  Scenario: Determine optimal n_envs
    Tool: Bash (python)
    Preconditions: Profiling完成
    Steps:
      1. Compare memory usage for 4, 8, 12, 16 envs
      2. Identify max envs without OOM
      3. Generate performance report
    Expected Result: Optimal n_envs determined
    Evidence: .sisyphus/evidence/task-11-optimal-envs.txt
  ```

  **Evidence to Capture**:
  - [ ] GPU memory usage for each n_envs
  - [ ] Performance report (optimal n_envs)

  **Commit**: YES
  - Message: `perf(train): GPU memory profiling`
  - Files: train/performance_report.txt
  - Pre-commit: None

- [ ] 12. **训练参数优化** - 超参数调优

  **What to do**:
  - 调优learning_rate（起始值3e-4）
  - 调优batch_size（起始值64）
  - 调优n_steps（起始值2048）
  - 调优gamma（起始值0.99）
  - 测试不同skill decision reward系数
  - 基于Task 11的GPU memory优化batch_size
  - 生成最佳参数配置
  
  **Must NOT do**:
  - 不实现自动超参数搜索（手动调优）
  - 不改变核心算法结构（仅参数调优）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 超参数调优需多次训练实验，分析结果
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (依赖Task 11)
  - **Blocks**: Tasks 13
  - **Blocked By**: Task 11

  **References**:
  - train/train_self_play.py - 训练脚本（参数调优）
  - train/config.py:TrainingConfig - 参数配置
  - ai/reward.py:RewardConfig - skill decision reward系数
  
  **WHY Each Reference Matters**:
  - train/train_self_play.py - 需测试不同参数组合
  - train/config.py - 参数配置类
  - ai/reward.py - 技能决策奖励系数需调优

  **Acceptance Criteria**:
  - [ ] Learning rate调优完成
  - [ ] Batch size优化完成
  - [ ] Skill decision reward系数调优
  - [ ] 最佳参数配置生成

  **QA Scenarios**:
  ```
  Scenario: Tune learning rate
    Tool: Bash (python)
    Preconditions: GPU memory profiling完成
    Steps:
      1. Run training with lr=1e-4, 3e-4, 1e-3
      2. Compare win_rate at 50K steps
      3. Identify best lr
    Expected Result: Best lr determined
    Evidence: .sisyphus/evidence/task-12-lr-tune.txt

  Scenario: Optimize batch_size
    Tool: Bash (python)
    Preconditions: GPU memory known
    Steps:
      1. Test batch_size=32, 64, 128
      2. Monitor GPU memory
      3. Choose batch_size fitting GPU
    Expected Result: Optimal batch_size determined
    Evidence: .sisyphus/evidence/task-12-batch-tune.txt

  Scenario: Tune skill decision reward
    Tool: Bash (python)
    Preconditions: Training running
    Steps:
      1. Test skill reward coeff=0.1, 0.2, 0.5
      2. Evaluate skill decision quality
      3. Choose best coeff
    Expected Result: Best skill reward coeff determined
    Evidence: .sisyphus/evidence/task-12-skill-reward.txt
  ```

  **Evidence to Capture**:
  - [ ] Learning rate comparison results
  - [ ] Batch size comparison results
  - [ ] Skill reward comparison results
  - [ ] Best parameter config

  **Commit**: YES
  - Message: `perf(train): hyperparameter tuning`
  - Files: train/best_config.json
  - Pre-commit: None

- [ ] 13. **2M步大规模训练验证** - 最终训练

  **What to do**:
  - 使用最佳参数配置（Task 12）
  - 运行完整2M步训练（最优n_envs from Task 11）
  - 验证训练收敛（胜率提升趋势）
  - 验证技能决策质量（观星、遗计等）
  - 验证ELO系统稳定（agent池ELO不爆炸）
  - 生成最终训练报告（胜率、ELO、技能质量）
  
  **Must NOT do**:
  - 不改变参数配置（使用Task 12最佳配置）
  - 不修改算法（仅验证大规模训练）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 大规模训练复杂，需长时间运行并分析结果
  - **Skills**: []
  - **Skills Evaluated but Omitted**: 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (依赖Task 12)
  - **Blocks**: Tasks F1-F4
  - **Blocked By**: Task 12

  **References**:
  - train/train_self_play.py - 训练脚本
  - train/best_config.json - 最佳参数
  - train/performance_report.txt - GPU memory配置
  
  **WHY Each Reference Matters**:
  - train/train_self_play.py - 主训练脚本
  - train/best_config.json - 使用最佳参数
  - train/performance_report.txt - 使用最优n_envs

  **Acceptance Criteria**:
  - [ ] 训练完成2M步
  - [ ] 胜率有提升趋势（vs初始agent）
  - [ ] ELO系统稳定
  - [ ] 技能决策质量提升

  **QA Scenarios**:
  ```
  Scenario: Full 2M steps training
    Tool: Bash (python)
    Preconditions: 最佳参数配置完成
    Steps:
      1. .venv/bin/python train/train_self_play.py --timesteps 2000000 --n-envs {optimal} --config train/best_config.json
      2. Wait for training to complete (timeout: 24 hours)
      3. Assert final_model.zip exists
    Expected Result: Training completes 2M steps
    Evidence: .sisyphus/evidence/task-13-2m-training.txt

  Scenario: Win rate improvement trend
    Tool: Bash (python)
    Preconditions: 训练完成
    Steps:
      1. Load evaluation.json at 0, 500K, 1M, 2M steps
      2. Compare win_rate trend
      3. Assert win_rate increases over training
    Expected Result: Win rate improves over time
    Evidence: .sisyphus/evidence/task-13-win-trend.txt

  Scenario: ELO system stable
    Tool: Bash (python)
    Preconditions: 训练完成，pool updated
    Steps:
      1. Check agent pool ELO distribution
      2. Assert ELO values not diverged absurdly (e.g., all > 5000)
    Expected Result: ELO distribution stable
    Evidence: .sisyphus/evidence/task-13-elo-stable.txt

  Scenario: Skill decision quality improved
    Tool: Bash (python)
    Preconditions: 训练完成
    Steps:
      1. Evaluate skill decision quality (观星 ordering correctness)
      2. Compare initial vs final quality
      3. Assert quality improved
    Expected Result: Skill decision quality higher
    Evidence: .sisyphus/evidence/task-13-skill-quality.txt
  ```

  **Evidence to Capture**:
  - [ ] Training completion log (2M steps)
  - [ ] Win rate curve (TensorBoard)
  - [ ] ELO distribution stats
  - [ ] Skill decision quality report

  **Commit**: YES
  - Message: `perf(train): 2M steps large-scale training`
  - Files: train/final_training_report.json
  - Pre-commit: None

---

## Final Verification Wave (MANDATORY)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest tests/` + linter. Review all changed files for: `as any`/`@ts-ignore`, unused imports, AI slop patterns.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute EVERY QA scenario from EVERY task. Test cross-module integration. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec. Check "Must NOT do" compliance.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **T1**: `feat(train): add training config system` - train/config.py, tests/test_config.py
- **T2**: `feat(train): add agent pool manager` - train/agent_pool_manager.py, tests/test_agent_pool_manager.py
- **T3**: `feat(train): add self-play wrapper` - train/self_play_wrapper.py, tests/test_self_play_wrapper.py
- **T4**: `feat(train): add training recovery` - train/recovery.py, tests/test_recovery.py
- **T5**: `feat(train): extend policy for skill decision` - train/policy_extension.py, tests/test_policy_extension.py
- **T6**: `test(train): validate skill decision masking` - tests/test_skill_decision_masking.py
- **T7**: `feat(train): add evaluation script` - train/evaluate.py, tests/test_evaluate.py
- **T8**: `feat(train): add self-play training loop` - train/train_self_play.py
- **T9**: `feat(train): add tensorboard visualization` - train/visualize.py, tests/test_visualize.py
- **T10**: `test(train): integration test 100K steps` - verify checkpoint and pool
- **T11-T13**: Optimization commits

---

## Success Criteria

### Verification Commands
```bash
# Checkpoint兼容性
.venv/bin/python -c "from ai.rl_ai import RLAI; ai = RLAI(RLAIConfig(model_path='train/logs/test_run/checkpoints/model_step_100000.zip')); print('OK')"
# Expected: OK

# Agent池工作
.venv/bin/python -c "from train.agent_pool_manager import AgentPoolManager; pool = AgentPoolManager(size=10); pool.add_agent('model.zip', elo=1000); stats = pool.get_stats(); assert stats['total_agents'] == 1; print('OK')"
# Expected: OK

# 技能决策掩码正确
pytest tests/test_skill_decision_masking.py -v
# Expected: All tests pass

# 训练脚本运行
.venv/bin/python train/train_self_play.py --test-mode --timesteps 1000 --n-envs 1
# Expected: Training completes, checkpoint saved

# 评估脚本运行
.venv/bin/python train/evaluate.py --model-path train/logs/test_run/final_model.zip --n-games 100
# Expected: JSON report with win_rate, identity_win_rates, elo
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All pytest tests pass
- [ ] Checkpoint兼容RLAI
- [ ] Agent池ELO系统工作
- [ ] 技能决策Step 3协议正确
- [ ] 评估生成JSON报告
- [ ] TensorBoard日志正常