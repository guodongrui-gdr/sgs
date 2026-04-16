# World Model + MARL Integration Plan

## TL;DR

> **Quick Summary**: Integrate World Model (DreamerV3-style RSSM) and MAPPO (Multi-Agent PPO) into the existing SGS RL training system to improve sample efficiency (10-50x) and team coordination (忠臣协作, 反贼集火).
> 
> **Deliverables**: 
> - `ai/world_model/rssm.py` - RSSM architecture (encoder, dynamics, reward)
> - `ai/world_model/imagination_env.py` - Imagination-based training environment
> - `ai/mappo/centralized_critic.py` - Centralized value function
> - `ai/mappo/mappo_policy.py` - MAPPO policy with decentralized actors
> - `train/train_mappo_world_model.py` - Integrated training script
> - `config/world_model_config.yaml` - Hyperparameters
> - `tests/test_world_model.py` - Unit tests
> - `tests/test_mappo.py` - Integration tests
> 
> **Estimated Effort**: Large (6-7 weeks with phased rollout)
> **Parallel Execution**: YES - 4 waves with rollback capability at each phase
> **Critical Path**: RSSM Core → Imagination Integration → MAPPO Centralized Critic → Team Reward → Training Integration → QA

---

## Context

### Original Request
User wants to apply World Model and MARL (MAPPO) to the existing SGS (三国杀) RL training project to:
1. Improve sample efficiency through imagination-based training
2. Enable team coordination (忠臣协作, 反贼集火) via centralized critic
3. Address long-term planning issues (e.g., 内奸策略)

### Interview Summary
**Key Discussions**:
- Current architecture uses IPPO with ~3000-dim state space
- User wants World Model for sample efficiency (10-50x target)
- User wants MAPPO for team coordination
- Existing components: TransformerPolicy, StateEncoder, ActionEncoder, MultiAgentEnv

**Research Findings** (from Metis consultation):
1. **MAPPO may not be optimal**: Recent ICLR 2024 research shows IPPO+global state often outperforms MAPPO on strategic games
2. **World Model complexity**: Requires extensive new infrastructure (~2-3x current RL system)
3. **State encoding**: Current 3000-dim encoding compresses well, can be reused
4. **SB3 limitations**: SB3 explicitly excludes MARL; extension is non-trivial

### Metis Review
**Identified Gaps** (addressed):
- **Critical**: Validation of IPPO vs MAPPO needed before full commitment
- **Critical**: Player elimination (死亡) changes agent count - need death masking
- **Critical**: 内奸's shifting objectives need phase-dependent rewards
- **Minor**: Skill decision interruption (观星, 遗计) needs hierarchical imagination
- **Minor**: Partial observability in imagination needs careful handling

---

## Work Objectives

### Core Objective
Implement World Model (RSSM-based) and MAPPO (centralized critic + decentralized actors) for the SGS RL training system, with validation gates at each phase to ensure rollback capability.

### Concrete Deliverables
- `ai/world_model/` - RSSM components with <10% prediction error
- `ai/mappo/` - Centralized critic and decentralized actors
- `train/train_mappo_world_model.py` - Integrated training pipeline
- `config/world_model_config.yaml` - Hyperparameters for all phases
- Test suite with automated QA scenarios

### Definition of Done
- [ ] Dynamics model achieves <10% prediction error on held-out rollouts
- [ ] Sample efficiency improves 2x+ over baseline IPPO
- [ ] Team coordination metrics show 20%+ improvement (忠臣协作 frequency, 反贼集火 success rate)
- [ ] All phases pass QA with automated evidence capture

### Must Have
- Reuse existing `StateEncoder` and `TransformerFeaturesExtractor` (no state space redesign)
- Maintain backward compatibility with `train/train_self_play.py`
- Add configuration flag for training mode (IPPO vs MAPPO)
- Death masking for eliminated players (维持tensor shapes)
- Asymmetric critic (critic sees true identities during training)

### Must NOT Have (Guardrails)
- **NO** modification of core game engine (`engine/`, `card/`, `skills/`)
- **NO** modification of existing reward calculation (`ai/reward.py`)
- **NO** full DreamerV3 implementation (start with RSSM dynamics only)
- **NO** commitment to MAPPO until Phase 0 validation shows 15%+ improvement
- **NO** imagination training until dynamics model achieves <10% error
- **NO** new dependencies without justification (use custom implementation)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (`tests/` directory exists, pytest configured)
- **Automated tests**: TDD for new modules, tests-after for integration
- **Framework**: pytest + custom RL-specific assertions

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/`.

- **Unit Tests**: Use Bash (`pytest`)
- **Training Verification**: Use Bash (run training script with small steps)
- **Model Accuracy**: Use Bash (Python script comparing predictions to ground truth)
- **Integration Tests**: Use Bash (full training loop with checkpoints)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Phase 0: Validation - Foundation):
├── Task 1: Implement IPPO+global baseline [quick]
├── Task 2: Implement vanilla MAPPO [unspecified-high]
├── Task 3: A/B test validation framework [quick]
└── Task 4: Config and documentation [quick]

Wave 2 (Phase 1: World Model Core):
├── Task 5: RSSM Encoder (reusing Transformer) [unspecified-high]
├── Task 6: RSSM Dynamics Model (GRU-based) [unspecified-high]
├── Task 7: RSSM Reward Model [quick]
└── Task 8: Standalone dynamics training script [unspecified-high]

Wave 3 (Phase 2: Imagination Integration):
├── Task 9: ImaginedEnvironment wrapper [unspecified-high]
├── Task 10: Mixed real/imagined training [unspecified-high]
├── Task 11: Imagination quality validation [quick]
└── Task 12: Integration with existing SelfPlayTrainer [unspecified-high]

Wave 4 (Phase 3: MAPPO Integration):
├── Task 13: CentralizedCritic with identity awareness [unspecified-high]
├── Task 14: Decentralized MAPPO actors [unspecified-high]
├── Task 15: Team reward allocation (主公+忠臣, 反贼, 内奸独立) [unspecified-high]
├── Task 16: Death masking for eliminated players [quick]
└── Task 17: MAPPO training loop integration [unspecified-high]

Wave FINAL (Phase 4: Polish + Verification):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
```

### Dependency Matrix

| Task | Wave | Dependencies | Blocks |
|------|------|--------------|--------|
| 1-4 | 1 | - | 5-17 |
| 5-8 | 2 | 1-4 | 9-12 |
| 9-12 | 3 | 5-8 | 13-17 |
| 13-17 | 4 | 9-12 | F1-F4 |
| F1-F4 | FINAL | 13-17 | - |

### Agent Dispatch Summary

- **Wave 1**: 4 tasks → `quick` (T1, T3, T4), `unspecified-high` (T2)
- **Wave 2**: 4 tasks → `unspecified-high` (T5-T6, T8), `quick` (T7)
- **Wave 3**: 4 tasks → `unspecified-high` (T9-T10, T12), `quick` (T11)
- **Wave 4**: 5 tasks → `unspecified-high` (T13-T17)
- **Wave FINAL**: 4 tasks → `oracle`, `unspecified-high`, `unspecified-high`, `deep`

---

## TODOs

### Phase 0: Validation (Gate 1)

- [ ] **1. Implement IPPO+global baseline**

  **What to do**:
  - Modify `ai/multi_agent_env.py` to provide global state to all agents
  - Create `train/train_ippo_global.py` as baseline
  - Run 10K steps to establish baseline metrics

  **Must NOT do**:
  - Modify core game engine
  - Change existing IPPO implementation
  - Add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - **Reason**: Requires understanding of both existing code and RL theory

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 5-17 (baseline needed for comparison)
  - **Blocked By**: None

  **References**:
  - `ai/multi_agent_env.py:120-150` - MultiAgentEnv class
  - `train/train_sb3.py` - Existing training script
  - `ai/state_encoder.py` - State encoding logic

  **Acceptance Criteria**:
  - [ ] IPPO+global runs without errors for 10K steps
  - [ ] Baseline win rate vs RuleAI recorded
  - [ ] Baseline sample efficiency metric logged

  **QA Scenarios**:
  ```
  Scenario: IPPO+global training runs successfully
    Tool: Bash (python training script)
    Preconditions: .venv activated, dependencies installed
    Steps:
      1. Run: python train/train_ippo_global.py --steps 10000 --eval-interval 1000
      2. Wait for completion
      3. Check logs/train_ippo_global_*.log exists
    Expected Result: Training completes, win rate logged, no errors
    Evidence: .sisyphus/evidence/task-1-ippo-global-training.log
  ```

  **Commit**: YES
  - Message: `feat(mappo): Add IPPO+global baseline for comparison`
  - Files: `train/train_ippo_global.py`, `ai/multi_agent_env_global.py`

---

- [ ] **2. Implement vanilla MAPPO**

  **What to do**:
  - Create `ai/mappo/` directory with centralized critic
  - Implement `CentralizedCritic` class in `ai/mappo/centralized_critic.py`
  - Implement `MAPPOActor` class in `ai/mappo/mappo_policy.py`
  - Create standalone MAPPO trainer (no World Model yet)

  **Must NOT do**:
  - Integrate with World Model (Phase 3)
  - Add team reward allocation yet (Phase 3)
  - Modify SB3 core classes (use custom implementation)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - **Reason**: Complex multi-agent RL implementation

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 13-17 (MAPPO foundation)
  - **Blocked By**: None (can implement independently)

  **References**:
  - `ai/models/transformer_policy.py:86-219` - TransformerFeaturesExtractor
  - External: MAPPO paper (Yu et al., 2022)
  - `ai/multi_agent_env.py` - Multi-agent environment interface

  **Acceptance Criteria**:
  - [ ] MAPPO trains for 10K steps without divergence
  - [ ] Centralized critic produces valid value estimates
  - [ ] Policy entropy doesn't collapse to zero

  **QA Scenarios**:
  ```
  Scenario: MAPPO basic training works
    Tool: Bash (python training script)
    Steps:
      1. Run: python train/train_mappo.py --steps 10000 --eval-interval 1000
      2. Monitor: Check entropy > 0.01, value std > 0.01
      3. Verify: No NaN in gradients or weights
    Expected Result: Training completes, critic values reasonable, no divergence
    Evidence: .sisyphus/evidence/task-2-mappo-training.log
  ```

  **Commit**: YES
  - Message: `feat(mappo): Implement vanilla MAPPO with centralized critic`
  - Files: `ai/mappo/centralized_critic.py`, `ai/mappo/mappo_policy.py`, `train/train_mappo.py`

---

- [ ] **3. A/B test validation framework**

  **What to do**:
  - Create `tests/validation_ab_test.py` comparing IPPO+global vs MAPPO
  - Define metrics: win rate, coordination frequency, sample efficiency
  - Run comparison on 10K steps each
  - **Decision Gate**: Only proceed if MAPPO shows 15%+ improvement

  **Must NOT do**:
  - Skip this validation
  - Proceed to Phase 1 if MAPPO underperforms

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - **Reason**: Simple comparison script

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T1, T2)
  - **Parallel Group**: Wave 1 (sequential after T1, T2)
  - **Blocks**: Tasks 5-17 (decision gate)
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `tests/` directory structure
  - `ai/reward.py` - Reward calculation for metrics

  **Acceptance Criteria**:
  - [ ] Comparison report generated
  - [ ] Decision documented (proceed/rollback)
  - [ ] If rollback: document why and suggest alternatives

  **QA Scenarios**:
  ```
  Scenario: A/B test completes and produces report
    Tool: Bash (python test script)
    Steps:
      1. Run: python tests/validation_ab_test.py --steps 10000
      2. Check: outputs/validation_report.json exists
      3. Verify: Contains win_rate_ippo, win_rate_mappo, improvement_pct
    Expected Result: Report shows clear comparison, decision criteria met
    Evidence: .sisyphus/evidence/task-3-ab-test-report.json
  ```

  **Commit**: YES
  - Message: `test(validation): Add IPPO vs MAPPO A/B test framework`
  - Files: `tests/validation_ab_test.py`

---

- [ ] **4. Config and documentation**

  **What to do**:
  - Create `config/world_model_config.yaml` with phase-specific hyperparameters
  - Document Phase 0 results in `docs/WORLD_MODEL_MAPPO.md`
  - Create rollback script `scripts/rollback_to_ippo.sh`

  **Must NOT do**:
  - Skip documentation

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1, T2)
  - **Parallel Group**: Wave 1
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `config.py` - Existing config structure
  - `AGENTS.md` - Documentation style

  **Acceptance Criteria**:
  - [ ] Config file parses correctly
  - [ ] Documentation includes Phase 0 decision
  - [ ] Rollback script tested

  **QA Scenarios**:
  ```
  Scenario: Config loads and rollback works
    Tool: Bash
    Steps:
      1. Run: python -c "import yaml; yaml.safe_load(open('config/world_model_config.yaml'))"
      2. Run: ./scripts/rollback_to_ippo.sh
      3. Verify: Training can run with IPPO mode
    Expected Result: Config valid, rollback successful
    Evidence: .sisyphus/evidence/task-4-config-test.log
  ```

  **Commit**: YES
  - Message: `docs(config): Add World Model config and Phase 0 documentation`
  - Files: `config/world_model_config.yaml`, `docs/WORLD_MODEL_MAPPO.md`, `scripts/rollback_to_ippo.sh`

---

### Phase 1: World Model Core

- [ ] **5. RSSM Encoder**

  **What to do**:
  - Create `ai/world_model/rssm.py`
  - Implement `RSSEncoder` class reusing `TransformerFeaturesExtractor`
  - Compress 3000-dim → 128-dim latent space
  - Add stochastic component (VAE-style)

  **Must NOT do**:
  - Redesign state encoding from scratch
  - Skip stochasticity (crucial for world model)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - **Reason**: Requires VAE/RSSM knowledge + integration with existing Transformer

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Phase 0 decision)
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 9-12 (imagination needs encoder)
  - **Blocked By**: Phase 0 decision (must proceed)

  **References**:
  - `ai/models/transformer_policy.py:86-219` - TransformerFeaturesExtractor
  - `ai/state_encoder.py:278` - State encoding
  - External: DreamerV3 paper (Hafner et al., 2023)

  **Acceptance Criteria**:
  - [ ] Encoder compresses 3000-dim → 128-dim
  - [ ] Reconstruction loss < 0.1 on test set
  - [ ] KL divergence stays within reasonable bounds (0.1-2.0)

  **QA Scenarios**:
  ```
  Scenario: Encoder compresses and reconstructs correctly
    Tool: Bash (python test)
    Steps:
      1. Run: python -c "from ai.world_model.rssm import RSSEncoder; e=RSSEncoder(); print(e(torch.randn(1,3066)).shape)"
      2. Verify: Output shape is (1, 128)
      3. Run: python tests/test_rssm_encoder.py
    Expected Result: All tests pass, reconstruction error < 0.1
    Evidence: .sisyphus/evidence/task-5-encoder-test.log
  ```

  **Commit**: YES
  - Message: `feat(world-model): Implement RSSM Encoder with VAE`
  - Files: `ai/world_model/rssm.py`, `tests/test_rssm_encoder.py`

---

- [ ] **6. RSSM Dynamics Model**

  **What to do**:
  - Implement `DynamicsModel` class in `ai/world_model/dynamics.py`
  - Use GRU-based sequence model (2 layers, hidden_dim=256)
  - Predict next latent state: (z_t, a_t) → z_{t+1}
  - Add uncertainty estimation (预测分布的方差)

  **Must NOT do**:
  - Use Transformer instead of GRU (GRU is simpler for Phase 1)
  - Skip uncertainty estimation (needed for exploration)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - **Reason**: Core world model component

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T5)
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 9-12
  - **Blocked By**: Phase 0 decision

  **References**:
  - `ai/world_model/rssm.py` - Encoder (Task 5)
  - `ai/action_encoder.py` - Action space encoding
  - External: RSSM paper (Hafner et al., 2019)

  **Acceptance Criteria**:
  - [ ] Dynamics model predicts next state
  - [ ] Uncertainty output is positive and bounded
  - [ ] Forward pass completes in <1ms on GPU

  **QA Scenarios**:
  ```
  Scenario: Dynamics model predicts correctly
    Tool: Bash (python test)
    Steps:
      1. Run: python tests/test_dynamics_model.py
      2. Verify: Prediction MSE < 0.1 on test transitions
      3. Verify: Uncertainty > 0 and < 10
    Expected Result: Model predicts reasonably, uncertainty calibrated
    Evidence: .sisyphus/evidence/task-6-dynamics-test.log
  ```

  **Commit**: YES
  - Message: `feat(world-model): Implement GRU-based Dynamics Model`
  - Files: `ai/world_model/dynamics.py`, `tests/test_dynamics_model.py`

---

- [ ] **7. RSSM Reward Model**

  **What to do**:
  - Implement `RewardModel` class in `ai/world_model/reward.py`
  - Predict reward: (z_t, a_t) → r_t
  - Support identity-aware rewards (复用 `ai/reward.py` 逻辑)
  - Simple MLP: input_dim=128+64, hidden=[256,256], output=1

  **Must NOT do**:
  - Modify existing `ai/reward.py` logic
  - Skip identity-awareness

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T5, T6)
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 9-12
  - **Blocked By**: Phase 0 decision

  **References**:
  - `ai/reward.py:293-637` - RewardSystem (reuse logic)
  - `ai/world_model/rssm.py` - Latent dimension

  **Acceptance Criteria**:
  - [ ] Reward model outputs scalar reward
  - [ ] Prediction error < 5 on test set
  - [ ] Supports all existing reward types

  **QA Scenarios**:
  ```
  Scenario: Reward model predicts identity-aware rewards
    Tool: Bash (python test)
    Steps:
      1. Run: python tests/test_reward_model.py
      2. Test: 主公杀反贼 → predicted reward ~+15
      3. Test: 主公杀忠臣 → predicted reward ~-30
    Expected Result: Rewards align with identity relationships
    Evidence: .sisyphus/evidence/task-7-reward-test.log
  ```

  **Commit**: YES
  - Message: `feat(world-model): Add identity-aware Reward Model`
  - Files: `ai/world_model/reward.py`, `tests/test_reward_model.py`

---

- [ ] **8. Standalone dynamics training script**

  **What to do**:
  - Create `train/train_dynamics.py` - standalone world model training
  - Collect rollouts from existing IPPO policy
  - Train encoder + dynamics + reward for 50K steps
  - **Gate**: Must achieve <10% prediction error before proceeding

  **Must NOT do**:
  - Integrate with policy training yet (Phase 2)
  - Skip validation gate

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - **Reason**: Complex training loop

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T5-T7)
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 9-12 (imagination needs trained model)
  - **Blocked By**: Tasks 5-7

  **References**:
  - `train/train_sb3.py` - Training script structure
  - `ai/self_play.py` - Rollout collection
  - `ai/world_model/` - All components from T5-T7

  **Acceptance Criteria**:
  - [ ] Training runs for 50K steps
  - [ ] Dynamics prediction error < 10%
  - [ ] Reward prediction error < 5
  - [ ] Model checkpoint saved

  **QA Scenarios**:
  ```
  Scenario: Dynamics training completes with target accuracy
    Tool: Bash (training script)
    Steps:
      1. Run: python train/train_dynamics.py --steps 50000
      2. Monitor: Check logs/dynamics_training_*.log for errors
      3. Verify: Final validation error < 0.1
      4. Check: checkpoints/dynamics_final.pt exists
    Expected Result: Training completes, accuracy target met
    Evidence: .sisyphus/evidence/task-8-dynamics-training.log
  ```

  **Commit**: YES
  - Message: `feat(world-model): Add standalone dynamics training script`
  - Files: `train/train_dynamics.py`

---

### Phase 2: Imagination Integration

- [ ] **9. ImaginedEnvironment wrapper**

  **What to do**:
  - Create `ai/world_model/imagination_env.py`
  - Wrap RSSM components as Gym-like environment
  - Support imagination rollouts: reset → step → step ...
  - Maintain action masking during imagination

  **Must NOT do**:
  - Skip action masking (critical for valid actions)
  - Allow imagination without trained model

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Phase 1 completion)
  - **Parallel Group**: Wave 3
  - **Blocks**: Tasks 10-12
  - **Blocked By**: Task 8 (trained dynamics model)

  **References**:
  - `ai/world_model/rssm.py`, `dynamics.py`, `reward.py`
  - `ai/gym_wrapper.py:89-540` - SGSEnv interface
  - `ai/action_encoder.py` - Action masking

  **Acceptance Criteria**:
  - [ ] Imagination runs 15 steps in <10ms
  - [ ] Returns match real environment structure
  - [ ] Action masks applied correctly

  **QA Scenarios**:
  ```
  Scenario: Imagination environment works correctly
    Tool: Bash (python test)
    Steps:
      1. Run: python tests/test_imagination_env.py
      2. Verify: 15-step imagination completes in <10ms
      3. Verify: Action masking prevents invalid actions
      4. Check: Rollout trajectory has correct structure
    Expected Result: Fast, valid, structure-compliant imagination
    Evidence: .sisyphus/evidence/task-9-imagination-env.log
  ```

  **Commit**: YES
  - Message: `feat(world-model): Implement ImaginedEnvironment wrapper`
  - Files: `ai/world_model/imagination_env.py`, `tests/test_imagination_env.py`

---

- [ ] **10. Mixed real/imagined training**

  **What to do**:
  - Modify `ai/self_play.py` to support imagination
  - Implement 50/50 real/imagined training mix (after 10K warmup)
  - Generate 50 imagined episodes per real episode
  - Use imagined data to train policy

  **Must NOT do**:
  - Start imagination before 10K warmup steps
  - Use >50% imagined data (risk of model exploitation)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T9)
  - **Parallel Group**: Wave 3
  - **Blocks**: Tasks 13-17
  - **Blocked By**: Task 9

  **References**:
  - `ai/self_play.py` - SelfPlayTrainer
  - `ai/world_model/imagination_env.py` - Task 9

  **Acceptance Criteria**:
  - [ ] Training runs with mixed data
  - [ ] Sample efficiency improves 2x+
  - [ ] No divergence from imagined data

  **QA Scenarios**:
  ```
  Scenario: Mixed training improves sample efficiency
    Tool: Bash (training script)
    Steps:
      1. Run: python train/train_mixed.py --steps 50000 --imagination-ratio 0.5
      2. Compare: Win rate at 25K steps vs baseline at 50K steps
      3. Verify: 2x+ sample efficiency improvement
    Expected Result: Faster convergence with imagination
    Evidence: .sisyphus/evidence/task-10-mixed-training.log
  ```

  **Commit**: YES
  - Message: `feat(world-model): Add mixed real/imagined training`
  - Files: `train/train_mixed.py`

---

- [ ] **11. Imagination quality validation**

  **What to do**:
  - Create `tests/test_imagination_quality.py`
  - Compare imagined rollouts vs real rollouts
  - Measure divergence at different horizons (1, 5, 10, 15 steps)
  - **Gate**: <20% divergence at 15-step horizon

  **Must NOT do**:
  - Skip validation
  - Proceed if divergence >20%

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T9, T10)
  - **Parallel Group**: Wave 3
  - **Blocks**: Tasks 13-17
  - **Blocked By**: Task 9

  **References**:
  - `tests/test_imagination_env.py`
  - `ai/gym_wrapper.py` - Real environment

  **Acceptance Criteria**:
  - [ ] Validation report generated
  - [ ] 15-step divergence < 20%
  - [ ] Decision documented

  **QA Scenarios**:
  ```
  Scenario: Imagination quality meets target
    Tool: Bash (test script)
    Steps:
      1. Run: python tests/test_imagination_quality.py --horizons 1,5,10,15
      2. Check: Divergence at each horizon
      3. Verify: 15-step divergence < 0.2
    Expected Result: Quality target met
    Evidence: .sisyphus/evidence/task-11-imagination-quality.json
  ```

  **Commit**: YES
  - Message: `test(world-model): Add imagination quality validation`
  - Files: `tests/test_imagination_quality.py`

---

- [ ] **12. Integration with existing SelfPlayTrainer**

  **What to do**:
  - Modify `ai/self_play.py` to optionally use imagination
  - Add config flag: `use_imagination: bool`
  - Maintain backward compatibility (IPPO mode still works)
  - Add imagination monitoring (KL divergence, reconstruction loss)

  **Must NOT do**:
  - Break existing SelfPlayTrainer
  - Force imagination on all users

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T10, T11)
  - **Parallel Group**: Wave 3
  - **Blocks**: Tasks 13-17
  - **Blocked By**: Tasks 10-11

  **References**:
  - `ai/self_play.py` - SelfPlayTrainer class
  - `ai/world_model/imagination_env.py`

  **Acceptance Criteria**:
  - [ ] Integration doesn't break existing tests
  - [ ] Config flag works correctly
  - [ ] Monitoring logs KL and reconstruction loss

  **QA Scenarios**:
  ```
  Scenario: Integration maintains backward compatibility
    Tool: Bash
    Steps:
      1. Run: python -m pytest tests/ -v (verify existing tests pass)
      2. Run: python train/train_self_play.py --use-imagination false
      3. Run: python train/train_self_play.py --use-imagination true
    Expected Result: Both modes work, existing tests pass
    Evidence: .sisyphus/evidence/task-12-integration-test.log
  ```

  **Commit**: YES
  - Message: `feat(world-model): Integrate imagination into SelfPlayTrainer`
  - Files: `ai/self_play.py` (modifications)

---

### Phase 3: MAPPO Integration

- [ ] **13. CentralizedCritic with identity awareness**

  **What to do**:
  - Create `ai/mappo/centralized_critic.py` (if not done in Phase 0)
  - Add identity-awareness: critic sees true identities
  - Support death masking (eliminated players produce zero gradients)
  - Input: global state + all agents' observations + true identities

  **Must NOT do**:
  - Skip death masking
  - Remove identity awareness (critical for coordination)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Phase 2)
  - **Parallel Group**: Wave 4
  - **Blocks**: Tasks 14-17
  - **Blocked By**: Phase 2 completion

  **References**:
  - `ai/mappo/centralized_critic.py` (from Phase 0 if exists, else new)
  - `ai/state_encoder.py` - Global state encoding
  - `ai/reward.py` - Identity relationships

  **Acceptance Criteria**:
  - [ ] Critic outputs valid value estimates
  - [ ] Death masking works (eliminated agents have zero gradients)
  - [ ] Identity information accessible to critic

  **QA Scenarios**:
  ```
  Scenario: Centralized critic with death masking works
    Tool: Bash (python test)
    Steps:
      1. Run: python tests/test_centralized_critic.py
      2. Test: Simulate player death, verify gradients become zero
      3. Test: Verify critic can distinguish identities
    Expected Result: Death masking works, identity-aware values
    Evidence: .sisyphus/evidence/task-13-critic-test.log
  ```

  **Commit**: YES
  - Message: `feat(mappo): Enhance CentralizedCritic with death masking`
  - Files: `ai/mappo/centralized_critic.py`, `tests/test_centralized_critic.py`

---

- [ ] **14. Decentralized MAPPO actors**

  **What to do**:
  - Create `ai/mappo/mappo_policy.py` (if not done in Phase 0)
  - Each actor only sees local observation
  - Support 5 actors (one per player)
  - Action masking through ActionEncoder

  **Must NOT do**:
  - Give actors global state (breaks CTDE paradigm)
  - Skip action masking

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T13)
  - **Parallel Group**: Wave 4
  - **Blocks**: Tasks 15-17
  - **Blocked By**: Phase 2 completion

  **References**:
  - `ai/mappo/mappo_policy.py` (from Phase 0 if exists)
  - `ai/models/transformer_policy.py` - Base policy
  - `ai/action_encoder.py` - Action masking

  **Acceptance Criteria**:
  - [ ] 5 actors run independently
  - [ ] Each actor uses only local observation
  - [ ] Action masking works for all actors

  **QA Scenarios**:
  ```
  Scenario: Decentralized actors work independently
    Tool: Bash (python test)
    Steps:
      1. Run: python tests/test_mappo_actors.py
      2. Verify: Each actor outputs actions for its player
      3. Verify: Actors don't share observations
    Expected Result: Independent decentralized execution
    Evidence: .sisyphus/evidence/task-14-actors-test.log
  ```

  **Commit**: YES
  - Message: `feat(mappo): Implement decentralized MAPPO actors`
  - Files: `ai/mappo/mappo_policy.py`, `tests/test_mappo_actors.py`

---

- [ ] **15. Team reward allocation**

  **What to do**:
  - Implement team reward logic in `ai/mappo/team_rewards.py`
  - Teams: 主公+忠臣 (共享胜利), 反贼 (共享胜利), 内奸 (独立)
  - Use advantage function decomposition (参考 COMA paper)
  - Support dynamic team sizes (players eliminated)

  **Must NOT do**:
  - Assign individual rewards only (breaks coordination)
  - Skip 内奸 special handling

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T13, T14)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 17
  - **Blocked By**: Tasks 13-14

  **References**:
  - `ai/reward.py` - Existing reward system
  - `ai/multi_agent_env.py` - Team configuration
  - External: COMA paper (Foerster et al., 2018)

  **Acceptance Criteria**:
  - [ ] Team rewards computed correctly
  - [ ] 忠臣协作 incentivized
  - [ ] 反贼集火 incentivized
  - [ ] 内奸独立 handled

  **QA Scenarios**:
  ```
  Scenario: Team rewards encourage coordination
    Tool: Bash (python test)
    Steps:
      1. Run: python tests/test_team_rewards.py
      2. Test: 主公胜利 → 主公+忠臣 get team bonus
      3. Test: 反贼胜利 → 反贼 get team bonus
      4. Test: 内奸独立, no team bonus
    Expected Result: Rewards align with team structure
    Evidence: .sisyphus/evidence/task-15-team-rewards.log
  ```

  **Commit**: YES
  - Message: `feat(mappo): Add team reward allocation with advantage decomposition`
  - Files: `ai/mappo/team_rewards.py`, `tests/test_team_rewards.py`

---

- [ ] **16. Death masking for eliminated players**

  **What to do**:
  - Implement death masking in CentralizedCritic
  - When player dies, mask their gradients to zero
  - Maintain tensor shapes (don't change agent count)
  - Handle in both training and inference

  **Must NOT do**:
  - Change agent count dynamically (breaks MAPPO)
  - Skip death masking (gradients from dead agents corrupt training)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T13-T15)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 17
  - **Blocked By**: Phase 2 completion

  **References**:
  - `ai/mappo/centralized_critic.py`
  - `engine/game_engine.py` - Player elimination logic

  **Acceptance Criteria**:
  - [ ] Death masking implemented
  - [ ] Gradients from dead agents are zero
  - [ ] Tensor shapes maintained

  **QA Scenarios**:
  ```
  Scenario: Death masking prevents gradient corruption
    Tool: Bash (python test)
    Steps:
      1. Run: python tests/test_death_masking.py
      2. Simulate: Player 2 dies at step 50
      3. Verify: Gradients for Player 2 are zero after death
      4. Verify: Other players' gradients remain non-zero
    Expected Result: Clean death masking
    Evidence: .sisyphus/evidence/task-16-death-masking.log
  ```

  **Commit**: YES
  - Message: `feat(mappo): Implement death masking for eliminated players`
  - Files: `ai/mappo/death_masking.py` (or integrated into critic)

---

- [ ] **17. MAPPO training loop integration**

  **What to do**:
  - Create `train/train_mappo_world_model.py` - full integration
  - Combine MAPPO with World Model imagination
  - Configurable: use_world_model, use_mappo flags
  - Add coordination metrics logging (忠臣协作, 反贼集火)

  **Must NOT do**:
  - Skip metrics logging
  - Force both features on (allow independent activation)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T13-T16)
  - **Parallel Group**: Wave 4
  - **Blocks**: Wave FINAL
  - **Blocked By**: Tasks 13-16

  **References**:
  - `train/train_mappo.py` (from Phase 0)
  - `train/train_mixed.py` (from Phase 2)
  - `ai/mappo/` - All MAPPO components
  - `ai/world_model/` - All World Model components

  **Acceptance Criteria**:
  - [ ] Full training runs without errors
  - [ ] Coordination metrics logged
  - [ ] 20%+ improvement in team coordination
  - [ ] Sample efficiency improves 2x+

  **QA Scenarios**:
  ```
  Scenario: Full MAPPO+World Model training succeeds
    Tool: Bash (training script)
    Steps:
      1. Run: python train/train_mappo_world_model.py --steps 100000
      2. Monitor: Coordination metrics in logs
      3. Evaluate: 20%+ improvement vs baseline
      4. Verify: Sample efficiency 2x+ vs baseline
    Expected Result: Successful training, target metrics met
    Evidence: .sisyphus/evidence/task-17-full-training.log
  ```

  **Commit**: YES
  - Message: `feat(integration): Add full MAPPO+World Model training`
  - Files: `train/train_mappo_world_model.py`

---

## Final Verification Wave

- [ ] **F1. Plan Compliance Audit**

  **What to do**:
  - Verify all "Must Have" present
  - Verify all "Must NOT Have" absent
  - Check evidence files exist
  - Compare deliverables against plan

  **Agent Profile**: `oracle`
  
  **Output**: `Must Have [10/10] | Must NOT Have [7/7] | Tasks [17/17] | VERDICT: APPROVE`

---

- [ ] **F2. Code Quality Review**

  **What to do**:
  - Run `pytest` on all tests
  - Check for `as any`, empty catches, unused imports
  - Review AI slop patterns
  - Verify type annotations

  **Agent Profile**: `unspecified-high`
  
  **Output**: `Tests [PASS] | Quality [CLEAN] | VERDICT: APPROVE`

---

- [ ] **F3. Real Manual QA**

  **What to do**:
  - Run full training for 10K steps
  - Test rollback script
  - Verify all QA scenarios produce evidence
  - Test edge cases (player elimination, etc.)

  **Agent Profile**: `unspecified-high`
  
  **Output**: `Scenarios [17/17 pass] | Integration [CLEAN] | VERDICT: APPROVE`

---

- [ ] **F4. Scope Fidelity Check**

  **What to do**:
  - Verify each task matches "What to do"
  - Check no scope creep
  - Verify "Must NOT do" compliance
  - Detect cross-task contamination

  **Agent Profile**: `deep`
  
  **Output**: `Tasks [17/17 compliant] | Scope [CLEAN] | VERDICT: APPROVE`

---

## Commit Strategy

| Commit | Message | Files | Pre-commit |
|--------|---------|-------|------------|
| 1 | `feat(mappo): Add IPPO+global baseline for comparison` | `train/train_ippo_global.py`, `ai/multi_agent_env_global.py` | `pytest tests/ -v` |
| 2 | `feat(mappo): Implement vanilla MAPPO with centralized critic` | `ai/mappo/centralized_critic.py`, `ai/mappo/mappo_policy.py`, `train/train_mappo.py` | `pytest tests/test_mappo*.py -v` |
| 3 | `test(validation): Add IPPO vs MAPPO A/B test framework` | `tests/validation_ab_test.py` | `python tests/validation_ab_test.py --dry-run` |
| 4 | `docs(config): Add World Model config and Phase 0 documentation` | `config/world_model_config.yaml`, `docs/WORLD_MODEL_MAPPO.md`, `scripts/rollback_to_ippo.sh` | `yaml.safe_load()` test |
| 5 | `feat(world-model): Implement RSSM Encoder with VAE` | `ai/world_model/rssm.py`, `tests/test_rssm_encoder.py` | `pytest tests/test_rssm_encoder.py -v` |
| 6 | `feat(world-model): Implement GRU-based Dynamics Model` | `ai/world_model/dynamics.py`, `tests/test_dynamics_model.py` | `pytest tests/test_dynamics_model.py -v` |
| 7 | `feat(world-model): Add identity-aware Reward Model` | `ai/world_model/reward.py`, `tests/test_reward_model.py` | `pytest tests/test_reward_model.py -v` |
| 8 | `feat(world-model): Add standalone dynamics training script` | `train/train_dynamics.py` | `python train/train_dynamics.py --dry-run` |
| 9 | `feat(world-model): Implement ImaginedEnvironment wrapper` | `ai/world_model/imagination_env.py`, `tests/test_imagination_env.py` | `pytest tests/test_imagination_env.py -v` |
| 10 | `feat(world-model): Add mixed real/imagined training` | `train/train_mixed.py` | `python train/train_mixed.py --steps 100 --dry-run` |
| 11 | `test(world-model): Add imagination quality validation` | `tests/test_imagination_quality.py` | `python tests/test_imagination_quality.py` |
| 12 | `feat(world-model): Integrate imagination into SelfPlayTrainer` | `ai/self_play.py` | `pytest tests/ -v` |
| 13 | `feat(mappo): Enhance CentralizedCritic with death masking` | `ai/mappo/centralized_critic.py`, `tests/test_centralized_critic.py` | `pytest tests/test_centralized_critic.py -v` |
| 14 | `feat(mappo): Implement decentralized MAPPO actors` | `ai/mappo/mappo_policy.py`, `tests/test_mappo_actors.py` | `pytest tests/test_mappo_actors.py -v` |
| 15 | `feat(mappo): Add team reward allocation with advantage decomposition` | `ai/mappo/team_rewards.py`, `tests/test_team_rewards.py` | `pytest tests/test_team_rewards.py -v` |
| 16 | `feat(mappo): Implement death masking for eliminated players` | `ai/mappo/death_masking.py` | `pytest tests/test_death_masking.py -v` |
| 17 | `feat(integration): Add full MAPPO+World Model training` | `train/train_mappo_world_model.py` | `python train/train_mappo_world_model.py --steps 100 --dry-run` |

---

## Success Criteria

### Verification Commands

```bash
# Test all components
pytest tests/ -v --tb=short

# Run Phase 0 validation
python tests/validation_ab_test.py --steps 10000

# Train dynamics model
python train/train_dynamics.py --steps 50000

# Train with imagination
python train/train_mixed.py --steps 50000 --imagination-ratio 0.5

# Train full MAPPO+World Model
python train/train_mappo_world_model.py --steps 100000

# Rollback test
./scripts/rollback_to_ippo.sh
python train/train_self_play.py --steps 1000
```

### Final Checklist

- [ ] Phase 0: MAPPO validation shows 15%+ improvement (or rollback documented)
- [ ] Phase 1: Dynamics model achieves <10% prediction error
- [ ] Phase 2: Imagination quality <20% divergence at 15-step horizon
- [ ] Phase 3: Team coordination improves 20%+
- [ ] Phase 4: Sample efficiency improves 2x+
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Rollback script tested
- [ ] Evidence files in `.sisyphus/evidence/`
