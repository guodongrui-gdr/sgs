# RL AI Win Rate Optimization Plan (80%+ Target)

## TL;DR

> **Quick Summary**: Enable RL-driven skill decisions and identity inference to achieve 80%+ win rate from 20% baseline. Fix critical MaskablePPO bug, integrate skill decision learning (观星 first), add identity belief state, run 5M-step curriculum training with VecNormalize comparison.
> 
> **Deliverables**:
> - Fixed RL agent with proper action masking (critical bug resolved)
> - Skill decision integration for 观星 (3其他技能 Phase 2后扩展)
> - Identity belief state in observation (~28 new dims)
> - Per-decision skill quality rewards
> - Two training runs comparing VecNormalize settings
> - 80%+ win rate evaluated over 100 episodes
> - Checkpoint evaluation every 50K steps
> 
> **Estimated Effort**: Large (3-4 weeks total)
> **Parallel Execution**: YES - 5 waves, max 6 tasks concurrent
> **Critical Path**: Task 1 → Task 3 → Task 5 → Task 8 → Task 11 → Task 13 → Training

---

## Context

### Original Request
优化目前强化学习AI的胜率至80%+

### Interview Summary
**User Decisions**:
- Current state: No trained models, 20% baseline (random policy)
- Target: 80%+ win rate (more ambitious than project's 50% goal)
- GPU: Available with CUDA support
- Time budget: Flexible, can iterate multiple times
- Evaluation: Frequent checkpointing every 50K steps
- VecNormalize: Test both approaches (norm_reward=True vs False)
- Skill rewards: Per-decision quality rewards

**Previous Attempts** (Failed):
- Hyperparameter tuning
- Reward shaping experiments
- Model architecture changes
- Result: "No learning progress, poor action quality"

### Metis Review Summary
**Root Cause Identified**:
1. 🔴 **CRITICAL**: Skill decisions AUTO-RESOLVED with `valid_options[0]` - RL never learns skill decisions
2. 🔴 **CRITICAL**: `rl_ai.py:141-150` does NOT pass `action_masks` to `model.predict()` - MaskablePPO bug
3. 🔴 Observation missing identity inference (hidden identity game)
4. VecNormalize `norm_reward=False` may cause instability

**Addressed Gaps**:
- LSP errors: Don't refactor (pre-existing, optional import guards)
- Testing strategy: TDD approach with pytest
- Skill decision complexity: Start with 观星 only before expanding
- Identity inference: Multi-task learning during RL training
- Observation extension: Fresh training (backward compatibility not critical)

---

## Work Objectives

### Core Objective
Achieve 80%+ win rate by enabling RL agent to learn complex skill decisions and infer hidden identities, fixing critical implementation bugs that prevented learning progress.

### Concrete Deliverables
1. `ai/rl_ai.py` - Fixed action_masks passing to model.predict()
2. `ai/gym_wrapper.py` - Removed auto-resolution, skill decision observation enabled
3. `ai/state_encoder.py` - Identity belief state added (~28 dims)
4. `ai/reward.py` - Skill decision quality rewards
5. `train/logs/run_A/` - Training with norm_reward=False
6. `train/logs/run_B/` - Training with norm_reward=True
7. Win rate > 80% evaluated over 100 episodes

### Definition of Done
- [ ] All pytest tests pass (skill decision, identity inference, action masks)
- [ ] Quick 500-step training validation completes without errors
- [ ] 观星 skill decision learning validated in 100K training
- [ ] Win rate > 80% (baseline: 20%) in final evaluation
- [ ] VecNormalize comparison documented with recommendation

### Must Have
- RL agent learns skill decisions (观星 at minimum)
- Action masks properly passed to MaskablePPO
- Identity belief state in observation
- 80%+ win rate achieved
- Evaluation with Wilson confidence intervals

### Must NOT Have (Guardrails from Metis)
- Do NOT refactor existing LSP errors (201 errors from optional imports - intentional)
- Do NOT modify existing reward values (only add skill decision rewards)
- Do NOT change curriculum stages (keep Easy/Medium/Hard progression)
- Do NOT add identity inference to all 8 skills simultaneously (start with 观星)
- Do NOT use ground-truth identity for targeting (belief state only)
- Do NOT change VecNormalize `norm_obs=True` (only `norm_reward`)
- Do NOT modify skill execution logic (only remove auto-resolution)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest in tests/)
- **Automated tests**: YES (TDD approach)
- **Framework**: pytest
- **TDD Workflow**: RED (write failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Python modules**: Use pytest - unit tests with exact expected values
- **Training validation**: Use Bash - run quick training, verify completion
- **Win rate evaluation**: Use Bash - run evaluate.py, parse JSON output

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Critical Fixes - foundation, max parallel 6):
├── Task 1: Fix action_masks bug in rl_ai.py [quick]
├── Task 2: Write test for action_masks fix [quick]
├── Task 3: Remove auto-resolution in gym_wrapper.py [quick]
├── Task 4: Write test for skill decision observation [quick]
├── Task 5: Add skill decision observation dimensions [quick]
├── Task 6: Quick 500-step training validation [quick]

Wave 2 (Skill Decision Rewards - 观星 focus):
├── Task 7: Add skill decision quality rewards [quick]
├── Task 8: Write test for skill decision rewards [quick]
├── Task 9: Implement 观星 skill decision mask [unspecified-low]
├── Task 10: Write test for 观星 mask [quick]
├── Task 11: 100K-step training with skill decisions [unspecified-high]

Wave 3 (Identity Inference - belief state):
├── Task 12: Add identity belief state dimensions [quick]
├── Task 13: Write test for identity belief encoding [quick]
├── Task 14: Implement belief update logic [unspecified-low]
├── Task 15: Write test for belief updates [quick]
├── Task 16: Identity-aware action masking [unspecified-low]
├── Task 17: 1M-step training with identity inference [unspecified-high]

Wave 4 (Full Training & VecNormalize Comparison):
├── Task 18: Setup training run A (norm_reward=False) [quick]
├── Task 19: Setup training run B (norm_reward=True) [quick]
├── Task 20: Run 5M training A [unspecified-high]
├── Task 21: Run 5M training B [unspecified-high]
├── Task 22: Compare VecNormalize results [quick]

Wave FINAL (Evaluation & Review):
├── Task F1: Final win rate evaluation (100 episodes) [oracle]
├── Task F2: Code quality review (pytest, lint) [unspecified-high]
├── Task F3: Skill decision learning verification [unspecified-high]
├── Task F4: Scope fidelity check [deep]
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 3 → Task 5 → Task 11 → Task 17 → Task 20-21 → F1
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 6 (Wave 1)
```

### Dependency Matrix

| Task | Blocked By | Blocks | Notes |
|------|------------|--------|-------|
| 1 | None | 2, 3, 5, 6 | Critical bug fix, must complete first |
| 2 | 1 | None | Test for Task 1, can run after |
| 3 | 1 | 4, 5, 11 | Remove auto-resolution, enables skill decisions |
| 4 | 3 | None | Test for Task 3 |
| 5 | 3 | 6, 11 | Add observation dims, breaks compatibility |
| 6 | 1, 3, 5 | 11 | Quick validation, all fixes required |
| 7 | None | 8, 11 | Reward additions, independent of bug fixes |
| 8 | 7 | None | Test for Task 7 |
| 9 | 5 | 10, 11 | 观星 mask implementation |
| 10 | 9 | None | Test for Task 9 |
| 11 | 6, 9 | 17 | 100K training with skill decisions |
| 12 | None | 13, 14, 17 | Identity belief dims, independent |
| 13 | 12 | None | Test for Task 12 |
| 14 | 12 | 15, 16, 17 | Belief update logic |
| 15 | 14 | None | Test for Task 14 |
| 16 | 14 | 17 | Identity-aware masking |
| 17 | 11, 16 | 20, 21 | 1M training with identity inference |
| 18-19 | None | 20, 21 | Config setup, parallel |
| 20-21 | 17 | 22 | Training runs A and B, parallel |
| 22 | 20, 21 | F1-F4 | Compare results |
| F1-F4 | 22 | User okay | Final review wave

### Agent Dispatch Summary

- **Wave 1**: **6** - T1-T4 → `quick`, T5 → `quick`, T6 → `quick`
- **Wave 2**: **5** - T7-T8 → `quick`, T9 → `unspecified-low`, T10 → `quick`, T11 → `unspecified-high`
- **Wave 3**: **6** - T12-T13 → `quick`, T14 → `unspecified-low`, T15 → `quick`, T16 → `unspecified-low`, T17 → `unspecified-high`
- **Wave 4**: **5** - T18-T19 → `quick`, T20-T21 → `unspecified-high` (parallel), T22 → `quick`
- **FINAL**: **4** - F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

- [x] 1. Fix action_masks bug in rl_ai.py

  **What to do**:
  - Modify `ai/rl_ai.py:141-150` to pass `action_masks` parameter to `model.predict()`
  - Current bug: MaskablePPO requires explicit action masks, but `select_action()` calls `model.predict(obs, deterministic=deterministic)` without masks
  - Fix: Change to `model.predict(obs, action_masks=action_masks, deterministic=deterministic)`
  - Generate action_masks using `ActionMaskGenerator` before calling predict
  - This is a CRITICAL bug preventing MaskablePPO from working correctly

  **Must NOT do**:
  - Do NOT refactor surrounding code (minimal change only)
  - Do NOT change VecNormalize settings in this file
  - Do NOT modify other methods in rl_ai.py

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-line fix, straightforward implementation
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**: All (simple bug fix, no domain overlap)

  **Parallelization**:
  - **Can Run In Parallel**: NO (blocks all other tasks)
  - **Parallel Group**: Wave 1 first task
  - **Blocks**: Tasks 2, 3, 5, 6
  - **Blocked By**: None (can start immediately)

  **References**:
  **Pattern References**:
  - `ai/action_encoder.py:ActionMaskGenerator` - Use to generate masks before predict
  - `ai/rl_ai.py:141-150` - Current predict() call location (needs fix)
  
  **API/Type References**:
  - MaskablePPO predict signature: `predict(observation, action_masks=None, deterministic=False)`
  
  **Test References**:
  - Write new test: `tests/test_rl_ai_action_masks.py`
  
  **External References**:
  - sb3-contrib MaskablePPO docs: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

  **Acceptance Criteria**:
  - [ ] Test file created: `tests/test_rl_ai_action_masks.py`
  - [ ] `pytest tests/test_rl_ai_action_masks.py` → PASS
  - [ ] Verify: Mock model.predict() receives action_masks parameter

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Action masks passed to model.predict()
    Tool: pytest
    Preconditions: Mock MaskablePPO model created
    Steps:
      1. Create mock model with predict() method that records parameters
      2. Create RLAI instance with mocked model
      3. Generate observation and action_masks using ActionMaskGenerator
      4. Call rl_ai.select_action(obs, action_masks)
      5. Assert: model.predict was called with action_masks=<non-None>
    Expected Result: action_masks parameter is non-None in predict() call
    Failure Indicators: action_masks is None or missing from call signature
    Evidence: .sisyphus/evidence/task-1-action-masks-passed.log

  Scenario: MaskablePPO prediction without masks fails
    Tool: pytest
    Preconditions: Real MaskablePPO model loaded (or mock that validates masks)
    Steps:
      1. Create observation from game state
      2. Call select_action WITHOUT providing action_masks
      3. Expected: Should still work (internal mask generation fallback)
    Expected Result: Method handles missing masks gracefully or generates internally
    Failure Indicators: AttributeError or TypeError on predict() call
    Evidence: .sisyphus/evidence/task-1-mask-fallback.log
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing action_masks parameter
  - [ ] Code diff showing the fix applied

  **Commit**: YES (Commit 1)
  - Message: `fix(rl_ai): pass action_masks to model.predict()`
  - Files: `ai/rl_ai.py`, `tests/test_rl_ai_action_masks.py`
  - Pre-commit: `pytest tests/test_rl_ai_action_masks.py`

- [x] 2. Write test for action_masks fix

  **What to do**:
  - Create `tests/test_rl_ai_action_masks.py`
  - Test 1: Verify action_masks parameter passed to model.predict()
  - Test 2: Verify RLAI handles missing masks gracefully
  - Test 3: Verify action mask generation before predict call
  - Use unittest.mock to intercept predict() and check parameters

  **Must NOT do**:
  - Do NOT test other rl_ai.py functionality (focus only on action_masks)
  - Do NOT modify rl_ai.py (test file only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard pytest test, straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Task 1 to establish what to test)
  - **Parallel Group**: Wave 1 (after Task 1)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:
  **Test References**:
  - `tests/` - Existing test patterns to follow
  
  **Pattern References**:
  - `ai/action_encoder.py:ActionMaskGenerator` - Understand mask generation
  - `ai/rl_ai.py:select_action()` - Method being tested

  **Acceptance Criteria**:
  - [ ] File created: `tests/test_rl_ai_action_masks.py`
  - [ ] Contains 3 test cases (parameter passed, fallback, generation)
  - [ ] `pytest tests/test_rl_ai_action_masks.py` → PASS

  **QA Scenarios**:
  ```
  Scenario: Test file executes successfully
    Tool: pytest
    Steps:
      1. Run: pytest tests/test_rl_ai_action_masks.py -v
      2. Verify: All 3 tests pass
    Expected Result: 3 passed, 0 failed
    Evidence: .sisyphus/evidence/task-2-tests-pass.log
  ```

  **Commit**: YES (part of Commit 1)
  - Files: `tests/test_rl_ai_action_masks.py`

- [x] 3. Remove auto-resolution in gym_wrapper.py

  **What to do**:
  - Modify `ai/gym_wrapper.py:546-556` to remove auto-resolution loop
  - Current code: `while self.skill_decision_context.has_pending_decision(): auto_action = valid_options[0]; self._handle_skill_decision(int(auto_action))`
  - Fix: Return skill decision observation instead of auto-resolving
  - When skill returns PAUSE signal, environment should:
    1. Set observation['current_step'] = 3 (skill decision phase)
    2. Set observation['skill_decision_type'] based on request.type
    3. Set observation['skill_decision_mask'] based on request.options
    4. Return (observation, 0.0, False, {}) to allow RL to make decision
  - Store pending decision context for next action
  - On next step() call with skill decision action, call `_handle_skill_decision()` with RL's choice

  **Must NOT do**:
  - Do NOT modify skill execution logic (only environment behavior)
  - Do NOT change reward calculation (handled separately)
  - Do NOT remove skill_decision_context infrastructure (keep framework)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Remove loop, return observation - straightforward refactor
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Task 1 action_masks fix)
  - **Parallel Group**: Wave 1 (after Task 1)
  - **Blocks**: Tasks 4, 5, 11
  - **Blocked By**: Task 1

  **References**:
  **Pattern References**:
  - `ai/gym_wrapper.py:546-556` - Current auto-resolution location (remove)
  - `ai/skill_decision.py:SkillDecisionRequest` - Understand request structure
  - `ai/skill_decision.py:SkillDecisionContext` - Pending decision management
  
  **API/Type References**:
  - SkillDecisionRequest.type: YES_NO, SELECT_ORDER, SELECT_PAIR, SELECT_CARDS, SELECT_TARGETS, DISTRIBUTE
  
  **Test References**:
  - Write new test: `tests/test_skill_decision_observation.py`

  **Acceptance Criteria**:
  - [ ] Auto-resolution loop removed (no `valid_options[0]` usage)
  - [ ] Skill decision observation returned when PAUSE signal received
  - [ ] `pytest tests/test_skill_decision_observation.py` → PASS
  - [ ] Verify: RL can make skill decision after observation returned

  **QA Scenarios**:

  ```
  Scenario: Skill decision observation returned (no auto-resolution)
    Tool: pytest
    Preconditions: Mock skill that returns PAUSE signal
    Steps:
      1. Create game state triggering 观星 skill
      2. Call env.step() leading to skill activation
      3. Skill returns PAUSE (requesting RL decision)
      4. Verify: env.step() returns observation with current_step=3
      5. Verify: observation['skill_decision_type'] == SELECT_ORDER
      6. Verify: No call to _handle_skill_decision with valid_options[0]
    Expected Result: Skill decision observation returned, not auto-resolved
    Failure Indicators: current_step != 3, or auto-resolution called
    Evidence: .sisyphus/evidence/task-3-skill-obs-returned.log

  Scenario: RL makes skill decision after observation
    Tool: pytest
    Preconditions: Skill decision observation returned
    Steps:
      1. RL agent receives skill decision observation
      2. Agent selects skill decision action (e.g., ordering choice)
      3. Call env.step(skill_decision_action)
      4. Verify: _handle_skill_decision called with RL's choice
      5. Verify: Skill execution continues
    Expected Result: RL's decision is applied, skill execution resumes
    Failure Indicators: Skill still paused, or auto-resolution called
    Evidence: .sisyphus/evidence/task-3-rl-decision-applied.log

  Scenario: Multiple sequential skill decisions
    Tool: pytest
    Preconditions: Multiple skill decisions pending (e.g., YES_NO then SELECT_ORDER)
    Steps:
      1. Trigger skill with YES_NO decision first
      2. RL makes YES_NO choice
      3. Skill execution continues, triggers SELECT_ORDER
      4. RL makes SELECT_ORDER choice
      5. Verify: Both decisions handled by RL
    Expected Result: All decisions handled sequentially by RL
    Failure Indicators: Auto-resolution on any decision
    Evidence: .sisyphus/evidence/task-3-multi-decision.log
  ```

  **Evidence to Capture**:
  - [ ] pytest output showing skill decision observation
  - [ ] Code diff showing auto-resolution removed

  **Commit**: YES (Commit 2)
  - Message: `refactor(gym_wrapper): remove skill decision auto-resolution`
  - Files: `ai/gym_wrapper.py`, `tests/test_skill_decision_observation.py`
  - Pre-commit: `pytest tests/test_skill_decision_observation.py`

- [x] 4. Write test for skill decision observation

  **What to do**:
  - Create `tests/test_skill_decision_observation.py`
  - Test 1: Skill decision observation returned (no auto-resolution)
  - Test 2: RL decision applied after observation
  - Test 3: Multiple sequential skill decisions handled
  - Test 4: Skill decision mask matches request.options length
  - Use mock skills to trigger PAUSE signals

  **Must NOT do**:
  - Do NOT test specific skill implementations (focus on observation flow)
  - Do NOT modify gym_wrapper.py (test file only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Task 3 to establish behavior)
  - **Parallel Group**: Wave 1 (after Task 3)
  - **Blocks**: None
  - **Blocked By**: Task 3

  **References**:
  **Test References**:
  - `tests/` - Existing patterns

  **Acceptance Criteria**:
  - [ ] File created: `tests/test_skill_decision_observation.py`
  - [ ] Contains 4 test cases
  - [ ] `pytest tests/test_skill_decision_observation.py` → PASS

  **Commit**: YES (part of Commit 2)

- [x] 5. Add skill decision observation dimensions

  **What to do**:
  - Modify `ai/state_encoder.py` to add skill decision observation fields
  - Add to observation dict:
    - `current_step`: 0-3 (0-2 game actions, 3 skill decision)
    - `skill_decision_type`: 0-6 (YES_NO, SELECT_ORDER, SELECT_PAIR, SELECT_CARDS, SELECT_TARGETS, DISTRIBUTE)
    - `skill_decision_request_id`: int (identifier for pending request)
    - `skill_decision_mask`: array matching request.options length
  - Extend observation space in `ai/gym_wrapper.py` to include these fields
  - Update `StateEncoder.encode()` to populate these fields when skill decision pending
  - Fields should be zeros when no skill decision pending

  **Must NOT do**:
  - Do NOT change existing observation dimensions (only add new fields)
  - Do NOT remove current_step encoding (already exists, extend range to 3)
  - Do NOT modify reward system (handled separately)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Add fields to existing encoding, straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Task 3 auto-resolution removal)
  - **Parallel Group**: Wave 1 (after Task 3)
  - **Blocks**: Tasks 6, 11
  - **Blocked By**: Task 3

  **References**:
  **Pattern References**:
  - `ai/state_encoder.py:StateEncoder` - Current encoding location
  - `ai/gym_wrapper.py:observation_space` - Extend observation dict
  
  **API/Type References**:
  - SkillDecisionType enum values: YES_NO=0, SELECT_ORDER=1, SELECT_PAIR=2, SELECT_CARDS=3, SELECT_TARGETS=4, DISTRIBUTE=5

  **Acceptance Criteria**:
  - [ ] Observation space extended with skill decision fields
  - [ ] Fields populated when skill decision pending
  - [ ] Fields are zeros when no skill decision pending
  - [ ] Backward compatibility: existing observations still work (fresh training OK)

  **QA Scenarios**:
  ```
  Scenario: Skill decision fields populated
    Tool: pytest
    Preconditions: Skill decision pending (current_step=3)
    Steps:
      1. Trigger skill decision state
      2. Encode observation with StateEncoder
      3. Verify: current_step == 3
      4. Verify: skill_decision_type matches request.type
      5. Verify: skill_decision_mask.shape matches request.options length
    Expected Result: All skill decision fields populated correctly
    Failure Indicators: current_step != 3 or mask mismatch
    Evidence: .sisyphus/evidence/task-5-fields-populated.log

  Scenario: Fields zero when no skill decision
    Tool: pytest
    Preconditions: No skill decision pending
    Steps:
      1. Encode normal game state (current_step=0-2)
      2. Verify: skill_decision_type == 0
      3. Verify: skill_decision_mask all zeros
    Expected Result: Skill decision fields are zero
    Failure Indicators: Non-zero values in skill decision fields
    Evidence: .sisyphus/evidence/task-5-fields-zero.log
  ```

  **Commit**: YES (part of infrastructure, before training)

- [x] 6. Quick 500-step training validation

  **What to do**:
  - Run quick training test to validate all fixes work together
  - Command: `.venv/bin/python train/train_sb3.py --n-steps 256 --timesteps 500 --n-envs 1`
  - Verify: Training completes without errors
  - Verify: Skill decision observations appear in training logs
  - Verify: Action masks passed correctly (no MaskablePPO errors)
  - This is a sanity check before committing to longer training runs

  **Must NOT do**:
  - Do NOT run longer training yet (just validation)
  - Do NOT modify training config (use defaults)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Quick validation run, straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Tasks 1, 3, 5 complete)
  - **Parallel Group**: Wave 1 (after Tasks 1, 3, 5)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 1, 3, 5

  **References**:
  **Pattern References**:
  - `train/train_sb3.py` - Training script to use
  
  **Acceptance Criteria**:
  - [ ] Training completes 500 steps without errors
  - [ ] TensorBoard logs show skill decision observations (current_step=3 appears)
  - [ ] No MaskablePPO action mask errors

  **QA Scenarios**:
  ```
  Scenario: Training completes without errors
    Tool: Bash
    Steps:
      1. Run: .venv/bin/python train/train_sb3.py --n-steps 256 --timesteps 500 --n-envs 1
      2. Check: Process exits with code 0
      3. Check: No Python exceptions in output
    Expected Result: Training completes successfully
    Failure Indicators: Exception, crash, or non-zero exit code
    Evidence: .sisyphus/evidence/task-6-training-complete.log

  Scenario: Skill decisions logged in TensorBoard
    Tool: Bash
    Preconditions: Training completed
    Steps:
      1. Check TensorBoard logs in train/logs/{run}/
      2. Verify: Skill decision observations logged (current_step=3 appears in rollout stats)
    Expected Result: Skill decision phase appears in training data
    Failure Indicators: Only current_step=0-2 in logs
    Evidence: .sisyphus/evidence/task-6-skill-decisions-logged.log
  ```

  **Commit**: NO (validation only, no code changes)

- [x] 7. Add skill decision quality rewards

  **What to do**:
  - Modify `ai/reward.py` to add skill decision quality rewards
  - Add method: `skill_decision_quality_reward(decision_type, decision_quality)`
  - For 观星 SELECT_ORDER:
    - Reward based on card positioning quality (0.5-2.0)
    - Simple metric: Reward if good cards placed on top (detect via game outcome correlation)
    - Alternative: Fixed reward for making any decision (activation reward 0.5)
  - For YES_NO:
    - Reward 0.5 for activating valuable skills (武圣, 龙胆)
    - Reward -0.5 for declining valuable skills
    - No reward for neutral decisions
  - Integrate into reward calculation in `gym_wrapper.py` when skill decision made

  **Must NOT do**:
  - Do NOT modify existing reward values (only add new rewards)
  - Do NOT change reward clipping (keep clip_reward=50.0)
  - Do NOT over-engineer quality metric (simple approach first)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Add reward method, straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (independent of bug fixes, can start immediately)
  - **Parallel Group**: Wave 2 (with Task 9)
  - **Blocks**: Tasks 8, 11
  - **Blocked By**: None

  **References**:
  **Pattern References**:
  - `ai/reward.py:RewardSystem` - Current reward system location
  - `ai/gym_wrapper.py:reward calculation` - Where to integrate
  
  **API/Type References**:
  - SkillDecisionType: SELECT_ORDER, YES_NO

  **Acceptance Criteria**:
  - [ ] `skill_decision_quality_reward()` method added to RewardSystem
  - [ ] Rewards integrated into gym_wrapper.py step() when skill decision made
  - [ ] Test passes verifying reward calculation

  **QA Scenarios**:
  ```
  Scenario: Skill decision rewards calculated
    Tool: pytest
    Preconditions: Skill decision made by RL
    Steps:
      1. Mock game state with skill decision
      2. Calculate reward with RewardSystem
      3. Verify: skill_decision_quality_reward returns 0.5-2.0 for 观星
      4. Verify: YES_NO reward is 0.5/-0.5 based on skill value
    Expected Result: Skill decision rewards calculated correctly
    Failure Indicators: Reward is 0 or incorrect value
    Evidence: .sisyphus/evidence/task-7-rewards-calculated.log
  ```

  **Commit**: YES (Commit 4)
  - Message: `feat(reward): add skill decision quality rewards`
  - Files: `ai/reward.py`, `tests/test_skill_decision_rewards.py`

- [x] 8. Write test for skill decision rewards

  **What to do**:
  - Create `tests/test_skill_decision_rewards.py`
  - Test 1: SELECT_ORDER reward calculated
  - Test 2: YES_NO reward for valuable skills
  - Test 3: YES_NO reward for declining valuable skills
  - Test 4: Rewards integrated into step()

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Task 7)
  - **Parallel Group**: Wave 2 (after Task 7)
  - **Blocked By**: Task 7

  **Commit**: YES (part of Commit 4)

- [x] 9. Implement 观星 skill decision mask

  **What to do**:
  - Modify `ai/action_encoder.py` to implement 观星 SELECT_ORDER mask
  - 观星: Select ordering of 5 cards from deck
  - Sequential selection approach: N steps for N cards
  - Mask calculation:
    - Step 1: Select position for card 0 (5 options: positions 0-4)
    - Step 2: Select position for card 1 (4 remaining options)
    - ...
    - Mask updates after each selection (removes filled positions)
  - Add method: `_get_guanxing_mask(request, current_step)` in ActionMaskGenerator
  - Handle variable card count (deck may have <5 cards)

  **Must NOT do**:
  - Do NOT use permutation encoding (5! = 120 too large)
  - Do NOT implement all 8 skills (观星 only for Phase 1)
  - Do NOT change existing mask logic (only add new method)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Sequential mask logic, moderate complexity
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 7)
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 10, 11
  - **Blocked By**: Task 5 (observation dims)

  **References**:
  **Pattern References**:
  - `ai/action_encoder.py:ActionMaskGenerator` - Current mask generator
  - `skills/shu.py:观星` - Skill implementation
  
  **API/Type References**:
  - SkillDecisionRequest for SELECT_ORDER
  - 观星 mechanics: 5 cards from deck top, reorder, place back

  **Acceptance Criteria**:
  - [ ] `_get_guanxing_mask()` method added
  - [ ] Mask handles variable card count
  - [ ] Mask updates after each selection
  - [ ] Test passes verifying mask calculation

  **QA Scenarios**:
  ```
  Scenario: 观星 mask for 5 cards
    Tool: pytest
    Preconditions: 观星 skill decision pending, 5 cards in deck
    Steps:
      1. Create SkillDecisionRequest for SELECT_ORDER with 5 cards
      2. Generate mask with _get_guanxing_mask()
      3. Verify: Mask has 5 valid options for step 1
      4. Select position 0
      5. Verify: Mask has 4 valid options for step 2
    Expected Result: Mask correctly reduces options after selection
    Failure Indicators: Mask doesn't update or wrong count
    Evidence: .sisyphus/evidence/task-9-guanxing-mask.log

  Scenario: 观星 mask for <5 cards
    Tool: pytest
    Preconditions: Deck has 3 cards only
    Steps:
      1. Create request with 3 cards
      2. Generate mask
      3. Verify: Mask has 3 options for step 1
    Expected Result: Mask handles variable count correctly
    Evidence: .sisyphus/evidence/task-9-variable-count.log
  ```

  **Commit**: YES (Commit 5)
  - Message: `feat(action_encoder): implement 观星 skill decision mask`
  - Files: `ai/action_encoder.py`, `tests/test_guanxing_mask.py`

- [x] 10. Write test for 观星 mask

  **What to do**:
  - Create `tests/test_guanxing_mask.py`
  - Test 1: Mask for 5 cards
  - Test 2: Mask updates after selection
  - Test 3: Mask for variable card count (<5)
  - Test 4: Mask handles empty positions correctly

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Blocked By**: Task 9

  **Commit**: YES (part of Commit 5)

- [x] 11. 100K-step training with skill decisions

  **What to do**:
  - Run 100K-step training to validate skill decision learning
  - Command: `.venv/bin/python train/train_sb3.py --n-steps 2048 --timesteps 100000 --n-envs 4`
  - Monitor in TensorBoard:
    - Skill decision observations appear (current_step=3)
    - Action distribution for skill decisions
    - Reward progress (should see skill decision rewards)
  - Expected: Agent begins learning 观星 ordering decisions
  - Check: No auto-resolution errors in logs

  **Must NOT do**:
  - Do NOT run 5M steps yet (just skill decision validation)
  - Do NOT add identity inference yet (Phase 3)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Long-running training task, needs monitoring
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Tasks 6, 9 complete)
  - **Parallel Group**: Wave 2 (after Tasks 6, 9)
  - **Blocks**: Task 17
  - **Blocked By**: Tasks 6, 9

  **References**:
  **Pattern References**:
  - `train/train_sb3.py` - Training script
  
  **Acceptance Criteria**:
  - [ ] Training completes 100K steps without errors
  - [ ] Skill decision actions logged in TensorBoard
  - [ ] Reward mean increases over training (evidence of learning)
  - [ ] 观星 decisions are RL-driven (not auto-resolved)

  **QA Scenarios**:
  ```
  Scenario: Skill decision learning visible
    Tool: Bash + TensorBoard
    Preconditions: 100K training completed
    Steps:
      1. Open TensorBoard: .venv/bin/tensorboard --logdir train/logs/{run}
      2. Check rollout/ep_rew_mean increases
      3. Check custom/skill_decision_count metric (if logged)
      4. Verify: current_step=3 appears in observations
    Expected Result: Learning progress visible, skill decisions used
    Failure Indicators: Reward flat, no skill decisions logged
    Evidence: .sisyphus/evidence/task-11-learning-progress.log
  ```

  **Commit**: NO (training run, no code changes)

- [x] 12. Add identity belief state dimensions

  **What to do**:
  - Modify `ai/state_encoder.py` to add identity belief state
  - 三国杀 hidden identity game: 主公 revealed, others hidden (忠臣/反贼/内奸)
  - Add 4 dims × 7 players = 28 new dimensions for belief state
  - For each non-主公 player: encode belief as [P(忠臣), P(反贼), P(内奸), P(unknown)]
  - 主公 belief: [1, 0, 0, 0] (known)
  - Others: uniform [0.33, 0.33, 0.33, 0] initially, updated based on observed actions
  - Add to observation space in gym_wrapper.py

  **Must NOT do**:
  - Do NOT use ground-truth identity (belief state only)
  - Do NOT remove existing identity encoding (extend only)
  - Do NOT modify reward system (handled separately)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Add dimensions to existing encoding
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (independent of skill decision tasks)
  - **Parallel Group**: Wave 3 (can start early)
  - **Blocks**: Tasks 13, 14, 17
  - **Blocked By**: None

  **References**:
  **Pattern References**:
  - `ai/state_encoder.py:StateEncoder._encode_other_players()` - Current encoding
  - `player/player.py:Player.identity` - Identity values (主公/忠臣/反贼/内奸)
  
  **API/Type References**:
  - Identity enum: 主公=0, 忠臣=1, 反贼=2, 内奸=3

  **Acceptance Criteria**:
  - [ ] 28 new dimensions added to observation
  - [ ] Belief state encoding implemented
  - [ ] 主公 belief is [1, 0, 0, 0]
  - [ ] Others start uniform [0.33, 0.33, 0.33, 0]

  **QA Scenarios**:
  ```
  Scenario: Identity belief state encoded
    Tool: pytest
    Preconditions: 5-player game state
    Steps:
      1. Create game state with 主公 revealed
      2. Encode observation with StateEncoder
      3. Verify: 主公 player belief = [1, 0, 0, 0]
      4. Verify: Other players belief = [0.33, 0.33, 0.33, 0]
      5. Verify: Total belief dims = 4 × 4 hidden players = 16 (for 5p game)
    Expected Result: Belief state encoded correctly for all players
    Failure Indicators: Wrong belief values or missing dimensions
    Evidence: .sisyphus/evidence/task-12-belief-encoded.log
  ```

  **Commit**: YES (Commit 3)
  - Message: `feat(state_encoder): add identity belief state dimensions`
  - Files: `ai/state_encoder.py`, `tests/test_identity_belief_encoding.py`

- [x] 13. Write test for identity belief encoding

  **What to do**:
  - Create `tests/test_identity_belief_encoding.py`
  - Test 1: Belief state encoded for all players
  - Test 2: 主公 belief is known
  - Test 3: Others belief starts uniform
  - Test 4: Belief dims correct count (4 × hidden players)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Blocked By**: Task 12

  **Commit**: YES (part of Commit 3)

- [x] 14. Implement belief update logic

  **What to do**:
  - Modify `ai/state_encoder.py` to add belief update logic
  - Update belief based on observed player actions:
    - Player attacks 主公 → increase P(反贼), decrease P(忠臣)
    - Player heals 主公 → increase P(忠臣), decrease P(反贼)
    - Player attacks 反贼 → increase P(忠臣), decrease P(反贼)
    - Player kills teammate → update based on victim identity belief
  - Add method: `update_identity_belief(player_idx, action_type, target_idx)`
  - Use simple Bayesian update: adjust probabilities based on observed behavior
  - Store belief history for temporal reasoning

  **Must NOT do**:
  - Do NOT over-engineer update logic (simple approach first)
  - Do NOT use perfect information (only observed actions)
  - Do NOT update belief for contradictory actions yet (Phase 2 refinement)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Update logic with moderate complexity
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Task 12 dimensions)
  - **Parallel Group**: Wave 3 (after Task 12)
  - **Blocks**: Tasks 15, 16, 17
  - **Blocked By**: Task 12

  **References**:
  **Pattern References**:
  - `ai/state_encoder.py:StateEncoder` - Where to add update logic
  - `engine/response.py:ResponseType` - Action types to observe
  
  **API/Type References**:
  - Attack action, heal action, skill usage

  **Acceptance Criteria**:
  - [ ] `update_identity_belief()` method added
  - [ ] Belief updates on attack/heal actions
  - [ ] Belief shifts towards correct faction based on behavior
  - [ ] Test passes verifying update logic

  **QA Scenarios**:
  ```
  Scenario: Belief updates on attack Lord
    Tool: pytest
    Preconditions: Player 2 attacks 主公 (player 0)
    Steps:
      1. Initial belief for player 2: [0.33, 0.33, 0.33, 0]
      2. Observe: player 2 uses 杀 on 主公
      3. Update belief with update_identity_belief(2, ATTACK, 0)
      4. Verify: P(反贼) increases (e.g., 0.33 → 0.5)
      5. Verify: P(忠臣) decreases (e.g., 0.33 → 0.2)
    Expected Result: Belief shifts towards 反贼 for attacker
    Failure Indicators: Belief unchanged or shifts wrong direction
    Evidence: .sisyphus/evidence/task-14-attack-lord-update.log

  Scenario: Belief updates on heal Lord
    Tool: pytest
    Preconditions: Player 3 heals 主公
    Steps:
      1. Observe: player 3 uses 桃 on 主公
      2. Update belief with update_identity_belief(3, HEAL, 0)
      3. Verify: P(忠臣) increases
      4. Verify: P(反贼) decreases
    Expected Result: Belief shifts towards 忠臣 for healer
    Evidence: .sisyphus/evidence/task-14-heal-lord-update.log

  Scenario: Belief accumulates over multiple actions
    Tool: pytest
    Preconditions: Player makes multiple actions
    Steps:
      1. Player 2 attacks 主公 once (belief shifts to 反贼)
      2. Player 2 heals 忠臣 later (belief shifts back towards 忠臣/内奸)
      3. Verify: Belief reflects combined action history
    Expected Result: Belief accumulates evidence over time
    Evidence: .sisyphus/evidence/task-14-accumulated-belief.log
  ```

  **Commit**: YES (Commit 6)
  - Message: `feat(state_encoder): implement identity belief update logic`
  - Files: `ai/state_encoder.py`, `tests/test_belief_update.py`

- [x] 15. Write test for belief updates

  **What to do**:
  - Create `tests/test_belief_update.py`
  - Test 1: Attack Lord belief update
  - Test 2: Heal Lord belief update
  - Test 3: Attack 反贼 belief update
  - Test 4: Accumulated belief over multiple actions

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Blocked By**: Task 14

  **Commit**: YES (part of Commit 6)

- [x] 16. Identity-aware action masking

  **What to do**:
  - Modify `ai/action_encoder.py` to use identity belief for targeting
  - Current: Action mask considers distance, alive status
  - Add: Consider identity belief for targeting decisions
  - Implementation:
    - Filter targets based on identity belief (prefer allies for beneficial actions, enemies for attacks)
    - Mask should reflect inferred ally/enemy status
    - Agent can still choose non-optimal targets (mask provides guidance, not hard constraint)
  - This enables strategic targeting based on inferred identities

  **Must NOT do**:
  - Do NOT hard-code ally/enemy targets (soft preference only)
  - Do NOT use ground-truth identity (belief only)
  - Do NOT prevent agent from targeting "believed enemies" (agent learns from mistakes)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Extend mask logic with belief integration
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Task 14 belief update)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 17
  - **Blocked By**: Task 14

  **References**:
  **Pattern References**:
  - `ai/action_encoder.py:ActionMaskGenerator._is_valid_target()` - Current validation
  - `ai/state_encoder.py` - Belief state access
  
  **Acceptance Criteria**:
  - [ ] Action mask considers identity belief
  - [ ] Beneficial actions prefer high P(忠臣) targets
  - [ ] Attack actions prefer high P(反贼) targets
  - [ ] Test passes verifying belief-aware masking

  **QA Scenarios**:
  ```
  Scenario: Attack mask prefers believed enemies
    Tool: pytest
    Preconditions: Player 2 belief: P(反贼)=0.7, others uniform
    Steps:
      1. Generate action mask for 杀 card
      2. Verify: Player 2 has higher mask weight or visibility
      3. Agent can still choose other targets (not hard-coded)
    Expected Result: Mask reflects belief preferences
    Failure Indicators: All targets equally weighted
    Evidence: .sisyphus/evidence/task-16-attack-mask.log
  ```

  **Commit**: YES (extend action_encoder)

- [x] 17. 1M-step training with identity inference

  **What to do**:
  - Run 1M-step training with identity inference enabled
  - Command: `.venv/bin/python train/train_sb3.py --n-steps 2048 --timesteps 1000000 --n-envs 4`
  - Monitor in TensorBoard:
    - Belief state changes over game (should see updates)
    - Targeting decisions based on belief
    - Win rate progress (should see improvement from identity-aware targeting)
  - Expected: Agent learns to infer identities and target strategically
  - This validates identity inference before full 5M training

  **Must NOT do**:
  - Do NOT run 5M yet (just identity inference validation)
  - Do NOT run VecNormalize comparison yet (Phase 4)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Long-running training
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Tasks 11, 16)
  - **Parallel Group**: Wave 3
  - **Blocks**: Tasks 20, 21
  - **Blocked By**: Tasks 11, 16

  **Acceptance Criteria**:
  - [ ] Training completes 1M steps
  - [ ] Belief updates logged in TensorBoard
  - [ ] Targeting decisions show identity awareness
  - [ ] Win rate improves over training

  **Commit**: NO (training run)

- [x] 18. Setup training run A (norm_reward=False)

  **What to do**:
  - Create training config for run A: VecNormalize norm_reward=False (current setting)
  - Copy `train/final_training.py` to `train/run_A_final_training.py`
  - Config: norm_reward=False (unchanged)
  - Set checkpoint_freq=50_000 (frequent checkpointing per user request)
  - Set eval_freq=50_000
  - Log directory: `train/logs/run_A/`
  - This preserves current baseline for comparison

  **Must NOT do**:
  - Do NOT change other VecNormalize settings (norm_obs=True stays)
  - Do NOT change hyperparameters (use optimal config from hyperparameter_analysis.md)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Config setup, straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 19)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 20
  - **Blocked By**: Task 17

  **Acceptance Criteria**:
  - [ ] Config file created: `train/run_A_final_training.py`
  - [ ] norm_reward=False confirmed
  - [ ] checkpoint_freq=50_000
  - [ ] Log dir: train/logs/run_A/

  **Commit**: YES (Commit 7 - config setup)

- [x] 19. Setup training run B (norm_reward=True)

  **What to do**:
  - Create training config for run B: VecNormalize norm_reward=True (experimental)
  - Copy `train/final_training.py` to `train/run_B_final_training.py`
  - Config: norm_reward=True (experimental setting)
  - Set checkpoint_freq=50_000
  - Set eval_freq=50_000
  - Log directory: `train/logs/run_B/`
  - This tests whether reward normalization improves training stability

  **Must NOT do**:
  - Do NOT change norm_obs (True)
  - Do NOT change hyperparameters (same as run A)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 18)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 21
  - **Blocked By**: Task 17

  **Acceptance Criteria**:
  - [ ] Config file created: `train/run_B_final_training.py`
  - [ ] norm_reward=True confirmed
  - [ ] checkpoint_freq=50_000
  - [ ] Log dir: train/logs/run_B/

  **Commit**: YES (part of Commit 7)

- [x] 20. Run 5M training A

  **What to do**:
  - Execute full 5M-step training with run A config
  - Command: `.venv/bin/python train/run_A_final_training.py --timesteps 5000000 --n-envs 8`
  - Duration: ~3.3 hours at 1500 steps/sec with GPU
  - Monitor TensorBoard:
    - Win rate progress through curriculum stages
    - Reward mean progression
    - Skill decision learning
    - Identity inference accuracy
  - Expected: 50%+ win rate (per project target), aiming for 80%+
  - Save checkpoints every 50K steps for evaluation

  **Must NOT do**:
  - Do NOT interrupt training (let it complete)
  - Do NOT change config mid-training

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Long-running training, needs monitoring
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 21 - both training runs parallel)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 22
  - **Blocked By**: Task 18

  **Acceptance Criteria**:
  - [ ] Training completes 5M steps
  - [ ] Curriculum stages completed (Easy → Medium → Hard)
  - [ ] Checkpoints saved every 50K steps
  - [ ] Final model saved: train/logs/run_A/final_model.zip

  **QA Scenarios**:
  ```
  Scenario: Training completes curriculum stages
    Tool: Bash + TensorBoard
    Steps:
      1. Monitor TensorBoard during training
      2. Verify: Easy stage completed (win rate 40%)
      3. Verify: Medium stage completed (win rate 45%)
      4. Verify: Hard stage in progress (win rate approaching 50-80%)
    Expected Result: Curriculum progression visible in logs
    Failure Indicators: Stuck in one stage or no progression
    Evidence: .sisyphus/evidence/task-20-curriculum-progress.log
  ```

  **Commit**: NO (training run)

- [x] 21. Run 5M training B

  **What to do**:
  - Execute full 5M-step training with run B config (norm_reward=True)
  - Command: `.venv/bin/python train/run_B_final_training.py --timesteps 5000000 --n-envs 8`
  - Duration: ~3.3 hours
  - Monitor same metrics as run A
  - Compare progress with run A to determine VecNormalize impact
  - Expected: Similar or better win rate if normalization helps

  **Must NOT do**:
  - Do NOT interrupt training

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 20)
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 22
  - **Blocked By**: Task 19

  **Acceptance Criteria**:
  - [ ] Training completes 5M steps
  - [ ] Checkpoints saved every 50K
  - [ ] Final model saved: train/logs/run_B/final_model.zip

  **Commit**: NO (training run)

- [x] 22. Compare VecNormalize results

  **What to do**:
  - Compare run A (norm_reward=False) vs run B (norm_reward=True)
  - Metrics to compare:
    - Final win rate (evaluate.py on both final models)
    - Training stability (reward variance in TensorBoard)
    - Curriculum progression speed
    - Convergence speed (when win rate plateaued)
  - Recommendation: Which setting to use for future training
  - Document findings in `docs/vecnormalize_comparison.md`

  **Must NOT do**:
  - Do NOT bias recommendation (objective comparison)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Compare two results, straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (requires Tasks 20, 21 complete)
  - **Parallel Group**: Wave 4
  - **Blocks**: Tasks F1-F4
  - **Blocked By**: Tasks 20, 21

  **Acceptance Criteria**:
  - [ ] Comparison documented
  - [ ] Recommendation provided
  - [ ] Winner model selected for final evaluation

  **QA Scenarios**:
  ```
  Scenario: VecNormalize comparison documented
    Tool: Bash + Read
    Steps:
      1. Evaluate both models: run_A and run_B
      2. Compare win rates: which is higher?
      3. Compare training stability: reward variance in TensorBoard
      4. Document findings in docs/vecnormalize_comparison.md
    Expected Result: Clear recommendation on norm_reward setting
    Evidence: .sisyphus/evidence/task-22-comparison.md
  ```

  **Commit**: YES (docs/vecnormalize_comparison.md)

---

## Final Verification Wave

### F1. Final Win Rate Evaluation
- Run: `.venv/bin/python train/evaluate.py --model-path train/logs/run_A/final_model.zip --episodes 100`
- Expected: Win rate > 80% with Wilson CI
- Compare: Run A (norm_reward=False) vs Run B (norm_reward=True)
- Output: JSON with identity-specific win rates, recommendation for VecNormalize setting

### F2. Code Quality Review
- Run: `pytest tests/` - all tests pass
- Run: `.venv/bin/python -m pytest tests/test_skill_decision.py tests/test_identity_inference.py tests/test_rl_ai.py`
- Expected: 0 failures
- Check: No new LSP errors introduced (compare baseline 201 errors)

### F3. Skill Decision Learning Verification
- Verify: 观星 skill decisions are RL-driven (not auto-resolved)
- Check: Agent learns to make strategic ordering decisions
- Evidence: TensorBoard logs showing skill decision action distribution

### F4. Scope Fidelity Check
- Verify: All "Must Have" present
- Verify: All "Must NOT Have" absent (no LSP refactoring, no ground-truth identity)
- Check: No scope creep (no new skills added, curriculum unchanged)

---

## Commit Strategy

### Atomic Commits (Per Metis Recommendations)

**Commit 1**: `fix(rl_ai): pass action_masks to model.predict()`
- File: `ai/rl_ai.py`
- Test: `tests/test_rl_ai_action_masks.py`
- Pre-commit: `pytest tests/test_rl_ai_action_masks.py`

**Commit 2**: `refactor(gym_wrapper): remove skill decision auto-resolution`
- File: `ai/gym_wrapper.py`
- Test: `tests/test_skill_decision_observation.py`
- Pre-commit: `pytest tests/test_skill_decision_observation.py`

**Commit 3**: `feat(state_encoder): add identity belief state dimensions`
- File: `ai/state_encoder.py`
- Test: `tests/test_identity_belief_encoding.py`
- Pre-commit: `pytest tests/test_identity_belief_encoding.py`

**Commit 4**: `feat(reward): add skill decision quality rewards`
- File: `ai/reward.py`
- Test: `tests/test_skill_decision_rewards.py`
- Pre-commit: `pytest tests/test_skill_decision_rewards.py`

**Commit 5**: `feat(action_encoder): implement 观星 skill decision mask`
- File: `ai/action_encoder.py`
- Test: `tests/test_guanxing_mask.py`
- Pre-commit: `pytest tests/test_guanxing_mask.py`

**Commit 6**: `feat(state_encoder): implement identity belief update logic`
- File: `ai/state_encoder.py`
- Test: `tests/test_belief_update.py`
- Pre-commit: `pytest tests/test_belief_update.py`

**Commit 7**: `config(training): setup VecNormalize comparison runs`
- File: `train/final_training.py` (create run_A.py and run_B.py variants)
- No tests (config change)

---

## Success Criteria

### Verification Commands
```bash
# Critical bug fix verification
pytest tests/test_rl_ai_action_masks.py
# Expected: PASS (action_masks parameter passed)

pytest tests/test_skill_decision_observation.py
# Expected: PASS (no auto-resolution, skill decision observation returned)

pytest tests/test_identity_belief_encoding.py
# Expected: PASS (28 dims added, belief state encoding works)

# Quick training validation
.venv/bin/python train/train_sb3.py --n-steps 256 --timesteps 500 --n-envs 1
# Expected: Training completes without errors

# Skill decision training
.venv/bin/python train/train_sb3.py --n-steps 2048 --timesteps 100000 --n-envs 4
# Expected: ep_rew_mean increases, skill decisions logged in TensorBoard

# Final win rate evaluation
.venv/bin/python train/evaluate.py --model-path train/logs/run_A/final_model.zip --episodes 100
# Expected: win_rate > 0.80 with Wilson CI

.venv/bin/python train/evaluate.py --model-path train/logs/run_B/final_model.zip --episodes 100
# Expected: win_rate > 0.80, compare with run_A
```

### Final Checklist
- [ ] All "Must Have" present (skill decisions, identity inference, 80% win rate)
- [ ] All "Must NOT Have" absent (no LSP refactoring, no ground-truth identity)
- [ ] All tests pass
- [ ] VecNormalize comparison documented
- [ ] User okay obtained after final review wave