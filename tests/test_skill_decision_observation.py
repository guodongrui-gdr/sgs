"""
Tests for skill decision observation flow

CRITICAL: Skill decisions should NOT be auto-resolved.
RL must receive skill decision observation and make decisions.

Tests verify:
1. Auto-resolution removed from step()
2. Skill decision observation returned when pending
3. RL decision applied correctly
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Skip if gym not available
pytest.importorskip("gymnasium")


class TestSkillDecisionObservation:
    """Test suite for skill decision observation flow"""

    def test_skill_decision_observation_returned_no_auto_resolve(self):
        """
        Test: Skill decision observation returned (no auto-resolution)

        Scenario: Mock skill returns PAUSE signal requesting RL decision
        Expected: env.step() returns observation with current_step=3
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig
        from ai.skill_decision import (
            SkillDecisionContext,
            SkillDecisionRequest,
            SkillDecisionType,
        )

        config = SGSConfig(player_num=5)
        env = SGSEnv(config)

        # Setup game state
        obs, info = env.reset(seed=42)

        # Simulate skill decision pending
        request = SkillDecisionRequest(
            skill_name="观星",
            decision_type=SkillDecisionType.SELECT_ORDER,
            options=[0, 1, 2, 3, 4],
            context={"num_cards": 5},
        )
        env.skill_decision_context.active_request = request

        # Get observation
        obs = env._get_observation()

        # Verify: current_step = 3 (skill decision phase)
        assert obs["current_step"] == 3
        assert obs["skill_decision_type"] == int(SkillDecisionType.SELECT_ORDER)
        # Mask should have valid options
        assert obs["skill_decision_mask"].sum() > 0

    def test_no_auto_resolution_loop(self):
        """
        Test: Auto-resolution loop removed

        Expected: No while loop with valid_options[0] auto-selection
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig
        from ai.skill_decision import SkillDecisionRequest, SkillDecisionType
        import inspect

        # Check source code for auto-resolution pattern
        source = inspect.getsource(SGSEnv.step)

        # Verify auto-resolution pattern is NOT present
        # The old pattern: "while self.skill_decision_context.has_pending_decision(): ... auto_action = valid_options[0]"
        assert "auto_action = valid_options[0]" not in source
        # Should only check pending once, not loop
        assert source.count("has_pending_decision()") <= 2

    def test_rl_decision_applied(self):
        """
        Test: RL decision is applied after observation

        Scenario: RL selects skill decision action
        Expected: _handle_skill_decision called with RL's choice
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig
        from ai.skill_decision import SkillDecisionRequest, SkillDecisionType

        config = SGSConfig(player_num=5)
        env = SGSEnv(config)

        obs, info = env.reset(seed=42)

        # Setup skill decision
        request = SkillDecisionRequest(
            skill_name="观星",
            decision_type=SkillDecisionType.SELECT_ORDER,
            options=[0, 1, 2, 3, 4],
            context={"num_cards": 5},
        )
        env.skill_decision_context.active_request = request

        # RL makes decision (select position 0)
        obs, reward, done, truncated, info = env.step(0)

        # Verify: Decision was handled (not auto-resolved)
        # The request should have recorded the selection
        if env.skill_decision_context.active_request:
            assert 0 in env.skill_decision_context.active_request.selections

    def test_skill_decision_mask_matches_options(self):
        """
        Test: Skill decision mask matches request.options length

        Expected: Mask has valid positions matching available options
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig
        from ai.skill_decision import SkillDecisionRequest, SkillDecisionType

        config = SGSConfig(player_num=5)
        env = SGSEnv(config)

        obs, info = env.reset(seed=42)

        # Setup SELECT_ORDER with 5 options
        request = SkillDecisionRequest(
            skill_name="观星",
            decision_type=SkillDecisionType.SELECT_ORDER,
            options=[0, 1, 2, 3, 4],
            context={"num_cards": 5},
        )
        env.skill_decision_context.active_request = request

        mask = env._get_skill_decision_mask()

        # Verify: Mask has exactly 5 valid options for first selection
        assert mask.sum() == 5.0
        # Positions 0-4 should be valid
        for i in range(5):
            assert mask[i] == 1.0

    def test_yes_no_skill_decision(self):
        """
        Test: YES_NO skill decision handled correctly

        Expected: Mask has 2 options (Yes=1, No=0)
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig
        from ai.skill_decision import SkillDecisionRequest, SkillDecisionType

        config = SGSConfig(player_num=5)
        env = SGSEnv(config)

        obs, info = env.reset(seed=42)

        request = SkillDecisionRequest(
            skill_name="武圣",
            decision_type=SkillDecisionType.YES_NO,
            options=[0, 1],
            context={},
        )
        env.skill_decision_context.active_request = request

        mask = env._get_skill_decision_mask()

        # YES_NO should have 2 options
        assert mask[0] == 1.0  # No
        assert mask[1] == 1.0  # Yes
        assert mask.sum() == 2.0


class TestSkillDecisionMultipleSteps:
    """Test sequential skill decisions"""

    def test_multiple_sequential_decisions(self):
        """
        Test: Multiple sequential skill decisions handled

        Expected: Each decision requires RL input
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig
        from ai.skill_decision import SkillDecisionRequest, SkillDecisionType

        config = SGSConfig(player_num=5)
        env = SGSEnv(config)

        obs, info = env.reset(seed=42)

        # First decision: YES_NO
        request1 = SkillDecisionRequest(
            skill_name="武圣",
            decision_type=SkillDecisionType.YES_NO,
            options=[0, 1],
            context={},
        )
        env.skill_decision_context.active_request = request1

        # RL says Yes (action=1)
        obs1, _, _, _, info1 = env.step(1)

        # Verify decision was applied
        assert info1.get("skill_decision_complete", False)

        # After YES_NO resolved, should return to normal gameplay
        # or next pending decision
        if env.skill_decision_context.active_request:
            # New pending decision exists
            assert obs1["current_step"] == 3
        else:
            # No pending decision, back to normal step
            assert obs1["current_step"] in [0, 1, 2]

    def test_select_order_multiple_selections(self):
        """
        Test: SELECT_ORDER requires multiple selections

        Expected: After first selection, mask updates for remaining options
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig
        from ai.skill_decision import SkillDecisionRequest, SkillDecisionType

        config = SGSConfig(player_num=5)
        env = SGSEnv(config)

        obs, info = env.reset(seed=42)

        request = SkillDecisionRequest(
            skill_name="观星",
            decision_type=SkillDecisionType.SELECT_ORDER,
            options=[0, 1, 2, 3, 4],
            context={"num_cards": 5},
        )
        env.skill_decision_context.active_request = request

        # First selection: position 2
        obs1, _, _, _, _ = env.step(2)

        # Check remaining options
        if env.skill_decision_context.active_request:
            remaining = (
                env.skill_decision_context.active_request.get_remaining_options()
            )
            # Should have 4 remaining (excluding 2)
            assert len(remaining) == 4
            assert 2 not in remaining


class TestSkillDecisionFields:
    """Test skill decision observation fields"""

    def test_skill_decision_fields_zero_when_no_pending(self):
        """
        Test: Fields are zeros when no skill decision pending

        Expected: skill_decision_type=0, mask zeros
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig

        config = SGSConfig(player_num=5)
        env = SGSEnv(config)

        obs, info = env.reset(seed=42)

        # Clear any pending decisions
        env.skill_decision_context.clear()

        obs = env._get_observation()

        # No skill decision pending
        assert obs["current_step"] in [0, 1, 2]
        assert obs["skill_decision_type"] == 0
        assert obs["skill_decision_mask"].sum() == 0.0

    def test_skill_decision_request_id_tracking(self):
        """
        Test: Skill decision request tracking

        Expected: Request context maintained across steps
        """
        from ai.gym_wrapper import SGSEnv, SGSConfig
        from ai.skill_decision import SkillDecisionRequest, SkillDecisionType

        config = SGSConfig(player_num=5)
        env = SGSEnv(config)

        obs, info = env.reset(seed=42)

        request = SkillDecisionRequest(
            skill_name="观星",
            decision_type=SkillDecisionType.SELECT_ORDER,
            options=[0, 1, 2, 3, 4],
            context={"num_cards": 5, "request_id": 123},
        )
        env.skill_decision_context.active_request = request

        # Request should persist until resolved
        assert env.skill_decision_context.active_request.skill_name == "观星"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
