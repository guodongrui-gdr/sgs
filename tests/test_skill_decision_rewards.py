"""
Test skill decision quality rewards.

This module tests the skill decision reward functionality added to ai/reward.py.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ai.reward import (
    RewardConfig,
    RewardSystem,
    RewardCalculator,
    RewardRecord,
)
from ai.skill_decision import SkillDecisionType, SkillDecisionRequest


class TestSkillDecisionRewards:
    """Test skill decision quality rewards."""

    def test_skill_yes_no_activation_reward(self):
        """Test YES_NO skill decision rewards activation."""
        config = RewardConfig()
        calculator = RewardCalculator(config)

        # Test activating valuable skill (should get positive reward)
        reward = calculator.calculate_reward(
            event_type="skill_decision_yes_no",
            source_identity="忠臣",
            target_identity="",
            current_identity="忠臣",
            is_source=True,
            value=1.0,  # Activated
            context={"skill_name": "武圣", "activated": True, "is_valuable": True},
        )

        # Should get positive reward for activating valuable skill
        assert reward > 0, (
            f"Expected positive reward for activating valuable skill, got {reward}"
        )

        # Test declining valuable skill (should get penalty)
        reward_decline = calculator.calculate_reward(
            event_type="skill_decision_yes_no",
            source_identity="忠臣",
            target_identity="",
            current_identity="忠臣",
            is_source=True,
            value=0.0,  # Declined
            context={"skill_name": "武圣", "activated": False, "is_valuable": True},
        )

        # Should get negative reward for declining valuable skill
        assert reward_decline < 0, (
            f"Expected negative reward for declining valuable skill, got {reward_decline}"
        )

    def test_skill_decision_recorded(self):
        """Test that skill decision rewards are properly recorded."""
        config = RewardConfig()
        calculator = RewardCalculator(config)

        # Trigger a skill decision reward
        calculator.calculate_reward(
            event_type="skill_decision_yes_no",
            source_identity="忠臣",
            target_identity="",
            current_identity="忠臣",
            is_source=True,
            value=1.0,
            context={"skill_name": "武圣"},
        )

        # Check that the record was created
        records = calculator.records
        skill_records = [r for r in records if "skill_decision" in r.event_type]
        assert len(skill_records) > 0, "Skill decision record not found"

    def test_skill_select_order_quality_reward(self):
        """Test SELECT_ORDER skill decision (观星) quality reward."""
        config = RewardConfig()
        system = RewardSystem(config=config)

        # Simulate 观星 decision with good ordering (strong cards on top)
        reward = system.get_reward(
            event_type="skill_decision_select_order",
            source_identity="忠臣",
            target_identity="",
            current_identity="忠臣",
            is_source=True,
            value=1.0,  # Quality score 0-1
            context={
                "skill_name": "观星",
                "decision_type": SkillDecisionType.SELECT_ORDER,
                "card_positions": [0, 1, 2, 3, 4],  # Ordering
                "quality_score": 0.8,  # High quality ordering
            },
        )

        # Should get positive reward proportional to quality
        assert reward > 0, f"Expected positive reward for SELECT_ORDER, got {reward}"

    def test_skill_distribute_reward(self):
        """Test DISTRIBUTE skill decision (遗计) reward."""
        config = RewardConfig()
        system = RewardSystem(config=config)

        # Simulate 遗计 distribution to allies
        reward = system.get_reward(
            event_type="skill_decision_distribute",
            source_identity="忠臣",
            target_identity="忠臣",
            current_identity="忠臣",
            is_source=True,
            value=2.0,  # Number of cards distributed
            context={
                "skill_name": "遗计",
                "decision_type": SkillDecisionType.DISTRIBUTE,
                "cards_distributed": 2,
                "ally_target": True,  # Distributed to ally
            },
        )

        # Should get positive reward for distributing to ally
        assert reward > 0, (
            f"Expected positive reward for DISTRIBUTE to ally, got {reward}"
        )

    def test_skill_activation_reward_in_reward_system(self):
        """Test skill activation reward through RewardSystem."""
        config = RewardConfig(skill_activation_reward=0.2)
        system = RewardSystem(config=config)

        # Simulate skill activation
        reward = system.get_reward(
            event_type="skill_activation",
            source_identity="忠臣",
            target_identity="",
            current_identity="忠臣",
            is_source=True,
            value=1.0,
            context={"skill_name": "武圣"},
        )

        # Should include skill_activation_reward
        assert reward > 0, (
            f"Expected positive reward for skill activation, got {reward}"
        )


class TestSkillDecisionRewardConfig:
    """Test skill decision reward configuration."""

    def test_skill_activation_reward_config(self):
        """Test that skill_activation_reward is configurable."""
        config = RewardConfig(skill_activation_reward=0.5)
        assert config.skill_activation_reward == 0.5

    def test_skill_decision_quality_enabled(self):
        """Test that skill decision quality rewards are enabled."""
        config = RewardConfig()
        # Check that the config has necessary fields for skill decisions
        assert hasattr(config, "skill_activation_reward")
        assert hasattr(config, "effective_card_use_reward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
