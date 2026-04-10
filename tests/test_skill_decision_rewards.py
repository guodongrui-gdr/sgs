"""
Test skill decision quality rewards.

This module tests the skill decision reward functionality added to ai/reward.py.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from ai.reward import (
    RewardConfig,
    RewardSystem,
    RewardCalculator,
    RewardRecord,
    IdentityRelationship,
)
from ai.skill_decision import (
    SkillDecisionType,
    SkillDecisionRequest,
    create_yes_no_request,
    create_select_order_request,
    create_distribute_request,
)


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


class TestSkillDecisionQualityRewardMethod:
    """Test the new skill_decision_quality_reward() method."""

    def setup_method(self):
        """Setup for each test"""
        self.config = RewardConfig()
        self.reward_system = RewardSystem(self.config, use_shaping=False)

    def test_yes_no_reward_activate_valuable(self):
        """Test发动有价值技能的奖励"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="YES_NO",
            skill_name="反馈",
            context={"activated": True, "is_valuable": True},
        )
        assert reward == 0.5, f"Expected 0.5, got {reward}"

    def test_yes_no_reward_decline_valuable(self):
        """Test拒绝有价值技能的惩罚"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="YES_NO",
            skill_name="反馈",
            context={"activated": False, "is_valuable": True},
        )
        assert reward == -0.5, f"Expected -0.5, got {reward}"

    def test_yes_no_reward_activate_not_valuable(self):
        """Test发动价值不高技能的小奖励"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="YES_NO",
            skill_name="测试技能",
            context={"activated": True, "is_valuable": False},
        )
        assert reward == 0.1, f"Expected 0.1, got {reward}"

    def test_yes_no_reward_decline_not_valuable(self):
        """Test正确拒绝价值不高技能"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="YES_NO",
            skill_name="测试技能",
            context={"activated": False, "is_valuable": False},
        )
        assert reward == 0.0, f"Expected 0.0, got {reward}"

    def test_select_order_reward_high_quality(self):
        """Test观星高质量牌序奖励"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="SELECT_ORDER",
            skill_name="观星",
            decision_quality=1.0,  # 最高质量
        )
        assert abs(reward - 2.0) < 0.01, f"Expected ~2.0, got {reward}"

    def test_select_order_reward_low_quality(self):
        """Test观星低质量牌序奖励"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="SELECT_ORDER",
            skill_name="观星",
            decision_quality=0.0,  # 最低质量
        )
        assert abs(reward - 0.5) < 0.01, f"Expected ~0.5, got {reward}"

    def test_select_order_reward_medium_quality(self):
        """Test观星中等质量牌序奖励"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="SELECT_ORDER",
            skill_name="观星",
            decision_quality=0.5,  # 中等质量
        )
        expected = 0.5 + (0.5 * 1.5)  # 1.25
        assert abs(reward - expected) < 0.01, f"Expected ~{expected}, got {reward}"

    def test_distribute_reward_to_ally(self):
        """Test分配卡牌给队友的奖励"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="DISTRIBUTE",
            skill_name="遗计",
            context={"ally_target": True},
        )
        assert reward == 0.5, f"Expected 0.5, got {reward}"

    def test_distribute_reward_to_enemy(self):
        """Test分配卡牌给敌人的惩罚"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="DISTRIBUTE",
            skill_name="遗计",
            context={"ally_target": False},
        )
        assert reward == -0.5, f"Expected -0.5, got {reward}"

    def test_select_cards_reward(self):
        """Test选择卡牌奖励"""
        reward = self.reward_system.skill_decision_quality_reward(
            decision_type="SELECT_CARDS",
            skill_name="制衡",
            decision_quality=0.8,
        )
        expected = 0.8 * 0.5  # 0.4
        assert abs(reward - expected) < 0.01, f"Expected ~{expected}, got {reward}"

    def test_reward_clipping(self):
        """Test奖励裁剪"""
        # 设置一个很低的clip_reward来测试裁剪
        config = RewardConfig(clip_reward=0.1)
        system = RewardSystem(config, use_shaping=False)

        # 这个应该返回2.0，但被裁剪到0.1
        reward = system.skill_decision_quality_reward(
            decision_type="SELECT_ORDER",
            skill_name="观星",
            decision_quality=1.0,
        )
        assert reward <= 0.1, f"Expected clipped to 0.1, got {reward}"

    def test_reward_records_added(self):
        """Test奖励记录被正确添加"""
        initial_count = len(self.reward_system.calculator.records)

        self.reward_system.skill_decision_quality_reward(
            decision_type="YES_NO",
            skill_name="测试",
            context={"activated": True, "is_valuable": True},
        )

        new_count = len(self.reward_system.calculator.records)
        assert new_count == initial_count + 1

        # 检查最后一条记录
        last_record = self.reward_system.calculator.records[-1]
        assert last_record.event_type == "skill_decision_yes_no"
        assert "测试" in str(last_record.context.get("skill_name", ""))

    def test_quality_bounds(self):
        """Test质量分数边界"""
        # Quality should always be between 0 and 1
        test_qualities = [-0.5, 0.0, 0.5, 1.0, 1.5]

        for quality in test_qualities:
            reward = self.reward_system.skill_decision_quality_reward(
                decision_type="SELECT_ORDER",
                skill_name="观星",
                decision_quality=quality,
            )
            # Reward should be between 0.5 and 2.0 (clipped)
            assert 0.5 <= reward <= 2.0, f"Quality {quality} gave reward {reward}"


class TestIdentityRelationships:
    """Test identity relationship logic for distribute decisions."""

    def test_identity_relationships(self):
        """Test identity relationship helper."""
        # Test identity relationships
        assert IdentityRelationship.get_relationship("主公", "忠臣") == "ally"
        assert IdentityRelationship.get_relationship("主公", "反贼") == "enemy"
        assert IdentityRelationship.get_relationship("忠臣", "反贼") == "enemy"
        assert IdentityRelationship.get_relationship("反贼", "反贼") == "ally"
        assert IdentityRelationship.get_relationship("内奸", "主公") == "enemy"
        assert IdentityRelationship.get_relationship("内奸", "内奸") == "ally"


class TestSkillDecisionRequests:
    """Test skill decision request creation and workflow."""

    def test_yes_no_request_creation(self):
        """Test creating YES_NO request."""
        request = create_yes_no_request("反馈", "是否发动反馈？")
        assert request.decision_type == SkillDecisionType.YES_NO
        assert request.skill_name == "反馈"
        assert request.options == ["否", "是"]

    def test_select_order_request_creation(self):
        """Test creating SELECT_ORDER request."""
        cards = ["card1", "card2", "card3"]
        request = create_select_order_request("观星", cards, "请排列牌的顺序")
        assert request.decision_type == SkillDecisionType.SELECT_ORDER
        assert request.skill_name == "观星"
        assert request.options == cards

    def test_distribute_request_creation(self):
        """Test creating DISTRIBUTE request."""
        items = ["card1", "card2"]
        targets = ["target1", "target2", "target3"]
        request = create_distribute_request("遗计", items, targets, "分配卡牌")
        assert request.decision_type == SkillDecisionType.DISTRIBUTE
        assert request.skill_name == "遗计"
        assert request.context.get("items") == items


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
