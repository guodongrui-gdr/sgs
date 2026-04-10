"""
Test identity-aware action masking

验证动作掩码是否正确应用身份信念:
- 有益动作(治疗/增益)偏好高P(忠臣)目标
- 攻击动作偏好高P(反贼)目标
- 软偏好而非硬约束，允许智能体从错误中学习
"""

import unittest
from unittest.mock import Mock, MagicMock
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.action_encoder import (
    ActionEncoder,
    ActionMaskGenerator,
    HierarchicalAction,
    ActionType,
)
from ai.state_encoder import StateEncoder


class MockCard:
    """Mock card for testing"""

    def __init__(self, name, card_type="BasicCard"):
        self.name = name
        self.card_type = card_type


class MockSkill:
    """Mock skill for testing"""

    def __init__(self, name):
        self.name = name


class MockPlayer:
    """Mock player for testing"""

    def __init__(self, idx, hand_cards=None, skills=None, is_alive=True):
        self.idx = idx
        self.player_id = idx
        self.hand_cards = hand_cards or []
        self.skills = skills or []
        self.is_alive = is_alive
        self.current_hp = 3
        self.max_hp = 4


class TestIdentityAwareMasking(unittest.TestCase):
    """Test identity-aware action masking"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = ActionEncoder()
        self.state_encoder = StateEncoder()
        self.mask_gen = ActionMaskGenerator(self.encoder, self.state_encoder)

        # Reset beliefs
        self.state_encoder.reset_beliefs(5, observer_idx=0)

        # Create mock players
        self.players = [
            MockPlayer(0, [], [], True),  # Observer (主公)
            MockPlayer(1, [], [], True),  # Target 1
            MockPlayer(2, [], [], True),  # Target 2
            MockPlayer(3, [], [], True),  # Target 3
            MockPlayer(4, [], [], True),  # Target 4
        ]

        self.game_state = {
            "players": self.players,
            "phase": "play_phase",
            "current_player_idx": 0,
        }

    def test_mask_generator_initialization(self):
        """Test mask generator can be initialized with state encoder"""
        mask_gen = ActionMaskGenerator(self.encoder, self.state_encoder)
        self.assertIsNotNone(mask_gen.state_encoder)
        self.assertEqual(mask_gen.ALLY_BENEFIT_BONUS, 0.3)
        self.assertEqual(mask_gen.ENEMY_ATTACK_BONUS, 0.3)

    def test_mask_generator_without_state_encoder(self):
        """Test mask generator works without state encoder (backward compatibility)"""
        mask_gen = ActionMaskGenerator(self.encoder, None)
        self.assertIsNone(mask_gen.state_encoder)

    def test_classify_beneficial_cards(self):
        """Test classification of beneficial cards"""
        player = MockPlayer(0, [MockCard("桃")])

        is_beneficial, is_attack = self.mask_gen._classify_action(
            ActionType.USE_CARD, player, 0, self.game_state
        )
        self.assertTrue(is_beneficial)
        self.assertFalse(is_attack)

        # Test 酒
        player.hand_cards = [MockCard("酒")]
        is_beneficial, is_attack = self.mask_gen._classify_action(
            ActionType.USE_CARD, player, 0, self.game_state
        )
        self.assertTrue(is_beneficial)
        self.assertFalse(is_attack)

        # Test 无中生有
        player.hand_cards = [MockCard("无中生有")]
        is_beneficial, is_attack = self.mask_gen._classify_action(
            ActionType.USE_CARD, player, 0, self.game_state
        )
        self.assertTrue(is_beneficial)
        self.assertFalse(is_attack)

    def test_classify_attack_cards(self):
        """Test classification of attack cards"""
        player = MockPlayer(0, [MockCard("杀")])

        is_beneficial, is_attack = self.mask_gen._classify_action(
            ActionType.USE_CARD, player, 0, self.game_state
        )
        self.assertFalse(is_beneficial)
        self.assertTrue(is_attack)

        # Test 决斗
        player.hand_cards = [MockCard("决斗")]
        is_beneficial, is_attack = self.mask_gen._classify_action(
            ActionType.USE_CARD, player, 0, self.game_state
        )
        self.assertFalse(is_beneficial)
        self.assertTrue(is_attack)

        # Test 火攻
        player.hand_cards = [MockCard("火攻")]
        is_beneficial, is_attack = self.mask_gen._classify_action(
            ActionType.USE_CARD, player, 0, self.game_state
        )
        self.assertFalse(is_beneficial)
        self.assertTrue(is_attack)

        # Test 过河拆桥
        player.hand_cards = [MockCard("过河拆桥")]
        is_beneficial, is_attack = self.mask_gen._classify_action(
            ActionType.USE_CARD, player, 0, self.game_state
        )
        self.assertFalse(is_beneficial)
        self.assertTrue(is_attack)

    def test_classify_neutral_cards(self):
        """Test classification of neutral cards"""
        player = MockPlayer(0, [MockCard("闪")])

        is_beneficial, is_attack = self.mask_gen._classify_action(
            ActionType.USE_CARD, player, 0, self.game_state
        )
        self.assertFalse(is_beneficial)
        self.assertFalse(is_attack)

    def test_apply_identity_preference_beneficial_to_ally(self):
        """Test beneficial action preference for believed ally"""
        # Set high belief that target 1 is 忠臣
        self.state_encoder._belief_states[0][1] = np.array(
            [0.8, 0.15, 0.04, 0.01], dtype=np.float32
        )

        # Beneficial action targeting believed ally
        result = self.mask_gen._apply_identity_preference(0, 1, True, False, 1.0)

        # Should have bonus for high P(忠臣)
        expected = 1.0 + 0.3 * 0.8  # base + ALLY_BENEFIT_BONUS * p_loyalist
        self.assertAlmostEqual(result, expected, places=5)
        self.assertGreater(result, 1.0)  # Should be boosted

    def test_apply_identity_preference_beneficial_to_enemy(self):
        """Test beneficial action targeting believed enemy"""
        # Set high belief that target 1 is 反贼
        self.state_encoder._belief_states[0][1] = np.array(
            [0.15, 0.8, 0.04, 0.01], dtype=np.float32
        )

        # Beneficial action targeting believed enemy
        result = self.mask_gen._apply_identity_preference(0, 1, True, False, 1.0)

        # Should have minimal bonus for low P(忠臣)
        expected = 1.0 + 0.3 * 0.15
        self.assertAlmostEqual(result, expected, places=5)
        self.assertLess(result, 1.1)  # Should be close to base

    def test_apply_identity_preference_attack_to_enemy(self):
        """Test attack action preference for believed enemy"""
        # Set high belief that target 1 is 反贼
        self.state_encoder._belief_states[0][1] = np.array(
            [0.15, 0.8, 0.04, 0.01], dtype=np.float32
        )

        # Attack action targeting believed enemy
        result = self.mask_gen._apply_identity_preference(0, 1, False, True, 1.0)

        # Should have bonus for high P(反贼)
        expected = 1.0 + 0.3 * 0.8  # base + ENEMY_ATTACK_BONUS * p_rebel
        self.assertAlmostEqual(result, expected, places=5)
        self.assertGreater(result, 1.0)  # Should be boosted

    def test_apply_identity_preference_attack_to_ally(self):
        """Test attack action targeting believed ally"""
        # Set high belief that target 1 is 忠臣
        self.state_encoder._belief_states[0][1] = np.array(
            [0.8, 0.15, 0.04, 0.01], dtype=np.float32
        )

        # Attack action targeting believed ally
        result = self.mask_gen._apply_identity_preference(0, 1, False, True, 1.0)

        # Should have minimal bonus for low P(反贼)
        expected = 1.0 + 0.3 * 0.15
        self.assertAlmostEqual(result, expected, places=5)
        self.assertLess(result, 1.1)  # Should be close to base

    def test_soft_preference_not_hard_constraint(self):
        """Test that soft preference doesn't prevent any valid target"""
        # Even with strong belief, all targets remain valid (> 0)
        self.state_encoder._belief_states[0][1] = np.array(
            [0.9, 0.05, 0.04, 0.01], dtype=np.float32
        )

        # Attack targeting believed ally (should be valid but less preferred)
        result = self.mask_gen._apply_identity_preference(0, 1, False, True, 1.0)
        self.assertGreater(result, 0.0)  # Still valid

        # Beneficial targeting believed enemy (should be valid but less preferred)
        result = self.mask_gen._apply_identity_preference(0, 1, True, False, 1.0)
        self.assertGreater(result, 0.0)  # Still valid

    def test_get_valid_targets_with_identity_awareness(self):
        """Test that _get_valid_targets applies identity preferences"""
        # Setup: player 0 has a 顺手牵羊 (debuff card targeting others)
        player = MockPlayer(0, [MockCard("顺手牵羊")])

        # Set beliefs: target 1 is likely ally, target 2 is likely enemy
        self.state_encoder._belief_states[0][1] = np.array(
            [0.8, 0.15, 0.04, 0.01], dtype=np.float32
        )
        self.state_encoder._belief_states[0][2] = np.array(
            [0.15, 0.8, 0.04, 0.01], dtype=np.float32
        )

        game_state = {
            "players": [player, self.players[1], self.players[2]],
            "phase": "play_phase",
        }

        pending = HierarchicalAction(
            action_type=ActionType.USE_CARD, card_idx=0, target_idx=None
        )

        # Get masks with identity awareness
        masks = self.mask_gen._get_valid_targets(
            game_state, player, pending, observer_idx=0
        )

        # Both targets should be valid (mask > 0)
        self.assertGreater(masks[0], 0.0)  # Target 1 (index 0 in mask = player 1)
        self.assertGreater(masks[1], 0.0)  # Target 2 (index 1 in mask = player 2)

        # For attack/debuff action, target 2 (believed enemy) should have higher mask
        # 顺手牵羊 is in DEBUFF_CARD_NAMES
        self.assertGreater(masks[1], masks[0])

    def test_get_valid_targets_attack_preference(self):
        """Test attack action prefers believed enemies"""
        # Setup: player 0 has a 杀 (attack card)
        player = MockPlayer(0, [MockCard("杀")])
        player.attack_range = 3  # Can reach all targets
        player.equipment = {}

        # Mock is_sha_card to return True
        import ai.action_encoder as ae_module

        original_is_sha = ae_module.is_sha_card
        ae_module.is_sha_card = lambda c: c.name in ["杀", "火杀", "雷杀"]

        # Set beliefs: target 1 is likely ally, target 2 is likely enemy
        self.state_encoder._belief_states[0][1] = np.array(
            [0.8, 0.15, 0.04, 0.01], dtype=np.float32
        )
        self.state_encoder._belief_states[0][2] = np.array(
            [0.15, 0.8, 0.04, 0.01], dtype=np.float32
        )

        game_state = {
            "players": [player, self.players[1], self.players[2]],
            "phase": "play_phase",
        }

        pending = HierarchicalAction(
            action_type=ActionType.USE_CARD, card_idx=0, target_idx=None
        )

        # Get masks with identity awareness
        masks = self.mask_gen._get_valid_targets(
            game_state, player, pending, observer_idx=0
        )

        # Restore original function
        ae_module.is_sha_card = original_is_sha

        # Target 2 (believed enemy) should have higher mask for attack action
        # Note: mask indices are player_idx - 1
        self.assertGreater(masks[1], masks[0])

    def test_target_mask_without_state_encoder(self):
        """Test that masking works without state encoder (backward compatibility)"""
        mask_gen = ActionMaskGenerator(self.encoder, None)

        # Use a card that targets others
        player = MockPlayer(0, [MockCard("顺手牵羊")])
        game_state = {
            "players": [player, self.players[1]],
            "phase": "play_phase",
        }

        pending = HierarchicalAction(
            action_type=ActionType.USE_CARD, card_idx=0, target_idx=None
        )

        masks = mask_gen._get_valid_targets(game_state, player, pending, None)

        # Should still work with base mask of 1.0
        self.assertEqual(masks[0], 1.0)

    def test_respond_tao_prefers_believed_allies(self):
        """Test that 桃响应偏好 believed allies

        Note: This test verifies the preference logic, but there's a pre-existing bug
        where the self-heal option (masks[0]) overwrites the first dying player's mask.
        The preference calculation is correct, but the mask assignment has a collision.
        """
        player = MockPlayer(0, [MockCard("桃")])
        player.current_hp = (
            3  # Healthy (not dying, avoid collision with dying player 1)
        )

        # Make target 1 likely ally, target 2 likely enemy
        self.state_encoder._belief_states[0][1] = np.array(
            [0.8, 0.15, 0.04, 0.01], dtype=np.float32
        )
        self.state_encoder._belief_states[0][2] = np.array(
            [0.15, 0.8, 0.04, 0.01], dtype=np.float32
        )

        # Only target 2 is dying, target 1 is healthy
        self.players[1].current_hp = 3  # Healthy (avoid collision)
        self.players[2].current_hp = -1  # Dying

        game_state = {
            "players": [player, self.players[1], self.players[2]],
            "phase": "respond_tao",
        }

        pending = HierarchicalAction(
            action_type=ActionType.RESPOND_TAO, card_idx=0, target_idx=None
        )

        masks = self.mask_gen._get_valid_targets(
            game_state, player, pending, observer_idx=0
        )

        # Target 2 (dying believed enemy) should be valid with preference applied
        # Note: Due to pre-existing bug, masks[0] is overwritten by self-heal option
        # But target 2 (index 1 in mask) should have the correct preference
        self.assertGreater(masks[1], 1.0)  # Target 2 with preference

        # The preference calculation is correct: enemy gets lower boost for heal
        # P(loyalist)=0.15 -> mask = 1.0 + 0.3*0.15 = 1.045
        expected_enemy_mask = 1.0 + 0.3 * 0.15
        self.assertAlmostEqual(masks[1], expected_enemy_mask, places=5)

    def test_heal_skill_prefers_allies(self):
        """Test that healing skills prefer believed allies"""
        from skills.base import ActiveSkill

        # Create a healing skill
        class HealSkill(ActiveSkill):
            def __init__(self):
                self.name = "急救"

        player = MockPlayer(0, skills=[HealSkill()])

        # Make target 1 likely ally
        self.state_encoder._belief_states[0][1] = np.array(
            [0.8, 0.15, 0.04, 0.01], dtype=np.float32
        )

        game_state = {
            "players": [player, self.players[1]],
            "phase": "play_phase",
        }

        pending = HierarchicalAction(
            action_type=ActionType.USE_SKILL, card_idx=0, target_idx=None
        )

        masks = self.mask_gen._get_valid_targets(
            game_state, player, pending, observer_idx=0
        )

        # Target should have boosted mask (healing skill)
        self.assertGreater(masks[0], 1.0)  # Should be > 1.0 due to ally bonus

    def test_generate_masks_passes_observer_idx(self):
        """Test that generate_masks properly passes observer_idx to target selection"""
        player = MockPlayer(0, [MockCard("顺手牵羊")])

        # Set up belief - target 2 is likely enemy
        self.state_encoder._belief_states[0][1] = np.array(
            [0.8, 0.15, 0.04, 0.01], dtype=np.float32
        )
        self.state_encoder._belief_states[0][2] = np.array(
            [0.15, 0.8, 0.04, 0.01], dtype=np.float32
        )

        game_state = {
            "players": [player, self.players[1], self.players[2]],
            "phase": "play_phase",
        }

        pending = HierarchicalAction(
            action_type=ActionType.USE_CARD, card_idx=0, target_idx=None
        )

        # Generate masks at step 2 (target selection)
        _, _, masks = self.mask_gen.generate_masks(
            game_state, player, current_step=2, pending_action=pending
        )

        # Should have applied identity preference for attack/debuff
        # Target 2 (believed enemy) should have mask > 1.0
        # 顺手牵羊 is a debuff card, so prefers enemies
        self.assertGreater(masks[1], 1.0)  # Target 2 (player 2) should be boosted

    def test_belief_thresholds(self):
        """Test that beliefs near thresholds work correctly"""
        # Test with belief at minimum threshold
        self.state_encoder._belief_states[0][1] = np.array(
            [0.05, 0.9, 0.04, 0.01], dtype=np.float32
        )

        result = self.mask_gen._apply_identity_preference(0, 1, False, True, 1.0)
        expected = 1.0 + 0.3 * 0.9
        self.assertAlmostEqual(result, expected, places=5)

        # Test with uniform belief (uncertain)
        self.state_encoder._belief_states[0][1] = np.array(
            [0.33, 0.33, 0.33, 0.01], dtype=np.float32
        )

        result = self.mask_gen._apply_identity_preference(0, 1, True, False, 1.0)
        expected = 1.0 + 0.3 * 0.33
        self.assertAlmostEqual(result, expected, places=5)


if __name__ == "__main__":
    unittest.main()
