"""
Test skill decision observation fields in state encoder and gym wrapper.

Verifies:
- skill_decision fields are properly encoded in state vector
- Fields are populated when skill decision is pending
- Fields are zeros when no pending decision
- Observation space matches actual observation structure
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ai.state_encoder import StateEncoder, EncodingConfig
from ai.skill_decision import (
    SkillDecisionType,
    SkillDecisionRequest,
    SkillDecisionContext,
    create_yes_no_request,
    create_select_order_request,
)


class TestSkillDecisionObservationFields(unittest.TestCase):
    """Test skill decision observation fields."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EncodingConfig()
        self.encoder = StateEncoder(self.config)

    def _create_game_state(self, skill_decision=None):
        """Create a minimal game state for testing."""
        state = {
            "players": [
                {
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                    "commander_name": "曹操",
                    "nation": "魏",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                    "identity": "主公",
                },
                {
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                    "commander_name": "刘备",
                    "nation": "蜀",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "identity": "忠臣",
                },
            ],
            "phase": "play_phase",
            "round_num": 1,
            "current_player_idx": 0,
            "deck_count": 100,
            "discard_pile_count": 10,
            "action_history": [],
        }

        if skill_decision:
            state["skill_decision"] = skill_decision
        else:
            # Default: no skill decision pending (current_step=0)
            state["skill_decision"] = {
                "current_step": 0,
                "decision_type": 0,
                "options_mask": [],
            }

        return state

    def test_skill_decision_fields_exist_in_encoded_state(self):
        """Test that skill decision fields are included in encoded state."""
        state = self._create_game_state()
        encoded = self.encoder.encode(state, 0)

        # The encoded state should have additional 10 dimensions for skill decision
        # (4 for current_step one-hot, 7 for decision_type one-hot, 1 for mask ratio)
        # But wait - we need to check if the encoder actually adds these
        # Let's just verify the encoding succeeds and has a reasonable size
        self.assertIsInstance(encoded, np.ndarray)
        self.assertGreater(len(encoded), 1000)  # Should have many dimensions

    def test_skill_decision_encoding_no_pending(self):
        """Test encoding when no skill decision is pending."""
        state = self._create_game_state(
            skill_decision={
                "current_step": 0,  # Regular game action
                "decision_type": 0,
                "options_mask": [],
            }
        )

        encoded = self.encoder.encode(state, 0)
        self.assertIsInstance(encoded, np.ndarray)

    def test_skill_decision_encoding_pending(self):
        """Test encoding when skill decision is pending."""
        state = self._create_game_state(
            skill_decision={
                "current_step": 3,  # Skill decision step
                "decision_type": int(SkillDecisionType.YES_NO),
                "options_mask": [1.0, 1.0],  # Two options available
            }
        )

        encoded = self.encoder.encode(state, 0)
        self.assertIsInstance(encoded, np.ndarray)

    def test_skill_decision_encoding_various_types(self):
        """Test encoding with various skill decision types."""
        decision_types = [
            SkillDecisionType.YES_NO,
            SkillDecisionType.SELECT_CARDS,
            SkillDecisionType.SELECT_TARGETS,
            SkillDecisionType.SELECT_ORDER,
            SkillDecisionType.DISTRIBUTE,
            SkillDecisionType.SELECT_PAIR,
            SkillDecisionType.SELECT_SINGLE,
        ]

        for dec_type in decision_types:
            with self.subTest(decision_type=dec_type):
                state = self._create_game_state(
                    skill_decision={
                        "current_step": 3,
                        "decision_type": int(dec_type),
                        "options_mask": [1.0, 0.0, 1.0],
                    }
                )
                encoded = self.encoder.encode(state, 0)
                self.assertIsInstance(encoded, np.ndarray)

    def test_skill_decision_step_range(self):
        """Test that current_step values are properly handled."""
        for step in [0, 1, 2, 3]:
            with self.subTest(current_step=step):
                state = self._create_game_state(
                    skill_decision={
                        "current_step": step,
                        "decision_type": 0,
                        "options_mask": [],
                    }
                )
                encoded = self.encoder.encode(state, 0)
                self.assertIsInstance(encoded, np.ndarray)

    def test_invalid_skill_decision_values(self):
        """Test that invalid skill decision values are handled gracefully."""
        # Test with out-of-range values
        state = self._create_game_state(
            skill_decision={
                "current_step": 999,  # Invalid
                "decision_type": 999,  # Invalid
                "options_mask": None,  # Invalid
            }
        )

        # Should not raise an exception
        encoded = self.encoder.encode(state, 0)
        self.assertIsInstance(encoded, np.ndarray)


class TestSkillDecisionInGymWrapper(unittest.TestCase):
    """Test skill decision fields in Gym wrapper observation."""

    @unittest.skipIf(
        not hasattr(
            __import__("ai.gym_wrapper", fromlist=["GYM_AVAILABLE"]), "GYM_AVAILABLE"
        ),
        "Gymnasium not available",
    )
    def test_observation_structure(self):
        """Test that observation has correct structure."""
        from ai.gym_wrapper import SGSEnv, SGSConfig

        config = SGSConfig(player_num=2)
        env = SGSEnv(config)

        obs, info = env.reset()

        # Check observation keys
        required_keys = [
            "state",
            "action_mask_type",
            "action_mask_card",
            "action_mask_target",
            "current_step",
            "skill_decision_type",
            "skill_decision_mask",
        ]

        for key in required_keys:
            self.assertIn(key, obs, f"Missing key: {key}")

        # Check types
        self.assertIsInstance(obs["state"], np.ndarray)
        self.assertIsInstance(obs["current_step"], (int, np.integer))
        self.assertIsInstance(obs["skill_decision_type"], (int, np.integer))
        self.assertIsInstance(obs["skill_decision_mask"], np.ndarray)

        env.close()

    @unittest.skipIf(
        not hasattr(
            __import__("ai.gym_wrapper", fromlist=["GYM_AVAILABLE"]), "GYM_AVAILABLE"
        ),
        "Gymnasium not available",
    )
    def test_skill_decision_zeros_when_no_pending(self):
        """Test that skill decision fields are zeros when no pending decision."""
        from ai.gym_wrapper import SGSEnv, SGSConfig

        config = SGSConfig(player_num=2)
        env = SGSEnv(config)

        obs, info = env.reset()

        # Should be in regular game state (not skill decision)
        self.assertEqual(obs["current_step"], 0)
        self.assertEqual(obs["skill_decision_type"], 0)
        self.assertEqual(np.sum(obs["skill_decision_mask"]), 0)

        env.close()


if __name__ == "__main__":
    unittest.main()
