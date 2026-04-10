"""
Tests for RLAI action_masks parameter fix

CRITICAL: MaskablePPO requires action_masks to be passed to predict().
This test verifies the fix in ai/rl_ai.py.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any

# Skip all tests if SB3 not available
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")


def create_mock_player():
    """Helper to create properly mocked player"""
    mock_player = Mock()
    mock_player.idx = 1
    mock_player.hand_cards = []
    mock_player.skills = []  # CRITICAL: must be iterable
    mock_player.is_alive = True
    mock_player.sha_count = 0
    mock_player.jiu_count = 0
    mock_player.current_hp = 4
    mock_player.max_hp = 4
    mock_player.equipment = {}
    mock_player.unlimited_sha = False
    mock_player.jiu_effect = 0
    mock_player.hand_limit = 4
    mock_player.commander_name = "测试武将"
    mock_player.nation = "群"
    return mock_player


def create_mock_engine():
    """Helper to create properly mocked engine"""
    mock_engine = Mock()
    mock_state = Mock()

    player_dict = {
        "hand_cards": [],
        "equipment": {},
        "judge_area": [],
        "skills": [],
        "is_alive": True,
        "current_hp": 4,
        "max_hp": 4,
        "idx": 1,
        "sha_count": 0,
        "jiu_count": 0,
        "unlimited_sha": False,
        "jiu_effect": 0,
        "hand_limit": 4,
        "commander_name": "测试武将",
        "nation": "群",
    }

    mock_state.to_dict = Mock(
        return_value={
            "players": [player_dict],
            "phase": "play_phase",
            "current_player_idx": 0,
            "round_num": 1,
            "deck_count": 50,
            "discard_pile_count": 10,
        }
    )
    mock_engine.get_state = Mock(return_value=mock_state)
    return mock_engine


class TestRLAIActionMasks:
    """Test suite for action_masks parameter in RLAI.select_action()"""

    def test_action_masks_passed_to_predict(self):
        """
        Test: action_masks parameter is passed to model.predict()

        Scenario: Mock MaskablePPO model records parameters passed to predict()
        Expected: action_masks is non-None in the predict() call
        """
        from ai.rl_ai import RLAI, RLAIConfig

        # Create mock model that records parameters
        mock_model = Mock()
        recorded_params = {}

        def record_predict(obs, action_masks=None, deterministic=False, **kwargs):
            recorded_params["obs"] = obs
            recorded_params["action_masks"] = action_masks
            recorded_params["deterministic"] = deterministic
            return (0, None)

        mock_model.predict.side_effect = record_predict

        mock_engine = create_mock_engine()
        mock_player = create_mock_player()

        config = RLAIConfig(model_path="dummy_path")

        with patch("ai.rl_ai.Path.exists", return_value=True):
            with patch("ai.rl_ai.MaskablePPO.load", return_value=mock_model):
                rl_ai = RLAI(config)
                rl_ai.use_masking = True
                rl_ai.model = mock_model

        # Call select_action
        result = rl_ai.select_action(mock_engine, mock_player)

        # Verify: action_masks was passed and is non-None
        assert "action_masks" in recorded_params
        assert recorded_params["action_masks"] is not None
        assert isinstance(recorded_params["action_masks"], np.ndarray)

    def test_mask_generation_before_predict(self):
        """
        Test: Action masks are generated before calling predict()

        Expected: Masks have valid structure (non-zero when actions available)
        """
        from ai.rl_ai import RLAI, RLAIConfig

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(0, None))

        mock_engine = create_mock_engine()
        mock_player = create_mock_player()

        config = RLAIConfig(model_path="dummy_path")

        with patch("ai.rl_ai.Path.exists", return_value=True):
            with patch("ai.rl_ai.MaskablePPO.load", return_value=mock_model):
                rl_ai = RLAI(config)
                rl_ai.use_masking = True
                rl_ai.model = mock_model

        # Generate masks directly
        masks = rl_ai._get_action_masks(mock_engine, mock_player)

        # Verify mask structure
        assert masks is not None
        assert isinstance(masks, np.ndarray)
        assert masks.dtype == np.float32
        # At least one action should be valid (END_TURN or PASS)
        assert masks.sum() >= 1.0

    def test_maskable_ppo_without_masks_fallback(self):
        """
        Test: Method handles missing masks gracefully

        Expected: Combined mask is non-empty (fallback prevents empty mask)
        """
        from ai.rl_ai import RLAI, RLAIConfig

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(0, None))

        mock_engine = Mock()
        mock_player = create_mock_player()

        mock_state = Mock()
        # Minimal state with unknown phase - triggers fallback
        mock_state.to_dict = Mock(
            return_value={
                "players": [
                    {
                        "hand_cards": [],
                        "equipment": {},
                        "judge_area": [],
                        "skills": [],
                        "is_alive": True,
                        "idx": 1,
                    }
                ],
                "phase": "unknown_phase",  # Invalid phase
                "current_player_idx": 0,
            }
        )
        mock_engine.get_state = Mock(return_value=mock_state)

        config = RLAIConfig(model_path="dummy_path")

        with patch("ai.rl_ai.Path.exists", return_value=True):
            with patch("ai.rl_ai.MaskablePPO.load", return_value=mock_model):
                rl_ai = RLAI(config)
                rl_ai.use_masking = True
                rl_ai.model = mock_model

        masks = rl_ai._get_action_masks(mock_engine, mock_player)

        # Fallback should ensure mask is non-empty
        assert masks.sum() >= 1.0

    def test_mask_dimensions_match_action_space(self):
        """
        Test: Mask dimensions match action space dimension

        Expected: action_masks.shape matches action_encoder.get_action_space_dim()
        """
        from ai.rl_ai import RLAI, RLAIConfig

        mock_model = Mock()
        mock_model.predict = Mock(return_value=(0, None))

        mock_engine = create_mock_engine()
        mock_player = create_mock_player()

        config = RLAIConfig(model_path="dummy_path")

        with patch("ai.rl_ai.Path.exists", return_value=True):
            with patch("ai.rl_ai.MaskablePPO.load", return_value=mock_model):
                rl_ai = RLAI(config)
                rl_ai.use_masking = True
                rl_ai.model = mock_model

        action_dim = rl_ai.action_encoder.get_action_space_dim()
        masks = rl_ai._get_action_masks(mock_engine, mock_player)

        assert len(masks) == action_dim


class TestActionMaskIntegration:
    """Integration tests for action mask flow"""

    def test_mask_used_in_select_action_flow(self):
        """
        Test: Masks flow through select_action to predict

        Expected: predict receives masks when use_masking=True
        """
        from ai.rl_ai import RLAI, RLAIConfig

        mock_model = Mock()
        recorded_action_masks = []

        def predict_with_mask_check(
            obs, action_masks=None, deterministic=False, **kwargs
        ):
            if action_masks is not None:
                recorded_action_masks.append(action_masks)
            return (0, None)

        mock_model.predict = Mock(side_effect=predict_with_mask_check)

        mock_engine = create_mock_engine()
        mock_player = create_mock_player()

        config = RLAIConfig(model_path="dummy_path")

        with patch("ai.rl_ai.Path.exists", return_value=True):
            with patch("ai.rl_ai.MaskablePPO.load", return_value=mock_model):
                rl_ai = RLAI(config)
                rl_ai.use_masking = True
                rl_ai.model = mock_model

        rl_ai.select_action(mock_engine, mock_player)

        # Verify masks were passed
        assert len(recorded_action_masks) > 0
        assert recorded_action_masks[0] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
