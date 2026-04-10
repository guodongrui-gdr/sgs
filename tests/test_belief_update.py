"""Tests for identity belief update logic in StateEncoder"""

import numpy as np
import pytest

from ai.state_encoder import StateEncoder, EncodingConfig


class TestBeliefInitialization:
    """Test belief state initialization"""

    def test_reset_beliefs_initializes_uniformly(self):
        """Test that beliefs are initialized with uniform distribution"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        # Check beliefs initialized for all other players
        assert 1 in encoder._belief_states[0]
        assert 2 in encoder._belief_states[0]
        assert 3 in encoder._belief_states[0]
        assert 4 in encoder._belief_states[0]

        # Check uniform distribution
        expected = np.array([0.33, 0.33, 0.33, 0.01], dtype=np.float32)
        for target_idx in range(1, 5):
            np.testing.assert_array_almost_equal(
                encoder._belief_states[0][target_idx], expected, decimal=2
            )

    def test_reset_skips_observer_self(self):
        """Test that observer doesn't have belief about themselves"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        assert 0 not in encoder._belief_states[0]

    def test_reset_clears_previous_beliefs(self):
        """Test that reset clears any previous belief states"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        # Modify a belief
        encoder._belief_states[0][1] = np.array([0.5, 0.3, 0.19, 0.01])

        # Reset
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        # Check reset to uniform
        expected = np.array([0.33, 0.33, 0.33, 0.01], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            encoder._belief_states[0][1], expected, decimal=2
        )


class TestBeliefUpdateAttack:
    """Test belief updates for attack actions"""

    def test_attack_lord_increases_rebel_probability(self):
        """Test that attacking 主公 increases P(反贼)"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
                {"identity": "未知"},
                {"identity": "未知"},
                {"identity": "未知"},
            ]
        }

        initial_belief = encoder._belief_states[0][1].copy()

        # Player 1 attacks 主公 (player 0)
        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="attack",
            target_idx=0,
            game_state=game_state,
        )

        updated_belief = encoder._belief_states[0][1]

        # P(反贼) should increase
        assert updated_belief[1] > initial_belief[1]
        # P(忠臣) should decrease
        assert updated_belief[0] < initial_belief[0]

    def test_attack_rebel_increases_loyalist_probability(self):
        """Test that attacking known 反贼 increases P(忠臣)"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        # Set player 2 as likely rebel
        encoder._belief_states[0][2] = np.array(
            [0.2, 0.7, 0.09, 0.01], dtype=np.float32
        )

        initial_belief = encoder._belief_states[0][1].copy()

        # Player 1 attacks player 2 (likely rebel)
        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="attack",
            target_idx=2,
            game_state={"players": []},
        )

        updated_belief = encoder._belief_states[0][1]

        # P(忠臣) should increase
        assert updated_belief[0] > initial_belief[0]
        # P(反贼) should decrease
        assert updated_belief[1] < initial_belief[1]

    def test_aoe_attack_general_aggression(self):
        """Test that AOE attacks signal slight 反贼/内奸 tendency"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        initial_belief = encoder._belief_states[0][1].copy()

        # Player 1 uses AOE (南蛮入侵/万箭齐发)
        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="aoe_attack",
            target_idx=None,
            game_state={"players": []},
        )

        updated_belief = encoder._belief_states[0][1]

        # P(反贼) and P(内奸) should increase slightly
        assert updated_belief[1] > initial_belief[1]
        assert updated_belief[2] > initial_belief[2]


class TestBeliefUpdateHeal:
    """Test belief updates for heal actions"""

    def test_heal_lord_increases_loyalist_probability(self):
        """Test that healing 主公 increases P(忠臣)"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
                {"identity": "未知"},
            ]
        }

        initial_belief = encoder._belief_states[0][1].copy()

        # Player 1 heals 主公 (player 0)
        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="heal",
            target_idx=0,
            game_state=game_state,
        )

        updated_belief = encoder._belief_states[0][1]

        # P(忠臣) should increase
        assert updated_belief[0] > initial_belief[0]
        # P(反贼) should decrease
        assert updated_belief[1] < initial_belief[1]

    def test_tao_lord_increases_loyalist_probability(self):
        """Test that using 桃 on 主公 increases P(忠臣)"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
            ]
        }

        initial_belief = encoder._belief_states[0][1].copy()

        # Player 1 uses 桃 on 主公
        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="tao",
            target_idx=0,
            game_state=game_state,
        )

        updated_belief = encoder._belief_states[0][1]

        # P(忠臣) should increase
        assert updated_belief[0] > initial_belief[0]

    def test_general_helpfulness_signal(self):
        """Test that general helpfulness signals slight 忠臣 tendency"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        initial_belief = encoder._belief_states[0][1].copy()

        # Player 1 buffs someone (not 主公)
        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="buff",
            target_idx=2,
            game_state={"players": []},
        )

        updated_belief = encoder._belief_states[0][1]

        # P(忠臣) should increase slightly
        assert updated_belief[0] > initial_belief[0]


class TestBeliefNormalization:
    """Test that beliefs are properly normalized"""

    def test_belief_probabilities_sum_to_one(self):
        """Test that belief probabilities sum to approximately 1"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
            ]
        }

        # Apply multiple updates
        for _ in range(5):
            encoder.update_identity_belief(
                observer_idx=0,
                actor_idx=1,
                action_type="attack",
                target_idx=0,
                game_state=game_state,
            )

        belief = encoder._belief_states[0][1]

        # Sum should be close to 1 (excluding small unknown)
        assert belief[0] + belief[1] + belief[2] <= 1.0
        assert belief[0] + belief[1] + belief[2] >= 0.99

    def test_belief_bounds_respected(self):
        """Test that beliefs stay within min/max bounds"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
            ]
        }

        # Apply many updates to push to bounds
        for _ in range(20):
            encoder.update_identity_belief(
                observer_idx=0,
                actor_idx=1,
                action_type="attack",
                target_idx=0,
                game_state=game_state,
            )

        belief = encoder._belief_states[0][1]

        # Check bounds (with tolerance for floating point)
        assert belief[0] >= encoder.BELIEF_MIN * 0.9  # Allow small tolerance
        assert belief[0] <= encoder.BELIEF_MAX * 1.1
        assert belief[1] >= encoder.BELIEF_MIN * 0.9
        assert belief[1] <= encoder.BELIEF_MAX * 1.1


class TestBeliefHistory:
    """Test belief history tracking"""

    def test_history_recorded_on_update(self):
        """Test that belief history is recorded"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
            ]
        }

        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="attack",
            target_idx=0,
            game_state=game_state,
        )

        history = encoder.get_belief_history(observer_idx=0)

        assert len(history) == 1
        assert history[0]["actor_idx"] == 1
        assert history[0]["action_type"] == "attack"
        assert history[0]["target_idx"] == 0
        assert "belief" in history[0]

    def test_history_temporal_ordering(self):
        """Test that history maintains temporal ordering"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {"players": [{"identity": "主公"}] + [{"identity": "未知"}] * 4}

        # Multiple updates
        encoder.update_identity_belief(0, 1, "attack", 0, game_state)
        encoder.update_identity_belief(0, 2, "heal", 0, game_state)
        encoder.update_identity_belief(0, 3, "attack", 1, game_state)

        history = encoder.get_belief_history(observer_idx=0)

        assert len(history) == 3
        assert history[0]["actor_idx"] == 1
        assert history[1]["actor_idx"] == 2
        assert history[2]["actor_idx"] == 3
        assert history[0]["timestamp"] == 0
        assert history[1]["timestamp"] == 1
        assert history[2]["timestamp"] == 2

    def test_get_belief_returns_current(self):
        """Test get_belief returns current belief"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
            ]
        }

        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="attack",
            target_idx=0,
            game_state=game_state,
        )

        belief = encoder.get_belief(observer_idx=0, target_idx=1)
        stored_belief = encoder._belief_states[0][1]

        np.testing.assert_array_equal(belief, stored_belief)

    def test_get_belief_default_for_unknown(self):
        """Test get_belief returns default for unknown targets"""
        encoder = StateEncoder()

        belief = encoder.get_belief(observer_idx=0, target_idx=1)
        expected = np.array([0.33, 0.33, 0.33, 0.01], dtype=np.float32)

        np.testing.assert_array_almost_equal(belief, expected, decimal=2)


class TestMultipleObservers:
    """Test belief tracking for multiple observers"""

    def test_separate_beliefs_per_observer(self):
        """Test that different observers maintain separate beliefs"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)
        encoder.reset_beliefs(player_num=5, observer_idx=1)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
                {"identity": "未知"},
            ]
        }

        # Observer 0 sees player 2 attack 主公
        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=2,
            action_type="attack",
            target_idx=0,
            game_state=game_state,
        )

        # Observer 1 sees player 2 heal 主公
        encoder.update_identity_belief(
            observer_idx=1,
            actor_idx=2,
            action_type="heal",
            target_idx=0,
            game_state=game_state,
        )

        belief_0 = encoder.get_belief(observer_idx=0, target_idx=2)
        belief_1 = encoder.get_belief(observer_idx=1, target_idx=2)

        # Observer 0 thinks player 2 is more likely rebel
        # Observer 1 thinks player 2 is more likely loyalist
        assert belief_0[1] > belief_1[1]  # P(反贼)
        assert belief_0[0] < belief_1[0]  # P(忠臣)


class TestClearBeliefs:
    """Test clearing belief states"""

    def test_clear_all_beliefs(self):
        """Test clearing all belief states"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        encoder.clear_beliefs()

        assert len(encoder._belief_states) == 0
        assert len(encoder._belief_history) == 0

    def test_clear_specific_observer(self):
        """Test clearing beliefs for specific observer only"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)
        encoder.reset_beliefs(player_num=5, observer_idx=1)

        encoder.clear_beliefs(observer_idx=0)

        assert 0 not in encoder._belief_states
        assert 0 not in encoder._belief_history
        assert 1 in encoder._belief_states
        assert 1 in encoder._belief_history


class TestLikelyRebelLoyalistHelpers:
    """Test helper methods for detecting likely rebels/loyalists"""

    def test_is_likely_rebel_true(self):
        """Test detecting likely rebel"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)
        encoder._belief_states[0][1] = np.array([0.2, 0.7, 0.09, 0.01])

        assert encoder._is_likely_rebel(observer_idx=0, target_idx=1, threshold=0.6)

    def test_is_likely_rebel_false(self):
        """Test non-rebel detection"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)
        encoder._belief_states[0][1] = np.array([0.5, 0.3, 0.19, 0.01])

        assert not encoder._is_likely_rebel(observer_idx=0, target_idx=1, threshold=0.6)

    def test_is_likely_loyalist_true(self):
        """Test detecting likely loyalist"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)
        encoder._belief_states[0][1] = np.array([0.7, 0.2, 0.09, 0.01])

        assert encoder._is_likely_loyalist(observer_idx=0, target_idx=1, threshold=0.6)


class TestIntegrationWithEncoding:
    """Test integration with state encoding"""

    def test_encoding_uses_learned_beliefs(self):
        """Test that state encoding uses learned belief states"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {
                    "identity": "主公",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "commander_name": "曹操",
                    "nation": "魏",
                    "skills": [],
                },
                {
                    "identity": "未知",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "commander_name": "刘备",
                    "nation": "蜀",
                    "skills": [],
                },
                {
                    "identity": "未知",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "commander_name": "孙权",
                    "nation": "吴",
                    "skills": [],
                },
            ]
        }

        # Update belief for player 1
        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="attack",
            target_idx=0,
            game_state=game_state,
        )

        # Encode state - should use updated beliefs
        encoded = encoder.encode(game_state, player_idx=0)

        # Beliefs should be encoded in the state vector
        # The encoding should complete without error
        assert len(encoded) > 0


class TestActionTypes:
    """Test various action types"""

    def test_duel_action(self):
        """Test that duel actions update beliefs"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
            ]
        }

        initial_belief = encoder._belief_states[0][1].copy()

        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="duel",
            target_idx=0,
            game_state=game_state,
        )

        updated_belief = encoder._belief_states[0][1]
        assert updated_belief[1] > initial_belief[1]  # P(反贼) increases

    def test_debuff_action(self):
        """Test that debuff actions update beliefs"""
        encoder = StateEncoder()
        encoder.reset_beliefs(player_num=5, observer_idx=0)

        game_state = {
            "players": [
                {"identity": "主公"},
                {"identity": "未知"},
            ]
        }

        initial_belief = encoder._belief_states[0][1].copy()

        encoder.update_identity_belief(
            observer_idx=0,
            actor_idx=1,
            action_type="debuff",
            target_idx=0,
            game_state=game_state,
        )

        updated_belief = encoder._belief_states[0][1]
        assert updated_belief[1] > initial_belief[1]  # P(反贼) increases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
