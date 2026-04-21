#!/usr/bin/env python
"""Unit tests for Cython core functions."""

import numpy as np
import pytest


class TestCalculateDistance:
    """Tests for calculate_distance_fast function."""

    def test_same_player_returns_zero(self):
        from engine.cython_distance import calculate_distance_fast

        assert calculate_distance_fast(0, 0, 5, False, False) == 0
        assert calculate_distance_fast(2, 2, 8, True, True) == 0

    def test_adjacent_players_distance_one(self):
        from engine.cython_distance import calculate_distance_fast

        assert calculate_distance_fast(0, 1, 5, False, False) == 1
        assert calculate_distance_fast(1, 2, 5, False, False) == 1
        assert calculate_distance_fast(4, 0, 5, False, False) == 1

    def test_distance_bidirectional_min(self):
        from engine.cython_distance import calculate_distance_fast

        assert calculate_distance_fast(0, 2, 5, False, False) == 2
        assert calculate_distance_fast(2, 0, 5, False, False) == 2
        assert calculate_distance_fast(0, 3, 5, False, False) == 2
        assert calculate_distance_fast(3, 0, 5, False, False) == 2

    def test_attack_horse_reduces_distance(self):
        from engine.cython_distance import calculate_distance_fast

        assert calculate_distance_fast(0, 2, 5, True, False) == 1
        assert calculate_distance_fast(0, 3, 5, True, False) == 1

    def test_defense_horse_increases_distance(self):
        from engine.cython_distance import calculate_distance_fast

        assert calculate_distance_fast(0, 2, 5, False, True) == 3
        assert calculate_distance_fast(0, 1, 5, False, True) == 2

    def test_both_horses(self):
        from engine.cython_distance import calculate_distance_fast

        assert calculate_distance_fast(0, 2, 5, True, True) == 2
        assert calculate_distance_fast(0, 3, 5, True, True) == 2

    def test_attack_horse_min_distance_one(self):
        from engine.cython_distance import calculate_distance_fast

        assert calculate_distance_fast(0, 1, 5, True, False) == 1


class TestIsInRange:
    """Tests for is_in_range_fast function."""

    def test_adjacent_in_range_one(self):
        from engine.cython_distance import is_in_range_fast

        assert is_in_range_fast(0, 1, 5, 1, False, False) == True
        assert is_in_range_fast(0, 2, 5, 1, False, False) == False

    def test_range_two(self):
        from engine.cython_distance import is_in_range_fast

        assert is_in_range_fast(0, 2, 5, 2, False, False) == True
        assert is_in_range_fast(0, 3, 5, 2, False, False) == True
        assert is_in_range_fast(0, 4, 5, 2, False, False) == True

    def test_attack_house_extends_range(self):
        from engine.cython_distance import is_in_range_fast

        assert is_in_range_fast(0, 2, 5, 1, True, False) == True
        assert is_in_range_fast(0, 3, 5, 1, True, False) == True

    def test_defense_horse_reduces_effective_range(self):
        from engine.cython_distance import is_in_range_fast

        assert is_in_range_fast(0, 2, 5, 2, False, True) == False
        assert is_in_range_fast(0, 2, 5, 3, False, True) == True


class TestCheckVictory:
    """Tests for check_victory_fast function."""

    def test_lord_dead_ends_game(self):
        from engine.cython_distance import check_victory_fast

        alive = np.array([1, 2, 3], dtype=np.int32)
        assert check_victory_fast(alive, 5) == True

    def test_no_rebels_or_spies_lord_wins(self):
        from engine.cython_distance import check_victory_fast

        alive = np.array([0, 1, 1], dtype=np.int32)
        assert check_victory_fast(alive, 5) == True

    def test_game_continues_with_rebels(self):
        from engine.cython_distance import check_victory_fast

        alive = np.array([0, 1, 2], dtype=np.int32)
        assert check_victory_fast(alive, 5) == False

    def test_game_continues_with_spy(self):
        from engine.cython_distance import check_victory_fast

        alive = np.array([0, 3], dtype=np.int32)
        assert check_victory_fast(alive, 5) == False


class TestEstimateThreat:
    """Tests for estimate_threat_fast function."""

    def test_dead_player_zero_threat(self):
        from engine.cython_distance import estimate_threat_fast

        assert estimate_threat_fast(0, 4, 0, False) == 0.0

    def test_full_hp_low_threat(self):
        from engine.cython_distance import estimate_threat_fast

        threat = estimate_threat_fast(4, 4, 0, True)
        assert threat == pytest.approx(0.0, abs=0.01)

    def test_low_hp_high_threat(self):
        from engine.cython_distance import estimate_threat_fast

        threat = estimate_threat_fast(1, 4, 0, True)
        assert threat >= 0.7

    def test_hand_cards_add_threat(self):
        from engine.cython_distance import estimate_threat_fast

        threat_no_cards = estimate_threat_fast(4, 4, 0, True)
        threat_with_cards = estimate_threat_fast(4, 4, 5, True)
        assert threat_with_cards > threat_no_cards


class TestGenerateRangeMask:
    """Tests for generate_range_mask_fast function."""

    def test_mask_shape(self):
        from engine.cython_distance import generate_range_mask_fast

        is_alive = np.array([1, 1, 1, 1, 1], dtype=np.uint8)
        attack_horse = np.zeros(5, dtype=np.uint8)
        defense_horse = np.zeros(5, dtype=np.uint8)

        mask = generate_range_mask_fast(0, 5, 1, is_alive, attack_horse, defense_horse)
        assert mask.shape == (5,)

    def test_adjacent_only_in_range_one(self):
        from engine.cython_distance import generate_range_mask_fast

        is_alive = np.array([1, 1, 1, 1, 1], dtype=np.uint8)
        attack_horse = np.zeros(5, dtype=np.uint8)
        defense_horse = np.zeros(5, dtype=np.uint8)

        mask = generate_range_mask_fast(0, 5, 1, is_alive, attack_horse, defense_horse)
        assert mask[1] == 1
        assert mask[4] == 1
        assert mask[2] == 0
        assert mask[3] == 0

    def test_dead_players_not_masked(self):
        from engine.cython_distance import generate_range_mask_fast

        is_alive = np.array([1, 1, 0, 1, 1], dtype=np.uint8)
        attack_horse = np.zeros(5, dtype=np.uint8)
        defense_horse = np.zeros(5, dtype=np.uint8)

        mask = generate_range_mask_fast(0, 5, 2, is_alive, attack_horse, defense_horse)
        assert mask[2] == 0

    def test_self_not_masked(self):
        from engine.cython_distance import generate_range_mask_fast

        is_alive = np.array([1, 1, 1, 1, 1], dtype=np.uint8)
        attack_horse = np.zeros(5, dtype=np.uint8)
        defense_horse = np.zeros(5, dtype=np.uint8)

        mask = generate_range_mask_fast(0, 5, 1, is_alive, attack_horse, defense_horse)
        assert mask[0] == 0


class TestCythonAvailability:
    """Tests for Cython module availability check."""

    def test_cython_available(self):
        from engine.cython_distance import is_cython_available

        result = is_cython_available()
        assert isinstance(result, bool)

    def test_fallback_works(self):
        from engine.cython_fallback import calculate_distance_fast as fallback_dist

        assert fallback_dist(0, 2, 5, True, False) == 1

    def test_cython_matches_fallback(self):
        from engine.cython_distance import calculate_distance_fast as cython_dist
        from engine.cython_fallback import calculate_distance_fast as fallback_dist

        assert cython_dist(0, 2, 5, True, False) == fallback_dist(0, 2, 5, True, False)
        assert cython_dist(0, 3, 5, False, True) == fallback_dist(0, 3, 5, False, True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
