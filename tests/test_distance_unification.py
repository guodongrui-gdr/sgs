#!/usr/bin/env python
"""Integration tests for unified distance calculation."""

import pytest
import numpy as np


class MockPlayer:
    """Mock player for testing distance calculation."""

    def __init__(self, idx, equipment=None):
        self.idx = idx
        self.equipment = equipment or {"进攻坐骑": None, "防御坐骑": None}
        self.next_player = None
        self.prev_player = None
        self.is_alive = True
        self.attack_range = 1


class TestCalculateDistanceFromPlayers:
    """Tests for calculate_distance_from_players function."""

    def test_same_player_returns_zero(self):
        from engine.cython_distance import calculate_distance_from_players

        p = MockPlayer(1)
        assert calculate_distance_from_players(p, p, 5) == 0

    def test_adjacent_players_distance_one(self):
        from engine.cython_distance import calculate_distance_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        assert calculate_distance_from_players(players[0], players[1], 5) == 1
        assert calculate_distance_from_players(players[4], players[0], 5) == 1

    def test_bidirectional_minimum(self):
        from engine.cython_distance import calculate_distance_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        assert calculate_distance_from_players(players[0], players[2], 5) == 2
        assert calculate_distance_from_players(players[2], players[0], 5) == 2
        assert calculate_distance_from_players(players[0], players[3], 5) == 2
        assert calculate_distance_from_players(players[3], players[0], 5) == 2

    def test_with_attack_horse(self):
        from engine.cython_distance import calculate_distance_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        players[0].equipment = {"进攻坐骑": "马", "防御坐骑": None}

        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        assert calculate_distance_from_players(players[0], players[2], 5) == 1
        assert calculate_distance_from_players(players[0], players[3], 5) == 1

    def test_with_defense_horse(self):
        from engine.cython_distance import calculate_distance_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        players[2].equipment = {"进攻坐骑": None, "防御坐骑": "马"}

        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        assert calculate_distance_from_players(players[0], players[2], 5) == 3

    def test_both_horses(self):
        from engine.cython_distance import calculate_distance_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        players[0].equipment = {"进攻坐骑": "马", "防御坐骑": None}
        players[2].equipment = {"进攻坐骑": None, "防御坐骑": "马"}

        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        assert calculate_distance_from_players(players[0], players[2], 5) == 2

    def test_matches_linked_list_implementation(self):
        from engine.cython_distance import calculate_distance_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        players[0].equipment = {"进攻坐骑": "马", "防御坐骑": None}
        players[3].equipment = {"进攻坐骑": None, "防御坐骑": "马"}

        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        def linked_list_distance(source, target):
            if source == target:
                return 0
            dist = 1
            current = source.next_player
            while current != target:
                dist += 1
                current = current.next_player
            reverse_dist = 1
            current = source.prev_player
            while current != target:
                reverse_dist += 1
                current = current.prev_player
            dist = min(dist, reverse_dist)
            if source.equipment.get("进攻坐骑"):
                dist = max(1, dist - 1)
            if target.equipment.get("防御坐骑"):
                dist += 1
            return dist

        for src in players:
            for tgt in players:
                if src != tgt:
                    expected = linked_list_distance(src, tgt)
                    actual = calculate_distance_from_players(src, tgt, 5)
                    assert actual == expected, (
                        f"src={src.idx}, tgt={tgt.idx}: expected {expected}, got {actual}"
                    )


class TestIsInRangeFromPlayers:
    """Tests for is_in_range_from_players function."""

    def test_range_1_adjacent_only(self):
        from engine.cython_distance import is_in_range_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        players[0].attack_range = 1

        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        assert is_in_range_from_players(players[0], players[1], 5) == True
        assert is_in_range_from_players(players[0], players[2], 5) == False

    def test_range_2_plus_attack_horse(self):
        from engine.cython_distance import is_in_range_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        players[0].attack_range = 1
        players[0].equipment = {"进攻坐骑": "马", "防御坐骑": None}

        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        assert is_in_range_from_players(players[0], players[2], 5) == True

    def test_defense_horse_effective_range(self):
        from engine.cython_distance import is_in_range_from_players

        players = [MockPlayer(i) for i in range(1, 6)]
        players[0].attack_range = 2
        players[2].equipment = {"进攻坐骑": None, "防御坐骑": "马"}

        for i in range(5):
            players[i].next_player = players[(i + 1) % 5]
            players[i].prev_player = players[(i - 1) % 5]

        assert is_in_range_from_players(players[0], players[2], 5) == False
        assert is_in_range_from_players(players[0], players[1], 5) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
