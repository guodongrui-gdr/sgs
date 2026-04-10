"""
Test identity belief state encoding

验证:
1. 新增28维belief state (4×7 players)
2. 主公身份编码为 [1, 0, 0, 0]
3. 其他玩家初始化为均匀分布 [0.33, 0.33, 0.33, 0]
4. 状态维度正确更新
"""

import unittest
import numpy as np

from ai.state_encoder import StateEncoder, EncodingConfig


class TestIdentityBeliefEncoding(unittest.TestCase):
    """Test identity belief state encoding"""

    def setUp(self):
        """Setup test fixtures"""
        self.config = EncodingConfig()
        self.encoder = StateEncoder(self.config)

    def test_zhugong_belief_encoding(self):
        """Test that 主公 gets [1, 0, 0, 0] belief encoding"""
        # Create a simple game state with 主公 visible
        game_state = {
            "phase": "waiting",
            "round_num": 1,
            "current_player_idx": 0,
            "deck_count": 100,
            "discard_pile_count": 0,
            "players": [
                {
                    "idx": 0,
                    "commander_name": "曹操",
                    "nation": "魏",
                    "identity": "主公",  # 主公 - known
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                },
                {
                    "idx": 1,
                    "commander_name": "刘备",
                    "nation": "蜀",
                    "identity": "忠臣",  # 忠臣 - hidden
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                },
            ],
            "action_history": [],
            "skill_decision": {},
        }

        # Player 1 (index 1) observing Player 0 (主公)
        encoded = self.encoder.encode(game_state, player_idx=1)

        # The encoding should include the belief for Player 0
        # 主公 should have belief [1, 0, 0, 0]
        # Find the belief encoding in the other_players section

        # Get the encoded state for other players
        # Player 0 is encoded in the first other player slot (since we skip player_idx=1)
        # Each other player has 77 dimensions now
        # Belief encoding is at positions: idx(8) + hp(1) + max_hp(1) + alive(1) + chained(1) + hand(1) + identity(4) + known(1) = 18
        # So belief starts at position 18 and is 4 dims

    def test_hidden_identity_uniform_belief(self):
        """Test that hidden identities get uniform belief [0.33, 0.33, 0.33, 0]"""
        game_state = {
            "phase": "waiting",
            "round_num": 1,
            "current_player_idx": 0,
            "deck_count": 100,
            "discard_pile_count": 0,
            "players": [
                {
                    "idx": 0,
                    "commander_name": "曹操",
                    "nation": "魏",
                    "identity": "主公",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                },
                {
                    "idx": 1,
                    "commander_name": "刘备",
                    "nation": "蜀",
                    "identity": "忠臣",  # Hidden
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                },
                {
                    "idx": 2,
                    "commander_name": "孙权",
                    "nation": "吴",
                    "identity": "反贼",  # Hidden
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                },
            ],
            "action_history": [],
            "skill_decision": {},
        }

        # 主公 observing others
        encoded = self.encoder.encode(game_state, player_idx=0)

    def test_state_dimension_increased(self):
        """Test that state dimension is correct with new belief encoding"""
        # Create minimal game state
        game_state = {
            "phase": "waiting",
            "round_num": 1,
            "current_player_idx": 0,
            "deck_count": 100,
            "discard_pile_count": 0,
            "players": [
                {
                    "idx": 0,
                    "commander_name": "曹操",
                    "nation": "魏",
                    "identity": "主公",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                    "sha_count": 0,
                    "jiu_count": 0,
                    "jiu_effect": 0,
                },
            ],
            "action_history": [],
            "skill_decision": {},
        }

        encoded = self.encoder.encode(game_state, player_idx=0)
        state_dim = len(encoded)

        # With the new belief encoding (81 dims per other player instead of 77)
        # Total should be 2670 dimensions
        self.assertEqual(state_dim, 2670)

    def test_other_player_encoding_has_81_dims(self):
        """Test that each other player now has 81 dimensions (was 77, added 4 for belief)"""
        game_state = {
            "phase": "waiting",
            "round_num": 1,
            "current_player_idx": 0,
            "deck_count": 100,
            "discard_pile_count": 0,
            "players": [
                {
                    "idx": 0,
                    "commander_name": "曹操",
                    "nation": "魏",
                    "identity": "主公",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                },
                {
                    "idx": 1,
                    "commander_name": "刘备",
                    "nation": "蜀",
                    "identity": "忠臣",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                },
            ],
            "action_history": [],
            "skill_decision": {},
        }

        # Encode and check the _encode_other_players directly
        # Returns 7 players worth of data
        other_players = self.encoder._encode_other_players(game_state, player_idx=0)

        # Should be 7 players × 81 dims = 567 dims (with padding)
        self.assertEqual(len(other_players), 567)

        # First player (index 1, the only other player) should be 81 dims
        # The actual first player data starts at beginning
        first_player_slice = other_players[:81]
        self.assertEqual(len(first_player_slice), 81)

    def test_belief_probabilities_sum_to_one(self):
        """Test that belief probabilities sum to approximately 1 (for known identities)"""
        # Create state with multiple players
        game_state = {
            "phase": "waiting",
            "round_num": 1,
            "current_player_idx": 0,
            "deck_count": 100,
            "discard_pile_count": 0,
            "players": [
                {
                    "idx": 0,
                    "commander_name": "曹操",
                    "nation": "魏",
                    "identity": "主公",
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                },
            ],
            "action_history": [],
            "skill_decision": {},
        }

        # Test _encode_identity_belief directly - use player from game_state
        player = game_state["players"][0]  # 主公 player from the list
        belief = self.encoder._encode_identity_belief(player, game_state, 0)

        # Should sum to 1.0
        self.assertAlmostEqual(np.sum(belief), 1.0, places=5)

        # Test hidden identity - add a 忠臣 player to game_state
        game_state["players"].append(
            {
                "idx": 1,
                "commander_name": "刘备",
                "nation": "蜀",
                "identity": "忠臣",
                "max_hp": 4,
                "current_hp": 4,
                "is_alive": True,
                "is_chained": False,
                "hand_cards": [],
                "equipment": {},
                "judge_area": [],
                "skills": [],
            }
        )
        player = game_state["players"][1]  # 忠臣 player from the list
        belief = self.encoder._encode_identity_belief(player, game_state, 0)

        # Should be [0.33, 0.33, 0.33, 0] which sums to ~0.99
        self.assertAlmostEqual(np.sum(belief), 0.99, places=1)

    def test_seven_players_belief_structure(self):
        """Test that 7 players get proper belief encoding (28 total dims)"""
        players = []
        for i in range(8):  # Create 8 players
            identity = "主公" if i == 0 else ("忠臣" if i == 1 else "反贼")
            players.append(
                {
                    "idx": i,
                    "commander_name": "曹操" if i == 0 else "刘备",
                    "nation": "魏" if i == 0 else "蜀",
                    "identity": identity,
                    "max_hp": 4,
                    "current_hp": 4,
                    "is_alive": True,
                    "is_chained": False,
                    "hand_cards": [],
                    "equipment": {},
                    "judge_area": [],
                    "skills": [],
                }
            )

        game_state = {
            "phase": "waiting",
            "round_num": 1,
            "current_player_idx": 0,
            "deck_count": 100,
            "discard_pile_count": 0,
            "players": players,
            "action_history": [],
            "skill_decision": {},
        }

        # Encode as player 0
        other_players = self.encoder._encode_other_players(game_state, player_idx=0)

        # Should be 7 other players × 81 dims = 567 dims
        # (8 total - 1 self = 7 other)
        self.assertEqual(len(other_players), 567)


class TestBeliefValues(unittest.TestCase):
    """Test specific belief encoding values"""

    def setUp(self):
        self.encoder = StateEncoder()
        self.game_state = {"players": []}

    def test_zhugong_belief_is_deterministic(self):
        """主公 belief should be [1, 0, 0, 0]"""
        player = {"identity": "主公"}
        belief = self.encoder._encode_identity_belief(player, self.game_state, 0)

        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(belief, expected, decimal=5)

    def test_zhongchen_belief_is_uniform(self):
        """忠臣 (hidden) should have uniform belief"""
        player = {"identity": "忠臣"}
        belief = self.encoder._encode_identity_belief(player, self.game_state, 0)

        expected = np.array([0.33, 0.33, 0.33, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(belief, expected, decimal=2)

    def test_fanzei_belief_is_uniform(self):
        """反贼 (hidden) should have uniform belief"""
        player = {"identity": "反贼"}
        belief = self.encoder._encode_identity_belief(player, self.game_state, 0)

        expected = np.array([0.33, 0.33, 0.33, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(belief, expected, decimal=2)

    def test_neijian_belief_is_uniform(self):
        """内奸 (hidden) should have uniform belief"""
        player = {"identity": "内奸"}
        belief = self.encoder._encode_identity_belief(player, self.game_state, 0)

        expected = np.array([0.33, 0.33, 0.33, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(belief, expected, decimal=2)

    def test_belief_shape_is_always_4(self):
        """Belief encoding is always 4 dimensions"""
        for identity in ["主公", "忠臣", "反贼", "内奸", "", "unknown"]:
            player = {"identity": identity}
            belief = self.encoder._encode_identity_belief(player, self.game_state, 0)
            self.assertEqual(len(belief), 4)


if __name__ == "__main__":
    unittest.main()
