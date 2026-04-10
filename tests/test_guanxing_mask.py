"""
Tests for 观星 skill decision mask generation

观星 (Guanxing): Select ordering of cards from deck top
- Sequential selection: N steps for N cards
- Step 1: Select position for card 0 (5 options: positions 0-4)
- Step 2: Select position for card 1 (4 remaining options)
- Mask updates after each selection (removes filled positions)
- Handle variable card count (deck may have <5 cards)
"""

import pytest
import numpy as np
from unittest.mock import Mock

from ai.action_encoder import ActionEncoder, ActionConfig, ActionMaskGenerator
from ai.skill_decision import (
    SkillDecisionRequest,
    SkillDecisionType,
    create_select_order_request,
)


class TestGuanxingMask:
    """Test suite for _get_guanxing_mask method"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = ActionConfig()
        self.encoder = ActionEncoder(self.config)
        self.mask_generator = ActionMaskGenerator(self.encoder)

    def test_guanxing_mask_initial_step(self):
        """
        Test: Initial step of guanxing shows all positions available

        Given: 5 cards to order, step 0
        Expected: All 5 positions (0-4) are valid
        """
        # Create request with 5 cards
        cards = [Mock(name=f"card_{i}") for i in range(5)]
        request = create_select_order_request("观星", cards, "选择牌序")

        # Get mask for step 0
        mask = self.mask_generator._get_guanxing_mask(request, current_step=0)

        # Verify all 5 positions are valid
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 1.0
        assert mask[4] == 1.0
        # Position 5+ should be 0
        assert mask[5] == 0.0

    def test_guanxing_mask_after_first_selection(self):
        """
        Test: Mask excludes already selected position

        Given: 5 cards, after selecting position 2
        Expected: Positions 0,1,3,4 available; position 2 excluded
        """
        cards = [Mock(name=f"card_{i}") for i in range(5)]
        request = create_select_order_request("观星", cards, "选择牌序")

        # Simulate selecting position 2
        request.add_selection(2)

        # Get mask for step 1
        mask = self.mask_generator._get_guanxing_mask(request, current_step=1)

        # Verify position 2 is excluded
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 0.0  # Already selected
        assert mask[3] == 1.0
        assert mask[4] == 1.0

    def test_guanxing_mask_after_multiple_selections(self):
        """
        Test: Mask excludes multiple selected positions

        Given: 5 cards, after selecting positions 0 and 4
        Expected: Positions 1,2,3 available
        """
        cards = [Mock(name=f"card_{i}") for i in range(5)]
        request = create_select_order_request("观星", cards, "选择牌序")

        # Simulate selecting positions 0 and 4
        request.add_selection(0)
        request.add_selection(4)

        # Get mask for step 2
        mask = self.mask_generator._get_guanxing_mask(request, current_step=2)

        # Verify selected positions are excluded
        assert mask[0] == 0.0  # Already selected
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 1.0
        assert mask[4] == 0.0  # Already selected

    def test_guanxing_mask_fewer_cards(self):
        """
        Test: Handle variable card count (<5 cards)

        Given: 3 cards to order
        Expected: Only positions 0,1,2 are valid
        """
        cards = [Mock(name=f"card_{i}") for i in range(3)]
        request = create_select_order_request("观星", cards, "选择牌序")

        # Get mask for step 0
        mask = self.mask_generator._get_guanxing_mask(request, current_step=0)

        # Verify only 3 positions are valid
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 0.0  # No card 3
        assert mask[4] == 0.0  # No card 4

    def test_guanxing_mask_single_card(self):
        """
        Test: Handle minimum card count (1 card)

        Given: 1 card to order
        Expected: Only position 0 is valid
        """
        cards = [Mock(name="card_0")]
        request = create_select_order_request("观星", cards, "选择牌序")

        mask = self.mask_generator._get_guanxing_mask(request, current_step=0)

        assert mask[0] == 1.0
        assert mask[1] == 0.0
        assert mask[2] == 0.0

    def test_guanxing_mask_no_cards(self):
        """
        Test: Handle empty card list

        Given: 0 cards
        Expected: All positions invalid
        """
        cards = []
        request = create_select_order_request("观星", cards, "选择牌序")

        mask = self.mask_generator._get_guanxing_mask(request, current_step=0)

        # All positions should be 0
        assert mask.sum() == 0.0

    def test_guanxing_mask_step_exceeds_card_count(self):
        """
        Test: Handle step exceeding card count

        Given: 3 cards, requesting step 5
        Expected: All positions invalid
        """
        cards = [Mock(name=f"card_{i}") for i in range(3)]
        request = create_select_order_request("观星", cards, "选择牌序")

        mask = self.mask_generator._get_guanxing_mask(request, current_step=5)

        # All positions should be 0
        assert mask.sum() == 0.0

    def test_guanxing_mask_final_step(self):
        """
        Test: Final step with only one position remaining

        Given: 5 cards, 4 selected, step 4
        Expected: Only 1 position available
        """
        cards = [Mock(name=f"card_{i}") for i in range(5)]
        request = create_select_order_request("观星", cards, "选择牌序")

        # Select 4 positions
        request.add_selection(0)
        request.add_selection(1)
        request.add_selection(2)
        request.add_selection(3)

        # Get mask for final step (step 4)
        mask = self.mask_generator._get_guanxing_mask(request, current_step=4)

        # Only position 4 should be available
        assert mask[0] == 0.0
        assert mask[1] == 0.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0
        assert mask[4] == 1.0
        assert mask.sum() == 1.0

    def test_guanxing_mask_returns_correct_dtype(self):
        """
        Test: Mask returns correct dtype (float32)
        """
        cards = [Mock(name=f"card_{i}") for i in range(5)]
        request = create_select_order_request("观星", cards, "选择牌序")

        mask = self.mask_generator._get_guanxing_mask(request, current_step=0)

        assert mask.dtype == np.float32

    def test_guanxing_mask_returns_correct_shape(self):
        """
        Test: Mask returns correct shape (matches encoder card_dim)
        """
        cards = [Mock(name=f"card_{i}") for i in range(5)]
        request = create_select_order_request("观星", cards, "选择牌序")

        mask = self.mask_generator._get_guanxing_mask(request, current_step=0)

        assert len(mask) == self.encoder.card_dim

    def test_guanxing_selection_updates_remaining(self):
        """
        Test: Selection correctly updates remaining options

        Validates the interaction between add_selection and mask generation
        """
        cards = [Mock(name=f"card_{i}") for i in range(5)]
        request = create_select_order_request("观星", cards, "选择牌序")

        # Initially 5 options
        remaining = request.get_remaining_options()
        assert len(remaining) == 5

        # Select position 2
        request.add_selection(2)
        remaining = request.get_remaining_options()
        assert 2 not in remaining
        assert len(remaining) == 4

        # Mask should reflect this
        mask = self.mask_generator._get_guanxing_mask(request, current_step=1)
        assert mask[2] == 0.0

    def test_guanxing_complete_sequence(self):
        """
        Test: Complete selection sequence (simulating all 5 steps)

        Validates the full flow of sequential selection
        """
        cards = [Mock(name=f"card_{i}") for i in range(5)]
        request = create_select_order_request("观星", cards, "选择牌序")

        # Step 0: All 5 positions available
        mask = self.mask_generator._get_guanxing_mask(request, current_step=0)
        assert mask.sum() == 5.0

        # Select position 3
        request.add_selection(3)

        # Step 1: 4 positions available
        mask = self.mask_generator._get_guanxing_mask(request, current_step=1)
        assert mask.sum() == 4.0
        assert mask[3] == 0.0

        # Select position 0
        request.add_selection(0)

        # Step 2: 3 positions available
        mask = self.mask_generator._get_guanxing_mask(request, current_step=2)
        assert mask.sum() == 3.0
        assert mask[0] == 0.0
        assert mask[3] == 0.0

        # Select position 4
        request.add_selection(4)

        # Step 3: 2 positions available
        mask = self.mask_generator._get_guanxing_mask(request, current_step=3)
        assert mask.sum() == 2.0

        # Select position 1
        request.add_selection(1)

        # Step 4: 1 position available (position 2)
        mask = self.mask_generator._get_guanxing_mask(request, current_step=4)
        assert mask.sum() == 1.0
        assert mask[2] == 1.0

        # Final selection
        request.add_selection(2)

        # Verify complete
        assert request.is_complete()
        result = request.get_result()
        assert result == [3, 0, 4, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
