"""
Tests for Curriculum Learning System

Tests cover:
- Stage progression logic
- Win rate threshold detection
- Environment configuration
- Reward multiplier application
- State persistence
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.curriculum import (
    CurriculumStage,
    StageConfig,
    CurriculumConfig,
    StageProgress,
    CurriculumManager,
    create_curriculum_manager,
    get_stage_from_string,
)


class TestCurriculumStage:
    """Tests for CurriculumStage enum"""

    def test_stage_ordering(self):
        """Test that stages are ordered correctly"""
        assert CurriculumStage.EASY < CurriculumStage.MEDIUM
        assert CurriculumStage.MEDIUM < CurriculumStage.HARD
        assert int(CurriculumStage.EASY) == 1
        assert int(CurriculumStage.MEDIUM) == 2
        assert int(CurriculumStage.HARD) == 3

    def test_stage_int_conversion(self):
        """Test stage to int conversion"""
        assert int(CurriculumStage.EASY) == 1
        assert int(CurriculumStage.HARD) == 3


class TestStageConfig:
    """Tests for StageConfig dataclass"""

    def test_default_stage_config(self):
        """Test creating a stage config with defaults"""
        config = StageConfig(
            stage=CurriculumStage.EASY,
            max_rounds=5,
            opponent_policy="random",
            reward_multiplier=1.5,
            min_win_rate=0.40,
            min_episodes=100,
            target_win_rate=0.50,
        )

        assert config.stage == CurriculumStage.EASY
        assert config.max_rounds == 5
        assert config.opponent_policy == "random"
        assert config.reward_multiplier == 1.5
        assert config.min_win_rate == 0.40
        assert config.learning_rate is None
        assert config.ent_coef is None

    def test_stage_config_to_dict(self):
        """Test serialization to dictionary"""
        config = StageConfig(
            stage=CurriculumStage.MEDIUM,
            max_rounds=10,
            opponent_policy="rule",
            reward_multiplier=1.0,
            min_win_rate=0.45,
            min_episodes=200,
            target_win_rate=0.50,
            ent_coef=0.05,
        )

        d = config.to_dict()

        assert d["stage"] == "MEDIUM"
        assert d["stage_value"] == 2
        assert d["max_rounds"] == 10
        assert d["opponent_policy"] == "rule"
        assert d["reward_multiplier"] == 1.0
        assert d["ent_coef"] == 0.05


class TestCurriculumConfig:
    """Tests for CurriculumConfig"""

    def test_default_config(self):
        """Test default curriculum configuration"""
        config = CurriculumConfig()

        assert len(config.stages) == 3
        assert config.stages[0].stage == CurriculumStage.EASY
        assert config.stages[1].stage == CurriculumStage.MEDIUM
        assert config.stages[2].stage == CurriculumStage.HARD

    def test_stage_1_config(self):
        """Test Stage 1 (Easy) default configuration"""
        config = CurriculumConfig()
        easy = config.get_stage(CurriculumStage.EASY)

        assert easy is not None
        assert easy.max_rounds == 5
        assert easy.opponent_policy == "random"
        assert easy.reward_multiplier == 1.5
        assert easy.min_win_rate == 0.40

    def test_stage_2_config(self):
        """Test Stage 2 (Medium) default configuration"""
        config = CurriculumConfig()
        medium = config.get_stage(CurriculumStage.MEDIUM)

        assert medium is not None
        assert medium.max_rounds == 10
        assert medium.opponent_policy == "rule"
        assert medium.reward_multiplier == 1.0
        assert medium.min_win_rate == 0.45

    def test_stage_3_config(self):
        """Test Stage 3 (Hard) default configuration"""
        config = CurriculumConfig()
        hard = config.get_stage(CurriculumStage.HARD)

        assert hard is not None
        assert hard.max_rounds == 15
        assert hard.opponent_policy == "self_play"
        assert hard.reward_multiplier == 1.0
        assert hard.min_win_rate == 0.50

    def test_get_next_stage(self):
        """Test getting next stage"""
        config = CurriculumConfig()

        assert config.get_next_stage(CurriculumStage.EASY) == CurriculumStage.MEDIUM
        assert config.get_next_stage(CurriculumStage.MEDIUM) == CurriculumStage.HARD
        assert config.get_next_stage(CurriculumStage.HARD) is None

    def test_custom_stages(self):
        """Test custom stage configuration"""
        custom_stages = [
            StageConfig(
                stage=CurriculumStage.EASY,
                max_rounds=3,
                opponent_policy="random",
                reward_multiplier=2.0,
                min_win_rate=0.30,
                min_episodes=50,
                target_win_rate=0.40,
            ),
        ]

        config = CurriculumConfig(stages=custom_stages)

        assert len(config.stages) == 1
        assert config.stages[0].max_rounds == 3
        assert config.stages[0].reward_multiplier == 2.0


class TestStageProgress:
    """Tests for StageProgress tracking"""

    def test_initial_progress(self):
        """Test initial progress state"""
        progress = StageProgress(
            stage=CurriculumStage.EASY,
            start_time="2025-01-01T00:00:00",
        )

        assert progress.episodes_completed == 0
        assert progress.win_count == 0
        assert progress.loss_count == 0
        assert progress.current_win_rate == 0.0
        assert progress.recent_win_rate == 0.0

    def test_add_episode_result(self):
        """Test adding episode results"""
        progress = StageProgress(
            stage=CurriculumStage.EASY,
            start_time="2025-01-01T00:00:00",
        )

        # Add wins and losses
        progress.add_episode_result(True, 100)
        progress.add_episode_result(True, 120)
        progress.add_episode_result(False, 80)

        assert progress.episodes_completed == 3
        assert progress.win_count == 2
        assert progress.loss_count == 1
        assert progress.current_win_rate == 2 / 3
        assert progress.timesteps_completed == 300

    def test_recent_win_rate_window(self):
        """Test rolling window for recent win rate"""
        progress = StageProgress(
            stage=CurriculumStage.EASY,
            start_time="2025-01-01T00:00:00",
        )

        # Add 100 episodes (50% win rate)
        for i in range(100):
            progress.add_episode_result(i % 2 == 0, 100, window_size=50)

        assert progress.episodes_completed == 100
        assert progress.win_count == 50
        assert progress.loss_count == 50
        assert progress.current_win_rate == 0.5

        # Recent 50 should also be 50%
        assert len(progress.recent_wins) == 50
        assert progress.recent_win_rate == 0.5

    def test_to_dict(self):
        """Test progress serialization"""
        progress = StageProgress(
            stage=CurriculumStage.MEDIUM,
            start_time="2025-01-01T00:00:00",
        )

        progress.add_episode_result(True, 100)
        progress.add_episode_result(False, 100)

        d = progress.to_dict()

        assert d["stage"] == "MEDIUM"
        assert d["episodes_completed"] == 2
        assert d["win_count"] == 1
        assert d["win_rate"] == 0.5


class TestCurriculumManager:
    """Tests for CurriculumManager"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def curriculum_config(self, temp_log_dir):
        """Create test curriculum config"""
        return CurriculumConfig(
            log_dir=temp_log_dir,
            advancement_window=50,
            advancement_patience=2,
            stages=[
                StageConfig(
                    stage=CurriculumStage.EASY,
                    max_rounds=5,
                    opponent_policy="random",
                    reward_multiplier=1.5,
                    min_win_rate=0.40,
                    min_episodes=10,
                    target_win_rate=0.50,
                ),
                StageConfig(
                    stage=CurriculumStage.MEDIUM,
                    max_rounds=10,
                    opponent_policy="rule",
                    reward_multiplier=1.0,
                    min_win_rate=0.45,
                    min_episodes=10,
                    target_win_rate=0.50,
                ),
                StageConfig(
                    stage=CurriculumStage.HARD,
                    max_rounds=15,
                    opponent_policy="self_play",
                    reward_multiplier=1.0,
                    min_win_rate=0.50,
                    min_episodes=10,
                    target_win_rate=0.50,
                ),
            ],
        )

    def test_initialization(self, curriculum_config):
        """Test manager initialization"""
        manager = CurriculumManager(curriculum_config)

        assert manager.current_stage == CurriculumStage.EASY
        assert manager.stage_config.max_rounds == 5
        assert manager.is_completed is False

    def test_get_environment_config(self, curriculum_config):
        """Test environment configuration retrieval"""
        manager = CurriculumManager(curriculum_config)

        env_config = manager.get_environment_config()

        assert env_config["max_rounds"] == 5
        assert env_config["other_player_policy"] == "random"
        assert env_config["reward_multiplier"] == 1.5

    def test_get_reward_multiplier(self, curriculum_config):
        """Test reward multiplier"""
        manager = CurriculumManager(curriculum_config)

        # Stage 1: 1.5x multiplier
        assert manager.get_reward_multiplier() == 1.5

        # Advance to Stage 2
        manager.advance_stage("test")
        assert manager.get_reward_multiplier() == 1.0

    def test_get_modified_reward(self, curriculum_config):
        """Test reward modification"""
        manager = CurriculumManager(curriculum_config)

        # Stage 1: 1.5x multiplier
        modified = manager.get_modified_reward(10.0)
        assert modified == 15.0

        # Negative reward should also be scaled
        modified = manager.get_modified_reward(-10.0)
        assert modified == -15.0

    def test_record_episode(self, curriculum_config):
        """Test episode recording"""
        manager = CurriculumManager(curriculum_config)

        # Record some episodes
        for i in range(10):
            manager.record_episode(won=i % 2 == 0, reward=10.0, timesteps=100)

        progress = manager.progress[CurriculumStage.EASY]

        assert progress.episodes_completed == 10
        assert progress.win_count == 5
        assert progress.loss_count == 5
        assert manager.total_episodes == 10
        assert manager.total_timesteps == 1000

    def test_should_advance_stage_not_enough_episodes(self, curriculum_config):
        """Test advancement check with insufficient episodes"""
        manager = CurriculumManager(curriculum_config)

        # Only 5 episodes, need 10
        for i in range(5):
            manager.record_episode(won=True, reward=10.0)

        should_advance, reason = manager.should_advance_stage()

        assert should_advance is False
        assert "Need" in reason

    def test_should_advance_stage_win_rate_below_threshold(self, curriculum_config):
        """Test advancement with win rate below threshold"""
        manager = CurriculumManager(curriculum_config)

        # 10 episodes with 30% win rate (below 40% threshold)
        for i in range(10):
            manager.record_episode(won=(i < 3), reward=10.0)

        should_advance, reason = manager.should_advance_stage()

        assert should_advance is False
        assert "Win rate" in reason

    def test_should_advance_stage_meets_threshold(self, curriculum_config):
        """Test advancement with sufficient win rate"""
        manager = CurriculumManager(curriculum_config)

        # 10 episodes with 50% win rate (above 40% threshold)
        for i in range(10):
            manager.record_episode(won=(i < 5), reward=10.0)

        # First check - should not advance yet (patience=2)
        should_advance, _ = manager.should_advance_stage()
        assert should_advance is False

        # Second check - should now advance
        should_advance, reason = manager.should_advance_stage()
        assert should_advance is True

    def test_advance_stage(self, curriculum_config):
        """Test stage advancement"""
        manager = CurriculumManager(curriculum_config)

        # Force advancement
        advanced = manager.advance_stage("manual test")

        assert advanced is True
        assert manager.current_stage == CurriculumStage.MEDIUM
        assert manager.stage_config.max_rounds == 10
        assert len(manager.stage_transitions) == 1

    def test_advance_to_final_stage(self, curriculum_config):
        """Test reaching final stage"""
        manager = CurriculumManager(curriculum_config)

        # Advance to medium
        manager.advance_stage("test 1")
        assert manager.current_stage == CurriculumStage.MEDIUM

        # Advance to hard
        manager.advance_stage("test 2")
        assert manager.current_stage == CurriculumStage.HARD

        # Try to advance again
        advanced = manager.advance_stage("test 3")

        assert advanced is False
        assert manager.is_completed is True

    def test_regress_stage(self, curriculum_config):
        """Test stage regression"""
        manager = CurriculumManager(curriculum_config)

        # Advance to medium
        manager.advance_stage("test")
        assert manager.current_stage == CurriculumStage.MEDIUM

        # Regress back
        regressed = manager.regress_stage("performance drop")

        assert regressed is True
        assert manager.current_stage == CurriculumStage.EASY

    def test_regress_from_first_stage(self, curriculum_config):
        """Test regression from first stage"""
        manager = CurriculumManager(curriculum_config)

        # Can't regress from first stage
        regressed = manager.regress_stage("test")

        assert regressed is False
        assert manager.current_stage == CurriculumStage.EASY

    def test_set_stage_manual(self, curriculum_config):
        """Test manual stage setting"""
        manager = CurriculumManager(curriculum_config)

        manager.set_stage(CurriculumStage.HARD)

        assert manager.current_stage == CurriculumStage.HARD
        assert manager.stage_config.max_rounds == 15

    def test_save_and_load_state(self, curriculum_config, temp_log_dir):
        """Test state persistence"""
        manager = CurriculumManager(curriculum_config)

        # Record some episodes
        for i in range(5):
            manager.record_episode(won=True, reward=10.0)

        # Advance stage
        manager.advance_stage("test")

        # Save state
        state_path = os.path.join(temp_log_dir, "test_state.json")
        manager.save_state(state_path)

        # Create new manager and load state
        new_manager = CurriculumManager(curriculum_config)
        new_manager.load_state(state_path)

        assert new_manager.current_stage == CurriculumStage.MEDIUM
        assert new_manager.total_episodes == 5
        assert new_manager.total_timesteps == 5

    def test_get_summary(self, curriculum_config):
        """Test summary generation"""
        manager = CurriculumManager(curriculum_config)

        manager.record_episode(won=True, reward=10.0)
        manager.record_episode(won=False, reward=-5.0)

        summary = manager.get_summary()

        assert "EASY" in summary
        assert "Win Rate" in summary
        # Format is "50.00%" with two decimal places
        assert "50.00%" in summary


class TestEnvironmentConfiguration:
    """Tests for environment configuration based on stage"""

    def test_easy_stage_env_config(self):
        """Test environment config for Stage 1"""
        config = CurriculumConfig()
        manager = CurriculumManager(config)

        env_config = manager.get_environment_config()

        assert env_config["max_rounds"] == 5
        assert env_config["other_player_policy"] == "random"

    def test_medium_stage_env_config(self):
        """Test environment config for Stage 2"""
        config = CurriculumConfig()
        manager = CurriculumManager(config, start_stage=CurriculumStage.MEDIUM)

        env_config = manager.get_environment_config()

        assert env_config["max_rounds"] == 10
        assert env_config["other_player_policy"] == "rule"

    def test_hard_stage_env_config(self):
        """Test environment config for Stage 3"""
        config = CurriculumConfig()
        manager = CurriculumManager(config, start_stage=CurriculumStage.HARD)

        env_config = manager.get_environment_config()

        assert env_config["max_rounds"] == 15
        # self_play maps to "none" (no external AI)
        assert env_config["other_player_policy"] == "none"


class TestRewardMultiplier:
    """Tests for reward multiplier functionality"""

    def test_stage_1_reward_boost(self):
        """Test 1.5x reward multiplier in Stage 1"""
        config = CurriculumConfig()
        manager = CurriculumManager(config)

        base_reward = 10.0
        modified = manager.get_modified_reward(base_reward)

        assert modified == 15.0

    def test_stage_2_no_boost(self):
        """Test 1.0x multiplier in Stage 2"""
        config = CurriculumConfig()
        manager = CurriculumManager(config)
        manager.set_stage(CurriculumStage.MEDIUM)

        base_reward = 10.0
        modified = manager.get_modified_reward(base_reward)

        assert modified == 10.0

    def test_negative_reward_scaling(self):
        """Test that negative rewards are scaled correctly"""
        config = CurriculumConfig()
        manager = CurriculumManager(config)

        # Stage 1: 1.5x
        assert manager.get_modified_reward(-10.0) == -15.0

        # Stage 2: 1.0x
        manager.set_stage(CurriculumStage.MEDIUM)
        assert manager.get_modified_reward(-10.0) == -10.0

    def test_terminal_reward_scaling(self):
        """Test that terminal rewards (victory/defeat) are scaled"""
        config = CurriculumConfig()
        manager = CurriculumManager(config)

        # Victory reward scaled in Stage 1
        victory_reward = 100.0
        assert manager.get_modified_reward(victory_reward) == 150.0

        # Defeat reward scaled
        defeat_reward = -100.0
        assert manager.get_modified_reward(defeat_reward) == -150.0


class TestStageTransitions:
    """Tests for stage transition logic"""

    @pytest.fixture
    def manager(self):
        """Create manager for transition tests"""
        config = CurriculumConfig(
            advancement_patience=1,  # Single check required
            stages=[
                StageConfig(
                    stage=CurriculumStage.EASY,
                    max_rounds=5,
                    opponent_policy="random",
                    reward_multiplier=1.5,
                    min_win_rate=0.40,
                    min_episodes=5,
                    target_win_rate=0.50,
                ),
                StageConfig(
                    stage=CurriculumStage.MEDIUM,
                    max_rounds=10,
                    opponent_policy="rule",
                    reward_multiplier=1.0,
                    min_win_rate=0.45,
                    min_episodes=5,
                    target_win_rate=0.50,
                ),
                StageConfig(
                    stage=CurriculumStage.HARD,
                    max_rounds=15,
                    opponent_policy="self_play",
                    reward_multiplier=1.0,
                    min_win_rate=0.50,
                    min_episodes=5,
                    target_win_rate=0.50,
                ),
            ],
        )
        return CurriculumManager(config)

    def test_transition_logging(self, manager):
        """Test that transitions are logged"""
        manager.advance_stage("test reason")

        assert len(manager.stage_transitions) == 1
        transition = manager.stage_transitions[0]

        assert transition["from_stage"] == "EASY"
        assert transition["to_stage"] == "MEDIUM"
        assert "test reason" in transition["reason"]

    def test_transition_updates_config(self, manager):
        """Test that transition updates stage config"""
        initial_max_rounds = manager.stage_config.max_rounds

        manager.advance_stage("test")

        assert manager.stage_config.max_rounds != initial_max_rounds
        assert manager.stage_config.max_rounds == 10

    def test_transition_resets_progress(self, manager):
        """Test that new stage gets fresh progress tracking"""
        # Record episodes in Stage 1
        for i in range(5):
            manager.record_episode(won=True, reward=10.0)

        initial_progress = manager.progress[CurriculumStage.EASY].episodes_completed

        # Advance
        manager.advance_stage("test")

        # Check new stage progress
        new_progress = manager.progress[CurriculumStage.MEDIUM]

        assert new_progress.episodes_completed == 0
        assert (
            manager.progress[CurriculumStage.EASY].episodes_completed
            == initial_progress
        )


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_get_stage_from_string(self):
        """Test stage name parsing"""
        assert get_stage_from_string("easy") == CurriculumStage.EASY
        assert get_stage_from_string("EASY") == CurriculumStage.EASY
        assert get_stage_from_string("1") == CurriculumStage.EASY

        assert get_stage_from_string("medium") == CurriculumStage.MEDIUM
        assert get_stage_from_string("2") == CurriculumStage.MEDIUM

        assert get_stage_from_string("hard") == CurriculumStage.HARD
        assert get_stage_from_string("3") == CurriculumStage.HARD

        # Unknown returns EASY
        assert get_stage_from_string("unknown") == CurriculumStage.EASY

    def test_create_curriculum_manager(self):
        """Test factory function"""
        manager = create_curriculum_manager(start_stage=CurriculumStage.MEDIUM)

        assert manager.current_stage == CurriculumStage.MEDIUM


class TestCurriculumIntegration:
    """Integration tests for curriculum system"""

    def test_full_curriculum_progression(self):
        """Test complete progression through all stages"""
        config = CurriculumConfig(
            advancement_patience=1,
            total_timesteps_per_stage=1000,
            stages=[
                StageConfig(
                    stage=CurriculumStage.EASY,
                    max_rounds=5,
                    opponent_policy="random",
                    reward_multiplier=1.5,
                    min_win_rate=0.40,
                    min_episodes=10,
                    target_win_rate=0.50,
                ),
                StageConfig(
                    stage=CurriculumStage.MEDIUM,
                    max_rounds=10,
                    opponent_policy="rule",
                    reward_multiplier=1.0,
                    min_win_rate=0.45,
                    min_episodes=10,
                    target_win_rate=0.50,
                ),
                StageConfig(
                    stage=CurriculumStage.HARD,
                    max_rounds=15,
                    opponent_policy="self_play",
                    reward_multiplier=1.0,
                    min_win_rate=0.50,
                    min_episodes=10,
                    target_win_rate=0.50,
                ),
            ],
        )

        manager = CurriculumManager(config)

        # Simulate training progression
        stages_completed = []

        while not manager.is_completed:
            # Record episodes
            for i in range(10):
                # Simulate varying win rates based on stage
                if manager.current_stage == CurriculumStage.EASY:
                    won = i < 5  # 50% win rate
                elif manager.current_stage == CurriculumStage.MEDIUM:
                    won = i < 5  # 50% win rate
                else:
                    won = i < 5  # 50% win rate

                manager.record_episode(won=won, reward=10.0)

            # Check for advancement
            should_advance, _ = manager.should_advance_stage()
            if should_advance:
                stages_completed.append(manager.current_stage)
                manager.advance_stage("win rate threshold met")

        assert manager.is_completed
        assert len(stages_completed) == 3  # All stages completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
