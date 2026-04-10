"""
Test configuration classes for the new training system.
TDD Phase: RED - Tests define expected behavior before implementation.
"""

import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class SamplingDistribution(Enum):
    """Sampling distribution for self-play opponent selection."""

    UNIFORM = "uniform"
    ELO_WEIGHTED = "elo_weighted"
    RECENT_WEIGHTED = "recent_weighted"
    HYBRID = "hybrid"


# Expected 43 skills from skills/__init__.py
ALL_SKILLS = [
    # Wei (9)
    "JianXiong",
    "Guicai",
    "Fankui",
    "Ganglie",
    "Tuxi",
    "Luoyi",
    "Tiandu",
    "Yiji",
    "Luoshen",
    "Qingguo",
    # Shu (10)
    "Rende",
    "Wusheng",
    "Paoxiao",
    "Guanxing",
    "Kongcheng",
    "Longdan",
    "Mashu",
    "Tieqi",
    "Jizhi",
    # Wu (13)
    "Zhiheng",
    "Qixi",
    "Keji",
    "Kurou",
    "Yingzi",
    "Fanjian",
    "Guose",
    "Liuli",
    "Qianxun",
    "Lianying",
    "Jieyin",
    "Xiaoji",
    # Qun (5)
    "Jiuji",
    "Qingnang",
    "Wushuang",
    "Lijian",
    "Biyue",
]


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_training_config_exists(self):
        """TrainingConfig should exist and be importable."""
        from train.config import TrainingConfig

        assert TrainingConfig is not None

    def test_training_config_defaults(self):
        """TrainingConfig should have sensible defaults."""
        from train.config import TrainingConfig

        config = TrainingConfig()

        assert config.timesteps == 1_000_000
        assert config.n_envs == 8
        assert config.checkpoint_freq == 100_000
        assert config.eval_freq == 50_000
        assert config.pool_size == 10

    def test_training_config_custom_values(self):
        """TrainingConfig should accept custom values."""
        from train.config import TrainingConfig

        config = TrainingConfig(
            timesteps=500_000,
            n_envs=4,
            checkpoint_freq=50_000,
            eval_freq=25_000,
            pool_size=5,
        )

        assert config.timesteps == 500_000
        assert config.n_envs == 4
        assert config.checkpoint_freq == 50_000
        assert config.eval_freq == 25_000
        assert config.pool_size == 5

    def test_training_config_field_access(self):
        """All fields should be accessible as attributes."""
        from train.config import TrainingConfig

        config = TrainingConfig()

        # Verify all expected fields exist
        _ = config.timesteps
        _ = config.n_envs
        _ = config.checkpoint_freq
        _ = config.eval_freq
        _ = config.pool_size


class TestSelfPlayConfig:
    """Tests for SelfPlayConfig dataclass."""

    def test_self_play_config_exists(self):
        """SelfPlayConfig should exist and be importable."""
        from train.config import SelfPlayConfig

        assert SelfPlayConfig is not None

    def test_self_play_config_defaults(self):
        """SelfPlayConfig should have sensible defaults."""
        from train.config import SelfPlayConfig

        config = SelfPlayConfig()

        assert config.sampling_distribution == SamplingDistribution.HYBRID
        assert config.elo_init == 1000.0

    def test_self_play_config_custom_values(self):
        """SelfPlayConfig should accept custom values."""
        from train.config import SelfPlayConfig

        config = SelfPlayConfig(
            sampling_distribution=SamplingDistribution.ELO_WEIGHTED, elo_init=1200.0
        )

        assert config.sampling_distribution == SamplingDistribution.ELO_WEIGHTED
        assert config.elo_init == 1200.0

    def test_self_play_config_field_access(self):
        """All fields should be accessible as attributes."""
        from train.config import SelfPlayConfig

        config = SelfPlayConfig()

        _ = config.sampling_distribution
        _ = config.elo_init


class TestSkillDecisionConfig:
    """Tests for SkillDecisionConfig dataclass."""

    def test_skill_decision_config_exists(self):
        """SkillDecisionConfig should exist and be importable."""
        from train.config import SkillDecisionConfig

        assert SkillDecisionConfig is not None

    def test_skill_decision_config_defaults(self):
        """SkillDecisionConfig should have sensible defaults."""
        from train.config import SkillDecisionConfig

        config = SkillDecisionConfig()

        # Should include all 43 skills
        assert len(config.skill_list) == 43
        assert config.reward_coeff == 1.0

    def test_skill_decision_config_all_skills_present(self):
        """SkillDecisionConfig should include all 43 skills."""
        from train.config import SkillDecisionConfig

        config = SkillDecisionConfig()

        # Verify all expected skills are present
        for skill in ALL_SKILLS:
            assert skill in config.skill_list, f"Missing skill: {skill}"

    def test_skill_decision_config_no_duplicates(self):
        """SkillDecisionConfig should not have duplicate skills."""
        from train.config import SkillDecisionConfig

        config = SkillDecisionConfig()

        assert len(config.skill_list) == len(set(config.skill_list))

    def test_skill_decision_config_custom_values(self):
        """SkillDecisionConfig should accept custom values."""
        from train.config import SkillDecisionConfig

        custom_skills = ["JianXiong", "Rende"]
        config = SkillDecisionConfig(skill_list=custom_skills, reward_coeff=0.5)

        assert config.skill_list == custom_skills
        assert config.reward_coeff == 0.5

    def test_skill_decision_config_field_access(self):
        """All fields should be accessible as attributes."""
        from train.config import SkillDecisionConfig

        config = SkillDecisionConfig()

        _ = config.skill_list
        _ = config.reward_coeff


class TestConfigIntegration:
    """Integration tests for config classes."""

    def test_all_configs_importable(self):
        """All config classes should be importable from train.config."""
        from train.config import TrainingConfig, SelfPlayConfig, SkillDecisionConfig

        assert TrainingConfig is not None
        assert SelfPlayConfig is not None
        assert SkillDecisionConfig is not None

    def test_configs_are_dataclasses(self):
        """All configs should be dataclasses."""
        from train.config import TrainingConfig, SelfPlayConfig, SkillDecisionConfig
        from dataclasses import is_dataclass

        assert is_dataclass(TrainingConfig)
        assert is_dataclass(SelfPlayConfig)
        assert is_dataclass(SkillDecisionConfig)
