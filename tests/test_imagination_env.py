"""
Tests for ImaginedEnvironment - World Model Phase 2

Tests the Gym-like wrapper for imagination-based policy training:
- reset/step interface
- Action masking
- 15-step rollout performance (<10ms)
- Model loading from checkpoint

Usage:
    pytest tests/test_imagination_env.py -v
"""

import pytest
import time
import numpy as np
import torch
from pathlib import Path

from ai.world_model.imagination_env import (
    ImaginedEnvironment,
    ImaginationConfig,
    ImaginationState,
    create_imagination_env,
)


class TestImaginationConfig:
    """Tests for ImaginationConfig"""

    def test_config_defaults(self):
        """ImaginationConfig should have sensible defaults"""
        config = ImaginationConfig()

        assert config.state_dim == 2670
        assert config.latent_dim == 128
        assert config.action_embed_dim == 64
        assert config.max_imagination_horizon == 15
        assert config.action_type_dim == 17
        assert config.max_hand_size == 20
        assert config.max_players == 8

    def test_config_custom_values(self):
        """ImaginationConfig should accept custom values"""
        config = ImaginationConfig(
            state_dim=1000,
            latent_dim=64,
            max_imagination_horizon=20,
        )

        assert config.state_dim == 1000
        assert config.latent_dim == 64
        assert config.max_imagination_horizon == 20

    def test_config_device_auto_detect(self):
        """ImaginationConfig should auto-detect device if None"""
        config = ImaginationConfig(device=None)

        assert config.device is None  # Will be set by environment


class TestImaginedEnvironmentInit:
    """Tests for ImaginedEnvironment initialization"""

    def test_init_without_checkpoint(self):
        """ImaginedEnvironment should work without checkpoint"""
        env = ImaginedEnvironment()

        assert env.encoder is not None
        assert env.dynamics is not None
        assert env.reward_model is not None
        assert env._state is None

    def test_init_with_config(self):
        """ImaginedEnvironment should accept config"""
        config = ImaginationConfig(latent_dim=128)
        env = ImaginedEnvironment(config=config)

        assert env.config.latent_dim == 128

    def test_models_in_eval_mode(self):
        """Models should be in eval mode for inference"""
        env = ImaginedEnvironment()

        assert not env.encoder.training
        assert not env.dynamics.training
        assert not env.reward_model.training

    def test_checkpoint_not_found_warning(self):
        """Should warn but not fail if checkpoint not found"""
        env = ImaginedEnvironment(checkpoint_path="nonexistent_path.pt")

        assert env.encoder is not None
        assert env.dynamics is not None


class TestImaginedEnvironmentReset:
    """Tests for reset() method"""

    def test_reset_with_state_vector(self):
        """reset() should encode state vector to latent"""
        env = ImaginedEnvironment()
        state = np.random.randn(2670).astype(np.float32)

        z_0, info = env.reset(initial_state=state)

        assert z_0.shape == (1, 128)
        assert str(z_0.device).startswith("cuda") or z_0.device.type == "cpu"
        assert "action_masks" in info
        assert "step" in info

    def test_reset_with_preencoded_latent(self):
        """reset() should accept pre-encoded latent"""
        env = ImaginedEnvironment()
        device = env.device
        z_input = torch.randn(128, device=device)

        z_0, info = env.reset(initial_latent=z_input)

        assert z_0.shape == (1, 128)
        assert torch.allclose(z_0.squeeze(0), z_input)

    def test_reset_with_action_masks(self):
        """reset() should accept custom action masks"""
        env = ImaginedEnvironment()
        masks = {
            "type": np.zeros(17, dtype=np.float32),
            "card": np.ones(20, dtype=np.float32),
            "target": np.ones(8, dtype=np.float32),
        }
        masks["type"][0] = 1.0

        z_0, info = env.reset(action_masks=masks)

        assert info["action_masks"] == masks

    def test_reset_without_state(self):
        """reset() should work without state (random latent for testing)"""
        env = ImaginedEnvironment()

        z_0, info = env.reset()

        assert z_0.shape == (1, 128)

    def test_reset_state_tracking(self):
        """reset() should properly initialize internal state"""
        env = ImaginedEnvironment()
        state = np.random.randn(2670).astype(np.float32)

        z_0, info = env.reset(initial_state=state)

        assert env._state is not None
        assert env._state.current_step == 0
        assert env._state.z_current is not None


class TestImaginedEnvironmentStep:
    """Tests for step() method"""

    def test_step_basic(self):
        """step() should predict next latent and reward"""
        env = ImaginedEnvironment()
        env.reset()

        z_next, reward, done, info = env.step(action_type=0)

        assert z_next.shape == (1, 128)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "uncertainty" in info

    def test_step_with_card_and_target(self):
        """step() should accept card and target indices"""
        env = ImaginedEnvironment()
        env.reset()

        z_next, reward, done, info = env.step(
            action_type=0,
            card_idx=3,
            target_idx=1,
        )

        assert z_next.shape == (1, 128)

    def test_step_without_reset_raises(self):
        """step() without reset() should raise error"""
        env = ImaginedEnvironment()

        with pytest.raises(RuntimeError, match="Must call reset"):
            env.step(action_type=0)

    def test_step_updates_state(self):
        """step() should update internal state"""
        env = ImaginedEnvironment()
        env.reset()

        env.step(action_type=0)

        assert env._state.current_step == 1
        assert len(env._state.action_sequence) == 1
        assert len(env._state.reward_sequence) == 1

    def test_step_reaches_horizon(self):
        """step() should reach done after max horizon"""
        env = ImaginedEnvironment(config=ImaginationConfig(max_imagination_horizon=5))
        env.reset()

        for i in range(5):
            z_next, reward, done, info = env.step(action_type=0)
            if i < 4:
                assert not done
            else:
                assert done

    def test_step_after_horizon_returns_done(self):
        """step() after horizon should return done=True"""
        env = ImaginedEnvironment(config=ImaginationConfig(max_imagination_horizon=3))
        env.reset()

        for _ in range(3):
            env.step(action_type=0)

        z_next, reward, done, info = env.step(action_type=0)

        assert done
        assert info.get("reason") == "max_horizon"


class TestImaginedEnvironmentActionMasking:
    """Tests for action masking"""

    def test_action_validity_check(self):
        """step() should check action validity"""
        env = ImaginedEnvironment()
        masks = {
            "type": np.zeros(17, dtype=np.float32),
            "card": np.ones(20, dtype=np.float32),
            "target": np.ones(8, dtype=np.float32),
        }
        masks["type"][0] = 1.0
        masks["type"][1] = 1.0

        env.reset(action_masks=masks)

        z_next, reward, _, info = env.step(action_type=0)
        assert info["action_valid"] == True

        z_next, reward, _, info = env.step(action_type=2)  # Invalid
        assert info["action_valid"] == False

    def test_invalid_action_penalty(self):
        """Invalid actions should receive penalty"""
        env = ImaginedEnvironment(config=ImaginationConfig(invalid_action_penalty=-5.0))
        masks = {
            "type": np.zeros(17, dtype=np.float32),
            "card": np.ones(20, dtype=np.float32),
            "target": np.ones(8, dtype=np.float32),
        }
        masks["type"][0] = 1.0

        env.reset(action_masks=masks)

        z_next1, reward1, _, info1 = env.step(action_type=0)
        z_next2, reward2, _, info2 = env.step(action_type=2)

        assert info1["action_valid"] == True
        assert info2["action_valid"] == False
        assert reward2 < reward1

    def test_mask_decay_over_steps(self):
        """Action masks should decay over imagination steps"""
        env = ImaginedEnvironment()
        masks = {
            "type": np.ones(17, dtype=np.float32),
            "card": np.ones(20, dtype=np.float32),
            "target": np.ones(8, dtype=np.float32),
        }

        env.reset(action_masks=masks)

        initial_type_mask = env._state.action_masks["type"].copy()

        for _ in range(5):
            env.step(action_type=0)

        later_type_mask = env._state.action_masks["type"]

        assert np.all(later_type_mask <= initial_type_mask)

    def test_pass_action_always_valid(self):
        """PASS (action_type=8) should always be valid after mask update"""
        env = ImaginedEnvironment()
        masks = {
            "type": np.zeros(17, dtype=np.float32),
            "card": np.ones(20, dtype=np.float32),
            "target": np.ones(8, dtype=np.float32),
        }
        masks["type"][0] = 1.0

        env.reset(action_masks=masks)

        z_next, reward, _, info = env.step(action_type=0)

        masks_after = env.get_action_masks()
        assert masks_after["type"][8] == 1.0


class TestImaginedEnvironmentPerformance:
    """Tests for performance requirements"""

    def test_single_step_performance(self):
        """Single step should complete in <1ms"""
        env = ImaginedEnvironment()
        env.reset()

        times = []
        for _ in range(100):
            start = time.perf_counter()
            env.step(action_type=0)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_time_ms = np.mean(times)
        assert avg_time_ms < 1.0, (
            f"Single step avg time {avg_time_ms:.2f}ms > 1ms threshold"
        )

    def test_15_step_rollout_performance(self):
        """15-step rollout should complete in <15ms"""
        env = ImaginedEnvironment()
        env.reset()

        start = time.perf_counter()
        for _ in range(15):
            z_next, reward, done, info = env.step(action_type=0)
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 15.0, f"15-step rollout {elapsed_ms:.2f}ms > 15ms threshold"

        stats = env.get_performance_stats()
        assert stats["avg_step_ms"] < 2.0

    def test_trajectory_method_performance(self):
        """imagine_trajectory() should be efficient"""
        env = ImaginedEnvironment()
        z_start = torch.randn(1, 128, device=env.device)
        actions = [{"action_type": i % 10} for i in range(15)]

        start = time.perf_counter()
        z_traj, rewards, info = env.imagine_trajectory(z_start, actions, horizon=15)
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 15.0, (
            f"Trajectory method {elapsed_ms:.2f}ms > 15ms threshold"
        )
        assert z_traj.shape[1] <= 15
        assert len(rewards) <= 15


class TestImaginedEnvironmentTrajectory:
    """Tests for trajectory methods"""

    def test_imagine_trajectory_returns_sequence(self):
        """imagine_trajectory() should return state sequence"""
        env = ImaginedEnvironment()
        z_start = torch.randn(1, 128, device=env.device)
        actions = [{"action_type": 0} for _ in range(10)]

        z_traj, rewards, info = env.imagine_trajectory(z_start, actions, horizon=10)

        assert z_traj.shape[0] == 1
        assert z_traj.shape[1] == 10
        assert z_traj.shape[2] == 128

    def test_imagine_trajectory_returns_rewards(self):
        """imagine_trajectory() should return reward sequence"""
        env = ImaginedEnvironment()
        z_start = torch.randn(1, 128, device=env.device)
        actions = [{"action_type": i % 17} for i in range(10)]

        z_traj, rewards, info = env.imagine_trajectory(z_start, actions, horizon=10)

        assert len(rewards) == 10
        assert isinstance(rewards, torch.Tensor)

    def test_imagine_trajectory_with_card_target(self):
        """imagine_trajectory() should handle card/target"""
        env = ImaginedEnvironment()
        z_start = torch.randn(1, 128, device=env.device)
        actions = [
            {"action_type": 0, "card_idx": 3, "target_idx": 1},
            {"action_type": 7, "card_idx": 0, "target_idx": 2},
            {"action_type": 8},
        ]

        z_traj, rewards, info = env.imagine_trajectory(z_start, actions, horizon=3)

        assert z_traj.shape[1] == 3

    def test_imagine_trajectory_stops_at_horizon(self):
        """imagine_trajectory() should respect horizon"""
        env = ImaginedEnvironment()
        z_start = torch.randn(1, 128, device=env.device)
        actions = [{"action_type": 0} for _ in range(30)]

        z_traj, rewards, info = env.imagine_trajectory(z_start, actions, horizon=15)

        assert z_traj.shape[1] == 15
        assert info["steps_completed"] == 15


class TestImaginedEnvironmentStateAccess:
    """Tests for state access methods"""

    def test_get_current_state(self):
        """get_current_state() should return latent"""
        env = ImaginedEnvironment()
        env.reset()

        z = env.get_current_state()

        assert z is not None
        assert z.shape == (1, 128)

    def test_get_action_masks(self):
        """get_action_masks() should return masks"""
        env = ImaginedEnvironment()
        env.reset()

        masks = env.get_action_masks()

        assert masks is not None
        assert "type" in masks
        assert "card" in masks
        assert "target" in masks

    def test_get_performance_stats(self):
        """get_performance_stats() should return timing stats"""
        env = ImaginedEnvironment()
        env.reset()

        for _ in range(5):
            env.step(action_type=0)

        stats = env.get_performance_stats()

        assert "avg_step_ms" in stats
        assert "total_steps" in stats
        assert stats["total_steps"] == 5


class TestImaginedEnvironmentBatched:
    """Tests for batched operations"""

    def test_step_batched_single(self):
        """Batched step with single batch should work"""
        env = ImaginedEnvironment()
        env.reset()

        batch_actions = {
            "action_type": torch.tensor([0]),
        }

        z_next, reward, done, info = env._step_batched(batch_actions)

        assert z_next.shape == (1, 128)
        assert reward.shape == (1, 1)

    def test_step_batched_multi(self):
        """Batched step with multiple batches"""
        env = ImaginedEnvironment()
        env.reset()

        batch_actions = {
            "action_type": torch.tensor([0, 1, 2]),
            "card_idx": torch.tensor([5, 3, 8]),
            "target_idx": torch.tensor([1, 2, 3]),
        }

        z_next, reward, done, info = env._step_batched(batch_actions)

        assert z_next.shape == (3, 128)
        assert reward.shape == (3, 1)


class TestFactoryFunction:
    """Tests for create_imagination_env factory"""

    def test_factory_basic(self):
        """Factory should create environment"""
        env = create_imagination_env()

        assert isinstance(env, ImaginedEnvironment)
        assert env.encoder is not None

    def test_factory_with_checkpoint(self):
        """Factory should accept checkpoint path"""
        env = create_imagination_env(checkpoint_path=None)

        assert env.checkpoint_path is None

    def test_factory_with_kwargs(self):
        """Factory should accept config kwargs"""
        env = create_imagination_env(max_imagination_horizon=20)

        assert env.config.max_imagination_horizon == 20


class TestCheckpointLoading:
    """Tests for checkpoint loading"""

    @pytest.mark.skipif(
        not Path("outputs/dynamics_training/dynamics_final.pt").exists(),
        reason="Checkpoint not available",
    )
    def test_load_real_checkpoint(self):
        """Should load real checkpoint if available"""
        env = ImaginedEnvironment(
            checkpoint_path="outputs/dynamics_training/dynamics_final.pt"
        )

        assert env.encoder is not None
        assert env.dynamics is not None
        assert env.reward_model is not None

    @pytest.mark.skipif(
        not Path("outputs/dynamics_training/dynamics_final.pt").exists(),
        reason="Checkpoint not available",
    )
    def test_reset_with_checkpoint(self):
        """Reset should work with loaded checkpoint"""
        env = ImaginedEnvironment(
            checkpoint_path="outputs/dynamics_training/dynamics_final.pt"
        )

        state = np.random.randn(2670).astype(np.float32)
        z_0, info = env.reset(initial_state=state)

        assert z_0.shape == (1, 128)

    @pytest.mark.skipif(
        not Path("outputs/dynamics_training/dynamics_final.pt").exists(),
        reason="Checkpoint not available",
    )
    def test_step_with_checkpoint(self):
        """Step should work with loaded checkpoint"""
        env = ImaginedEnvironment(
            checkpoint_path="outputs/dynamics_training/dynamics_final.pt"
        )

        env.reset()
        z_next, reward, done, info = env.step(action_type=0)

        assert z_next.shape == (1, 128)


class TestIntegration:
    """Integration tests"""

    def test_full_imagination_cycle(self):
        """Full imagination cycle: reset -> steps -> trajectory"""
        env = ImaginedEnvironment()

        state = np.random.randn(2670).astype(np.float32)
        z_0, info = env.reset(initial_state=state)

        rewards = []
        uncertainties = []

        for i in range(15):
            z_next, reward, done, info = env.step(
                action_type=i % 17,
                card_idx=i % 20,
                target_idx=i % 8,
            )
            rewards.append(reward)
            uncertainties.append(info["uncertainty"])

            if done:
                break

        assert len(rewards) == 15
        assert len(env._state.action_sequence) == 15

        stats = env.get_performance_stats()
        assert stats["avg_step_ms"] < 1.0

    def test_multiple_reset_cycles(self):
        """Multiple reset cycles should work"""
        env = ImaginedEnvironment()

        for cycle in range(3):
            state = np.random.randn(2670).astype(np.float32)
            z_0, _ = env.reset(initial_state=state)

            for _ in range(5):
                env.step(action_type=0)

            assert env._state.current_step == 5
