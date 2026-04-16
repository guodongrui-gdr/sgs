"""
Tests for Phase 1 World Model components

Tests:
- RSSEncoder: state compression and VAE properties
- DynamicsModel: latent prediction and uncertainty
- RewardModel: reward prediction with event types
- Integration: end-to-end flow
"""

import pytest
import torch

from ai.world_model import (
    RSSEncoder,
    RSSEncoderConfig,
    RSSMDecoder,
    RSSMVAE,
    DynamicsModel,
    DynamicsConfig,
    RewardModel,
    RewardModelConfig,
    compute_kl_loss,
)


class TestRSSEncoder:
    def test_encoder_output_shape(self):
        config = RSSEncoderConfig(state_dim=3066, latent_dim=128)
        encoder = RSSEncoder(config)
        state = torch.randn(4, 3066)
        output = encoder.encode(state)

        assert output.latent.shape == (4, 128)
        assert output.stochastic.shape == (4, 64)
        assert output.deterministic.shape == (4, 64)
        assert output.mean.shape == (4, 64)
        assert output.log_std.shape == (4, 64)

    def test_kl_loss_positive(self):
        config = RSSEncoderConfig(state_dim=3066, latent_dim=128)
        encoder = RSSEncoder(config)
        state = torch.randn(4, 3066)
        output = encoder.encode(state)

        kl_loss = compute_kl_loss(output.mean, output.log_std, config.free_bits)
        assert kl_loss.item() > 0

    def test_free_bits_clamping(self):
        config = RSSEncoderConfig(state_dim=3066, latent_dim=128, free_bits=2.0)
        encoder = RSSEncoder(config)
        state = torch.randn(4, 3066)
        output = encoder.encode(state)

        kl_loss = compute_kl_loss(output.mean, output.log_std, config.free_bits)
        assert kl_loss.item() >= 2.0


class TestRSSMDecoder:
    def test_decoder_output_shape(self):
        config = RSSEncoderConfig(state_dim=3066, latent_dim=128)
        decoder = RSSMDecoder(config)
        latent = torch.randn(4, 128)

        reconstructed = decoder.decode(latent)
        assert reconstructed.shape == (4, 3066)


class TestDynamicsModel:
    def test_forward_output_shape(self):
        config = DynamicsConfig()
        model = DynamicsModel(config)

        z_t = torch.randn(4, 128)
        action_type = torch.randint(0, 17, (4,))
        card_idx = torch.randint(0, 20, (4,))
        target_idx = torch.randint(0, 8, (4,))

        z_next, uncertainty = model.forward(z_t, action_type, card_idx, target_idx)

        assert z_next.shape == (4, 128)
        assert uncertainty.shape == (4, 1)

    def test_uncertainty_bounded(self):
        config = DynamicsConfig()
        model = DynamicsModel(config)

        z_t = torch.randn(4, 128)
        action_type = torch.randint(0, 17, (4,))
        card_idx = torch.randint(0, 20, (4,))
        target_idx = torch.randint(0, 8, (4,))

        z_next, uncertainty = model.forward(z_t, action_type, card_idx, target_idx)

        assert torch.all(uncertainty >= 0) and torch.all(uncertainty <= 1)

    def test_sequence_forward(self):
        config = DynamicsConfig()
        model = DynamicsModel(config)

        batch_size = 2
        seq_len = 10
        z_seq = torch.randn(batch_size, seq_len, 128)
        action_types = torch.randint(0, 17, (batch_size, seq_len))
        card_indices = torch.randint(0, 20, (batch_size, seq_len))
        target_indices = torch.randint(0, 8, (batch_size, seq_len))

        z_next_seq, uncertainty_seq = model.forward_sequence(
            z_seq, action_types, card_indices, target_indices
        )

        assert z_next_seq.shape == (batch_size, seq_len, 128)
        assert uncertainty_seq.shape == (batch_size, seq_len, 1)


class TestRewardModel:
    def test_forward_output_shape(self):
        config = RewardModelConfig()
        model = RewardModel(config)

        z_t = torch.randn(4, 128)
        action_embed = torch.randn(4, 64)

        reward = model.forward(z_t, action_embed)

        assert reward.shape == (4, 1)

    def test_reward_clipping(self):
        config = RewardModelConfig(reward_clip_min=-50, reward_clip_max=50)
        model = RewardModel(config)

        z_t = torch.randn(100, 128)
        action_embed = torch.randn(100, 64)

        rewards = model.forward(z_t, action_embed)

        assert torch.all(rewards >= -50) and torch.all(rewards <= 50)

    def test_event_type_conditioning(self):
        config = RewardModelConfig(use_event_conditioning=True)
        model = RewardModel(config)

        z_t = torch.randn(4, 128)
        action_embed = torch.randn(4, 64)

        reward_damage = model.forward(z_t, action_embed, event_type="damage")
        reward_kill = model.forward(z_t, action_embed, event_type="kill_enemy")

        assert reward_damage.shape == (4, 1)
        assert reward_kill.shape == (4, 1)


class TestIntegration:
    def test_end_to_end_flow(self):
        encoder_config = RSSEncoderConfig(state_dim=3066, latent_dim=128)
        encoder = RSSEncoder(encoder_config)
        decoder = RSSMDecoder(encoder_config)

        dynamics_config = DynamicsConfig()
        dynamics = DynamicsModel(dynamics_config)

        reward_config = RewardModelConfig()
        reward_model = RewardModel(reward_config)

        state = torch.randn(4, 3066)

        encoder_output = encoder.encode(state)
        assert encoder_output.latent.shape == (4, 128)

        z_next, uncertainty = dynamics.forward(
            encoder_output.latent,
            torch.randint(0, 17, (4,)),
            torch.randint(0, 20, (4,)),
            torch.randint(0, 8, (4,)),
        )
        assert z_next.shape == (4, 128)

        action_embed = dynamics.action_embedder(
            torch.randint(0, 17, (4,)),
            torch.randint(0, 20, (4,)),
            torch.randint(0, 8, (4,)),
        )
        reward = reward_model.forward(z_next, action_embed, event_type="damage")
        assert reward.shape == (4, 1)

        reconstructed = decoder.decode(encoder_output.latent)
        assert reconstructed.shape == (4, 3066)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
