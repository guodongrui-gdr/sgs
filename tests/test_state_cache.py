"""
Unit tests for GlobalStateCache class.

Tests cover:
- LRU eviction behavior
- Hit/miss tracking
- Memory usage estimation
- Cache invalidation
- Episode boundary clearing
"""

import pytest
import numpy as np
from collections import OrderedDict

from ai.state_cache import GlobalStateCache, CacheStats, create_state_cache


class MockStateEncoder:
    def __init__(self, state_dim: int = 100):
        self.state_dim = state_dim
        self.encode_count = 0

    def encode(self, game_state: dict, player_idx: int) -> np.ndarray:
        self.encode_count += 1
        return np.random.randn(self.state_dim).astype(np.float32)


class TestCacheStats:
    def test_initial_values(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_reset(self):
        stats = CacheStats(hits=10, misses=5, evictions=2)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0


class TestGlobalStateCacheInit:
    def test_default_init(self):
        cache = GlobalStateCache()
        assert cache.maxsize == 1000
        assert cache.encoder is None
        assert len(cache) == 0

    def test_custom_maxsize(self):
        cache = GlobalStateCache(maxsize=500)
        assert cache.maxsize == 500

    def test_with_encoder(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)
        assert cache.encoder is encoder

    def test_invalid_maxsize(self):
        with pytest.raises(ValueError):
            GlobalStateCache(maxsize=0)
        with pytest.raises(ValueError):
            GlobalStateCache(maxsize=-1)


class TestGlobalStateCacheEncode:
    def test_encode_for_agent_miss(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=10, state_encoder=encoder)
        game_state = {"phase": "play_phase", "players": []}

        result = cache.encode_for_agent(game_state, agent_idx=0)

        assert result.shape == (encoder.state_dim,)
        assert encoder.encode_count == 1
        assert cache.get_stats()["misses"] == 1
        assert cache.get_stats()["hits"] == 0

    def test_encode_for_agent_hit(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=10, state_encoder=encoder)
        game_state = {"phase": "play_phase", "players": []}

        cache.encode_for_agent(game_state, agent_idx=0)
        result = cache.encode_for_agent(game_state, agent_idx=0)

        assert encoder.encode_count == 1
        assert cache.get_stats()["hits"] == 1
        assert cache.get_stats()["misses"] == 1

    def test_encode_for_different_agents(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=10, state_encoder=encoder)
        game_state = {"phase": "play_phase", "players": []}

        cache.encode_for_agent(game_state, agent_idx=0)
        cache.encode_for_agent(game_state, agent_idx=1)

        assert encoder.encode_count == 2
        assert cache.get_stats()["misses"] == 2

    def test_encode_without_encoder_raises(self):
        cache = GlobalStateCache(maxsize=10)
        game_state = {"phase": "play_phase"}

        with pytest.raises(ValueError, match="state_encoder not set"):
            cache.encode_for_agent(game_state, agent_idx=0)


class TestGlobalStateCacheBatch:
    def test_encode_batch_single_game_state(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=10, state_encoder=encoder)
        game_state = {"phase": "play_phase", "players": []}

        result = cache.encode_batch([game_state], num_agents=5)

        assert result.shape == (5, encoder.state_dim)
        assert encoder.encode_count == 5

    def test_encode_batch_multiple_game_states(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=10, state_encoder=encoder)
        game_states = [
            {"phase": "play_phase", "players": []},
            {"phase": "draw_phase", "players": []},
        ]

        result = cache.encode_batch(game_states, num_agents=3)

        assert result.shape == (6, encoder.state_dim)
        assert encoder.encode_count == 6

    def test_encode_batch_cache_hit_on_same_game_state(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=10, state_encoder=encoder)
        game_state = {"phase": "play_phase", "players": []}

        cache.encode_batch([game_state], num_agents=5)
        cache.encode_batch([game_state], num_agents=5)

        assert encoder.encode_count == 5
        assert cache.get_stats()["hits"] == 1


class TestGlobalStateCacheGlobal:
    def test_encode_global_returns_flattened(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=10, state_encoder=encoder)
        game_state = {"phase": "play_phase", "players": []}

        result = cache.encode_global(game_state, num_agents=5)

        assert result.shape == (5 * encoder.state_dim,)

    def test_encode_global_caches_batch(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=10, state_encoder=encoder)
        game_state = {"phase": "play_phase", "players": []}

        cache.encode_global(game_state, num_agents=5)
        cache.encode_global(game_state, num_agents=5)

        assert encoder.encode_count == 5
        assert cache.get_stats()["hits"] == 1


class TestLRUEviction:
    def test_eviction_at_maxsize(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=3, state_encoder=encoder)

        for i in range(5):
            game_state = {"id": i}
            cache.encode_for_agent(game_state, agent_idx=0)

        assert len(cache) == 3
        assert cache.get_stats()["evictions"] == 2

    def test_lru_order_on_access(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=3, state_encoder=encoder)

        gs1 = {"id": 1}
        gs2 = {"id": 2}
        gs3 = {"id": 3}

        cache.encode_global(gs1, 2)
        cache.encode_global(gs2, 2)
        cache.encode_global(gs3, 2)

        cache.encode_global(gs1, 2)

        gs4 = {"id": 4}
        cache.encode_global(gs4, 2)

        assert gs1 in cache
        assert gs2 not in cache
        assert gs3 in cache
        assert gs4 in cache

    def test_is_full_property(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=2, state_encoder=encoder)

        assert not cache.is_full

        cache.encode_global({"id": 1}, 2)
        assert not cache.is_full

        cache.encode_global({"id": 2}, 2)
        assert cache.is_full


class TestHitRate:
    def test_hit_rate_zero_requests(self):
        cache = GlobalStateCache()
        assert cache.get_hit_rate() == 0.0

    def test_hit_rate_all_misses(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)

        cache.encode_for_agent({"id": 1}, 0)
        cache.encode_for_agent({"id": 2}, 0)

        assert cache.get_hit_rate() == 0.0

    def test_hit_rate_all_hits(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)
        game_state = {"id": 1}

        cache.encode_for_agent(game_state, 0)
        cache.encode_for_agent(game_state, 0)
        cache.encode_for_agent(game_state, 0)

        assert cache.get_hit_rate() == pytest.approx(2 / 3)

    def test_miss_rate_complement(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)
        game_state = {"id": 1}

        cache.encode_for_agent(game_state, 0)
        cache.encode_for_agent(game_state, 0)

        assert cache.get_miss_rate() == pytest.approx(1 - cache.get_hit_rate())


class TestMemoryUsage:
    def test_empty_cache_zero_memory(self):
        cache = GlobalStateCache()
        assert cache.get_cache_memory_usage() == 0.0

    def test_memory_usage_with_entries(self):
        encoder = MockStateEncoder(state_dim=100)
        cache = GlobalStateCache(state_encoder=encoder)

        cache.encode_for_agent({"id": 1}, 0)
        cache.encode_for_agent({"id": 2}, 0)

        expected_mb = (2 * 100 * 4) / (1024 * 1024)
        assert cache.get_cache_memory_usage() == pytest.approx(expected_mb, rel=0.1)


class TestCacheClear:
    def test_clear_empties_cache(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)

        cache.encode_for_agent({"id": 1}, 0)
        cache.encode_for_agent({"id": 2}, 0)
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0

    def test_clear_resets_stats(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)
        game_state = {"id": 1}

        cache.encode_for_agent(game_state, 0)
        cache.encode_for_agent(game_state, 0)

        cache.clear()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestCacheInvalidation:
    def test_invalidate_existing_entry(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)
        game_state = {"id": 1}

        cache.encode_global(game_state, 2)
        assert game_state in cache

        result = cache.invalidate(game_state)
        assert result is True
        assert game_state not in cache

    def test_invalidate_nonexistent_entry(self):
        cache = GlobalStateCache()
        game_state = {"id": 1}

        result = cache.invalidate(game_state)
        assert result is False


class TestContainsAndLen:
    def test_contains_with_cached_state(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)
        game_state = {"id": 1}

        cache.encode_global(game_state, 2)
        assert game_state in cache

    def test_contains_with_uncached_state(self):
        cache = GlobalStateCache()
        game_state = {"id": 1}
        assert game_state not in cache

    def test_len(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(state_encoder=encoder)

        assert len(cache) == 0
        cache.encode_global({"id": 1}, 2)
        assert len(cache) == 1
        cache.encode_global({"id": 2}, 2)
        assert len(cache) == 2


class TestFactoryFunction:
    def test_create_state_cache(self):
        encoder = MockStateEncoder()
        cache = create_state_cache(maxsize=500, state_encoder=encoder)

        assert isinstance(cache, GlobalStateCache)
        assert cache.maxsize == 500
        assert cache.encoder is encoder


class TestGetStats:
    def test_get_stats_structure(self):
        encoder = MockStateEncoder()
        cache = GlobalStateCache(maxsize=100, state_encoder=encoder)
        game_state = {"id": 1}

        cache.encode_for_agent(game_state, 0)
        cache.encode_for_agent(game_state, 0)

        stats = cache.get_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "evictions" in stats
        assert "size" in stats
        assert "maxsize" in stats
        assert stats["maxsize"] == 100
        assert stats["size"] == 1
