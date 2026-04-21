"""
Global State Cache with LRU Eviction - Caches game state encodings for multi-agent training

Purpose:
- Avoid redundant encoding of the same game_state for multiple agents
- In multi-agent envs, the same game_state is encoded N times (once per agent)
- Cache uses content-based hash as key (NOT id() - unreliable due to Python memory reuse)
- LRU eviction prevents unbounded memory growth

Key Design Decisions:
- Cache GLOBAL state only (game_state dict), NOT per-agent local states
- Use content hash as cache key - handles Python object reuse in loops
- Clear cache on episode boundaries (game_state objects change between episodes)
- Bounded cache with LRU eviction to prevent memory leaks

Performance Impact:
- Expected hit rate: 80%+ in multi-agent scenarios (N-1 hits per game_state)
- Memory: O(maxsize * state_dim * 4 bytes) per cache entry
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
import sys
import json

import numpy as np


def _hash_game_state(game_state: Dict, agent_idx: Optional[int] = None) -> int:
    """
    Generate a content-based hash for game_state.

    Uses JSON serialization with sorted keys for consistent hashing.
    Handles numpy arrays and other non-serializable types via default=str.
    """
    try:
        key_data = (game_state, agent_idx) if agent_idx is not None else game_state
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return hash(serialized)
    except (TypeError, ValueError):
        return hash(str(game_state))


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0


class GlobalStateCache:
    """
    LRU cache for game state encodings in multi-agent training.

    Usage:
        cache = GlobalStateCache(maxsize=1000, state_encoder=encoder)
        encoded_states = cache.encode_batch(game_states, num_agents)
        cache.clear()  # Call on episode boundaries
        hit_rate = cache.get_hit_rate()
        memory_mb = cache.get_cache_memory_usage()

    Thread Safety: NOT thread-safe. Use one cache per thread.
    """

    def __init__(
        self,
        maxsize: int = 1000,
        state_encoder: Optional[Any] = None,
    ):
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")

        self.maxsize = maxsize
        self.encoder = state_encoder
        self._cache: OrderedDict[Union[int, Tuple[int, int]], np.ndarray] = (
            OrderedDict()
        )
        self._stats = CacheStats()
        self._state_dim: Optional[int] = None

    def encode_batch(
        self,
        game_states: List[Dict],
        num_agents: int,
    ) -> np.ndarray:
        """
        Encode a batch of game states for all agents, caching global state.

        Args:
            game_states: List of game_state dicts to encode
            num_agents: Number of agents (encodes state for each agent)

        Returns:
            np.ndarray of shape (batch_size * num_agents, state_dim)
        """
        if self.encoder is None:
            raise ValueError("state_encoder not set - cannot encode states")

        all_encoded = []

        for game_state in game_states:
            cache_key = _hash_game_state(game_state)

            if cache_key in self._cache:
                encoded_state = self._cache[cache_key]
                self._cache.move_to_end(cache_key)
                self._stats.hits += 1
            else:
                agent_states = [
                    self.encoder.encode(game_state, agent_idx)
                    for agent_idx in range(num_agents)
                ]
                encoded_state = np.stack(agent_states)
                self._cache[cache_key] = encoded_state
                self._stats.misses += 1
                self._evict_if_needed()

                if self._state_dim is None and encoded_state is not None:
                    self._state_dim = encoded_state.shape[-1]

            all_encoded.append(encoded_state)

        return np.concatenate(all_encoded, axis=0) if all_encoded else np.array([])

    def encode_for_agent(
        self,
        game_state: Dict,
        agent_idx: int,
    ) -> np.ndarray:
        """
        Encode game state for a single agent with caching.

        Args:
            game_state: Game state dict to encode
            agent_idx: Agent index for player-specific encoding

        Returns:
            Encoded state vector (state_dim,)
        """
        if self.encoder is None:
            raise ValueError("state_encoder not set - cannot encode states")

        cache_key = _hash_game_state(game_state, agent_idx)

        if cache_key in self._cache:
            encoded = self._cache[cache_key]
            self._cache.move_to_end(cache_key)
            self._stats.hits += 1
            return encoded

        encoded = self.encoder.encode(game_state, agent_idx)
        self._cache[cache_key] = encoded
        self._stats.misses += 1

        if self._state_dim is None:
            self._state_dim = encoded.size

        self._evict_if_needed()
        return encoded

    def encode_global(
        self,
        game_state: Dict,
        num_agents: int,
    ) -> np.ndarray:
        """
        Encode game state for ALL agents and cache the result.

        Args:
            game_state: Game state dict to encode
            num_agents: Number of agents

        Returns:
            Concatenated encoded states: (num_agents * state_dim,)
        """
        if self.encoder is None:
            raise ValueError("state_encoder not set - cannot encode states")

        cache_key = _hash_game_state(game_state)

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            self._cache.move_to_end(cache_key)
            self._stats.hits += 1
            return cached.flatten() if len(cached.shape) > 1 else cached

        agent_states = [
            self.encoder.encode(game_state, agent_idx)
            for agent_idx in range(num_agents)
        ]

        batch = np.stack(agent_states)
        self._cache[cache_key] = batch
        self._stats.misses += 1

        if self._state_dim is None and agent_states:
            self._state_dim = agent_states[0].size

        self._evict_if_needed()
        return batch.flatten()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    def get_hit_rate(self) -> float:
        """Get cache hit rate in [0.0, 1.0]. Returns 0.0 if no requests."""
        total = self._stats.hits + self._stats.misses
        return self._stats.hits / total if total > 0 else 0.0

    def get_miss_rate(self) -> float:
        return 1.0 - self.get_hit_rate()

    def get_stats(self) -> Dict[str, int]:
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": self._stats.evictions,
            "size": len(self._cache),
            "maxsize": self.maxsize,
        }

    def get_cache_memory_usage(self) -> float:
        """Estimate cache memory usage in megabytes."""
        if not self._cache:
            return 0.0

        total_bytes = sum(
            arr.nbytes if isinstance(arr, np.ndarray) else sys.getsizeof(arr)
            for arr in self._cache.values()
        )
        return total_bytes / (1024 * 1024)

    def clear(self) -> None:
        """Clear cache and reset statistics. Call at episode boundaries."""
        self._cache.clear()
        self._stats.reset()

    def invalidate(self, game_state: Dict) -> bool:
        """Invalidate a specific game_state from cache. Returns True if found."""
        cache_key = _hash_game_state(game_state)
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, game_state: Dict) -> bool:
        return _hash_game_state(game_state) in self._cache

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def is_full(self) -> bool:
        return len(self._cache) >= self.maxsize


def create_state_cache(
    maxsize: int = 1000,
    state_encoder: Optional[Any] = None,
) -> GlobalStateCache:
    """Factory function to create a GlobalStateCache."""
    return GlobalStateCache(maxsize=maxsize, state_encoder=state_encoder)
