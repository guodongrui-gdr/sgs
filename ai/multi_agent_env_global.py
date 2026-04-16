"""
IPPO Global State Environment - Wrapper for IPPO with global state baseline

This module provides a GlobalStateWrapper that:
- Concatenates all agent observations into a single global state
- Maintains local action masking (each agent only sees valid actions)
- Provides the same global state to all agents (IPPO baseline)

Purpose: Phase 0 validation - baseline for comparing against MAPPO

Architecture:
- GlobalStateWrapper wraps MultiAgentEnv
- Each agent receives the SAME global observation (concatenated states)
- Each agent maintains LOCAL action masks (valid actions per agent)
- Independent policies learn from global state (IPPO = Independent PPO)

State dimension:
- Per-agent: ~3066 dims
- Global state: 3066 * 5 = 15330 dims (for 5 players)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Conditional imports for Gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces

        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        gym = None
        spaces = None

from ai.gym_wrapper import SGSEnv, SGSConfig, GYM_AVAILABLE
from ai.state_encoder import StateEncoder, EncodingConfig
from ai.action_encoder import ActionEncoder, ActionConfig, ActionType


@dataclass
class GlobalStateConfig:
    """Configuration for GlobalStateWrapper"""

    player_num: int = 5
    use_shared_state: bool = True  # All agents receive same global state
    normalize_state: bool = False  # Whether to normalize concatenated state
    state_encoder_config: Optional[EncodingConfig] = None
    action_encoder_config: Optional[ActionConfig] = None


if GYM_AVAILABLE:
    _BaseEnv = gym.Env
else:
    _BaseEnv = object


class GlobalStateWrapper(_BaseEnv):
    """
    Wrapper that provides global state to all agents for IPPO baseline.

    Key Features:
    - get_global_state(): concatenates all 5 agent observations → (batch, 3066*5)
    - get_local_obs_for_agent(agent_id): returns global state (same for all)
    - Maintains local action masking (each agent only sees valid actions)

    IPPO vs MAPPO:
    - IPPO: Each agent has independent policy, receives same global state
    - MAPPO: Centralized training with shared critic, decentralized execution

    This wrapper implements IPPO baseline where:
    - All agents receive FULL global state (no information hiding)
    - Each agent has INDEPENDENT policy network
    - Action masks are LOCAL (agent-specific valid actions)
    """

    def __init__(self, config: GlobalStateConfig = None):
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium or gym is required")

        super().__init__()

        self.config = config or GlobalStateConfig()
        self.player_num = self.config.player_num

        # Initialize encoders
        self.state_encoder = StateEncoder(
            self.config.state_encoder_config or EncodingConfig()
        )
        self.action_encoder = ActionEncoder(
            self.config.action_encoder_config or ActionConfig()
        )

        # Base environment
        self.sgs_config = SGSConfig(player_num=self.player_num)
        self.base_env: Optional[SGSEnv] = None

        # State tracking
        self._current_agent_idx: int = 0
        self._step_count: int = 0
        self._done: bool = False

        # Pre-compute dimensions
        self._single_state_dim = self.state_encoder.get_state_dim(self.player_num)
        self._global_state_dim = self._single_state_dim * self.player_num

        # Setup observation/action spaces for Gymnasium compatibility
        self._setup_spaces()

        # Cached global state (recomputed each step)
        self._cached_global_state: Optional[np.ndarray] = None
        self._cached_action_masks: Dict[int, np.ndarray] = {}

    def _setup_spaces(self):
        """Setup observation and action spaces for Gymnasium"""
        if not GYM_AVAILABLE:
            return

        total_mask_dim = (
            self.action_encoder.action_type_dim
            + self.action_encoder.card_dim
            + self.action_encoder.target_dim
        )

        # Global state observation space (concatenated states)
        self.observation_space = spaces.Dict(
            {
                "global_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._global_state_dim,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(total_mask_dim,), dtype=np.float32
                ),
                "agent_id": spaces.Discrete(self.player_num),
            }
        )

        # Action space (same as base env)
        self.action_space = spaces.Discrete(total_mask_dim)
        self._action_dim = total_mask_dim

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset environment and return initial global state"""
        # Create fresh base environment
        self.base_env = SGSEnv(self.sgs_config)
        obs, info = self.base_env.reset(seed=seed)

        self._current_agent_idx = self.base_env.current_player_idx
        self._step_count = 0
        self._done = False

        # Compute and cache global state
        self._compute_global_state()

        # Return global observation for current agent
        global_obs = self._get_global_obs_for_agent(self._current_agent_idx)

        info["agent_id"] = self._current_agent_idx
        info["global_state_dim"] = self._global_state_dim

        return global_obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action for current agent, return global state observation"""
        if self._done:
            # Episode ended - return cached state
            global_obs = self._get_global_obs_for_agent(self._current_agent_idx)
            return global_obs, 0.0, True, False, {"agent_id": self._current_agent_idx}

        # Execute action in base environment
        base_obs, reward, done, truncated, info = self.base_env.step(action)

        self._step_count += 1
        self._done = done or truncated

        # Update current agent index
        self._current_agent_idx = self.base_env.current_player_idx

        # Recompute global state
        self._compute_global_state()

        # Return global observation for NEW current agent
        global_obs = self._get_global_obs_for_agent(self._current_agent_idx)

        # Add metadata to info
        info["agent_id"] = self._current_agent_idx
        info["step_count"] = self._step_count
        info["global_state_dim"] = self._global_state_dim

        return global_obs, reward, done, truncated, info

    def _compute_global_state(self):
        """
        Compute global state by concatenating all agent observations.

        Global state = concat([state_i for i in range(player_num)])
        Result shape: (player_num * single_state_dim,)
        """
        if self.base_env is None or self.base_env.players is None:
            self._cached_global_state = np.zeros(
                self._global_state_dim, dtype=np.float32
            )
            return

        # Get game state dict from base env
        game_state = self.base_env._get_game_state_dict()

        # Encode state for each agent
        agent_states = []
        for i in range(self.player_num):
            encoded_state = self.state_encoder.encode(game_state, i)
            agent_states.append(encoded_state)

        # Concatenate all agent states
        self._cached_global_state = np.concatenate(agent_states)

        # Compute action masks for each agent
        self._compute_action_masks()

    def _compute_action_masks(self):
        """
        Compute action masks for each agent.

        Each agent has LOCAL action mask (only their valid actions).
        Masks are cached for quick retrieval.
        """
        if self.base_env is None:
            return

        game_state = self.base_env._get_game_state_dict()

        for i in range(self.player_num):
            if i == self._current_agent_idx:
                mask_type, mask_card, mask_target = self.base_env._get_action_masks()
                total_dim = len(mask_type) + len(mask_card) + len(mask_target)
                combined_mask = np.zeros(total_dim, dtype=np.float32)
                combined_mask[: len(mask_type)] = mask_type
                combined_mask[len(mask_type) : len(mask_type) + len(mask_card)] = (
                    mask_card
                )
                combined_mask[len(mask_type) + len(mask_card) :] = mask_target
                self._cached_action_masks[i] = combined_mask
                self._action_dim = total_dim
            else:
                mask_type, mask_card, mask_target = self.base_env._get_action_masks()
                total_dim = len(mask_type) + len(mask_card) + len(mask_target)
                self._cached_action_masks[i] = np.zeros(total_dim, dtype=np.float32)

    def _get_global_obs_for_agent(self, agent_idx: int) -> Dict:
        """
        Get global observation for specific agent.

        Returns:
            Dict with:
            - global_state: concatenated state from all agents
            - action_mask: LOCAL action mask for this agent
            - agent_id: which agent this observation belongs to
        """
        global_state = (
            self._cached_global_state.copy()
            if self._cached_global_state is not None
            else np.zeros(self._global_state_dim, dtype=np.float32)
        )

        action_mask = self._cached_action_masks.get(
            agent_idx, np.zeros(self._action_dim, dtype=np.float32)
        )

        return {
            "global_state": global_state,
            "action_mask": action_mask,
            "agent_id": agent_idx,
        }

    def get_global_state(self) -> np.ndarray:
        """
        Get concatenated global state for all agents.

        Returns:
            np.ndarray of shape (global_state_dim,) = (player_num * single_state_dim,)
        """
        return (
            self._cached_global_state.copy()
            if self._cached_global_state is not None
            else np.zeros(self._global_state_dim, dtype=np.float32)
        )

    def get_local_obs_for_agent(self, agent_id: int) -> Dict:
        """
        Get observation for specific agent.

        For IPPO baseline: returns SAME global state for all agents.

        Args:
            agent_id: Agent index (0 to player_num-1)

        Returns:
            Dict with global_state, action_mask, agent_id
        """
        return self._get_global_obs_for_agent(agent_id)

    def get_all_agent_obs(self) -> Dict[int, Dict]:
        """
        Get observations for all agents.

        Returns:
            Dict mapping agent_id -> observation dict
        """
        observations = {}
        for i in range(self.player_num):
            observations[i] = self._get_global_obs_for_agent(i)
        return observations

    def action_masks(self) -> np.ndarray:
        """
        Get action mask for current agent (MaskablePPO compatibility).

        Returns:
            Action mask for current agent
        """
        return self._cached_action_masks.get(
            self._current_agent_idx, np.zeros(self._action_dim, dtype=np.float32)
        )

    def render(self, mode: str = "human"):
        """Render environment"""
        if self.base_env is not None:
            return self.base_env.render(mode)

    def close(self):
        """Close environment"""
        if self.base_env is not None:
            self.base_env.close()

    @property
    def current_agent_idx(self) -> int:
        """Get current agent index"""
        return self._current_agent_idx

    @property
    def single_state_dim(self) -> int:
        """Get single agent state dimension"""
        return self._single_state_dim

    @property
    def global_state_dim(self) -> int:
        """Get global state dimension"""
        return self._global_state_dim

    @property
    def num_agents(self) -> int:
        """Get number of agents"""
        return self.player_num


class IPPOGlobalVecEnvWrapper:
    """
    Vectorized environment wrapper for IPPO global state training.

    Wraps multiple GlobalStateWrapper instances for parallel training.
    Compatible with stable_baselines3 VecEnv interface.

    Key Methods:
    - reset(): Reset all environments, return stacked observations
    - step(actions): Execute actions across all environments
    - action_masks(): Get stacked action masks for MaskablePPO
    """

    def __init__(self, envs: List[GlobalStateWrapper]):
        self.envs = envs
        self.num_envs = len(envs)

        # Extract spaces from first env
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

        # State dimensions
        self.global_state_dim = envs[0].global_state_dim
        self.num_agents = envs[0].num_agents

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset all environments"""
        observations = []
        infos = []

        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs["global_state"])
            infos.append(info)

        # Stack observations
        stacked_obs = np.stack(observations, axis=0)

        # Combine infos
        combined_info = {
            "agent_ids": [info.get("agent_id", 0) for info in infos],
        }

        # Return Dict observation format
        return {
            "global_state": stacked_obs,
            "action_mask": self.action_masks(),
        }, combined_info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Execute actions across all environments"""
        observations = []
        rewards = []
        dones = []
        truncateds = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, truncated, info = env.step(action)
            observations.append(obs["global_state"])
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)

        # Stack results
        stacked_obs = np.stack(observations, axis=0)

        return (
            {
                "global_state": stacked_obs,
                "action_mask": self.action_masks(),
            },
            np.array(rewards),
            np.array(dones),
            np.array(truncateds),
            {"infos": infos},
        )

    def action_masks(self) -> np.ndarray:
        """Get stacked action masks"""
        masks = []
        for env in self.envs:
            mask = env.action_masks()
            masks.append(mask)
        return np.stack(masks, axis=0)

    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name: str) -> List[Any]:
        """Get attribute from all environments"""
        return [getattr(env, attr_name) for env in self.envs]


def make_global_state_env(player_num: int = 5, seed: int = None) -> GlobalStateWrapper:
    """
    Factory function to create GlobalStateWrapper.

    Args:
        player_num: Number of players (default 5)
        seed: Random seed

    Returns:
        GlobalStateWrapper instance
    """
    config = GlobalStateConfig(player_num=player_num)
    env = GlobalStateWrapper(config)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_global_state_env(
    n_envs: int = 4, player_num: int = 5
) -> IPPOGlobalVecEnvWrapper:
    """
    Factory function to create vectorized global state environment.

    Args:
        n_envs: Number of parallel environments
        player_num: Number of players per environment

    Returns:
        IPPOGlobalVecEnvWrapper instance
    """
    envs = [make_global_state_env(player_num) for _ in range(n_envs)]
    return IPPOGlobalVecEnvWrapper(envs)
