"""
多智能体环境 - 支持自博弈训练

功能:
- 多智能体并行执行
- 独立观察空间
- 身份感知奖励
- 支持随机策略和策略池
"""

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ai.action_encoder import ActionEncoder, ActionType
from ai.gym_wrapper import SGSEnv, SGSConfig, GYM_AVAILABLE
from ai.reward import RewardSystem, IdentityRelationship
from ai.state_encoder import StateEncoder
from engine.game_engine import GameEngine
from player.player import Player


class AgentType(IntEnum):
	RANDOM = 0
	RULE_BASED = 1
	RL_POLICY = 2
	HUMAN = 3


@dataclass
class AgentConfig:
	agent_type: AgentType = AgentType.RANDOM
	policy_path: Optional[str] = None
	identity: str = ""
	player_idx: int = 0


@dataclass
class MultiAgentConfig:
	sgs_config: SGSConfig = field(default_factory=SGSConfig)
	agent_configs: List[AgentConfig] = field(default_factory=list)
	training_agent_idx: int = 0

	def __post_init__(self):
		if not self.agent_configs:
			self.agent_configs = [
				AgentConfig(agent_type=AgentType.RANDOM, player_idx=i)
				for i in range(self.sgs_config.player_num)
			]


class RandomAgent:
	def __init__(self, player_idx: int = 0):
		self.player_idx = player_idx

	def get_action(self, obs: Dict, legal_actions: List[int]) -> int:
		if not legal_actions:
			return 0
		return random.choice(legal_actions)

	def update(self, *args, **kwargs):
		pass


class RuleBasedAgent:
	def __init__(self, player_idx: int = 0):
		self.player_idx = player_idx

	def get_action(
			self, obs: Dict, legal_actions: List[int], game_state: Dict = None
	) -> int:
		if not legal_actions:
			return 0

		if ActionType.END_TURN in legal_actions:
			for action in legal_actions:
				if action != ActionType.END_TURN:
					return action

		if ActionType.USE_CARD in legal_actions and random.random() < 0.7:
			return ActionType.USE_CARD

		return random.choice(legal_actions)

	def update(self, *args, **kwargs):
		pass


class PolicyAgent:
	def __init__(self, policy, player_idx: int = 0, use_masking: bool = True):
		self.policy = policy
		self.player_idx = player_idx
		self.use_masking = use_masking

	def get_action(
			self, obs: Dict, legal_actions: List[int], action_mask: np.ndarray = None
	) -> int:
		if self.policy is None:
			return random.choice(legal_actions) if legal_actions else 0

		if self.use_masking and action_mask is not None:
			action, _ = self.policy.predict(
				obs, action_masks=action_mask, deterministic=False
			)
		else:
			action, _ = self.policy.predict(obs, deterministic=False)

		if action not in legal_actions and legal_actions:
			action = random.choice(legal_actions)

		return action

	def update(self, *args, **kwargs):
		pass


class MultiAgentEnv:
	"""
	多智能体环境

	每个智能体独立行动，环境返回所有智能体的观察和奖励
	"""

	def __init__(self, config: MultiAgentConfig = None):
		if not GYM_AVAILABLE:
			raise ImportError("gymnasium or gym is required")

		self.config = config or MultiAgentConfig()
		self.sgs_config = self.config.sgs_config

		self.state_encoder = StateEncoder()
		self.action_encoder = ActionEncoder()
		self.reward_systems: List[RewardSystem] = []

		self.base_env = SGSEnv(self.sgs_config)

		self.agents: List[Any] = []
		self._setup_agents()

		self.engine: Optional[GameEngine] = None
		self.players: List[Player] = []
		self.current_agent_idx: int = 0

		self._step_count: int = 0
		self._done: bool = False
		self._winner: Optional[str] = None

	def _setup_agents(self):
		self.agents = []

		for agent_config in self.config.agent_configs:
			if agent_config.agent_type == AgentType.RANDOM:
				self.agents.append(RandomAgent(agent_config.player_idx))
			elif agent_config.agent_type == AgentType.RULE_BASED:
				self.agents.append(RuleBasedAgent(agent_config.player_idx))
			elif agent_config.agent_type == AgentType.RL_POLICY:
				if agent_config.policy_path:
					try:
						from stable_baselines3 import PPO

						policy = PPO.load(agent_config.policy_path)
						self.agents.append(PolicyAgent(policy, agent_config.player_idx))
					except ImportError:
						self.agents.append(RandomAgent(agent_config.player_idx))
				else:
					self.agents.append(RandomAgent(agent_config.player_idx))
			else:
				self.agents.append(RandomAgent(agent_config.player_idx))

		while len(self.agents) < self.sgs_config.player_num:
			self.agents.append(RandomAgent(len(self.agents)))

		self.reward_systems = [RewardSystem() for _ in self.agents]

	def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
		obs, info = self.base_env.reset(seed=seed)

		self.engine = self.base_env.engine
		self.players = self.base_env.players
		self.current_agent_idx = self.base_env.current_player_idx

		self._step_count = 0
		self._done = False
		self._winner = None

		for rs in self.reward_systems:
			rs.reset()

		observations = self._get_all_observations()
		infos = self._get_all_infos()

		return observations, infos

	def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
		if self._done:
			return (
				self._get_all_observations(),
				{},
				{"__all__": True},
				{},
				self._get_all_infos(),
			)

		active_agent_idx = self.current_agent_idx
		action = actions.get(active_agent_idx, 0)

		obs, reward, done, truncated, info = self.base_env.step(action)

		self._step_count += 1
		self._done = done or truncated

		if self._done:
			self._winner = info.get("winner")

		observations = self._get_all_observations()
		rewards = self._get_all_rewards(info)
		dones = {"__all__": self._done}
		truncations = {"__all__": truncated}
		infos = self._get_all_infos()

		self.current_agent_idx = self.base_env.current_player_idx

		return observations, rewards, dones, truncations, infos

	def _get_all_observations(self) -> Dict[int, Dict]:
		observations = {}

		for i, player in enumerate(self.players):
			if player.is_alive:
				obs = self._get_observation_for_agent(i)
				observations[i] = obs

		return observations

	def _get_observation_for_agent(self, agent_idx: int) -> Dict:
		state = self.base_env._get_game_state_dict()

		encoded_state = self.state_encoder.encode(state, agent_idx)

		if agent_idx == self.current_agent_idx:
			mask_type, mask_card, mask_target = self.base_env._get_action_masks()
		else:
			mask_type = np.zeros(self.action_encoder.action_type_dim, dtype=np.float32)
			mask_card = np.zeros(self.action_encoder.card_dim, dtype=np.float32)
			mask_target = np.zeros(self.action_encoder.target_dim, dtype=np.float32)

		return {
			"state": encoded_state,
			"action_mask_type": mask_type,
			"action_mask_card": mask_card,
			"action_mask_target": mask_target,
			"current_step": np.array([self.base_env.current_step], dtype=np.int32),
		}

	def _get_all_rewards(self, info: Dict) -> Dict[int, float]:
		rewards = {}

		if self._done and self._winner:
			for i, player in enumerate(self.players):
				identity = player.identity
				if IdentityRelationship.is_victory(identity, self._winner):
					rewards[i] = 10.0
				else:
					rewards[i] = -10.0
		else:
			for i in range(len(self.players)):
				rewards[i] = self.reward_systems[i].get_total_reward()

		return rewards

	def _get_all_infos(self) -> Dict[int, Dict]:
		infos = {}

		for i, player in enumerate(self.players):
			infos[i] = {
				"player_idx": i,
				"identity": player.identity,
				"is_alive": player.is_alive,
				"current_hp": player.current_hp,
				"hand_count": len(player.hand_cards),
			}

		infos["__all__"] = {
			"step_count": self._step_count,
			"round_num": self.base_env.round_count if self.base_env else 0,
			"current_agent_idx": self.current_agent_idx,
			"winner": self._winner,
		}

		return infos

	def get_legal_actions(self, agent_idx: int) -> List[int]:
		if agent_idx != self.current_agent_idx:
			return []

		return self.base_env.get_legal_actions()

	def get_action_mask(self, agent_idx: int) -> np.ndarray:
		if agent_idx != self.current_agent_idx:
			return np.zeros(
				self.action_encoder.get_action_space_dim(), dtype=np.float32
			)

		return self.base_env.action_masks()

	def render(self, mode: str = "human"):
		return self.base_env.render(mode)

	def close(self):
		self.base_env.close()

	@property
	def num_agents(self) -> int:
		return len(self.players) if self.players else self.sgs_config.player_num

	@property
	def observation_space(self):
		return self.base_env.observation_space

	@property
	def action_space(self):
		return self.base_env.action_space


class SelfPlayEnv(MultiAgentEnv):
	"""
	自博弈环境

	训练智能体对抗历史版本的自己或其他策略
	"""

	def __init__(
			self,
			config: MultiAgentConfig = None,
			training_agent_idx: int = 0,
			opponent_pool: List[Any] = None,
	):
		super().__init__(config)

		self.training_agent_idx = training_agent_idx
		self.opponent_pool = opponent_pool or []

	def set_opponent_pool(self, pool: List[Any]):
		self.opponent_pool = pool

	def sample_opponents(self) -> List[Any]:
		opponents = []

		for i, agent_config in enumerate(self.config.agent_configs):
			if i == self.training_agent_idx:
				opponents.append(None)
			elif self.opponent_pool:
				opponents.append(random.choice(self.opponent_pool))
			else:
				opponents.append(RandomAgent(i))

		return opponents

	def step_with_policy(self, policy=None) -> Tuple[Dict, float, bool, bool, Dict]:
		if self._done:
			obs = self._get_observation_for_agent(self.training_agent_idx)
			return obs, 0.0, True, False, {"winner": self._winner}

		current_idx = self.current_agent_idx

		if current_idx == self.training_agent_idx and policy is not None:
			obs = self._get_observation_for_agent(current_idx)
			legal_actions = self.get_legal_actions(current_idx)
			action_mask = self.get_action_mask(current_idx)

			if hasattr(policy, "predict"):
				if hasattr(policy, "use_masking") and policy.use_masking:
					action, _ = policy.predict(
						obs, action_masks=action_mask, deterministic=False
					)
				else:
					action, _ = policy.predict(obs, deterministic=False)

				if action not in legal_actions and legal_actions:
					action = random.choice(legal_actions)
			else:
				action = random.choice(legal_actions) if legal_actions else 0
		else:
			agent = self.agents[current_idx]
			obs = self._get_observation_for_agent(current_idx)
			legal_actions = self.get_legal_actions(current_idx)
			action_mask = self.get_action_mask(current_idx)

			if hasattr(agent, "get_action"):
				action = agent.get_action(obs, legal_actions, action_mask)
			else:
				action = random.choice(legal_actions) if legal_actions else 0

		actions = {current_idx: action}
		observations, rewards, dones, truncations, infos = self.step(actions)

		train_obs = observations.get(self.training_agent_idx, {})
		train_reward = rewards.get(self.training_agent_idx, 0.0)
		done = dones.get("__all__", False)
		info = infos.get("__all__", {})

		return train_obs, train_reward, done, False, info


def make_multi_agent_env(
		player_num: int = 5,
		training_agent_idx: int = 0,
		opponent_pool: List[Any] = None,
) -> SelfPlayEnv:
	sgs_config = SGSConfig(player_num=player_num)

	agent_configs = [
		AgentConfig(
			agent_type=AgentType.RL_POLICY
			if i == training_agent_idx
			else AgentType.RANDOM,
			player_idx=i,
		)
		for i in range(player_num)
	]

	multi_config = MultiAgentConfig(
		sgs_config=sgs_config,
		agent_configs=agent_configs,
		training_agent_idx=training_agent_idx,
	)

	return SelfPlayEnv(
		config=multi_config,
		training_agent_idx=training_agent_idx,
		opponent_pool=opponent_pool,
	)
