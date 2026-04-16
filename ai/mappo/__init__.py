"""
MAPPO (Multi-Agent PPO) Module

Implements vanilla MAPPO with centralized critic and decentralized actors.

Key Components:
- CentralizedCritic: Shared critic that sees global state + all agent actions
- MAPPOActor: Decentralized actor that sees local observation only
- MAPPOAgent: Coordinator for 5 actors + 1 shared critic
- TeamRewardAllocator: Team-based reward allocation with advantage decomposition

Architecture:
- Centralized Critic: global_state (2670*5) + joint_actions (5 agents) -> value
- Decentralized Actor: local_obs (2670) -> action_logits (20)
- Team Reward Allocation: counterfactual baseline (COMA-style) for team coordination

Reference:
- Yu et al., 2022 "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
- Foerster et al., 2018 "Counterfactual Multi-Agent Policy Gradients" (COMA)
"""

from .centralized_critic import CentralizedCritic, CentralizedCriticConfig
from .mappo_policy import MAPPOActor, MAPPOActorConfig
from .mappo_agent import MAPPOAgent, MAPPOAgentConfig
from .team_rewards import (
    TeamRewardAllocator,
    TeamRewardConfig,
    Team,
    TeamState,
    ActionContext,
    create_team_reward_allocator,
)

__all__ = [
    "CentralizedCritic",
    "CentralizedCriticConfig",
    "MAPPOActor",
    "MAPPOActorConfig",
    "MAPPOAgent",
    "MAPPOAgentConfig",
    "TeamRewardAllocator",
    "TeamRewardConfig",
    "Team",
    "TeamState",
    "ActionContext",
    "create_team_reward_allocator",
]
