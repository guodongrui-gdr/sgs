"""
Team Reward Allocation for MAPPO Phase 3

Implements team-based reward allocation with advantage decomposition
for the SGS (三国杀) multi-agent reinforcement learning system.

Key Components:
- TeamRewardAllocator: Maps identities to teams, computes team bonuses
- Counterfactual Baseline: COMA-style advantage decomposition
- Coordination Incentives: Bonuses for coordinated team actions

Teams in SGS:
- Lord Team (主公队): 主公 + 忠臣 - shared victory
- Rebel Team (反贼队): 反贼 players - shared victory
- Spy (内奸): Independent agent - unique objective

Reference:
- Foerster et al., 2018 "Counterfactual Multi-Agent Policy Gradients" (COMA)
- Yu et al., 2022 "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

import numpy as np

if TYPE_CHECKING:
    from player.player import Player

logger = logging.getLogger(__name__)


class Team(Enum):
    """Team enumeration for SGS factions"""

    LORD = "lord_team"  # 主公 + 忠臣
    REBEL = "rebel_team"  # 反贼
    SPY = "spy"  # 内奸 (independent, not a team)
    UNKNOWN = "unknown"  # Identity not yet revealed


@dataclass
class TeamRewardConfig:
    """Configuration for team reward allocation"""

    # Team coordination bonuses
    lord_loyalist_coordination_bonus: float = 2.0  # 主公+忠臣协作奖励
    rebel_focus_fire_bonus: float = (
        1.5  # 反贼集火奖励 (multiple rebels attacking same target)
    )
    protect_lord_bonus: float = 3.0  # 忠臣保护主公奖励

    # Team damage bonuses/penalties
    team_damage_bonus_multiplier: float = 0.3  # Team damage contribution bonus
    harm_team_penalty: float = -5.0  # Penalty for harming teammates

    # Spy (内奸) specific
    spy_balance_bonus: float = 1.0  # Reward for maintaining balance
    spy_no_team_bonus: float = 0.0  # No team bonus for spy (independent objective)

    # Counterfactual baseline parameters
    counterfactual_samples: int = 10  # Number of samples for baseline estimation
    baseline_decay: float = 0.95  # Decay factor for baseline smoothing

    # Team size dynamics
    dynamic_team_weight: bool = True  # Adjust rewards based on alive team members
    team_size_penalty_factor: float = 0.1  # Penalty when team size decreases

    # Reward scaling
    max_team_bonus: float = 10.0  # Maximum team bonus per step
    team_bonus_clip: float = 5.0  # Clip team bonuses


@dataclass
class ActionContext:
    """Context for an agent's action"""

    agent_idx: int
    agent_identity: str
    action_type: str  # "attack", "defend", "heal", "skill", etc.
    target_idx: Optional[int] = None
    target_identity: Optional[str] = None
    damage_value: float = 0.0
    card_type: Optional[str] = None
    is_effective: bool = True  # Whether action was effective
    assisted_lord: bool = False  # Whether action helped 主公
    focused_target: bool = False  # Whether targeting same as other rebels


@dataclass
class TeamState:
    """Current state of teams"""

    lord_alive: bool = True
    loyalists_alive: List[int] = field(default_factory=list)
    rebels_alive: List[int] = field(default_factory=list)
    spy_alive: bool = True

    def get_team_size(self, team: Team) -> int:
        """Get current size of a team"""
        if team == Team.LORD:
            return (
                1 + len(self.loyalists_alive)
                if self.lord_alive
                else len(self.loyalists_alive)
            )
        elif team == Team.REBEL:
            return len(self.rebels_alive)
        elif team == Team.SPY:
            return 1 if self.spy_alive else 0
        return 0

    def get_team_members(self, team: Team) -> List[int]:
        """Get indices of alive team members"""
        if team == Team.LORD:
            members = []
            # Note: 主公 index is typically 0
            if self.lord_alive:
                members.append(0)  # Assuming 主公 is always index 0
            members.extend(self.loyalists_alive)
            return members
        elif team == Team.REBEL:
            return self.rebels_alive.copy()
        elif team == Team.SPY:
            return []  # Spy is independent, no team members
        return []


class TeamRewardAllocator:
    """
    Team-based reward allocator for MAPPO

    Handles:
    1. Identity to team mapping
    2. Team coordination bonuses
    3. Counterfactual advantage decomposition
    4. Dynamic team size handling
    5. Spy (内奸) special case
    """

    def __init__(self, config: Optional[TeamRewardConfig] = None, num_agents: int = 5):
        self.config = config or TeamRewardConfig()
        self.num_agents = num_agents

        # Team mappings
        self._team_map: Dict[int, Team] = {}
        self._identity_map: Dict[int, str] = {}

        # Team state tracking
        self._team_state: Optional[TeamState] = None

        # Counterfactual baseline tracking (per agent)
        self._baseline_values: Dict[int, float] = {}
        self._baseline_history: Dict[int, List[float]] = {}

        # Coordination tracking
        self._recent_targets: Dict[
            int, List[int]
        ] = {}  # agent_idx -> recent target indices
        self._coordination_events: List[Dict] = []

        # Reward accumulation
        self._team_rewards: Dict[int, float] = {}
        self._individual_rewards: Dict[int, float] = {}

    def initialize_teams(self, players: List[Player]) -> None:
        """
        Initialize team mappings from player identities

        Args:
            players: List of Player objects with identity assigned
        """
        self._team_map.clear()
        self._identity_map.clear()

        lord_idx = None
        loyalists = []
        rebels = []
        spy_idx = None

        for i, player in enumerate(players):
            identity = player.identity
            self._identity_map[i] = identity

            if identity == "主公":
                self._team_map[i] = Team.LORD
                lord_idx = i
            elif identity == "忠臣":
                self._team_map[i] = Team.LORD
                loyalists.append(i)
            elif identity == "反贼":
                self._team_map[i] = Team.REBEL
                rebels.append(i)
            elif identity == "内奸":
                self._team_map[i] = Team.SPY
                spy_idx = i
            else:
                self._team_map[i] = Team.UNKNOWN

        # Initialize team state
        self._team_state = TeamState(
            lord_alive=lord_idx is not None,
            loyalists_alive=loyalists,
            rebels_alive=rebels,
            spy_alive=spy_idx is not None,
        )

        # Initialize baselines
        for i in range(len(players)):
            self._baseline_values[i] = 0.0
            self._baseline_history[i] = []
            self._recent_targets[i] = []

        logger.debug(
            f"[TeamRewardAllocator] Teams initialized: "
            f"Lord team: {[lord_idx] + loyalists if lord_idx else loyalists}, "
            f"Rebel team: {rebels}, Spy: {spy_idx}"
        )

    def get_team(self, agent_idx: int) -> Team:
        """Get team for an agent"""
        return self._team_map.get(agent_idx, Team.UNKNOWN)

    def get_identity(self, agent_idx: int) -> str:
        """Get identity string for an agent"""
        return self._identity_map.get(agent_idx, "")

    def is_spy(self, agent_idx: int) -> bool:
        """Check if agent is the spy (内奸)"""
        return self.get_team(agent_idx) == Team.SPY

    def is_team_member(self, agent_a: int, agent_b: int) -> bool:
        """Check if two agents are on the same team"""
        team_a = self.get_team(agent_a)
        team_b = self.get_team(agent_b)

        # Spy is not part of any team
        if team_a == Team.SPY or team_b == Team.SPY:
            return False

        return team_a == team_b and team_a != Team.UNKNOWN

    def update_team_state(self, players: List[Player]) -> None:
        """
        Update team state based on current player status

        Args:
            players: Current list of Player objects
        """
        if self._team_state is None:
            return

        loyalists_alive = []
        rebels_alive = []
        lord_alive = False
        spy_alive = False

        for i, player in enumerate(players):
            if not player.is_alive:
                continue

            identity = self._identity_map.get(i, player.identity)

            if identity == "主公":
                lord_alive = True
            elif identity == "忠臣":
                loyalists_alive.append(i)
            elif identity == "反贼":
                rebels_alive.append(i)
            elif identity == "内奸":
                spy_alive = True

        prev_lord_size = self._team_state.get_team_size(Team.LORD)
        prev_rebel_size = self._team_state.get_team_size(Team.REBEL)

        self._team_state.lord_alive = lord_alive
        self._team_state.loyalists_alive = loyalists_alive
        self._team_state.rebels_alive = rebels_alive
        self._team_state.spy_alive = spy_alive

        # Apply team size penalties if configured
        if self.config.dynamic_team_weight:
            curr_lord_size = self._team_state.get_team_size(Team.LORD)
            curr_rebel_size = self._team_state.get_team_size(Team.REBEL)

            if curr_lord_size < prev_lord_size:
                # Lord team member died
                for idx in self._team_state.get_team_members(Team.LORD):
                    if idx in self._team_rewards:
                        self._team_rewards[idx] -= (
                            self.config.team_size_penalty_factor * 5
                        )

            if curr_rebel_size < prev_rebel_size:
                # Rebel team member died
                for idx in self._team_state.get_team_members(Team.REBEL):
                    if idx in self._team_rewards:
                        self._team_rewards[idx] -= (
                            self.config.team_size_penalty_factor * 3
                        )

    def compute_team_bonus(self, action_context: ActionContext) -> float:
        """
        Compute team coordination bonus for an action

        Args:
            action_context: Context of the agent's action

        Returns:
            Team bonus value
        """
        agent_idx = action_context.agent_idx
        team = self.get_team(agent_idx)

        # Spy gets no team bonus (independent objective)
        if team == Team.SPY:
            return self._compute_spy_bonus(action_context)

        if team == Team.UNKNOWN:
            return 0.0

        bonus = 0.0

        # Lord team coordination
        if team == Team.LORD:
            bonus = self._compute_lord_team_bonus(action_context)

        # Rebel team coordination (focus fire)
        elif team == Team.REBEL:
            bonus = self._compute_rebel_team_bonus(action_context)

        # Clip bonus
        bonus = np.clip(
            bonus, -self.config.team_bonus_clip, self.config.team_bonus_clip
        )

        return bonus

    def _compute_lord_team_bonus(self, action_context: ActionContext) -> float:
        """Compute bonus for Lord team (主公+忠臣) coordination"""
        bonus = 0.0
        agent_identity = action_context.agent_identity

        # 忠臣 protecting 主公
        if agent_identity == "忠臣" and action_context.assisted_lord:
            bonus += self.config.protect_lord_bonus

        # 忠臣 attacking enemies together with 主公
        if agent_identity == "忠臣" and action_context.action_type == "attack":
            if action_context.target_idx is not None:
                target_team = self.get_team(action_context.target_idx)
                # Attacking enemy (rebel or spy)
                if target_team in [Team.REBEL, Team.SPY]:
                    # Check if 主公 recently attacked same target
                    lord_idx = 0  # Assuming 主公 is index 0
                    if lord_idx in self._recent_targets:
                        recent_targets = self._recent_targets[lord_idx]
                        if action_context.target_idx in recent_targets[-3:]:
                            bonus += self.config.lord_loyalist_coordination_bonus

        # 主公 attacking with 忠臣 support
        if agent_identity == "主公" and action_context.action_type == "attack":
            # Check if any 忠臣 attacked same target recently
            for loyalist_idx in (
                self._team_state.loyalists_alive if self._team_state else []
            ):
                if loyalist_idx in self._recent_targets:
                    recent_targets = self._recent_targets[loyalist_idx]
                    if action_context.target_idx in recent_targets[-3:]:
                        bonus += self.config.lord_loyalist_coordination_bonus * 0.5

        # Damage contribution to team effort
        if action_context.damage_value > 0 and action_context.is_effective:
            if action_context.target_idx is not None:
                target_team = self.get_team(action_context.target_idx)
                if target_team in [Team.REBEL, Team.SPY]:
                    bonus += (
                        action_context.damage_value
                        * self.config.team_damage_bonus_multiplier
                    )

        return bonus

    def _compute_rebel_team_bonus(self, action_context: ActionContext) -> float:
        """Compute bonus for Rebel team (反贼) coordination - focus fire"""
        bonus = 0.0

        # Focus fire: multiple rebels attacking same target
        if (
            action_context.action_type == "attack"
            and action_context.target_idx is not None
        ):
            target_idx = action_context.target_idx

            # Count how many other rebels attacked this target recently
            focus_count = 0
            for rebel_idx in self._team_state.rebels_alive if self._team_state else []:
                if rebel_idx != action_context.agent_idx:
                    if rebel_idx in self._recent_targets:
                        recent_targets = self._recent_targets[rebel_idx]
                        if target_idx in recent_targets[-5:]:
                            focus_count += 1

            # Focus fire bonus scales with number of coordinated attacks
            if focus_count >= 1:
                bonus += self.config.rebel_focus_fire_bonus * (1 + 0.5 * focus_count)

        # Damage contribution
        if action_context.damage_value > 0 and action_context.is_effective:
            if action_context.target_idx is not None:
                target_team = self.get_team(action_context.target_idx)
                if target_team == Team.LORD:
                    bonus += (
                        action_context.damage_value
                        * self.config.team_damage_bonus_multiplier
                        * 1.5
                    )
                elif target_team == Team.SPY:
                    bonus += (
                        action_context.damage_value
                        * self.config.team_damage_bonus_multiplier
                    )

        return bonus

    def _compute_spy_bonus(self, action_context: ActionContext) -> float:
        """Compute bonus for Spy (内奸) - independent objective, no team bonus"""
        # Spy gets no team coordination bonus
        # Instead, reward maintaining balance between factions

        bonus = 0.0

        if (
            action_context.action_type == "attack"
            and action_context.target_idx is not None
            and self._team_state is not None
        ):
            lord_size = self._team_state.get_team_size(Team.LORD)
            rebel_size = self._team_state.get_team_size(Team.REBEL)

            target_team = self.get_team(action_context.target_idx)

            # Balance bonus: attack the stronger faction
            if lord_size > rebel_size and target_team == Team.LORD:
                bonus += self.config.spy_balance_bonus
            elif rebel_size > lord_size and target_team == Team.REBEL:
                bonus += self.config.spy_balance_bonus

        # No team bonus for spy
        bonus += self.config.spy_no_team_bonus

        return bonus

    def compute_harm_penalty(self, action_context: ActionContext) -> float:
        """
        Compute penalty for harming teammates

        Args:
            action_context: Context of the agent's action

        Returns:
            Penalty value (negative)
        """
        if action_context.target_idx is None:
            return 0.0

        agent_idx = action_context.agent_idx
        target_idx = action_context.target_idx

        # Check if harming teammate
        if self.is_team_member(agent_idx, target_idx):
            return self.config.harm_team_penalty

        return 0.0

    def record_action(self, action_context: ActionContext) -> None:
        """
        Record an action for coordination tracking

        Args:
            action_context: Context of the agent's action
        """
        agent_idx = action_context.agent_idx

        # Track recent targets for focus fire detection
        if action_context.target_idx is not None:
            if agent_idx not in self._recent_targets:
                self._recent_targets[agent_idx] = []
            self._recent_targets[agent_idx].append(action_context.target_idx)

            # Keep only recent history (last 10 actions)
            if len(self._recent_targets[agent_idx]) > 10:
                self._recent_targets[agent_idx] = self._recent_targets[agent_idx][-10:]

        # Record coordination event
        self._coordination_events.append(
            {
                "agent_idx": agent_idx,
                "team": self.get_team(agent_idx).value,
                "target_idx": action_context.target_idx,
                "action_type": action_context.action_type,
                "damage": action_context.damage_value,
            }
        )

    def compute_counterfactual_baseline(
        self,
        agent_idx: int,
        team_reward: float,
        team_action_contributions: Dict[int, float],
    ) -> float:
        """
        Compute counterfactual baseline for advantage decomposition (COMA-style)

        The counterfactual baseline estimates what the team reward would be
        if this agent took a different action (average over possible actions).

        Advantage_i = TeamReward - Baseline_i

        Args:
            agent_idx: Agent to compute baseline for
            team_reward: Total team reward
            team_action_contributions: Dict mapping agent_idx to their contribution

        Returns:
            Counterfactual baseline value
        """
        team = self.get_team(agent_idx)

        # Spy uses individual baseline, not counterfactual
        if team == Team.SPY:
            return self._baseline_values.get(agent_idx, 0.0)

        # Get historical baseline
        if agent_idx in self._baseline_history:
            history = self._baseline_history[agent_idx]
            if len(history) > 0:
                # Weighted average with decay
                weights = [self.config.baseline_decay**i for i in range(len(history))]
                historical_baseline = sum(
                    h * w for h, w in zip(history, weights)
                ) / sum(weights)
            else:
                historical_baseline = 0.0
        else:
            historical_baseline = 0.0

        if self._team_state is None:
            return 0.0

        # Estimate counterfactual: what if this agent's contribution was average?
        team_members = self._team_state.get_team_members(team)
        if len(team_members) <= 1:
            return team_reward  # Single agent team

        other_contributions = [
            team_action_contributions.get(m, 0.0)
            for m in team_members
            if m != agent_idx
        ]

        if other_contributions:
            avg_other_contribution = np.mean(other_contributions)
        else:
            avg_other_contribution = 0.0

        # Counterfactual team reward if this agent had average contribution
        agent_contribution = team_action_contributions.get(agent_idx, 0.0)

        # Baseline = team_reward - (agent_contribution - avg_contribution)
        # This represents what team reward would be if agent was replaced by average agent
        counterfactual_baseline = (
            team_reward - agent_contribution + avg_other_contribution
        )

        # Blend with historical baseline
        blended_baseline = float(
            0.7 * counterfactual_baseline + 0.3 * historical_baseline
        )

        if agent_idx not in self._baseline_history:
            self._baseline_history[agent_idx] = []
        self._baseline_history[agent_idx].append(blended_baseline)

        if len(self._baseline_history[agent_idx]) > self.config.counterfactual_samples:
            self._baseline_history[agent_idx] = self._baseline_history[agent_idx][
                -self.config.counterfactual_samples :
            ]

        self._baseline_values[agent_idx] = blended_baseline

        return blended_baseline

    def decompose_team_reward(
        self, team_reward: float, agent_contributions: Dict[int, float], team: Team
    ) -> Dict[int, float]:
        """
        Decompose team reward into individual advantages using counterfactual baseline

        Advantage_i = TeamReward - Baseline_i

        Args:
            team_reward: Total team reward to decompose
            agent_contributions: Dict mapping agent_idx to their contribution estimate
            team: Which team this reward belongs to

        Returns:
            Dict mapping agent_idx to their decomposed reward (advantage)
        """
        decomposed = {}
        if self._team_state is None:
            return decomposed

        team_members = self._team_state.get_team_members(team)

        for agent_idx in team_members:
            baseline = self.compute_counterfactual_baseline(
                agent_idx, team_reward, agent_contributions
            )
            advantage = team_reward - baseline
            decomposed[agent_idx] = advantage

        return decomposed

    def allocate_rewards(
        self,
        base_rewards: Dict[int, float],
        action_contexts: List[ActionContext],
        players: List[Player],
    ) -> Dict[int, float]:
        """
        Allocate final rewards combining base rewards with team bonuses

        Args:
            base_rewards: Base rewards from existing reward system
            action_contexts: List of action contexts for this step
            players: Current player states

        Returns:
            Dict mapping agent_idx to final shaped reward
        """
        self.update_team_state(players)

        final_rewards = base_rewards.copy()

        # Compute team bonuses for each action
        team_bonuses: Dict[int, float] = {i: 0.0 for i in range(len(players))}
        team_contributions: Dict[int, float] = {i: 0.0 for i in range(len(players))}

        for ctx in action_contexts:
            self.record_action(ctx)

            agent_idx = ctx.agent_idx

            # Team coordination bonus
            team_bonus = self.compute_team_bonus(ctx)
            team_bonuses[agent_idx] += team_bonus

            # Harm teammate penalty
            harm_penalty = self.compute_harm_penalty(ctx)
            team_bonuses[agent_idx] += harm_penalty

            # Track contribution for decomposition
            team_contributions[agent_idx] += ctx.damage_value + team_bonus

        # Compute team-level rewards and decompose
        lord_team_reward = sum(
            team_bonuses[i]
            for i in (
                self._team_state.get_team_members(Team.LORD) if self._team_state else []
            )
        )
        rebel_team_reward = sum(
            team_bonuses[i]
            for i in (
                self._team_state.get_team_members(Team.REBEL)
                if self._team_state
                else []
            )
        )

        # Decompose Lord team reward
        lord_decomposed = self.decompose_team_reward(
            lord_team_reward, team_contributions, Team.LORD
        )

        # Decompose Rebel team reward
        rebel_decomposed = self.decompose_team_reward(
            rebel_team_reward, team_contributions, Team.REBEL
        )

        # Combine base rewards with decomposed team rewards
        for agent_idx in range(len(players)):
            team = self.get_team(agent_idx)

            if team == Team.LORD:
                final_rewards[agent_idx] += lord_decomposed.get(agent_idx, 0.0)
            elif team == Team.REBEL:
                final_rewards[agent_idx] += rebel_decomposed.get(agent_idx, 0.0)
            elif team == Team.SPY:
                # Spy gets individual team bonus (no decomposition)
                final_rewards[agent_idx] += team_bonuses.get(agent_idx, 0.0)

            # Clip final reward
            final_rewards[agent_idx] = np.clip(
                final_rewards[agent_idx],
                -self.config.max_team_bonus,
                self.config.max_team_bonus,
            )

        # Store for tracking
        self._team_rewards = {
            i: lord_decomposed.get(i, 0.0) + rebel_decomposed.get(i, 0.0)
            for i in range(len(players))
        }
        self._individual_rewards = final_rewards

        return final_rewards

    def get_team_rewards(self) -> Dict[int, float]:
        """Get current team rewards per agent"""
        return self._team_rewards.copy()

    def get_individual_rewards(self) -> Dict[int, float]:
        """Get current individual rewards per agent"""
        return self._individual_rewards.copy()

    def get_baseline_values(self) -> Dict[int, float]:
        """Get current counterfactual baseline values"""
        return self._baseline_values.copy()

    def get_coordination_stats(self) -> Dict:
        """Get coordination statistics"""
        lord_focus = 0
        rebel_focus = 0

        # Analyze recent coordination events
        recent_events = self._coordination_events[-20:]

        lord_targets = {}
        rebel_targets = {}

        for event in recent_events:
            target = event.get("target_idx")
            team = Team(event.get("team", "unknown"))

            if target is not None:
                if team == Team.LORD:
                    lord_targets[target] = lord_targets.get(target, 0) + 1
                elif team == Team.REBEL:
                    rebel_targets[target] = rebel_targets.get(target, 0) + 1

        # Focus fire score: max count of same target
        lord_focus = max(lord_targets.values()) if lord_targets else 0
        rebel_focus = max(rebel_targets.values()) if rebel_targets else 0

        return {
            "lord_focus_fire_count": lord_focus,
            "rebel_focus_fire_count": rebel_focus,
            "lord_team_size": self._team_state.get_team_size(Team.LORD)
            if self._team_state
            else 0,
            "rebel_team_size": self._team_state.get_team_size(Team.REBEL)
            if self._team_state
            else 0,
            "spy_alive": self._team_state.spy_alive if self._team_state else False,
            "coordination_events_count": len(self._coordination_events),
        }

    def reset(self) -> None:
        """Reset allocator state for new episode"""
        self._team_map.clear()
        self._identity_map.clear()
        self._team_state = None
        self._baseline_values.clear()
        self._baseline_history.clear()
        self._recent_targets.clear()
        self._coordination_events.clear()
        self._team_rewards.clear()
        self._individual_rewards.clear()

    def integrate_with_reward_system(
        self, base_reward: float, agent_idx: int, event_type: str, context: Dict
    ) -> Tuple[float, float]:
        """
        Integration interface with existing reward system

        Takes base reward from RewardSystem and adds team-based shaping.

        Args:
            base_reward: Base reward from ai/reward.py
            agent_idx: Agent index
            event_type: Event type from reward system
            context: Additional context (target, damage, etc.)

        Returns:
            Tuple of (shaped_reward, team_bonus)
        """
        team_bonus = 0.0

        # Create action context from event
        action_context = ActionContext(
            agent_idx=agent_idx,
            agent_identity=self.get_identity(agent_idx),
            action_type=event_type,
            target_idx=context.get("target_idx"),
            target_identity=context.get("target_identity"),
            damage_value=context.get("damage_value", 0.0),
            is_effective=context.get("is_effective", True),
            assisted_lord=context.get("assisted_lord", False),
        )

        # Compute team bonus
        team_bonus = self.compute_team_bonus(action_context)
        team_bonus += self.compute_harm_penalty(action_context)

        # Shaped reward
        shaped_reward = base_reward + team_bonus

        # Clip
        shaped_reward = np.clip(
            shaped_reward, -self.config.max_team_bonus, self.config.max_team_bonus
        )

        return shaped_reward, team_bonus


def create_team_reward_allocator(
    num_agents: int = 5, config_kwargs: Optional[Dict] = None
) -> TeamRewardAllocator:
    """
    Factory function to create TeamRewardAllocator

    Args:
        num_agents: Number of agents in the game
        config_kwargs: Optional configuration overrides

    Returns:
        TeamRewardAllocator instance
    """
    config = TeamRewardConfig(**(config_kwargs or {}))
    return TeamRewardAllocator(config=config, num_agents=num_agents)
