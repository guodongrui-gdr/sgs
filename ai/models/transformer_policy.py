"""
Transformer策略网络 - 使用注意力机制处理游戏状态

架构:
1. 输入拆分为多个token序列 (全局状态、手牌、玩家、历史)
2. Transformer Encoder处理序列
3. 策略头和价值头输出

状态序列化设计:
- 全局token: 1个 (游戏阶段、回合数等)
- 当前玩家token: 1个 (HP、装备等)
- 手牌tokens: 最多20个
- 其他玩家tokens: 最多4个
- 历史动作tokens: 最多10个
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import math

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.policies import ActorCriticPolicy

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseFeaturesExtractor = nn.Module
    ActorCriticPolicy = object


@dataclass
class TransformerConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: int = 0.1
    activation: str = "gelu"

    max_hand_cards: int = 20
    max_players: int = 5
    max_history: int = 10

    global_dim: int = 34
    player_dim: int = 136
    card_dim: int = 81
    action_dim: int = 78

    output_dim: int = 256


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer特征提取器

    将游戏状态转换为token序列，通过Transformer编码
    """

    def __init__(
        self,
        observation_space,
        config: TransformerConfig = None,
        state_dim: int = 3000,
    ):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required")

        self.config = config or TransformerConfig()
        self.state_dim = state_dim

        dummy_features_dim = self.config.output_dim
        super().__init__(observation_space, features_dim=dummy_features_dim)

        c = self.config

        self.global_embed = TokenEmbedding(c.global_dim, c.d_model)
        self.player_embed = TokenEmbedding(c.player_dim, c.d_model)
        self.card_embed = TokenEmbedding(c.card_dim, c.d_model)
        self.action_embed = TokenEmbedding(c.action_dim, c.d_model)

        self.token_type_embed = nn.Embedding(5, c.d_model)
        self.positional_encoding = PositionalEncoding(c.d_model, dropout=c.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            activation=c.activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=c.num_encoder_layers,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(c.d_model, c.d_model),
            nn.GELU(),
            nn.LayerNorm(c.d_model),
        )

        self._features_dim = c.output_dim

    def _parse_state(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将观察字典解析为token序列

        返回:
            tokens: (batch, seq_len, d_model)
            mask: (batch, seq_len) - True表示有效token
        """
        c = self.config
        batch_size = obs["state"].shape[0] if "state" in obs else 1

        if "state" in obs:
            state = obs["state"]
        else:
            state = obs.get("observation", torch.zeros(batch_size, self.state_dim))

        device = state.device
        tokens_list = []
        token_types = []
        mask_list = []

        global_state = state[:, : c.global_dim]
        global_token = self.global_embed(global_state).unsqueeze(1)
        tokens_list.append(global_token)
        token_types.append(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
        mask_list.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        player_start = c.global_dim
        player_end = player_start + c.player_dim
        current_player = state[:, player_start:player_end]
        player_token = self.player_embed(current_player).unsqueeze(1)
        tokens_list.append(player_token)
        token_types.append(torch.ones(batch_size, 1, dtype=torch.long, device=device))
        mask_list.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        card_start = player_end
        card_dim = c.card_dim
        for i in range(c.max_hand_cards):
            start = card_start + i * card_dim
            end = start + card_dim
            card_state = state[:, start:end]
            card_token = self.card_embed(card_state).unsqueeze(1)
            tokens_list.append(card_token)
            token_types.append(
                torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
            )
            mask_list.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        other_player_start = card_start + c.max_hand_cards * card_dim
        for i in range(c.max_players - 1):
            start = other_player_start + i * c.player_dim
            end = start + c.player_dim
            other_player = state[:, start:end]
            other_token = self.player_embed(other_player).unsqueeze(1)
            tokens_list.append(other_token)
            token_types.append(
                torch.full((batch_size, 1), 3, dtype=torch.long, device=device)
            )
            mask_list.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        tokens = torch.cat(tokens_list, dim=1)
        token_types = torch.cat(token_types, dim=1)
        mask = torch.cat(mask_list, dim=1)

        type_embed = self.token_type_embed(token_types)
        tokens = tokens + type_embed

        return tokens, mask

    def forward(self, observations: Dict) -> torch.Tensor:
        tokens, mask = self._parse_state(observations)

        tokens = self.positional_encoding(tokens)

        encoded = self.transformer_encoder(tokens, src_key_padding_mask=~mask)

        pooled = encoded[:, 0, :]

        output = self.output_proj(pooled)

        return output


class TransformerPolicy(ActorCriticPolicy):
    """
    使用Transformer的Actor-Critic策略
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        config: TransformerConfig = None,
        *args,
        **kwargs,
    ):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required")

        self.transformer_config = config or TransformerConfig()

        kwargs["features_extractor_class"] = TransformerFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {
            "config": self.transformer_config,
            "state_dim": kwargs.pop("state_dim", 3000),
        }

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def forward(
        self,
        obs: Dict,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播，支持动作掩码
        """
        features = self.extract_features(obs)

        latent_pi = self.mlp_extractor.forward_actor(features)
        latent_vf = self.mlp_extractor.forward_critic(features)

        values = self.value_net(latent_vf)
        logits = self.action_net(latent_pi)

        if action_masks is not None:
            logits = logits.clone()
            logits[action_masks == 0] = float("-inf")

        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.distribution.logits = logits

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def predict(
        self,
        observation: Dict,
        state: Optional[Tuple] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        预测动作，支持动作掩码
        """
        self.set_training_mode(False)

        obs_tensor, vectorized_env = self.prepare_obs(observation)

        with torch.no_grad():
            actions, values, log_prob = self.forward(
                obs_tensor, deterministic=deterministic, action_masks=action_masks
            )

        actions = actions.cpu().numpy()

        if isinstance(self.action_space, dict):
            actions = {
                key: actions[i] for i, key in enumerate(self.action_space.keys())
            }

        return actions, state


def create_transformer_policy_kwargs(
    config: TransformerConfig = None,
    state_dim: int = 3000,
) -> Dict[str, Any]:
    """
    创建Transformer策略的policy_kwargs
    """
    config = config or TransformerConfig()

    return {
        "policy_class": TransformerPolicy,
        "features_extractor_class": TransformerFeaturesExtractor,
        "features_extractor_kwargs": {
            "config": config,
            "state_dim": state_dim,
        },
        "state_dim": state_dim,
    }
