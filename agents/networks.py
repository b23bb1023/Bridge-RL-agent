"""Q-networks for the 31-action bridge maintenance problem.

FlatQNetwork: an MLP over the full 103-dim observation.
SharedBridgeQNetwork: one encoder shared by all bridges. Each bridge's inspect / minor / major
advantages come from its own embedding plus a pooled network context, so what is learned about
one bridge transfers to the others, and relabelling the bridges relabels the actions consistently.
DecomposedQNetwork: the same shared encoder, but the value is split into one cost-to-go per bridge
linked by a learned price of money (see the class docstring).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

FEATURES_PER_BRIDGE = 10
GLOBAL_FEATURES = 3


def mlp(sizes: list[int], layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if k < len(sizes) - 2:
            if layer_norm and k == 0:
                layers.append(nn.LayerNorm(b))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def combine(value: torch.Tensor | None, advantage: torch.Tensor) -> torch.Tensor:
    if value is None:
        return advantage
    return value + advantage - advantage.mean(dim=1, keepdim=True)


class FlatQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, dueling: bool = True) -> None:
        super().__init__()
        self.trunk = mlp([obs_dim, hidden, hidden], layer_norm=True)
        self.trunk.append(nn.ReLU())
        self.advantage = mlp([hidden, hidden // 2, n_actions])
        self.value = mlp([hidden, hidden // 2, 1]) if dueling else None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        return combine(self.value(h) if self.value is not None else None, self.advantage(h))


class SharedBridgeQNetwork(nn.Module):
    def __init__(self, n_bridges: int, emb: int = 64, hidden: int = 128, dueling: bool = True) -> None:
        super().__init__()
        self.n = n_bridges
        self.encoder = mlp([FEATURES_PER_BRIDGE + GLOBAL_FEATURES, emb, emb], layer_norm=True)
        self.encoder.append(nn.ReLU())
        self.context = mlp([2 * emb + GLOBAL_FEATURES, hidden, hidden])
        self.context.append(nn.ReLU())
        self.bridge_head = mlp([emb + hidden, emb, 3])
        self.defer_head = nn.Linear(hidden, 1)
        self.value = mlp([hidden, emb, 1]) if dueling else None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        per = obs[:, : self.n * FEATURES_PER_BRIDGE].reshape(batch, self.n, FEATURES_PER_BRIDGE)
        glob = obs[:, self.n * FEATURES_PER_BRIDGE :]
        e = self.encoder(torch.cat([per, glob[:, None, :].expand(-1, self.n, -1)], dim=-1))
        ctx = self.context(torch.cat([e.mean(dim=1), e.amax(dim=1), glob], dim=-1))
        per_bridge = self.bridge_head(torch.cat([e, ctx[:, None, :].expand(-1, self.n, -1)], dim=-1))
        # Action order: defer, inspect x N, minor x N, major x N.
        advantage = torch.cat([self.defer_head(ctx), per_bridge.transpose(1, 2).reshape(batch, 3 * self.n)], dim=1)
        return combine(self.value(ctx) if self.value is not None else None, advantage)


class DecomposedQNetwork(nn.Module):
    """Per-bridge decomposition of the action value: Q(s, a) = sum_i Q_i(s, a).

    Q_i is bridge i's own discounted cost-to-go. A work order on bridge i acts on it directly
    (one learned value per work type); a work order on another bridge j affects bridge i only by
    spending money, priced by a learned non-negative shadow price lambda_i(s):
        Q_i(s, defer)          = base_i(s)
        Q_i(s, work k on i)    = own_i(s, k)
        Q_i(s, work k on j!=i) = base_i(s) - lambda_i(s) * cost(k, j)
    Each Q_i is trained on bridge i's share of the reward, with the joint greedy action as target.
    """

    def __init__(self, n_bridges: int, cost_ratios=(0.01, 0.1, 1.0), emb: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.n = n_bridges
        self.encoder = mlp([FEATURES_PER_BRIDGE + GLOBAL_FEATURES, emb, emb], layer_norm=True)
        self.encoder.append(nn.ReLU())
        self.context = mlp([2 * emb + GLOBAL_FEATURES, hidden, hidden])
        self.context.append(nn.ReLU())
        self.head = mlp([emb + hidden, emb, 5])
        self.register_buffer("cost_ratios", torch.tensor(cost_ratios, dtype=torch.float32))
        own = torch.zeros(3, n_bridges, 1 + 3 * n_bridges)
        for k in range(3):
            own[k, torch.arange(n_bridges), 1 + k * n_bridges + torch.arange(n_bridges)] = 1.0
        self.register_buffer("own_mask", own)

    def components(self, obs: torch.Tensor) -> torch.Tensor:
        batch, n = obs.shape[0], self.n
        per = obs[:, : n * FEATURES_PER_BRIDGE].reshape(batch, n, FEATURES_PER_BRIDGE)
        glob = obs[:, n * FEATURES_PER_BRIDGE :]
        e = self.encoder(torch.cat([per, glob[:, None, :].expand(-1, n, -1)], dim=-1))
        ctx = self.context(torch.cat([e.mean(dim=1), e.amax(dim=1), glob], dim=-1))
        h = self.head(torch.cat([e, ctx[:, None, :].expand(-1, n, -1)], dim=-1))
        base, own, price = h[..., 0], h[..., 1:4], F.softplus(h[..., 4])

        # Cost of every action in $M, recovered from the observed (log-scaled) major-repair cost.
        major = (torch.pow(10.0, 4.5 * per[..., 8]) - 1.0) / 1e3
        cost = torch.cat([torch.zeros(batch, 1, device=obs.device),
                          (major[:, None, :] * self.cost_ratios[None, :, None]).reshape(batch, 3 * n)], dim=1)
        q = base[..., None] - price[..., None] * cost[:, None, :]
        own_mask = self.own_mask.sum(dim=0)
        own_values = torch.einsum("bik,kia->bia", own, self.own_mask)
        return q * (1.0 - own_mask) + own_values

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.components(obs).sum(dim=1)


def build_q_network(arch: str, obs_dim: int, n_actions: int, n_bridges: int, hidden: int, dueling: bool,
                    cost_ratios=(0.01, 0.1, 1.0)) -> nn.Module:
    if arch == "flat":
        return FlatQNetwork(obs_dim, n_actions, hidden=hidden, dueling=dueling)
    if arch == "shared":
        return SharedBridgeQNetwork(n_bridges, hidden=hidden // 2, dueling=dueling)
    if arch == "decomposed":
        return DecomposedQNetwork(n_bridges, cost_ratios, hidden=hidden // 2)
    raise ValueError(f"unknown architecture {arch!r}")
