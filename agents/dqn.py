"""Masked dueling double DQN.

Infeasible actions are excluded everywhere a maximum over actions is taken: when acting
(greedy and exploratory) and when forming the double-DQN bootstrap target, using the
feasibility mask of the next state stored in replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.networks import build_q_network

NEG = -1e9


@dataclass
class DQNConfig:
    arch: str = "decomposed"          # "decomposed", "shared" or "flat"
    hidden: int = 256
    dueling: bool = True
    double: bool = True
    masked: bool = True               # False gives the vanilla ablation (invalid actions are penalised by the env)
    gamma: float = 0.99
    n_step: int = 3
    lr: float = 2.5e-4
    lr_final: float = -1.0            # >= 0: decay the learning rate linearly to this value by the end of training
    batch_size: int = 64
    buffer_size: int = 200_000
    learning_starts: int = 10_000
    train_every: int = 4              # environment steps per gradient update
    n_envs: int = 8                   # environments stepped together when collecting experience
    target_update: int = 2_000        # environment steps between target-network syncs (when tau == 1)
    tau: float = 1.0                  # < 1: soft target update after every gradient step
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3
    explore_defer_prob: float = 0.0   # share of exploratory actions that defer instead of sampling uniformly
    cost_ratios: tuple = (0.01, 0.1, 1.0)   # inspection / minor / major cost relative to major (decomposed network)
    demo_policy: str = ""             # name of a tuned rule to learn from at the start ("" = learn from scratch)
    demo_episodes: int = 200
    demo_pretrain_updates: int = 20_000
    demo_ratio: float = 0.25          # share of each batch drawn from demonstrations during RL
    margin: float = 0.8               # large-margin loss: demonstrated actions must beat others by this much
    margin_weight: float = 1.0
    margin_decay_steps: int = 200_000 # margin weight fades linearly over these RL steps ...
    margin_floor: float = 0.0         # ... down to this fraction of its initial value
    per: bool = False                 # prioritised replay
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    max_grad_norm: float = 10.0
    huber_delta: float = 1.0
    total_steps: int = 1_000_000
    eval_every: int = 25_000
    extra: dict = field(default_factory=dict)

    def epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.eps_fraction * self.total_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, n_bridges: int, cfg: DQNConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.obs_dim, self.n_actions, self.n_bridges = obs_dim, n_actions, n_bridges
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.decomposed = cfg.arch == "decomposed"
        self.online = build_q_network(cfg.arch, obs_dim, n_actions, n_bridges, cfg.hidden, cfg.dueling, tuple(cfg.cost_ratios))
        self.target = build_q_network(cfg.arch, obs_dim, n_actions, n_bridges, cfg.hidden, cfg.dueling, tuple(cfg.cost_ratios))
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        try:
            self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr, fused=True)
        except (RuntimeError, TypeError):
            self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.updates = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
        return self.online(x).numpy()

    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        return int(self.act_batch(obs[None, :], mask[None, :], epsilon)[0])

    def act_batch(self, obs: np.ndarray, masks: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Epsilon-greedy actions for a batch of states; exploration samples uniformly among allowed actions."""
        allowed = masks if self.cfg.masked else np.ones_like(masks)
        actions = np.argmax(np.where(allowed, self.q_values(obs), NEG), axis=1)
        if epsilon > 0.0:
            explore = self.rng.random(len(actions)) < epsilon
            for k in np.flatnonzero(explore):
                if self.rng.random() < self.cfg.explore_defer_prob:
                    actions[k] = 0
                else:
                    actions[k] = self.rng.choice(np.flatnonzero(allowed[k]))
        return actions

    def policy(self):
        """Greedy policy in the evaluation interface `policy(env, obs) -> action`."""
        return lambda env, obs: self.act(obs, env.action_masks(), 0.0)

    # ------------------------------------------------------------------ learning
    def update(self, batch: dict[str, np.ndarray], margin_weight: float = 0.0) -> float:
        obs = torch.as_tensor(batch["obs"])
        action = torch.as_tensor(batch["action"])
        reward = torch.as_tensor(batch["reward"])
        next_obs = torch.as_tensor(batch["next_obs"])
        discount = torch.as_tensor(batch["discount"])
        next_mask = torch.as_tensor(batch["next_mask"])
        if not self.cfg.masked:
            next_mask = torch.ones_like(next_mask)

        if self.decomposed:
            # One TD target per bridge component, all evaluated at the joint greedy next action.
            with torch.no_grad():
                if "next_action" in batch:   # evaluate a given policy instead of the greedy one
                    choice = torch.as_tensor(batch["next_action"])
                else:
                    choice = self.online(next_obs).masked_fill(~next_mask, NEG).argmax(dim=1)
                comp_next = self.target.components(next_obs)
                index = choice[:, None, None].expand(-1, comp_next.shape[1], 1)
                target = reward + discount[:, None] * comp_next.gather(2, index).squeeze(2)
            comp = self.online.components(obs)
            q = comp.gather(2, action[:, None, None].expand(-1, comp.shape[1], 1)).squeeze(2)
            elementwise = F.smooth_l1_loss(q, target, beta=self.cfg.huber_delta, reduction="none").sum(dim=1)
            q, target = q.sum(dim=1), target.sum(dim=1)
        else:
            with torch.no_grad():
                q_next_target = self.target(next_obs)
                if "next_action" in batch:
                    choice = torch.as_tensor(batch["next_action"])[:, None]
                elif self.cfg.double:
                    choice = self.online(next_obs).masked_fill(~next_mask, NEG).argmax(dim=1, keepdim=True)
                else:
                    choice = q_next_target.masked_fill(~next_mask, NEG).argmax(dim=1, keepdim=True)
                target = reward + discount * q_next_target.gather(1, choice).squeeze(1)
            q = self.online(obs).gather(1, action[:, None]).squeeze(1)
            elementwise = F.smooth_l1_loss(q, target, beta=self.cfg.huber_delta, reduction="none")
        if "weights" in batch:
            loss = (torch.as_tensor(batch["weights"]) * elementwise).mean()
        else:
            loss = elementwise.mean()
        if margin_weight > 0.0 and "is_demo" in batch:
            # Large-margin loss on demonstrated transitions: max_a [Q(s,a) + m*1(a != a_demo)] - Q(s, a_demo).
            is_demo = torch.as_tensor(batch["is_demo"], dtype=torch.float32)
            mask = torch.as_tensor(batch["mask"])
            q_all = self.online(obs)
            bonus = torch.full_like(q_all, self.cfg.margin).scatter(1, action[:, None], 0.0)
            best = (q_all + bonus).masked_fill(~mask, NEG).amax(dim=1)
            q_demo = q_all.gather(1, action[:, None]).squeeze(1)
            loss = loss + margin_weight * ((best - q_demo) * is_demo).sum() / is_demo.sum().clamp(min=1.0)
        self.last_td_error = (q - target).detach().numpy()
        if self.cfg.lr_final >= 0.0:
            total_updates = max(1, self.cfg.total_steps // self.cfg.train_every)
            frac = min(1.0, self.updates / total_updates)
            for group in self.optim.param_groups:
                group["lr"] = self.cfg.lr + frac * (self.cfg.lr_final - self.cfg.lr)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.max_grad_norm, foreach=True)
        self.optim.step()

        self.updates += 1
        if self.cfg.tau < 1.0:
            with torch.no_grad():
                for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
                    t_param.lerp_(o_param, self.cfg.tau)
        elif self.updates % max(1, self.cfg.target_update // self.cfg.train_every) == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.online.state_dict(),
                "config": asdict(self.cfg),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "n_bridges": self.n_bridges,
                "meta": meta or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DQNAgent":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)   # tensors and plain values only
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], ckpt["n_bridges"], DQNConfig(**ckpt["config"]))
        agent.online.load_state_dict(ckpt["state_dict"])
        agent.target.load_state_dict(ckpt["state_dict"])
        agent.meta = ckpt.get("meta", {})
        return agent
